"""The autonomous audit agent (Algorithm 1 in the paper).

Given a model and the robustness a defense would *report* (the REFERENCE attack:
one adaptive white-box RIS run at M=128, 80 iters), the agent escalates the
*realizable* attack over a schedule of (aperture M, optimizer iters, restarts),
plus a decoupled transfer adversary crafted on the undefended victim. It keeps,
per user, the effective channel with the lowest achieved-SE ratio, so the audited
robustness is a genuine per-user worst case, not a best-of-means. Every reported
violation carries a per-user witness tuple <v, h_eff, w_pred, w_opt, SE-ratio>
that any third party can recompute. A feature-space PGD run gives an unrealizable
upper bound for context.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import rfcore as rf
from . import settings as S
from .oracle import (AttackResult, clean_eval, eval_effective, eval_feature,
                     per_user_se_ratio, tau_verdicts)


def ris_best(model, H, M, kappa=S.KAPPA, iters=80, restarts=1, seed=0,
             keep_theta=False):
    """Strongest per-user RIS attack over `restarts` random restarts: keep, per
    user, the effective channel with the lowest achieved-SE ratio. With
    `keep_theta` it also returns the per-user winning restart index, so a
    witness can name the exact geometry seed (seed + restart) that produced it
    — without this, released witnesses are not independently recomputable."""
    worst_H = worst_se = worst_theta = worst_r = None
    for r in range(restarts):
        g = rf.build_geometry(H, M, kappa, seed=seed + r)
        H_adv, theta = rf.ris_attack(model, g, iters=iters, lr=S.ATTACK_LR,
                                     seed=seed + r, return_theta=True)
        se = per_user_se_ratio(model, H_adv, rng=np.random.default_rng(seed))
        theta_np = theta.cpu().numpy() if keep_theta else None
        if worst_H is None:
            worst_H, worst_se, worst_theta = H_adv.copy(), se, theta_np
            worst_r = np.zeros(len(se), dtype=int)
        else:
            take = se < worst_se
            worst_H[take], worst_se[take] = H_adv[take], se[take]
            worst_r[take] = r
            if keep_theta:
                worst_theta[take] = theta_np[take]
    return (worst_H, worst_theta, worst_r) if keep_theta else worst_H


@dataclass
class ModelAudit:
    model_name: str
    scenario: str
    model_seed: int
    audit_seed: int
    clean: AttackResult
    reference: AttackResult        # claimed robustness (REFERENCE attack)
    battery: list[AttackResult]    # escalated realizable attacks + transfer
    feature_bound: AttackResult    # unrealizable upper bound (context only)
    per_user_audited: np.ndarray = field(default=None, repr=False)
    witnesses: list[dict] = field(default_factory=list, repr=False)

    @property
    def audited_se(self) -> float:
        """Per-user worst over the realizable battery, then population mean."""
        return float(self.per_user_audited.mean())

    @property
    def robustness_gap(self) -> float:
        return self.reference.se_ratio - self.audited_se

    @property
    def audited_by(self) -> str:
        return min((a for a in self.battery if a.realizable),
                   key=lambda a: a.se_ratio).name

    def survives(self, tau=S.TAU_DEFAULT) -> bool:
        return self.audited_se >= tau

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "scenario": self.scenario,
            "model_seed": self.model_seed,
            "audit_seed": self.audit_seed,
            "clean_se": round(self.clean.se_ratio, 4),
            "claimed_se": round(self.reference.se_ratio, 4),
            "audited_se": round(self.audited_se, 4),
            "audited_by": self.audited_by,
            "robustness_gap": round(self.robustness_gap, 4),
            "tau_survives": tau_verdicts(self.audited_se),
            "battery": [a.to_dict() for a in self.battery],
            "feature_bound_se": round(self.feature_bound.se_ratio, 4),
            "n_witnesses": len(self.witnesses),
        }


def _witnesses(H_adv, theta, r_idx, model, rng, tau, *, cfg, kappa, audit_seed, k=5):
    """Complete witness records for the k worst violating users: everything a
    third party (holding the substrate's channels) needs to rebuild the RIS
    geometry (audit_seed + winning restart), apply the stored phase vector, and
    recompute the SE ratio. h_eff is stored too, so the SE-ratio arithmetic is
    checkable even without rebuilding the geometry."""
    pred = rf.predict_probs(model, rf.complex_to_feat(rf.add_cn_noise(H_adv, rf.PILOT_SNR_DB, rng))).argmax(1)
    W = rf.dft_codebook()
    gains = np.abs(H_adv.astype(np.complex128) @ W.conj().T) ** 2
    opt = gains.argmax(1)
    ratio = per_user_se_ratio(model, H_adv, rng=np.random.default_rng(0))
    order = np.argsort(ratio)[:k]
    out = []
    for u in order:
        if ratio[u] >= tau:
            continue
        out.append({
            "user": int(u),
            "config": {"M": int(cfg[0]), "iters": int(cfg[1]), "restarts": int(cfg[2])},
            "kappa": float(kappa),
            "audit_seed": int(audit_seed),
            "geometry_seed": int(audit_seed + r_idx[u]),
            "theta": [round(float(x), 4) for x in np.atleast_1d(theta[u])],
            "h_eff": [[round(float(z.real), 6), round(float(z.imag), 6)]
                      for z in np.atleast_1d(H_adv[u])],
            "w_pred": int(pred[u]),
            "w_opt": int(opt[u]),
            "se_ratio": round(float(ratio[u]), 4),
        })
    return out


def audit_model(model_name, model, data, victim, *, scenario="asu", model_seed=42,
                audit_seed=0, schedule=S.SCHEDULE, kappa=S.KAPPA, n_users=S.N_USERS,
                eps=S.PGD_EPS) -> ModelAudit:
    H = np.ascontiguousarray(data["H_test"][:n_users]).astype(np.complex64)
    rng = lambda: np.random.default_rng(audit_seed)
    clean = clean_eval(model, H, rng=rng())

    battery: list[AttackResult] = []
    reference: AttackResult | None = None
    per_user_worst = np.ones(len(H))
    worst_H = worst_theta = worst_r = worst_cfg = None   # strongest single config, for witnesses
    worst_mean = 2.0
    for (M, iters, restarts) in schedule:
        t0 = time.time()
        H_adv, theta, r_idx = ris_best(model, H, M=M, kappa=kappa, iters=iters,
                                       restarts=restarts, seed=audit_seed, keep_theta=True)
        res = eval_effective(model, H_adv, name=f"RIS M={M} it={iters} x{restarts}",
                             realizable=True, kind="whitebox", rng=rng())
        res.seconds = time.time() - t0
        battery.append(res)
        per_user_worst = np.minimum(per_user_worst, res.per_user)
        if res.se_ratio < worst_mean:
            worst_mean, worst_H, worst_theta = res.se_ratio, H_adv, theta
            worst_r, worst_cfg = r_idx, (M, iters, restarts)
        if (M, iters, restarts) == S.REFERENCE:
            reference = res

    # decoupled transfer attacker: RIS crafted on the undefended victim, applied here
    if victim is not model:
        for M in S.TRANSFER_M:
            g = rf.build_geometry(H, M, kappa, seed=audit_seed)
            H_tr = rf.ris_attack(victim, g, iters=80, lr=S.ATTACK_LR, seed=audit_seed)
            res = eval_effective(model, H_tr, name=f"RIS transfer M={M}",
                                 realizable=True, kind="transfer", rng=rng())
            battery.append(res)
            per_user_worst = np.minimum(per_user_worst, res.per_user)

    reference = reference or battery[0]

    # feature-space PGD: unrealizable upper bound for context
    y_eff, _ = rf.best_beam_labels(H, rf.dft_codebook())
    X_pgd = rf.pgd(model, rf.complex_to_feat(H), y_eff, eps=eps, iters=S.PGD_ITERS)
    feature_bound = eval_feature(model, X_pgd, H, name=f"feature-PGD eps={eps} (unrealizable)")

    for a in [*battery, feature_bound]:
        a.se_drop = clean.se_ratio - a.se_ratio

    witnesses = (_witnesses(worst_H, worst_theta, worst_r, model, rng(), S.TAU_DEFAULT,
                            cfg=worst_cfg, kappa=kappa, audit_seed=audit_seed)
                 if worst_H is not None else [])

    return ModelAudit(model_name, scenario, model_seed, audit_seed, clean, reference,
                      battery, feature_bound, per_user_audited=per_user_worst,
                      witnesses=witnesses)

"""Obfuscated-gradient / gradient-masking diagnostics (Athalye et al.).

A defense that merely *masks gradients* would show tell-tale signatures: a black-box
transfer or random perturbation beating the white-box attack, the loss failing to
converge, or an unbounded-budget attack still failing. We run the battery so the
"not gradient-masked" claim rests on evidence, not a single control.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import rfcore as rf
from . import settings as S
from .audit import ris_best
from .oracle import eval_effective, per_user_se_ratio


def aperture_sweep(model, victim, H, *, audit_seed=0, Ms=S.SWEEP_M,
                   iters=S.SWEEP_ITERS, restarts=S.SWEEP_RESTARTS, kappa=S.KAPPA):
    """White-box, black-box-transfer, and random-RIS achieved-SE at each aperture.
    The random-RIS control runs on THIS model at every M (not only the undefended
    one), so a flat/rising white-box curve cannot be an artifact of a broken attack."""
    rng = lambda: np.random.default_rng(audit_seed)
    curve = {"whitebox": {}, "transfer": {}, "random": {}}
    for M in Ms:
        H_adv = ris_best(model, H, M=M, kappa=kappa, iters=iters, restarts=restarts, seed=audit_seed)
        curve["whitebox"][M] = eval_effective(model, H_adv, rng=rng()).se_ratio

        g = rf.build_geometry(H, M, kappa, seed=audit_seed)
        H_tr = rf.ris_attack(victim, g, iters=iters, lr=S.ATTACK_LR, seed=audit_seed)
        curve["transfer"][M] = eval_effective(model, H_tr, rng=rng()).se_ratio

        theta = torch.rand(H.shape[0], M, device=rf.DEVICE) * 2 * np.pi
        H_rand = rf.h_eff_torch(g, theta).detach().cpu().numpy()
        curve["random"][M] = eval_effective(model, H_rand, rng=rng()).se_ratio
    return curve


def convergence(model, H, *, audit_seed=0, Ms=S.CONVERGENCE_M, iters=S.CONVERGENCE_ITERS,
                kappa=S.KAPPA, stride=10):
    """Loss-vs-iteration of the RIS phase optimizer. A converging (then flat) curve
    rules out gradient masking via non-convergence; a monotone-in-M final loss shows
    aperture is a genuine budget knob."""
    rng = np.random.default_rng(audit_seed)
    out = {}
    for M in Ms:
        g = rf.build_geometry(H, M, kappa, seed=audit_seed)
        torch.manual_seed(audit_seed)
        U = g["H_d"].shape[0]
        theta = (torch.rand(U, M, device=rf.DEVICE) * 2 * np.pi).requires_grad_(True)
        opt = torch.optim.Adam([theta], lr=S.ATTACK_LR)
        Wc = g["W"].conj()
        losses = []
        for it in range(iters):
            opt.zero_grad()
            Hh = rf.h_eff_torch(g, theta)
            p = torch.softmax(model(rf.feats_torch(Hh)), dim=1)
            gains = (Hh @ Wc.T).abs() ** 2
            opt_gain = gains.max(dim=1, keepdim=True).values.detach()
            loss = ((p * gains).sum(dim=1, keepdim=True) / (opt_gain + 1e-30)).mean()
            loss.backward(); opt.step()
            if it % stride == 0 or it == iters - 1:
                losses.append((it, float(loss.detach())))
        out[M] = losses
    return out


def unbounded(model, H, *, audit_seed=0, budget=S.UNBOUNDED, kappa=S.KAPPA):
    """Very large iteration/restart budget: the strongest positive evidence. If the
    defended model still stays above tau here, robustness is not a budget artifact."""
    M, iters, restarts = budget
    t0 = time.time()
    H_adv = ris_best(model, H, M=M, kappa=kappa, iters=iters, restarts=restarts, seed=audit_seed)
    r = eval_effective(model, H_adv, name=f"unbounded M={M} it={iters} x{restarts}", rng=np.random.default_rng(audit_seed))
    return {"se_ratio": r.se_ratio, "M": M, "iters": iters, "restarts": restarts,
            "seconds": round(time.time() - t0, 1)}


def universal(model, H, *, audit_seed=0, M=128, iters=S.UNIVERSAL_ITERS, kappa=S.KAPPA):
    """CSI-blind attacker: one shared RIS config for all users (knows the model and
    environment distribution, not any victim's per-user CSI)."""
    g = rf.build_geometry(H, M, kappa, seed=audit_seed)
    theta = rf.ris_attack_universal(model, g, iters=iters, seed=audit_seed)
    H_adv = rf.heff_from_theta(g, theta)
    return eval_effective(model, H_adv, name=f"universal M={M}", kind="universal",
                          rng=np.random.default_rng(audit_seed)).se_ratio


def kappa_sweep(model, victim, H, *, audit_seed=0, kappas=S.KAPPA_SWEEP,
                M=S.KAPPA_SWEEP_M, iters=S.KAPPA_SWEEP_ITERS,
                restarts=S.KAPPA_SWEEP_RESTARTS):
    """Sweep the RIS amplitude budget kappa (the remaining physical budget axis
    beside aperture M and phase resolution b) at fixed aperture, with the same
    white-box / transfer / random controls as the aperture sweep. Tests whether
    the audit's conclusion (defended >> undefended, but overstated) is stable as
    the rogue RIS is made physically stronger. Runnable on the substrate; opt-in
    via `qed-rf run --kappa-sweep` because it re-attacks all users per kappa."""
    rng = lambda: np.random.default_rng(audit_seed)
    curve = {"whitebox": {}, "transfer": {}, "random": {}}
    for kap in kappas:
        H_adv = ris_best(model, H, M=M, kappa=kap, iters=iters, restarts=restarts, seed=audit_seed)
        curve["whitebox"][kap] = eval_effective(model, H_adv, rng=rng()).se_ratio

        g = rf.build_geometry(H, M, kap, seed=audit_seed)
        H_tr = rf.ris_attack(victim, g, iters=iters, lr=S.ATTACK_LR, seed=audit_seed)
        curve["transfer"][kap] = eval_effective(model, H_tr, rng=rng()).se_ratio

        theta = torch.rand(H.shape[0], M, device=rf.DEVICE) * 2 * np.pi
        H_rand = rf.h_eff_torch(g, theta).detach().cpu().numpy()
        curve["random"][kap] = eval_effective(model, H_rand, rng=rng()).se_ratio
    return curve


def bbit_audit(model, H, *, audit_seed=0, budget=S.BBIT_M, bbits=S.BBITS, kappa=S.KAPPA):
    """Discrete b-bit RIS phases at the audited worst-case budget: does the violation
    survive realistic few-bit hardware phase quantization?"""
    M, iters, restarts = budget
    out = {}
    rng = lambda: np.random.default_rng(audit_seed)
    # continuous reference (b = inf)
    H_adv = ris_best(model, H, M=M, kappa=kappa, iters=iters, restarts=restarts, seed=audit_seed)
    out["inf"] = eval_effective(model, H_adv, rng=rng()).se_ratio
    for b in bbits:
        worst_H = worst_se = None
        for r in range(restarts):
            g = rf.build_geometry(H, M, kappa, seed=audit_seed + r)
            H_b = rf.ris_attack(model, g, iters=iters, lr=S.ATTACK_LR, seed=audit_seed + r, bbit=b)
            se = per_user_se_ratio(model, H_b, rng=np.random.default_rng(audit_seed))
            if worst_H is None:
                worst_H, worst_se = H_b.copy(), se
            else:
                take = se < worst_se
                worst_H[take], worst_se[take] = H_b[take], se[take]
        out[str(b)] = eval_effective(model, worst_H, rng=rng()).se_ratio
    return out

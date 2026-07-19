"""The physical robustness oracle.

Robustness is scored by the *achieved spectral-efficiency ratio* on the TRUE
(effective) channel a user experiences. Under the geometric RIS-cascade channel
model this is a physical measurement the attacker cannot game: the model always
predicts from a *noisy* estimate of the effective CSI (the substrate's honest
protocol), so neither side can exploit a perfect-CSI assumption, and a defense
cannot look robust unless the beam it selects actually delivers SE.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import rfcore as rf
from .settings import TAU_DEFAULT, TAUS


@dataclass
class AttackResult:
    name: str
    realizable: bool
    se_ratio: float          # population mean of the per-user ratio
    beam_acc: float          # fraction choosing the effective channel's best beam
    kind: str = "whitebox"   # whitebox | transfer | random | universal | feature
    se_drop: float = 0.0     # clean_se - se_ratio (filled by the auditor)
    # per-user evidence (not serialized into the summary dict):
    per_user: np.ndarray | None = field(default=None, repr=False)
    theta: np.ndarray | None = field(default=None, repr=False)   # RIS phases (U,M) or (M,)
    pred: np.ndarray | None = field(default=None, repr=False)    # predicted beam index (U,)
    opt: np.ndarray | None = field(default=None, repr=False)     # optimal beam index (U,)
    h_eff: np.ndarray | None = field(default=None, repr=False)   # effective channels (U,N)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "realizable": self.realizable,
            "se_ratio": round(self.se_ratio, 4),
            "beam_acc": round(self.beam_acc, 4),
            "se_drop": round(self.se_drop, 4),
            "seconds": round(self.seconds, 1),
        }


def _noisy_feats(H, rng):
    return rf.complex_to_feat(rf.add_cn_noise(H, rf.PILOT_SNR_DB, rng))


def _gains(H):
    H = H.astype(np.complex128)
    W = rf.dft_codebook().astype(np.complex128)
    return np.abs(H @ W.conj().T) ** 2


def _per_user_ratio(gains, pred):
    g_pred = gains[np.arange(len(pred)), pred]
    g_opt = gains.max(1)
    rho = 10 ** (rf.EVAL_SNR_DB / 10.0) / g_opt.mean()
    return np.log2(1 + rho * g_pred) / np.log2(1 + rho * g_opt)


def eval_effective(model, H_eff, name="attack", realizable=True, kind="whitebox",
                   rng=None, keep=False) -> AttackResult:
    """Score a realizable attack: model predicts from noisy effective CSI; SE on
    the true h_eff. `keep=True` retains per-user evidence for witnesses/CDFs."""
    rng = rng or np.random.default_rng(rf.SEED)
    pred = rf.predict_probs(model, _noisy_feats(H_eff, rng)).argmax(1)
    gains = _gains(H_eff)
    ratio = _per_user_ratio(gains, pred)
    res = AttackResult(name, realizable, float(ratio.mean()),
                       float(np.mean(pred == gains.argmax(1))), kind=kind,
                       per_user=ratio)
    if keep:
        res.pred, res.opt, res.h_eff = pred, gains.argmax(1), H_eff
    return res


def eval_feature(model, X_adv, H_orig, name="feature") -> AttackResult:
    """Score an (unrealizable) feature-space attack: SE on the ORIGINAL channel."""
    pred = rf.predict_probs(model, X_adv).argmax(1)
    gains = _gains(H_orig)
    ratio = _per_user_ratio(gains, pred)
    return AttackResult(name, realizable=False, se_ratio=float(ratio.mean()),
                        beam_acc=float(np.mean(pred == gains.argmax(1))),
                        kind="feature", per_user=ratio)


def clean_eval(model, H, rng=None) -> AttackResult:
    return eval_effective(model, H, name="clean", realizable=True, kind="clean", rng=rng)


def per_user_se_ratio(model, H_eff, rng=None) -> np.ndarray:
    """Per-user achieved/optimal SE ratio (for picking the worst attack per user)."""
    rng = rng or np.random.default_rng(rf.SEED)
    pred = rf.predict_probs(model, _noisy_feats(H_eff, rng)).argmax(1)
    return _per_user_ratio(_gains(H_eff), pred)


def is_violation(result: AttackResult, tau: float = TAU_DEFAULT) -> bool:
    """A realizable attack that drops achieved-SE below tau is an audited violation."""
    return result.realizable and result.se_ratio < tau


def tau_verdicts(audited_se: float, taus=TAUS) -> dict[str, bool]:
    """Survival verdict of an audited SE ratio at each threshold (the verdict is
    threshold-dependent by construction; the paper reports the full sweep)."""
    return {f"{t:.2f}": bool(audited_se >= t) for t in taus}

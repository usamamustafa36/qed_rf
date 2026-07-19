"""Per-seed retraining of the victim and RIS-adversarially-trained defended
models, mirroring the substrate's stage-1/stage-4 recipe with explicit seed
control. The substrate is read-only; retrained weights land in qed_rf/artifacts/
so audited numbers carry a mean and a std instead of a single point estimate.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from . import rfcore as rf
from . import settings as S


def _model_dir(scenario_key: str, model_seed: int) -> str:
    return os.path.join(rf.LOCAL_ART, "models", scenario_key, f"seed{model_seed}")


def _best_beam(H, W):
    return np.abs(H.astype(np.complex128) @ W.conj().T).argmax(1)


def _noisy_feats(H, rng):
    return rf.complex_to_feat(rf.add_cn_noise(H, rf.PILOT_SNR_DB, rng))


def build_augmented(victim, H_tr, W, geo_seed: int, rng):
    """Substrate stage-4 recipe: clean + malicious-RIS (M=64,128 vs the victim)
    + random-RIS effective channels, each labelled with its own true best beam."""
    Xs, ys = [_noisy_feats(H_tr, rng)], [_best_beam(H_tr, W)]
    for M in (64, 128):
        g = rf.build_geometry(H_tr, M, S.KAPPA, seed=geo_seed)
        H_adv = rf.ris_attack(victim, g, iters=60, lr=S.ATTACK_LR, seed=geo_seed)
        Xs.append(_noisy_feats(H_adv, rng)); ys.append(_best_beam(H_adv, W))
        theta_rand = torch.rand(H_tr.shape[0], M, device=rf.DEVICE) * 2 * np.pi
        H_rand = rf.h_eff_torch(g, theta_rand).detach().cpu().numpy()
        Xs.append(_noisy_feats(H_rand, rng)); ys.append(_best_beam(H_rand, W))
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys)


def train_pair(scenario_key: str, model_seed: int, data=None, force=False):
    """Train (or load from cache) the victim/defended pair for one model seed.
    Returns (victim, defended)."""
    d = _model_dir(scenario_key, model_seed)
    vpath, dpath = os.path.join(d, "victim.pt"), os.path.join(d, "defended.pt")
    if not force and os.path.exists(vpath) and os.path.exists(dpath):
        return rf.load_model(vpath), rf.load_model(dpath)

    data = data or rf.build_dataset_for(scenario_key)
    os.makedirs(d, exist_ok=True)

    victim = rf.train_model(data["Xtr"], data["ytr"], data["Xva"], data["yva"],
                            data["n_beams"], epochs=60, verbose=False, seed=model_seed)
    torch.save(victim.state_dict(), vpath)

    rng = np.random.default_rng(model_seed)
    Xaug, yaug = build_augmented(victim, data["H_tr"], data["W"],
                                 geo_seed=S.AUG_GEO_SEED + model_seed, rng=rng)
    defended = rf.train_model(Xaug, yaug, data["Xva"], data["yva"], data["n_beams"],
                              epochs=40, verbose=False, seed=model_seed)
    torch.save(defended.state_dict(), dpath)
    return victim, defended


def ensure_models(scenario_key: str, model_seeds=S.MODEL_SEEDS, data=None):
    """Train every missing (victim, defended) pair; returns {seed: (victim, defended)}."""
    data = data or rf.build_dataset_for(scenario_key)
    return {s: train_pair(scenario_key, s, data=data) for s in model_seeds}

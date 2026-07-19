"""Adapter to the lab's 6G beam-prediction substrate (read-only).

We build on `usama/paper2026` rather than fork it, so results stay consistent with
the already-validated attack/defense pipeline. Point `QED_RF_PAPER2026` at that
directory if it lives elsewhere.
"""
from __future__ import annotations

import os
import sys

import torch

from .settings import LOCAL_ART  # noqa: F401  (re-exported; train/experiments use rf.LOCAL_ART)

_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "usama", "paper2026")
PAPER2026 = os.path.abspath(os.environ.get("QED_RF_PAPER2026", _DEFAULT))
ARTIFACTS = os.path.join(PAPER2026, "artifacts")

if not os.path.isdir(PAPER2026):
    raise RuntimeError(
        "6G substrate not found.\n"
        f"  looked in: {PAPER2026}\n"
        "  Expected the lab's paper2026 directory, containing the modules\n"
        "  attacks.py, beamdata.py, model.py, ris.py and an artifacts/ folder\n"
        "  with the cached channels (channels_64ant.npy, channels_mmw.npy).\n"
        "  Point QED_RF_PAPER2026 at it:\n"
        "    export QED_RF_PAPER2026=/path/to/usama/paper2026          # POSIX\n"
        "    $env:QED_RF_PAPER2026 = 'C:\\path\\to\\usama\\paper2026'  # PowerShell\n"
        "  Note: cached results in artifacts/runs_*.json can still be aggregated\n"
        "  without the substrate via `python -m qedrf bench` / `audit`."
    )
if PAPER2026 not in sys.path:
    sys.path.insert(0, PAPER2026)

import attacks as _attacks   # noqa: E402
import beamdata as bd        # noqa: E402
import model as _model       # noqa: E402
import ris as _ris           # noqa: E402

# --- re-exports -----------------------------------------------------------
DEVICE = _model.DEVICE
BeamMLP = _model.BeamMLP
predict_probs = _model.predict_probs
train_model = _model.train_model

build_dataset = bd.build_dataset
dft_codebook = bd.dft_codebook
se_ratio = bd.se_ratio
topk_acc = bd.topk_acc
complex_to_feat = bd.complex_to_feat
best_beam_labels = bd.best_beam_labels
add_cn_noise = bd.add_cn_noise
PILOT_SNR_DB = bd.PILOT_SNR_DB
EVAL_SNR_DB = bd.EVAL_SNR_DB
SEED = bd.SEED

fgsm, pgd, cw = _attacks.fgsm, _attacks.pgd, _attacks.cw
build_geometry = _ris.build_geometry
ris_attack = _ris.ris_attack
ris_attack_universal = _ris.ris_attack_universal
heff_from_theta = _ris.heff_from_theta
h_eff_torch = _ris._h_eff             # torch effective channel (for gradient-free search)
feats_torch = _ris._feats_torch


def load_model(path: str, in_dim: int = 128, n_beams: int = 64):
    """Load a trained victim/defended state_dict (absolute path, or a filename
    resolved against the substrate's artifacts)."""
    if not os.path.isabs(path):
        path = os.path.join(ARTIFACTS, path)
    m = BeamMLP(in_dim, n_beams).to(DEVICE)
    obj = torch.load(path, map_location=DEVICE)
    m.load_state_dict(obj if isinstance(obj, dict) else obj.state_dict())
    m.eval()
    return m


def build_dataset_for(scenario_key: str):
    """Dataset dict for a scenario key from settings.SCENARIOS ('asu', 'mmw')."""
    from .settings import SCENARIOS
    cfg = SCENARIOS[scenario_key]
    return build_dataset(scenario=cfg["scenario"], cache_name=cfg["cache_name"])

"""Every knob of the audit, in one place, so the schedule that produces the
audited numbers is fully specified (and can be emitted verbatim as the paper's
settings table). No other module hard-codes an attack budget.
"""
from __future__ import annotations

import os

# QED-RF's own artifacts (cached raw runs, retrained per-seed weights). The
# substrate stays read-only; everything this package produces lands here.
# Kept in settings (not rfcore) so cache-only workflows never touch the substrate.
LOCAL_ART = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))

# --- scenarios (DeepMIMO ray-traced, cached in the substrate's artifacts) ----
SCENARIOS = {
    "asu": dict(scenario="asu_campus_3p5", cache_name="channels_64ant.npy",
                label="ASU campus 3.5 GHz"),
    "mmw": dict(scenario="city_0_newyork_28", cache_name="channels_mmw.npy",
                label="NYC 28 GHz mmWave"),
}

# --- seeds -------------------------------------------------------------------
MODEL_SEEDS = (42, 1, 2)     # victim + defended training seeds (per scenario)
AUDIT_SEEDS = (0, 1, 2)      # RIS geometry / phase-init / CSI-noise seeds (main grid)
DIAG_AUDIT_SEEDS = (0, 1)    # audit seeds for the heavier diagnostic battery
AUG_GEO_SEED = 700           # + model_seed -> geometry seed of the training-time attack

# --- audit population --------------------------------------------------------
N_USERS = 2000               # test users audited (first N of the test split)

# --- the escalation schedule (aperture M, optimizer iters, random restarts) ---
# REFERENCE is the single configuration a defense paper typically reports
# ("claimed" robustness); the rest escalate compute and aperture.
REFERENCE = (128, 80, 1)
SCHEDULE = [
    (128, 80, 1),
    (128, 200, 3),
    (256, 80, 1),
    (256, 200, 3),
    (512, 200, 3),
]

# RIS phase optimizer (identical to the substrate's attack): Adam on the
# unconstrained phase vector theta, v = exp(j*theta) (unit modulus by construction).
ATTACK_LR = 0.1
KAPPA = 1.0                  # RIS-to-direct amplitude budget at M_ref = 64

# --- RIS amplitude-budget (kappa) sweep -------------------------------------
# The paper argues a physically-realizable attacker's strength is a *vector* of
# resources (aperture M, phase bits b, compute) with no canonical operating
# point; kappa (the RIS-to-direct amplitude budget) is the remaining axis. This
# sweep re-audits the same models at each kappa. It requires the substrate
# (re-attack), so it is opt-in behind `qed-rf run --kappa-sweep`; the exact
# command and feasibility tier are recorded in the experiment manifest.
KAPPA_SWEEP = (0.5, 1.0, 2.0, 4.0)
KAPPA_SWEEP_M = 256          # aperture held fixed while kappa varies
KAPPA_SWEEP_ITERS = 200
KAPPA_SWEEP_RESTARTS = 3

# --- controls / diagnostics ---------------------------------------------------
SWEEP_M = (16, 32, 64, 128, 256, 512)   # aperture sweep (iters/restarts below)
SWEEP_ITERS = 120
SWEEP_RESTARTS = 1
TRANSFER_M = (64, 128, 256)  # black-box: crafted on undefended victim, applied to defended
UNIVERSAL_ITERS = 200        # one shared RIS config for all users (CSI-blind attacker)
CONVERGENCE_M = (64, 128, 256)
CONVERGENCE_ITERS = 400
UNBOUNDED = (256, 1000, 3)   # (M, iters, restarts): the "unbounded budget" run
                             # (37x the reference's 80-iter/1-restart compute)
BBITS = (1, 2, 3)            # discrete RIS phase resolutions audited at BBIT_M
BBIT_M = (256, 200, 3)

# feature-space PGD upper bound (unrealizable over the air; context only)
PGD_EPS = 0.1
PGD_ITERS = 20

# --- verdict thresholds -------------------------------------------------------
TAUS = (0.85, 0.90, 0.95)
TAU_DEFAULT = 0.90

# --- statistics (deterministic; numpy-only, cache-derivable) -----------------
# Confidence intervals accompany the mean/std so the paper reports interval
# estimates, not just point + spread. Seed-grid CIs use a Student-t interval
# over the N=|MODEL_SEEDS|*|AUDIT_SEEDS| runs; per-user quantities (outage,
# audited mean) use a fixed-seed bootstrap so the interval is reproducible from
# the released cache alone.
CI_LEVEL = 0.95
N_BOOT = 2000
BOOT_SEED = 20260707

# --- extensibility registry (scaffold) ---------------------------------------
# The audit protocol is defense- and architecture-agnostic: it only needs a
# trained model plus the substrate's channel/RIS/oracle primitives. This
# registry names the families the harness is wired to audit. Only "risadv" has
# released weights in artifacts/models/; the others are runnable scaffolds whose
# exact commands and required substrate hooks are in the experiment manifest
# (`qed-rf manifest`). Adding a family is a training recipe in train.py plus an
# entry here -- the audit, diagnostics, and reporting code are unchanged.
DEFENSE_FAMILIES = {
    "risadv": dict(label="RIS-adversarial training", released=True,
                   recipe="train.build_augmented (stage-4)"),
    "smoothing": dict(label="randomized-smoothing (median beam over RIS draws)",
                      released=False, recipe="train.build_smoothed (scaffold)"),
    "detector": dict(label="detector-gated (abstain on RIS-anomalous CSI)",
                     released=False, recipe="train.build_detector (scaffold)"),
}
ARCHITECTURES = {
    "beammlp": dict(label="BeamMLP (2-layer MLP)", released=True),
    "beamcnn": dict(label="1-D CNN over antenna CSI", released=False),
    "beamattn": dict(label="self-attention beam predictor", released=False),
}

# Aggregation rule (formal): for user u, audited ratio r_u = min over every
# realizable attack in the battery of the per-user achieved/optimal SE ratio;
# audited SE = mean_u r_u. "Claimed" = population mean under REFERENCE only.
AGGREGATION = "per-user worst over the realizable battery, then population mean"


def forward_backward_passes(n_users: int = N_USERS) -> dict[str, int]:
    """Attacker compute budget per schedule entry: model forward+backward passes."""
    return {f"M={M} it={it} x{r}": 2 * it * r * n_users for (M, it, r) in SCHEDULE}

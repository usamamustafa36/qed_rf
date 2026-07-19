"""The experiment manifest: one machine-readable place that names every
experiment in the audit, the exact command that produces it, what it consumes
and emits, and its reproducibility tier. `qed-rf manifest` prints it.

Tiers (matching REPRODUCIBILITY.md):
  cache   -- derivable from the released artifacts/runs_*.json (numpy only);
  figure  -- cache + matplotlib;
  substrate -- needs torch + the (unreleased) paper2026 substrate to (re)run.

The manifest is the contract between the paper's claims and the code: every
Results paragraph should trace to a manifest entry, and every substrate-tier
entry carries the literal command a lab user runs to reproduce it. Keeping it in
code (not just prose) lets `qed-rf validate`/CI enumerate what is and isn't
reproducible off-lab without guessing.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    key: str
    title: str
    tier: str            # cache | figure | substrate
    command: str
    consumes: str
    produces: str
    supports: str        # what paper claim / table / figure it backs


EXPERIMENTS: list[Experiment] = [
    Experiment(
        "main_grid", "Multi-seed claimed-vs-audited robustness", "substrate",
        "python -m qedrf run --force",
        "substrate channels + retrained victim/defended pairs (3 seeds x 2 scenarios)",
        "artifacts/runs_{asu,mmw}.json",
        "Table main; Results (1)-(2); the audited/claimed/gap numbers"),
    Experiment(
        "aggregate", "Aggregate cache -> tables, macros, figures", "figure",
        "python -m qedrf bench --out-dir paper",
        "artifacts/runs_*.json",
        "paper/results.*, tau.tex, settings.tex, diagnostics.tex, ablation.tex, "
        "phasebits.tex, witnesses.tex, results_macros.tex, *.png",
        "every table/figure/inline number in the paper"),
    Experiment(
        "findings", "Research findings + claims check", "cache",
        "python -m qedrf findings",
        "artifacts/runs_*.json",
        "findings report; non-zero exit if any headline claim fails",
        "reviewer gate for Results (1)-(5) and the diagnostics"),
    Experiment(
        "validate", "Artifact schema + reproducibility + macro coverage", "cache",
        "python -m qedrf validate",
        "artifacts/runs_*.json + paper/",
        "pass/fail per gate; non-zero exit on any failure",
        "artifact integrity claim in Discussion"),
    Experiment(
        "escalation_ablation", "Schedule ablation (compute vs aperture)", "cache",
        "python -m qedrf findings   # 'Schedule ablation' section",
        "battery configs in artifacts/runs_*.json",
        "paper/ablation.tex, ablation.png; DefComputeDrop/DefApertureDrop macros",
        "Results (4); Table ablation -- adaptive schedule is necessary"),
    Experiment(
        "aperture_sweep", "Aperture (M) sensitivity + masking margin", "figure",
        "python -m qedrf bench   # sweep figure",
        "sweep{whitebox,transfer,random} in artifacts/runs_*.json",
        "paper/robustness_sweep.png; DefBelowM macro",
        "Results (3); Fig sweep -- decline with aperture, no masking"),
    Experiment(
        "phasebit_sensitivity", "RIS phase-resolution (b-bit) sensitivity", "figure",
        "python -m qedrf bench   # phasebits table/figure",
        "bbit in artifacts/runs_*.json",
        "paper/phasebits.tex, phasebits.png; DefBbitGap macro",
        "Results (1) realizability; saturates by 3 bits"),
    Experiment(
        "outage_witness", "Per-user outage + worst-case witnesses", "cache",
        "python -m qedrf findings   # outage + witness sections",
        "per_user vectors + witnesses in artifacts/runs_*.json",
        "paper/witnesses.tex, cdf.png; DefOutage/DefOutageCi/Witness macros",
        "Results (5); outage story + case-study table"),
    Experiment(
        "statistics", "Confidence intervals + sign test", "cache",
        "python -m qedrf findings   # interval estimates",
        "seed grid + per-user vectors",
        "DefGapCi/DefAuditedCi/Sign* macros",
        "Results (2)/(4): gap CI excludes zero; per-run sign test"),
    Experiment(
        "kappa_sweep", "RIS amplitude-budget (kappa) sweep", "substrate",
        "python -m qedrf run --force --kappa-sweep",
        "substrate re-attack at kappa in settings.KAPPA_SWEEP (fixed M/iters/restarts)",
        "artifacts/runs_*.json['kappa_sweep']; paper/kappa.tex, kappa.png (when present)",
        "Discussion scope: does the verdict hold across kappa (not yet run)"),
    Experiment(
        "second_defense", "Second defense family (e.g. smoothing/detector)", "substrate",
        "python -m qedrf run --force --family smoothing   # after adding recipe to train.py",
        "substrate + a new training recipe registered in settings.DEFENSE_FAMILIES",
        "artifacts/runs_<family>_*.json",
        "external validity beyond one defense family (recommended rerun #1)"),
    Experiment(
        "second_architecture", "Second architecture (CNN/attention beam predictor)", "substrate",
        "python -m qedrf run --force --arch beamcnn   # after adding model class to substrate",
        "substrate model class registered in settings.ARCHITECTURES",
        "artifacts/runs_<arch>_*.json",
        "external validity beyond BeamMLP (recommended rerun #1)"),
    Experiment(
        "surrogate_transfer", "Independently-trained surrogate / query-based black box", "substrate",
        "python -m qedrf run --force --strong-transfer",
        "substrate + a surrogate trained on a disjoint split",
        "extra transfer rows in the battery",
        "Discussion: a stronger black box only makes the audit harsher (rerun #3)"),
]

TIERS = {"cache": "cache-only (numpy)", "figure": "cache + matplotlib",
         "substrate": "needs torch + paper2026 substrate"}


def manifest_markdown() -> str:
    L = ["# QED-RF experiment manifest", "",
         "| Experiment | Tier | Command | Supports |",
         "|---|---|---|---|"]
    for e in EXPERIMENTS:
        L.append(f"| {e.title} | {e.tier} | `{e.command}` | {e.supports} |")
    L += ["", "## Reproducibility tiers", ""]
    for k, v in TIERS.items():
        n = sum(e.tier == k for e in EXPERIMENTS)
        L.append(f"- **{k}** ({n}): {v}")
    L += ["", "## Detail", ""]
    for e in EXPERIMENTS:
        L += [f"### {e.title} (`{e.key}`, tier: {e.tier})",
              f"- command: `{e.command}`",
              f"- consumes: {e.consumes}",
              f"- produces: {e.produces}",
              f"- supports: {e.supports}", ""]
    return "\n".join(L) + "\n"

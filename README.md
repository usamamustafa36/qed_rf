# QED-RF — Autonomous Adaptive-Robustness Audit of AI-Native Beam Prediction

QED-RF asks a narrow, testable question about 6G physical-layer AI: **does the
robustness a defended beam predictor *reports* — one adaptive attack at one fixed
budget — survive a stronger, still physically-realizable attacker?** It is an
autonomous audit agent that escalates a malicious-RIS attack (larger aperture,
more optimizer iterations, random restarts, a decoupled transfer adversary),
keeps the **per-user worst** effective channel, and scores it with a physical
spectral-efficiency (SE) oracle on the *true* effective channel — bringing the
*"false sense of security"* evaluation discipline (Athalye et al.) to AI-native
beam management. This repository holds the audit harness (`qedrf/`) and its
test suite; the cached multi-seed results and the paper they generate are kept
with the full project. Every inline number in the paper is produced by the
harness, nothing is hand-typed.

## Key contributions

- **An escalating, realizable audit** (Algorithm 1 in the paper): a schedule
  over (RIS aperture M, iterations, restarts) plus a transfer control, with
  per-user worst-case aggregation and recomputable per-user witness tuples
  ⟨v, h_eff, w_pred, w_opt, SE-ratio⟩ for every violation.
- **A multi-seed audit** (3 model seeds × 3 audit seeds × 2 ray-traced DeepMIMO
  scenarios: ASU campus 3.5 GHz and NYC 28 GHz mmWave), mean ± std **with 95%
  confidence intervals and an exact per-run sign test** on every headline number,
  a **schedule ablation** attributing the loss to the compute vs aperture budget
  axes, and a gradient-masking diagnostic battery (transfer, random-RIS,
  optimizer convergence, unbounded budget, b-bit phases).
- **An honest, quantified finding**: RIS-adversarial training gives *genuine,
  large* robustness (audited SE 0.107±0.008 → 0.746±0.030 at 3.5 GHz;
  0.215 → 0.706 at 28 GHz) with no gradient-masking signature on our
  diagnostics — **but** the single-configuration number a defense would report
  (0.857±0.021) overstates the audited worst case (0.746±0.030) by a gap
  (0.110) far exceeding seed noise (0.010), and audited SE declines with
  attacker aperture, crossing τ = 0.90 at M ≥ 64. The audited worst case fails
  at every τ ∈ {0.85, 0.90, 0.95} on both bands; the *reported* number is
  itself threshold-sensitive. Fixed-budget evaluation hides all of this.

## Repository structure

```
qed_rf/
├── qedrf/                  # the audit harness (Python package)
│   ├── settings.py         #   every audit knob in one place; artifacts path
│   ├── rfcore.py           #   read-only adapter to the lab substrate (channels · model · RIS · SE oracle)
│   ├── train.py            #   per-seed victim + RIS-adversarially-trained defended pairs
│   ├── oracle.py           #   achieved-SE ratio on the TRUE effective channel; τ verdicts
│   ├── audit.py            #   Algorithm 1: escalate, keep per-user worst, emit witnesses
│   ├── diagnostics.py      #   gradient-masking battery
│   ├── experiments.py      #   orchestrate the grid; cache raw runs (substrate needed only on cache miss)
│   ├── benchmark.py        #   aggregate → mean±std tables, LaTeX macros, figures
│   ├── analysis.py         #   research findings from the cache: gap decomposition, outage,
│   │                       #   variance structure, cost, claims check (`qed-rf findings`)
│   ├── validate.py         #   artifact gate: schema + byte-identical regeneration + macros
│   ├── manifest.py         #   experiment manifest: analysis → command → reproducibility tier
│   └── cli.py              #   `qed-rf run | bench | audit | findings`
├── tests/                  # substrate-free suite + substrate smoke tests (self-skipping)
└── Makefile · pyproject.toml · requirements.txt · LICENSE (MIT)
```

## Installation

Python **≥ 3.10**. Linux, macOS, and Windows are supported for everything that
doesn't need the substrate (see *Reproducibility status*).

```bash
pip install -r requirements.txt      # torch, numpy, scipy, scikit-learn, matplotlib
pip install -e .                     # optional: installs the `qed-rf` console script
```

For cache-only aggregation (regenerating tables/figures), `numpy` and
`matplotlib` alone are enough. For a **full rerun** you additionally need the
lab substrate:

```bash
export QED_RF_PAPER2026=/path/to/usama/paper2026      # POSIX
$env:QED_RF_PAPER2026 = "C:\path\to\usama\paper2026"  # PowerShell
```

The substrate directory must contain the modules `attacks.py`, `beamdata.py`,
`model.py`, `ris.py` and an `artifacts/` folder with the cached ray-traced
channels (`channels_64ant.npy`, `channels_mmw.npy`). It is the (currently
unreleased) companion codebase; see the *Reproducibility status* table below
for exactly what works without it.

## Quick start

```bash
python -m qedrf --help                       # works on any machine
python -m qedrf bench --out-dir paper        # regenerate all tables/macros/figures from the cache
python -m unittest discover -s tests -t .    # test suite (substrate tests skip with a reason)
```

## Commands

| Command | What it does | Needs |
|---|---|---|
| `python -m qedrf run` | train (3 seeds × 2 scenarios) + audit + diagnostics; caches to `artifacts/` | torch + substrate (on cache miss) |
| `python -m qedrf run --force` | ignore cache; retrain + re-audit everything | torch + substrate |
| `python -m qedrf audit` | quick Markdown summary from the cache | numpy |
| `python -m qedrf bench --out-dir paper` | aggregate cache → tables, macros, figures | numpy + matplotlib |
| `python -m qedrf findings [--out FILE]` | research-findings report + **claims check**: re-verifies every headline claim in the paper against the cache (non-zero exit if any fails) | numpy |
| `python -m qedrf validate [--paper-dir paper]` | **artifact gate**: cache schema + byte-identical regeneration + macro coverage (non-zero exit on any gating failure) | numpy |
| `python -m qedrf manifest [--out FILE]` | experiment manifest: every analysis, its command, and its reproducibility tier | (stdlib) |
| `python -m qedrf run --kappa-sweep` | full pipeline **plus** the RIS amplitude-budget (κ) sweep | torch + substrate |
| `python -m unittest discover -s tests -t .` | tests | numpy |
| `cd paper && pdflatex main.tex` (×2) | build the paper; numbers auto-fill from `results_macros.tex` | TeX |

Flags: `--scenarios asu mmw`, `--n-users N`, `--no-diagnostics`, `--force`.
POSIX shells can use `make help / test / run / audit / bench / paper / clean`;
on Windows run the `python -m qedrf ...` commands directly.

## Inputs and outputs

**Inputs:** the substrate's ray-traced DeepMIMO channels and pipeline
(read-only). Every audit hyperparameter — escalation schedule, seeds, τ set,
RIS budget κ, scenario constants — lives in [`qedrf/settings.py`](qedrf/settings.py)
and nowhere else.

**Outputs:** `run` caches raw per-seed numbers to `artifacts/runs_*.json` and
retrained weights to `artifacts/models/`; `bench` aggregates them into
`paper/results.{json,md,tex}`, `tau.tex`, `settings.tex`, `diagnostics.tex`,
`results_macros.tex`, and the three figures. `python -m qedrf manifest` prints
a full per-file manifest.

## Reproducibility status

This repository is the audit harness; the cached runs and the built paper are
kept with the full project, so the cache-dependent rows below need those files.

| What | Needs | Status |
|---|---|---|
| Inspect all numbers, witnesses, tables, figures, PDF | the cached runs + built paper | ✅ |
| Regenerate every table/macro/figure from cached runs | numpy + matplotlib | ✅ **verified byte-identical** on a machine with no torch and no substrate |
| Rebuild `main.pdf` | TeX distribution | ✅ (on any machine with TeX) |
| Full retrain + re-audit | torch + the unreleased substrate | ⚠️ lab-only until the substrate is released |

`tests/test_artifacts.py` enforces the middle row: it regenerates the paper's
text artifacts from the cached `runs_*.json` and fails if they differ from the
reference copies, so code, cache, and paper cannot silently drift apart. For
determinism notes and the full artifact manifest, run `python -m qedrf manifest`.

## Limitations

- **Full reruns are not yet externally reproducible**: the channels, model
  architecture, attack primitives, and SE oracle come from the unreleased lab
  substrate. Cached artifacts allow verification of every reported table and
  figure; they do not substitute for an independent end-to-end rerun.
- Simulated (ray-traced) channels only; no over-the-air validation.
- One defense family (RIS-adversarial training) and one architecture
  (`BeamMLP`) audited; two scenarios. The audit protocol is model-agnostic, and
  second-defense/second-architecture/κ-sweep extensions are wired as runnable
  scaffolds behind documented commands (`python -m qedrf manifest`), awaiting the
  substrate rerun — they are *not* claimed as results until run.
- Verdicts are threshold-dependent by construction; the paper reports the full
  τ sweep rather than a single binary.

## Citation

The associated paper is
*"Adaptive Robustness Auditing of RIS Defenses for AI-Native Beam Prediction:
Do Reported Numbers Survive Stronger Attacks?"* (preprint).

## Authors and license

Usama Mustafa (MCS, NUST) · Sana Mustafa (Institute of Space Technology) ·
Imran Rashid (MCS, NUST). MIT licensed (see `LICENSE`). The substrate
(`usama/paper2026`) is © the lab and used read-only. The malicious-RIS threat
model is studied on simulated channels, with no live RF transmission, to
*defend* AI-native networks.

## Troubleshooting

- **`RuntimeError: 6G substrate not found`** — you triggered a full pipeline
  run (cache miss or `--force`) without the substrate. Set `QED_RF_PAPER2026`
  (see *Installation*), or work from the cache (`bench` / `audit` without
  `--force`).
- **`No module named 'torch'`** — same situation: torch is only needed for
  full reruns and the substrate smoke tests. `pip install torch` if you have
  the substrate; otherwise the cache-only commands work without it.
- **Tests report `skipped=5`** — expected off-lab: the substrate smoke tests
  in `tests/test_rf.py` skip and print the reason. The other ~20 tests must pass.
- **`make` not found (Windows)** — run the `python -m qedrf ...` commands from
  the *Commands* table directly.
- **Regenerated figures differ from committed PNGs** — expected across
  matplotlib versions; only the text artifacts are byte-stable (and tested).

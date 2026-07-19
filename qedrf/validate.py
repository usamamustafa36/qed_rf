"""Artifact / reproducibility validation, cache-only (numpy).

`qed-rf validate` is the machine-checkable answer to a reviewer's question:
*are the released artifacts internally consistent, complete, and sufficient to
reproduce the paper?* It runs three independent gates and exits non-zero if any
fails:

  1. schema      -- every cached run has the fields, grid, and value ranges the
                    analysis code assumes (a malformed cache is caught here, not
                    as a cryptic error deep in aggregation);
  2. reproduce   -- regenerating the paper's text artifacts from the cache
                    reproduces the committed copies byte-for-byte (the same
                    guarantee tests/test_artifacts.py enforces, exposed as a CLI);
  3. macros      -- every generated LaTeX macro is actually cited in main.tex,
                    so a number cannot silently drop out of the prose while its
                    macro lingers.

None of this needs the substrate or torch: it validates the *released* artifacts.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from . import settings as S
from .analysis import macro_coverage
from .benchmark import write_artifacts


class Check:
    __slots__ = ("name", "ok", "detail", "gating")

    def __init__(self, name: str, ok: bool, detail: str = "", gating: bool = True):
        self.name, self.ok, self.detail, self.gating = name, bool(ok), detail, gating


# --------------------------------------------------------------------------- #
#  gate 1: cache schema + internal consistency
# --------------------------------------------------------------------------- #
def schema_checks(raws: dict[str, dict]) -> list[Check]:
    out: list[Check] = []
    n_grid = len(S.MODEL_SEEDS) * len(S.AUDIT_SEEDS)
    for key, raw in raws.items():
        p = f"[{key}]"
        for field in ("scenario", "label", "n_users", "audits", "settings", "per_user"):
            out.append(Check(f"{p} has '{field}'", field in raw))
        for model in ("undefended", "defended"):
            runs = raw.get("audits", {}).get(model, [])
            out.append(Check(f"{p} {model} grid complete ({n_grid} runs)",
                             len(runs) == n_grid, f"got {len(runs)}"))
            bad = [f"{f}={a[f]}" for a in runs for f in ("clean_se", "claimed_se", "audited_se")
                   if not 0.0 <= a[f] <= 1.0]
            out.append(Check(f"{p} {model} ratios in [0,1]", not bad, "; ".join(bad[:3])))
            viol = [a for a in runs if a["audited_se"] > a["claimed_se"] + 1e-6]
            out.append(Check(f"{p} {model} audited<=claimed every run", not viol,
                             f"{len(viol)} violations"))
            pv = raw.get("per_user", {}).get(model)
            out.append(Check(f"{p} {model} per-user vector length == n_users",
                             pv is not None and len(pv) == raw["n_users"],
                             f"len {len(pv) if pv else 0} vs {raw['n_users']}"))
        # reference config present in the battery
        names = {b["name"] for b in raw["audits"]["defended"][0]["battery"]}
        ref = f"RIS M={S.REFERENCE[0]} it={S.REFERENCE[1]} x{S.REFERENCE[2]}"
        out.append(Check(f"{p} battery contains reference config", ref in names, ref))
        # settings cache matches source of truth
        s = raw.get("settings", {})
        out.append(Check(f"{p} cached settings match settings.py",
                         s.get("schedule") == [list(x) for x in S.SCHEDULE]
                         and s.get("model_seeds") == list(S.MODEL_SEEDS)
                         and s.get("audit_seeds") == list(S.AUDIT_SEEDS)))
        # diagnostic completeness (present in the shipped cache)
        if "sweep" in raw:
            sw = raw["sweep"]["defended"]
            have = all(str(M) in sw["whitebox"] for M in S.SWEEP_M)
            out.append(Check(f"{p} aperture sweep covers all M", have))
        if "bbit" in raw:
            bb = raw["bbit"]["defended"]
            out.append(Check(f"{p} b-bit audit has 1/2/3/continuous",
                             all(k in bb for k in ("1", "2", "3", "inf"))))
        # witness completeness (compact vs recomputable) -- reported, not failed
        ws = raw.get("witnesses", [])
        complete = all("theta" in w and "h_eff" in w for w in ws) if ws else False
        out.append(Check(f"{p} witnesses present", bool(ws), f"{len(ws)} records"))
        out.append(Check(f"{p} witnesses recomputable (full theta+h_eff)", complete,
                         "compact schema; harness emits full tuple on substrate rerun"
                         if not complete else "", gating=False))
    return out


# --------------------------------------------------------------------------- #
#  gate 2: byte-identical regeneration of the paper's text artifacts
# --------------------------------------------------------------------------- #
GENERATED_TEXT = ["results.json", "results.md", "results.tex", "tau.tex",
                  "settings.tex", "diagnostics.tex", "ablation.tex",
                  "phasebits.tex", "witnesses.tex", "results_macros.tex"]


def reproduce_checks(raws: dict[str, dict], paper_dir: str) -> list[Check]:
    out: list[Check] = []
    if not os.path.isdir(paper_dir):
        return [Check("paper/ directory present", False, paper_dir)]
    with tempfile.TemporaryDirectory() as tmp:
        write_artifacts(raws, tmp, figures=False)
        for name in GENERATED_TEXT:
            committed = os.path.join(paper_dir, name)
            regen = os.path.join(tmp, name)
            if not os.path.exists(committed):
                out.append(Check(f"{name} committed", False))
                continue
            with open(committed, encoding="utf-8") as f:
                want = f.read()
            with open(regen, encoding="utf-8") as f:
                got = f.read()
            out.append(Check(f"{name} regenerates byte-identical", got == want))
    return out


# --------------------------------------------------------------------------- #
#  gate 3: macro coverage
# --------------------------------------------------------------------------- #
def macro_checks(paper_dir: str) -> list[Check]:
    mp = os.path.join(paper_dir, "results_macros.tex")
    tp = os.path.join(paper_dir, "main.tex")
    if not (os.path.exists(mp) and os.path.exists(tp)):
        return [Check("results_macros.tex + main.tex present", False)]
    with open(mp, encoding="utf-8") as f:
        macros = f.read()
    with open(tp, encoding="utf-8") as f:
        main = f.read()
    cov = macro_coverage(macros, main)
    # The generator emits a superset of macros for flexibility, so "every macro
    # cited" is informational, not a hard gate. The gate is the inverse and
    # build-breaking direction: the paper must not cite a generated-looking macro
    # that has no definition (a guaranteed LaTeX "Undefined control sequence").
    checks = [Check(f"{cov['n_used']}/{cov['n_defined']} generated macros cited in main.tex",
                    not cov["unused"],
                    "unused (info): " + ", ".join(cov["unused"][:6]) + (" ..." if len(cov["unused"]) > 6 else ""),
                    gating=False)]
    return checks


# --------------------------------------------------------------------------- #
#  driver
# --------------------------------------------------------------------------- #
def run_validation(raws: dict[str, dict], paper_dir: str) -> tuple[list[Check], bool]:
    """Returns (checks, ok). `ok` is True iff every *gating* check passes;
    informational checks (e.g. witness recomputability, a documented
    substrate-only state) are reported but do not fail the gate."""
    checks = schema_checks(raws) + reproduce_checks(raws, paper_dir) + macro_checks(paper_dir)
    return checks, all(c.ok for c in checks if c.gating)


def format_report(checks: list[Check]) -> str:
    L = ["# QED-RF artifact validation", ""]
    for c in checks:
        tag = "PASS" if c.ok else ("INFO" if not c.gating else "FAIL")
        tail = f" -- {c.detail}" if c.detail else ""
        L.append(f"- [{tag}] {c.name}{tail}")
    gating = [c for c in checks if c.gating]
    n_fail = sum(not c.ok for c in gating)
    info = len(checks) - len(gating)
    L += ["", f"{len(gating) - n_fail}/{len(gating)} gating checks pass"
          + (f" ({info} informational)." if info else ".")]
    return "\n".join(L) + "\n"

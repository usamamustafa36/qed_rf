"""Validate the committed raw runs (artifacts/runs_*.json) and check that the
committed paper tables/macros are exactly what the aggregation code produces
from them. This is the artifact-integrity guarantee of the repo: every number
in the paper is derivable from the released cache. Needs only numpy.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from qedrf import settings as S
from qedrf.benchmark import aggregate, to_markdown, write_artifacts

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"
RUNS = {k: Path(S.LOCAL_ART) / f"runs_{k}.json" for k in S.SCENARIOS}

# text artifacts write_artifacts must reproduce byte-for-byte from the cache
GENERATED = ["results.json", "results.md", "results.tex", "tau.tex",
             "settings.tex", "diagnostics.tex", "ablation.tex", "phasebits.tex",
             "witnesses.tex", "results_macros.tex"]


def _load_runs():
    return {k: json.loads(p.read_text(encoding="utf-8")) for k, p in RUNS.items()}


@unittest.skipUnless(all(p.exists() for p in RUNS.values()),
                     "cached runs (artifacts/runs_*.json) not present")
class RunsSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raws = _load_runs()

    def test_top_level_schema(self):
        for key, raw in self.raws.items():
            for field in ("scenario", "label", "n_users", "audits", "settings"):
                self.assertIn(field, raw, f"{key} missing {field}")
            self.assertEqual(raw["scenario"], key)

    def test_grid_is_complete(self):
        n_expected = len(S.MODEL_SEEDS) * len(S.AUDIT_SEEDS)
        for key, raw in self.raws.items():
            for model in ("undefended", "defended"):
                self.assertEqual(len(raw["audits"][model]), n_expected,
                                 f"{key}/{model}: expected {n_expected} runs")

    def test_audit_numbers_are_valid_ratios(self):
        for raw in self.raws.values():
            for audits in raw["audits"].values():
                for a in audits:
                    for field in ("clean_se", "claimed_se", "audited_se"):
                        self.assertTrue(0.0 <= a[field] <= 1.0,
                                        f"{field}={a[field]} out of [0,1]")

    def test_audited_never_exceeds_claimed(self):
        # the audited number is a per-user worst over a battery that includes
        # the reference attack, so it can never exceed the claimed number
        for raw in self.raws.values():
            for audits in raw["audits"].values():
                for a in audits:
                    self.assertLessEqual(a["audited_se"], a["claimed_se"] + 1e-6)

    def test_cached_settings_match_current_settings(self):
        # guards against silently changing settings.py without re-running:
        # the cache must have been produced under the settings we document
        for raw in self.raws.values():
            s = raw["settings"]
            self.assertEqual(s["model_seeds"], list(S.MODEL_SEEDS))
            self.assertEqual(s["audit_seeds"], list(S.AUDIT_SEEDS))
            self.assertEqual(s["reference"], list(S.REFERENCE))
            self.assertEqual(s["schedule"], [list(x) for x in S.SCHEDULE])
            self.assertEqual(s["taus"], list(S.TAUS))

    def test_aggregate_and_markdown_run(self):
        for raw in self.raws.values():
            agg = aggregate(raw)
            self.assertIn("defended", agg)
            md = to_markdown(raw)
            self.assertIn("| Model |", md)


@unittest.skipUnless(all(p.exists() for p in RUNS.values()) and PAPER.is_dir(),
                     "cached runs or paper/ directory not present")
class PaperConsistencyTest(unittest.TestCase):
    """Regenerate every text artifact from the cache and compare it to the
    committed copy in paper/. A failure means code, cache, and paper have
    drifted apart — the one thing this repo promises never happens."""

    def test_committed_paper_artifacts_match_cache(self):
        raws = _load_runs()
        with tempfile.TemporaryDirectory() as tmp:
            write_artifacts(raws, tmp, figures=False)
            for name in GENERATED:
                committed = PAPER / name
                if not committed.exists():
                    self.skipTest(f"paper/{name} not committed")
                got = Path(tmp, name).read_text(encoding="utf-8")
                want = committed.read_text(encoding="utf-8")
                self.assertEqual(got, want,
                                 f"paper/{name} does not match regeneration "
                                 "from artifacts/runs_*.json — re-run "
                                 "`python -m qedrf bench --out-dir paper`")


if __name__ == "__main__":
    unittest.main()

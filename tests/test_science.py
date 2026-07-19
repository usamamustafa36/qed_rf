"""Tests for the strengthened analysis (CIs, ablation, sensitivity, sign test),
the experiment manifest, and the validation gate. Cache-only (numpy); these run
everywhere the committed runs exist, no substrate needed.
"""
import json
import unittest
from pathlib import Path

import numpy as np

from qedrf import analysis as A
from qedrf import settings as S
from qedrf import validate as V
from qedrf.manifest import EXPERIMENTS, manifest_markdown

REPO = Path(__file__).resolve().parents[1]
RUNS = {k: Path(S.LOCAL_ART) / f"runs_{k}.json" for k in S.SCENARIOS}
PAPER = REPO / "paper"


def _load():
    return {k: json.loads(p.read_text(encoding="utf-8")) for k, p in RUNS.items()}


@unittest.skipUnless(all(p.exists() for p in RUNS.values()), "cached runs not present")
class StatsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raws = _load()

    def test_t_ci_matches_known_case(self):
        # constant sample -> zero-width interval at the mean
        ci = A.t_ci([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(ci["mean"], 0.5)
        self.assertAlmostEqual(ci["ci_lo"], 0.5)
        self.assertAlmostEqual(ci["ci_hi"], 0.5)

    def test_bootstrap_is_deterministic(self):
        v = np.linspace(0, 1, 200)
        a = A.bootstrap_ci(v, seed=123)
        b = A.bootstrap_ci(v, seed=123)
        self.assertEqual(a, b)
        self.assertLessEqual(a["ci_lo"], a["stat"])
        self.assertGreaterEqual(a["ci_hi"], a["stat"])

    def test_gap_ci_excludes_zero(self):
        for raw in self.raws.values():
            ci = A.confidence_intervals(raw)
            self.assertGreater(ci["robustness_gap"]["ci_lo"], 0.0)

    def test_ci_ordering(self):
        for raw in self.raws.values():
            ci = A.confidence_intervals(raw)
            for key in ("claimed_se", "audited_se", "robustness_gap"):
                self.assertLessEqual(ci[key]["ci_lo"], ci[key]["mean"])
                self.assertLessEqual(ci[key]["mean"], ci[key]["ci_hi"])

    def test_sign_test_all_runs_below(self):
        for raw in self.raws.values():
            ct = A.consistency_test(raw)
            self.assertEqual(ct["k_audited_below_claimed"], ct["n"])
            self.assertLess(ct["p_value"], 0.01)

    def test_binom_sf_bounds(self):
        self.assertAlmostEqual(A._binom_sf_ge(0, 5, 0.5), 1.0)
        self.assertAlmostEqual(A._binom_sf_ge(5, 5, 0.5), 0.5**5)


@unittest.skipUnless(all(p.exists() for p in RUNS.values()), "cached runs not present")
class AblationSensitivityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raws = _load()

    def test_ablation_components_positive_and_sum(self):
        for raw in self.raws.values():
            sa = A.schedule_ablation(raw)
            self.assertGreater(sa["compute_component"], 0.01)
            self.assertGreater(sa["aperture_component"], 0.01)
            # components reconstruct the fixed->strongest drop
            self.assertAlmostEqual(
                sa["compute_component"] + sa["aperture_component"],
                sa["escalation_drop"], places=6)

    def test_aperture_monotone_and_margin(self):
        for raw in self.raws.values():
            ap = A.aperture_sensitivity(raw)
            self.assertTrue(ap["monotone_nonincreasing"])
            self.assertGreaterEqual(ap["min_margin"], -1e-6)
            self.assertIsNotNone(ap["crossing_M"])

    def test_bbit_saturates_by_3(self):
        for raw in self.raws.values():
            bs = A.bbit_sensitivity(raw)
            self.assertTrue(bs["monotone_1to3"])
            self.assertLessEqual(abs(bs["realizability_gap_3_vs_cont"]), 0.02)

    def test_kappa_sensitivity_empty_without_sweep(self):
        # the shipped cache has no kappa sweep -> analysis returns {} gracefully
        for raw in self.raws.values():
            self.assertEqual(A.kappa_sensitivity(raw), {})

    def test_witness_table_sorted_and_typed(self):
        for raw in self.raws.values():
            wt = A.witness_table(raw, k=3)
            self.assertTrue(wt)
            ratios = [w["se_ratio"] for w in wt]
            self.assertEqual(ratios, sorted(ratios))
            for w in wt:
                self.assertIsInstance(w["user"], int)
                self.assertIn("complete", w)


@unittest.skipUnless(all(p.exists() for p in RUNS.values()), "cached runs not present")
class ClaimsAndValidateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raws = _load()

    def test_all_claims_hold(self):
        checks = A.claims_check(self.raws)
        failed = [c["claim"] for c in checks if not c["holds"]]
        self.assertEqual(failed, [], f"claims failed: {failed}")
        self.assertGreaterEqual(len(checks), 30)

    def test_schema_and_reproduce_gates_pass(self):
        checks, ok = V.run_validation(self.raws, str(PAPER))
        gating_fail = [c.name for c in checks if c.gating and not c.ok]
        self.assertEqual(gating_fail, [], f"gating failures: {gating_fail}")
        self.assertTrue(ok)

    def test_manifest_covers_all_tiers(self):
        tiers = {e.tier for e in EXPERIMENTS}
        self.assertEqual(tiers, {"cache", "figure", "substrate"})
        self.assertIn("kappa", manifest_markdown().lower())


if __name__ == "__main__":
    unittest.main()

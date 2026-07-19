"""Sanity checks on the audit configuration. These guard the invariants the
rest of the pipeline (and the paper) silently rely on. No torch/substrate needed."""
import unittest

from qedrf import settings as S


class SettingsTest(unittest.TestCase):
    def test_reference_is_in_schedule(self):
        # audit_model only fills `reference` when it encounters REFERENCE in the
        # schedule; the claimed-vs-audited comparison depends on this.
        self.assertIn(S.REFERENCE, S.SCHEDULE)

    def test_default_tau_is_in_the_sweep(self):
        self.assertIn(S.TAU_DEFAULT, S.TAUS)

    def test_taus_sorted_and_in_unit_interval(self):
        self.assertEqual(tuple(sorted(S.TAUS)), tuple(S.TAUS))
        for t in S.TAUS:
            self.assertTrue(0.0 < t < 1.0)

    def test_seeds_are_unique(self):
        self.assertEqual(len(set(S.MODEL_SEEDS)), len(S.MODEL_SEEDS))
        self.assertEqual(len(set(S.AUDIT_SEEDS)), len(S.AUDIT_SEEDS))
        # diagnostic seeds must be a subset of the audit seeds they summarize
        self.assertTrue(set(S.DIAG_AUDIT_SEEDS) <= set(S.AUDIT_SEEDS))

    def test_scenarios_have_required_keys(self):
        for key, cfg in S.SCENARIOS.items():
            for field in ("scenario", "cache_name", "label"):
                self.assertIn(field, cfg, f"{key} missing {field}")

    def test_schedule_entries_are_positive_triples(self):
        for entry in S.SCHEDULE:
            M, iters, restarts = entry
            self.assertGreater(M, 0)
            self.assertGreater(iters, 0)
            self.assertGreater(restarts, 0)

    def test_forward_backward_passes(self):
        passes = S.forward_backward_passes(n_users=100)
        self.assertEqual(len(passes), len(S.SCHEDULE))
        for label, count in passes.items():
            self.assertGreater(count, 0, label)

    def test_positive_scalars(self):
        self.assertGreater(S.N_USERS, 0)
        self.assertGreater(S.KAPPA, 0)
        self.assertGreater(S.ATTACK_LR, 0)


if __name__ == "__main__":
    unittest.main()

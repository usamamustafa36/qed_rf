"""Fast smoke tests for the RF audit (small subsets; skipped if the 6G substrate
or a torch runtime is unavailable)."""
import unittest

_WHY = ""
try:
    from qedrf import rfcore as rf
    from qedrf.audit import audit_model
    from qedrf.oracle import (AttackResult, eval_effective, is_violation,
                              tau_verdicts)
    _READY = True
except Exception as _e:  # noqa: BLE001 — substrate/torch may be absent in some CI
    _READY = False
    _WHY = str(_e).splitlines()[0]


@unittest.skipUnless(_READY, f"6G substrate or torch unavailable: {_WHY}")
class RFTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = rf.build_dataset()
        cls.victim = rf.load_model("victim_model.pt")

    def test_clean_se_is_high(self):
        r = eval_effective(self.victim, self.data["H_test"][:300])
        self.assertGreater(r.se_ratio, 0.9)

    def test_audit_proves_realizable_degradation(self):
        a = audit_model("victim", self.victim, self.data, self.victim,
                        schedule=[(128, 40, 1)], n_users=200)
        self.assertLess(a.audited_se, a.clean.se_ratio)   # the attack degrades achieved SE
        self.assertLess(a.audited_se, 0.9)                # undefended is provably vulnerable
        self.assertFalse(a.survives(0.90))

    def test_per_user_worst_not_above_any_battery_mean(self):
        a = audit_model("victim", self.victim, self.data, self.victim,
                        schedule=[(128, 40, 1), (256, 40, 1)], n_users=200)
        worst_battery_mean = min(r.se_ratio for r in a.battery if r.realizable)
        # per-user worst aggregation can only be <= the best single-config mean
        self.assertLessEqual(a.audited_se, worst_battery_mean + 1e-6)

    def test_violation_requires_realizable_proof(self):
        self.assertTrue(is_violation(AttackResult("x", True, 0.50, 0.1)))    # realizable + low SE
        self.assertFalse(is_violation(AttackResult("x", True, 0.95, 0.9)))   # realizable but robust
        self.assertFalse(is_violation(AttackResult("x", False, 0.10, 0.1)))  # unrealizable != proof

    def test_tau_verdicts_are_threshold_dependent(self):
        v = tau_verdicts(0.83)
        self.assertFalse(v["0.90"])   # below 0.90
        self.assertTrue(v["0.85"] is False and v["0.90"] is False)
        self.assertTrue(tau_verdicts(0.83)["0.85"] is False)


if __name__ == "__main__":
    unittest.main()

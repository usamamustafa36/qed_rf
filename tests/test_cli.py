"""CLI smoke tests. Must pass on a machine without torch or the substrate:
the CLI is required to at least parse arguments and print help anywhere."""
import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path

from qedrf.cli import build_parser

REPO = Path(__file__).resolve().parents[1]


class ParserTest(unittest.TestCase):
    def test_help_exits_zero(self):
        buf = io.StringIO()
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stdout(buf):
            build_parser().parse_args(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("qed-rf", buf.getvalue())

    def test_run_defaults(self):
        args = build_parser().parse_args(["run"])
        self.assertEqual(sorted(args.scenarios), ["asu", "mmw"])
        self.assertFalse(args.force)
        self.assertFalse(args.no_diagnostics)

    def test_bench_out_dir(self):
        args = build_parser().parse_args(["bench", "--out-dir", "somewhere"])
        self.assertEqual(args.out_dir, "somewhere")

    def test_unknown_scenario_rejected(self):
        err = io.StringIO()
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stderr(err):
            build_parser().parse_args(["run", "--scenarios", "nonexistent"])
        self.assertNotEqual(ctx.exception.code, 0)


class ModuleInvocationTest(unittest.TestCase):
    def test_python_dash_m_help(self):
        """`python -m qedrf --help` must work even without torch/substrate."""
        proc = subprocess.run([sys.executable, "-m", "qedrf", "--help"],
                              cwd=REPO, capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("qed-rf", proc.stdout)


if __name__ == "__main__":
    unittest.main()

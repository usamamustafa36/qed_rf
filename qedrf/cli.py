"""QED-RF command line.

    qed-rf run      [--scenarios asu mmw] [--force]  # train + audit + diagnostics (cached)
    qed-rf bench    [--out-dir paper]                # aggregate cache -> tables + figures
    qed-rf audit    [--scenarios asu]                # quick Markdown summary from cache
    qed-rf findings [--out FILE]                     # research findings + claims check from cache
    qed-rf validate [--paper-dir paper]             # artifact schema + reproducibility + macros
    qed-rf manifest [--out FILE]                     # the experiment manifest (commands + tiers)
"""
from __future__ import annotations

import argparse

from . import settings as S
from .benchmark import to_markdown, write_artifacts

# NOTE: .experiments (via .rfcore) needs the paper2026 substrate on disk and
# raises at import time if it is missing, so it is imported lazily inside the
# subcommands. This keeps `python -m qedrf --help` usable everywhere.


def _cmd_run(args) -> int:
    from .experiments import run_all
    raws = run_all(scenarios=tuple(args.scenarios), force=args.force,
                   n_users=args.n_users, diagnostics=not args.no_diagnostics,
                   kappa_sweep=args.kappa_sweep)
    for v in raws.values():
        print("\n" + to_markdown(v))
    return 0


def _cmd_bench(args) -> int:
    from .experiments import run_all
    raws = run_all(scenarios=tuple(args.scenarios), force=args.force,
                   n_users=args.n_users, diagnostics=not args.no_diagnostics,
                   kappa_sweep=args.kappa_sweep)
    write_artifacts(raws, args.out_dir, figures=not args.no_diagnostics)
    for v in raws.values():
        print("\n" + to_markdown(v))
    print(f"\n[qed-rf] artifacts -> {args.out_dir}/ "
          "(results.json/md/tex, tau.tex, settings.tex, diagnostics.tex, "
          "results_macros.tex, *.png)")
    return 0


def _cmd_audit(args) -> int:
    from .experiments import run_all
    raws = run_all(scenarios=tuple(args.scenarios), force=False,
                   n_users=args.n_users, diagnostics=False)
    for v in raws.values():
        print("\n" + to_markdown(v))
    return 0


def _cmd_findings(args) -> int:
    """Research-findings report + claims check, cache-only. Exits non-zero if
    any headline claim fails to hold in the cached data (a reviewer gate)."""
    from .analysis import claims_check, findings_markdown
    from .experiments import run_all
    raws = run_all(scenarios=tuple(args.scenarios), force=False,
                   n_users=args.n_users, diagnostics=False)
    report = findings_markdown(raws)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(report)
        print(f"[qed-rf] findings -> {args.out}")
    return 0 if all(c["holds"] for c in claims_check(raws)) else 1


def _cmd_validate(args) -> int:
    """Artifact integrity gate, cache-only. Schema + byte-identical regeneration
    + macro coverage. Non-zero exit on any failure (a CI/reviewer gate)."""
    import json

    from . import settings as S
    from .validate import format_report, run_validation
    raws = {}
    for k in args.scenarios:
        path = f"{S.LOCAL_ART}/runs_{k}.json"
        with open(path, encoding="utf-8") as f:
            raws[k] = json.load(f)
    checks, ok = run_validation(raws, args.paper_dir)
    print(format_report(checks))
    return 0 if ok else 1


def _cmd_manifest(args) -> int:
    from .manifest import manifest_markdown
    report = manifest_markdown()
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(report)
        print(f"[qed-rf] manifest -> {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qed-rf",
        description="Autonomous adaptive-robustness audit of AI-native beam prediction.")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--scenarios", nargs="+", default=["asu", "mmw"],
                        choices=list(S.SCENARIOS))
        sp.add_argument("--n-users", type=int, default=S.N_USERS)
        sp.add_argument("--force", action="store_true", help="ignore cache; retrain + re-audit")
        sp.add_argument("--no-diagnostics", action="store_true",
                        help="skip the (heavier) gradient-masking battery")
        sp.add_argument("--kappa-sweep", action="store_true",
                        help="also sweep the RIS amplitude budget kappa (substrate; opt-in)")

    r = sub.add_parser("run", help="train + audit + diagnostics across the seed grid")
    common(r); r.set_defaults(func=_cmd_run)

    b = sub.add_parser("bench", help="aggregate the grid -> tables, macros, figures")
    common(b); b.add_argument("--out-dir", metavar="DIR", default="paper")
    b.set_defaults(func=_cmd_bench)

    a = sub.add_parser("audit", help="quick Markdown summary (no diagnostics)")
    common(a); a.set_defaults(func=_cmd_audit)

    f = sub.add_parser("findings",
                       help="research findings + claims check from cached runs")
    common(f); f.add_argument("--out", metavar="FILE", default=None,
                              help="also write the Markdown report to FILE")
    f.set_defaults(func=_cmd_findings)

    v = sub.add_parser("validate",
                       help="artifact schema + reproducibility + macro coverage (cache-only)")
    v.add_argument("--scenarios", nargs="+", default=["asu", "mmw"],
                   choices=list(S.SCENARIOS))
    v.add_argument("--paper-dir", metavar="DIR", default="paper")
    v.set_defaults(func=_cmd_validate)

    m = sub.add_parser("manifest", help="print the experiment manifest (commands + tiers)")
    m.add_argument("--out", metavar="FILE", default=None,
                   help="also write the Markdown manifest to FILE")
    m.set_defaults(func=_cmd_manifest)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Aggregate the raw multi-seed experiment grid into the paper's artifacts:
mean +/- std tables (Markdown + LaTeX), a macros file for inline numbers, and the
figures (aperture sweep with white-box/transfer/random controls, per-user CDF,
optimizer convergence). Consumes the cached runs from `experiments.run_all`.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import settings as S


# --------------------------------------------------------------------------- #
#  aggregation helpers
# --------------------------------------------------------------------------- #
def _col(audits, key):
    return np.array([a[key] for a in audits], dtype=float)


def _ms(arr):
    return float(np.mean(arr)), float(np.std(arr))


def aggregate(raw: dict) -> dict:
    """Per-scenario, per-model mean/std of every headline number + tau verdicts."""
    agg = {}
    for name, audits in raw["audits"].items():
        clean_m, clean_s = _ms(_col(audits, "clean_se"))
        claim_m, claim_s = _ms(_col(audits, "claimed_se"))
        aud = _col(audits, "audited_se")
        aud_m, aud_s = _ms(aud)
        gap_m, gap_s = _ms(_col(audits, "robustness_gap"))
        # verdict per tau: does the MEAN audited SE clear tau, and how many runs do
        tau = {f"{t:.2f}": {"survives": bool(aud_m >= t),
                            "frac_runs": float(np.mean(aud >= t))} for t in S.TAUS}
        agg[name] = dict(n_runs=len(audits), clean=(clean_m, clean_s),
                         claimed=(claim_m, claim_s), audited=(aud_m, aud_s),
                         gap=(gap_m, gap_s), tau=tau,
                         feature_bound=_ms(_col(audits, "feature_bound_se")))
    return agg


def _sweep_stats(rec_kind: dict) -> dict:
    return {int(M): _ms(np.array(v)) for M, v in rec_kind.items()}


# --------------------------------------------------------------------------- #
#  tables
# --------------------------------------------------------------------------- #
def to_markdown(raw: dict) -> str:
    agg = aggregate(raw)
    lines = [f"### {raw['label']} ({raw['audits']['defended'][0]['scenario']}), "
             f"n={raw['n_users']} users, {agg['defended']['n_runs']} runs/model",
             "| Model | Clean SE | Claimed SE | Audited SE | Gap | Survives tau=0.90 |",
             "|---|--:|--:|--:|--:|:--:|"]
    for name, a in agg.items():
        surv = "yes" if a["tau"]["0.90"]["survives"] else "NO"
        lines.append(f"| {name} | {a['clean'][0]:.3f}±{a['clean'][1]:.3f} | "
                     f"{a['claimed'][0]:.3f}±{a['claimed'][1]:.3f} | "
                     f"{a['audited'][0]:.3f}±{a['audited'][1]:.3f} | "
                     f"{a['gap'][0]:.3f}±{a['gap'][1]:.3f} | {surv} |")
    return "\n".join(lines)


def to_latex_main(raws: list[dict]) -> str:
    """Multi-scenario main table with mean +/- std for every number."""
    head = [r"\begin{tabular}{llccccc}", r"\toprule",
            r"Scenario & Model & Clean SE & Claimed SE & Audited SE & Gap & Surv.\ $\tau{=}0.90$ \\",
            r"\midrule"]
    rows = []
    for raw in raws:
        agg = aggregate(raw)
        for i, (name, a) in enumerate(agg.items()):
            scen = raw["label"] if i == 0 else ""
            surv = "yes" if a["tau"]["0.90"]["survives"] else r"\textbf{no}"
            rows.append(
                r"%s & %s & $%.3f{\pm}%.3f$ & $%.3f{\pm}%.3f$ & $%.3f{\pm}%.3f$ & "
                r"$%.3f{\pm}%.3f$ & %s \\" % (
                    scen, name, *a["clean"], *a["claimed"], *a["audited"], *a["gap"], surv))
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    return "\n".join(head + rows + [r"\end{tabular}"])


def to_latex_tau(raws: list[dict]) -> str:
    """Threshold sensitivity for the DEFENDED model: the verdict under the number a
    defense would report (reference/claimed) vs. the audited worst case, at each tau.
    Makes concrete that the reported number is itself threshold-dependent, while the
    audited worst case fails regardless of tau."""
    def verdict(x, t):
        return "yes" if x >= t else r"\textbf{no}"
    head = [r"\begin{tabular}{ll" + "c" * len(S.TAUS) + "}", r"\toprule",
            r"Scenario & Verdict on & " + " & ".join(fr"$\tau{{=}}{t:.2f}$" for t in S.TAUS) + r" \\",
            r"\midrule"]
    rows = []
    for raw in raws:
        agg = aggregate(raw)["defended"]
        claimed, audited = agg["claimed"][0], agg["audited"][0]
        rows.append(f"{raw['label']} & reported ({claimed:.3f}) & "
                    + " & ".join(verdict(claimed, t) for t in S.TAUS) + r" \\")
        rows.append(f" & audited ({audited:.3f}) & "
                    + " & ".join(verdict(audited, t) for t in S.TAUS) + r" \\")
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    return "\n".join(head + rows + [r"\end{tabular}"])


def to_latex_settings(raw: dict) -> str:
    """The audit schedule + scenario, so the paper self-contains the method."""
    s = raw["settings"]
    sched = "; ".join(f"$M{{=}}{M}$/{it}it/{r}$\\times$" for M, it, r in s["schedule"])
    rows = [
        ("Array / codebook", f"{s['n_ant']}-elem ULA, {s['n_beams']}-beam DFT"),
        ("Pilot / eval SNR", f"{s['pilot_snr_db']:.0f} / {s['eval_snr_db']:.0f} dB"),
        ("Audit users", f"{s['n_users']}"),
        ("Model / audit seeds", f"{len(s['model_seeds'])} $\\times$ {len(s['audit_seeds'])}"),
        ("RIS budget $\\kappa$, $M_{\\mathrm{ref}}$", f"{s['kappa']:.1f}, 64"),
        ("Phase optimizer", f"Adam, lr {s['attack_lr']}, unit-modulus $v_m{{=}}e^{{j\\theta_m}}$"),
        ("Reference (claimed)", f"$M{{=}}{s['reference'][0]}$, {s['reference'][1]} it, "
                                f"{s['reference'][2]}$\\times$"),
        ("Escalation schedule", sched),
        ("Aggregation", "per-user worst over battery, then mean"),
    ]
    head = [r"\begin{tabular}{@{}l p{0.60\linewidth}@{}}", r"\toprule",
            r"Setting & Value \\", r"\midrule"]
    body = [f"{k} & {v} \\\\" for k, v in rows]
    return "\n".join(head + body + [r"\bottomrule", r"\end{tabular}"])


def to_latex_diag(raws: list[dict]) -> str:
    """Gradient-masking battery on the DEFENDED model (audited SE ratio)."""
    head = [r"\begin{tabular}{lcccccc}", r"\toprule",
            r"Scenario & Clean & White-box & Transfer & Random-RIS & Universal & Unbounded \\",
            r"\midrule"]
    rows = []
    for raw in raws:
        d = raw["audits"]["defended"]
        clean = _ms(_col(d, "clean_se"))[0]
        sw = raw["sweep"]["defended"]
        wb = _sweep_stats(sw["whitebox"])
        tr = _sweep_stats(sw["transfer"])
        rd = _sweep_stats(sw["random"])
        Mmax = max(wb)
        uni = _ms(np.array(raw["universal"]["defended"]))[0]
        unb = _ms(np.array(raw["unbounded"]["defended"]))[0]
        rows.append(r"%s & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\" % (
            raw["label"], clean, wb[Mmax][0], tr[Mmax][0], rd[Mmax][0], uni, unb))
    return "\n".join(head + rows + [r"\bottomrule", r"\end{tabular}"])


def to_latex_ablation(raws: list[dict]) -> str:
    """Schedule ablation on the DEFENDED model: the fixed reference budget, then
    the marginal achieved-SE loss attributable to more optimizer compute vs. to
    larger aperture. Shows that an *adaptive* schedule (not a single fixed
    attack) is necessary, and which budget axis dominates."""
    from .analysis import schedule_ablation
    head = [r"\begin{tabular}{lcc}", r"\toprule",
            r"Escalation step & 3.5\,GHz & 28\,GHz \\", r"\midrule"]
    sa = {raw["scenario"]: schedule_ablation(raw) for raw in raws}
    order = [raw["scenario"] for raw in raws]

    def row(label, key, sub=None):
        vals = []
        for s in order:
            d = sa[s]["steps"][key] if sub is None else sa[s][key]
            v = d["drop"] if isinstance(d, dict) and "drop" in d else d
            vals.append(f"${v:+.3f}$" if label != "Reference (fixed budget)" else f"${v:.3f}$")
        return f"{label} & " + " & ".join(vals) + r" \\"

    rows = [
        f"Reference (fixed budget) & "
        + " & ".join(f"${sa[s]['reference_mean']:.3f}$" for s in order) + r" \\",
        r"\midrule",
        f"$+$ optimizer compute ($M{{=}}128$) & "
        + " & ".join(f"${-sa[s]['steps']['compute_at_M128']['drop']:+.3f}$" for s in order) + r" \\",
        f"$+$ aperture $128\\!\\to\\!256$ & "
        + " & ".join(f"${-sa[s]['steps']['aperture_hi_128to256']['drop']:+.3f}$" for s in order) + r" \\",
        f"$+$ aperture $256\\!\\to\\!512$ & "
        + " & ".join(f"${-sa[s]['steps']['aperture_256to512']['drop']:+.3f}$" for s in order) + r" \\",
        r"\midrule",
        f"Strongest single config & "
        + " & ".join(f"${sa[s]['strongest_single_mean']:.3f}$" for s in order) + r" \\",
    ]
    return "\n".join(head + rows + [r"\bottomrule", r"\end{tabular}"])


def to_latex_phasebits(raws: list[dict]) -> str:
    """Phase-resolution sensitivity on the DEFENDED model: achieved-SE at b-bit
    RIS phases and continuous, with the saturation gap b=3 vs continuous."""
    from .analysis import bbit_sensitivity
    head = [r"\begin{tabular}{lccccc}", r"\toprule",
            r"Scenario & $b{=}1$ & $b{=}2$ & $b{=}3$ & cont. & $b{=}3{-}$cont. \\",
            r"\midrule"]
    rows = []
    for raw in raws:
        bs = bbit_sensitivity(raw)
        if not bs:
            continue
        bybit = bs["by_bits"]
        rows.append(r"%s & %.3f & %.3f & %.3f & %.3f & $%+.3f$ \\" % (
            raw["label"], bybit["1"], bybit["2"], bybit["3"], bybit["inf"],
            bs["realizability_gap_3_vs_cont"]))
    return "\n".join(head + rows + [r"\bottomrule", r"\end{tabular}"])


def to_latex_witness(raws: list[dict], k: int = 3) -> str:
    """Worst-user witness case studies: for the k most-violated users, the audited
    SE ratio and the predicted vs. optimal beam index -- evidence that failures
    are gross misdirection, not adjacent-beam error. Records are drawn from the
    released cache; `complete` in the harness marks whether the full recomputable
    tuple (theta, h_eff) is attached (compact in the shipped cache)."""
    from .analysis import witness_table
    head = [r"\begin{tabular}{llccc}", r"\toprule",
            r"Scenario & User & SE ratio & Pred.\ beam & Opt.\ beam \\", r"\midrule"]
    rows = []
    for raw in raws:
        wt = witness_table(raw, k=k)
        for i, w in enumerate(wt):
            scen = raw["label"] if i == 0 else ""
            rows.append(r"%s & %d & %.4f & %d & %d \\" % (
                scen, w["user"], w["se_ratio"], w["w_pred"], w["w_opt"]))
        rows.append(r"\midrule")
    if rows:
        rows[-1] = r"\bottomrule"
    return "\n".join(head + rows + [r"\end{tabular}"])


def to_latex_kappa(raws: list[dict]) -> str | None:
    """Per-kappa white-box SE on the DEFENDED model (rendered only when a kappa
    sweep is present in the cache; None otherwise so the driver can skip it)."""
    from .analysis import kappa_sensitivity
    sens = {raw["scenario"]: kappa_sensitivity(raw) for raw in raws}
    if not any(sens.values()):
        return None
    kappas = sorted({k for s in sens.values() if s for k in s["per_kappa"]})
    head = [r"\begin{tabular}{l" + "c" * len(kappas) + "}", r"\toprule",
            r"Scenario & " + " & ".join(fr"$\kappa{{=}}{k:g}$" for k in kappas) + r" \\",
            r"\midrule"]
    rows = []
    for raw in raws:
        s = sens.get(raw["scenario"])
        if not s:
            continue
        cells = " & ".join(f"{s['per_kappa'][k][0]:.3f}" if k in s["per_kappa"] else "--"
                           for k in kappas)
        rows.append(f"{raw['label']} & {cells} " + r"\\")
    return "\n".join(head + rows + [r"\bottomrule", r"\end{tabular}"])


def write_macros(raws: list[dict], path: str) -> None:
    """LaTeX \newcommand macros so every inline number in the prose is generated
    (including the escalation-decomposition, outage, stability, quantization and
    cost analyses from qedrf.analysis)."""
    from .analysis import (bbit_sensitivity, bbit_summary, confidence_intervals,
                           consistency_test, cost_summary, escalation_decomposition,
                           outage, schedule_ablation, variance_decomposition)

    def cmd(name, val):
        return r"\newcommand{\%s}{%s}" % (name, val)
    lines = []
    tag = {"asu": "Asu", "mmw": "Mmw"}
    n_runs_total = 0
    total_wall_min = 0.0
    for raw in raws:
        t = tag[raw["scenario"]]
        agg = aggregate(raw)
        for mname, mtag in (("undefended", "Undef"), ("defended", "Def")):
            a = agg[mname]
            lines += [
                cmd(f"{t}{mtag}Clean", f"{a['clean'][0]:.3f}"),
                cmd(f"{t}{mtag}Claimed", f"{a['claimed'][0]:.3f}"),
                cmd(f"{t}{mtag}ClaimedStd", f"{a['claimed'][1]:.3f}"),
                cmd(f"{t}{mtag}Audited", f"{a['audited'][0]:.3f}"),
                cmd(f"{t}{mtag}AuditedStd", f"{a['audited'][1]:.3f}"),
                cmd(f"{t}{mtag}Gap", f"{a['gap'][0]:.3f}"),
                cmd(f"{t}{mtag}GapStd", f"{a['gap'][1]:.3f}"),
            ]
        # smallest aperture at which the defended white-box curve first drops below 0.90
        wb = _sweep_stats(raw["sweep"]["defended"]["whitebox"])
        below = sorted(M for M, (m, s) in wb.items() if m < S.TAU_DEFAULT)
        lines.append(cmd(f"{t}DefBelowM", str(below[0]) if below else r"\text{none}"))
        lines.append(cmd(f"{t}DefUnbounded", f"{_ms(np.array(raw['unbounded']['defended']))[0]:.3f}"))
        lines.append(cmd(f"{t}Nusers", str(raw["n_users"])))

        # gap decomposition: escalation vs per-user-worst aggregation
        esc = escalation_decomposition(raw)
        n_runs_total += esc["n_runs"]
        lines += [cmd(f"{t}EscDrop", f"{esc['escalation_drop']:.3f}"),
                  cmd(f"{t}AggDrop", f"{esc['aggregation_drop']:.3f}")]
        # uncertainty structure: audit-seed vs model-seed variance
        var = variance_decomposition(raw)
        lines += [cmd(f"{t}DefStdModelSeed", f"{var['model_seed_std']:.3f}"),
                  cmd(f"{t}DefStdAuditSeed", f"{var['audit_seed_std']:.3f}")]
        # per-user outage under the audited worst case (first cached seed pair)
        o = outage(raw, "defended")
        lines += [cmd(f"{t}DefOutage", str(round(100 * o["frac_below"][f"{S.TAU_DEFAULT:.2f}"]))),
                  cmd(f"{t}DefMedianUser", f"{o['median']:.2f}")]
        # hardware phase quantization at the audited budget
        bb = bbit_summary(raw)
        if bb:
            lines += [cmd(f"{t}DefBbitThree", f"{bb['3']:.3f}"),
                      cmd(f"{t}DefBbitCont", f"{bb['inf']:.3f}")]
        # worst-user witness example (first cached seed pair)
        if raw.get("witnesses"):
            w = raw["witnesses"][0]
            lines += [cmd(f"{t}WitnessPred", str(w["w_pred"])),
                      cmd(f"{t}WitnessOpt", str(w["w_opt"]))]
        # interval estimates: 95% CIs on the headline numbers
        ci = confidence_intervals(raw)
        lines += [cmd(f"{t}DefAuditedCiLo", f"{ci['audited_se']['ci_lo']:.3f}"),
                  cmd(f"{t}DefAuditedCiHi", f"{ci['audited_se']['ci_hi']:.3f}"),
                  cmd(f"{t}DefGapCiLo", f"{ci['robustness_gap']['ci_lo']:.3f}"),
                  cmd(f"{t}DefGapCiHi", f"{ci['robustness_gap']['ci_hi']:.3f}"),
                  cmd(f"{t}DefOutageCiLo", str(round(100 * ci['outage_frac']['ci_lo']))),
                  cmd(f"{t}DefOutageCiHi", str(round(100 * ci['outage_frac']['ci_hi'])))]
        # schedule ablation: compute vs aperture components of the escalation drop
        sa = schedule_ablation(raw)
        lines += [cmd(f"{t}DefComputeDrop", f"{sa['compute_component']:.3f}"),
                  cmd(f"{t}DefApertureDrop", f"{sa['aperture_component']:.3f}")]
        # per-run consistency (sign test) that audited < claimed in every run
        ct = consistency_test(raw)
        lines += [cmd(f"{t}SignK", str(ct["k_audited_below_claimed"])),
                  cmd(f"{t}SignN", str(ct["n"])),
                  cmd(f"{t}SignP", f"{ct['p_value']:.4f}")]
        # phase-resolution saturation gap (b=3 vs continuous)
        bs = bbit_sensitivity(raw)
        if bs:
            lines.append(cmd(f"{t}DefBbitGap", f"{bs['realizability_gap_3_vs_cont']:+.3f}"))
        total_wall_min += cost_summary(raw)["wall_minutes"]
    lines.append(cmd("NmodelSeeds", str(len(S.MODEL_SEEDS))))
    lines.append(cmd("NauditSeeds", str(len(S.AUDIT_SEEDS))))
    lines.append(cmd("NrunsTotal", str(n_runs_total)))
    lines.append(cmd("PassesPerUser", str(sum(2 * it * r for (_, it, r) in S.SCHEDULE))))
    lines.append(cmd("TotalWallMin", str(round(total_wall_min))))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
#  figures
# --------------------------------------------------------------------------- #
def make_sweep_figure(raws: list[dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(raws), figsize=(4.6 * len(raws), 3.4), squeeze=False)
    for ax, raw in zip(axes[0], raws):
        for mname, style in (("undefended", dict(color="#c0392b")),
                             ("defended", dict(color="#1f6f8b"))):
            wb = _sweep_stats(raw["sweep"][mname]["whitebox"])
            Ms = sorted(wb)
            mean = np.array([wb[M][0] for M in Ms]); std = np.array([wb[M][1] for M in Ms])
            ax.plot(Ms, mean, marker="o", label=f"{mname} white-box", **style)
            ax.fill_between(Ms, mean - std, mean + std, alpha=0.15, color=style["color"])
        # controls on the defended model
        for kind, ls, lab in (("transfer", ":", "transfer (black-box)"),
                              ("random", "-.", "random-RIS")):
            st = _sweep_stats(raw["sweep"]["defended"][kind])
            Ms = sorted(st)
            ax.plot(Ms, [st[M][0] for M in Ms], ls=ls, color="#1f6f8b", alpha=0.8,
                    label=f"defended {lab}")
        ax.axhline(S.TAU_DEFAULT, ls="--", lw=1, color="grey", label=r"$\tau=0.90$")
        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"realizable RIS aperture $M$")
        ax.set_title(raw["label"], fontsize=10)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("achieved / optimal SE ratio")
    axes[0][-1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_cdf_figure(raws: list[dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(raws), figsize=(4.4 * len(raws), 3.2), squeeze=False)
    for ax, raw in zip(axes[0], raws):
        for mname, color in (("undefended", "#c0392b"), ("defended", "#1f6f8b")):
            v = np.sort(np.array(raw["per_user"][mname]))
            ax.plot(v, np.linspace(0, 1, len(v)), label=mname, color=color)
        ax.axvline(S.TAU_DEFAULT, ls="--", lw=1, color="grey", label=r"$\tau=0.90$")
        ax.set_xlabel("per-user audited SE ratio")
        ax.set_title(raw["label"], fontsize=10)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("CDF over audit users")
    axes[0][-1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_convergence_figure(raws: list[dict], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(raws), figsize=(4.4 * len(raws), 3.2), squeeze=False)
    for ax, raw in zip(axes[0], raws):
        conv = raw["convergence"]["defended"]
        for M, series in sorted(conv.items(), key=lambda kv: int(kv[0])):
            its = [it for it, _ in series]; ls = [l for _, l in series]
            ax.plot(its, ls, marker=".", ms=3, label=f"M={M}")
        ax.set_xlabel("RIS phase-optimizer iteration")
        ax.set_title(raw["label"], fontsize=10)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("attack loss (achieved/opt SE)")
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_ablation_figure(raws: list[dict], path: str) -> None:
    """Waterfall of the escalation ladder on the defended model: from the fixed
    reference budget down through each added budget knob to the strongest single
    config, per scenario."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .analysis import schedule_ablation
    labels = ["reference", "+compute", "+M 256", "+M 512"]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for raw, color in zip(raws, ("#1f6f8b", "#c0392b")):
        sa = schedule_ablation(raw)
        ref = sa["reference_mean"]
        pts = [ref,
               ref - sa["steps"]["compute_at_M128"]["drop"],
               ref - sa["steps"]["compute_at_M128"]["drop"] - sa["steps"]["aperture_hi_128to256"]["drop"],
               sa["strongest_single_mean"]]
        ax.plot(labels, pts, marker="o", color=color, label=raw["label"])
    ax.axhline(S.TAU_DEFAULT, ls="--", lw=1, color="grey", label=r"$\tau=0.90$")
    ax.set_ylabel("defended white-box SE ratio")
    ax.set_xlabel("cumulative escalation over the fixed reference budget")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def make_bbit_figure(raws: list[dict], path: str) -> None:
    """Achieved-SE vs. RIS phase resolution on the defended model: the attack
    strengthens from 1 to 3 bits and saturates at continuous phases."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .analysis import bbit_sensitivity
    order = ["1", "2", "3", "inf"]
    xt = ["1", "2", "3", r"$\infty$"]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for raw, color in zip(raws, ("#1f6f8b", "#c0392b")):
        bs = bbit_sensitivity(raw)
        if not bs:
            continue
        ax.plot(xt, [bs["by_bits"][k] for k in order], marker="s", color=color,
                label=raw["label"])
    ax.axhline(S.TAU_DEFAULT, ls="--", lw=1, color="grey", label=r"$\tau=0.90$")
    ax.set_xlabel("RIS phase resolution (bits)")
    ax.set_ylabel("defended audited SE ratio")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def make_kappa_figure(raws: list[dict], path: str) -> bool:
    """Defended white-box SE vs. RIS amplitude budget kappa. Returns False (and
    draws nothing) unless a kappa sweep is present in the cache."""
    from .analysis import kappa_sensitivity
    sens = [(raw, kappa_sensitivity(raw)) for raw in raws]
    if not any(s for _, s in sens):
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for (raw, s), color in zip(sens, ("#1f6f8b", "#c0392b")):
        if not s:
            continue
        ks = sorted(s["per_kappa"])
        ax.plot(ks, [s["per_kappa"][k][0] for k in ks], marker="D", color=color,
                label=raw["label"])
    ax.axhline(S.TAU_DEFAULT, ls="--", lw=1, color="grey", label=r"$\tau=0.90$")
    ax.set_xlabel(r"RIS amplitude budget $\kappa$")
    ax.set_ylabel("defended white-box SE ratio")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
#  driver
# --------------------------------------------------------------------------- #
def write_artifacts(raws: dict[str, dict], out_dir: str, figures=True) -> str:
    os.makedirs(out_dir, exist_ok=True)
    order = [r for r in raws.values()]

    # UTF-8 + LF everywhere so regenerated artifacts are byte-identical across OSes.
    def _open_w(name):
        return open(os.path.join(out_dir, name), "w", encoding="utf-8", newline="\n")

    payload = {k: {"aggregate": aggregate(v), "n_users": v["n_users"],
                   "label": v["label"], "settings": v.get("settings"),
                   "witnesses": v.get("witnesses", [])} for k, v in raws.items()}
    with _open_w("results.json") as f:
        json.dump(payload, f, indent=2, default=lambda o: list(o) if hasattr(o, "__iter__") else o)

    with _open_w("results.md") as f:
        f.write("\n\n".join(to_markdown(v) for v in order) + "\n")

    with _open_w("results.tex") as f:
        f.write(to_latex_main(order) + "\n")
    with _open_w("tau.tex") as f:
        f.write(to_latex_tau(order) + "\n")
    with _open_w("settings.tex") as f:
        f.write(to_latex_settings(order[0]) + "\n")
    with _open_w("diagnostics.tex") as f:
        f.write(to_latex_diag(order) + "\n")
    with _open_w("ablation.tex") as f:
        f.write(to_latex_ablation(order) + "\n")
    with _open_w("phasebits.tex") as f:
        f.write(to_latex_phasebits(order) + "\n")
    with _open_w("witnesses.tex") as f:
        f.write(to_latex_witness(order) + "\n")
    kappa_tex = to_latex_kappa(order)
    if kappa_tex is not None:   # only after `qed-rf run --kappa-sweep`
        with _open_w("kappa.tex") as f:
            f.write(kappa_tex + "\n")
    write_macros(order, os.path.join(out_dir, "results_macros.tex"))

    if figures and "sweep" in order[0]:
        make_sweep_figure(order, os.path.join(out_dir, "robustness_sweep.png"))
        make_cdf_figure(order, os.path.join(out_dir, "cdf.png"))
        make_convergence_figure(order, os.path.join(out_dir, "convergence.png"))
        make_ablation_figure(order, os.path.join(out_dir, "ablation.png"))
        make_bbit_figure(order, os.path.join(out_dir, "phasebits.png"))
        make_kappa_figure(order, os.path.join(out_dir, "kappa.png"))  # no-op unless swept
    return out_dir

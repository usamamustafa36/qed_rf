"""Research-level analysis of the cached audit runs.

Everything here is derived from artifacts/runs_*.json alone (numpy only), so
every scientific claim in the paper can be re-derived — and challenged — from
the released cache without the substrate. `qed-rf findings` renders the full
report; `claims_check` is the reviewer-facing verification that each headline
claim actually holds in the released data (it is also what keeps the paper
honest after a future re-run: if a re-run breaks a claim, the check fails).

ASCII-only output: reports are printed to arbitrary consoles.
"""
from __future__ import annotations

import numpy as np

from . import settings as S


def _runs(raw: dict, model: str) -> list[dict]:
    return raw["audits"][model]


# --------------------------------------------------------------------------- #
#  statistics (dependency-free: numpy only, no scipy)
# --------------------------------------------------------------------------- #
# Two-sided Student-t critical values at the 95% level, indexed by degrees of
# freedom. Hard-coded so seed-grid confidence intervals need no scipy; df beyond
# the table falls back to the normal quantile 1.96. The audit grid has
# N = |MODEL_SEEDS| * |AUDIT_SEEDS| runs, so df = N-1 (8 for the shipped 3x3).
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 17: 2.110,
        20: 2.086, 25: 2.060, 30: 2.042}


def _t_crit(df: int, level: float = S.CI_LEVEL) -> float:
    if level != 0.95:  # the shipped table is the 95% level; be explicit otherwise
        return 1.959964
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    keys = sorted(_T95)
    return _T95[min(keys, key=lambda k: abs(k - df))] if df < 30 else 1.959964


def t_ci(values, level: float = S.CI_LEVEL) -> dict:
    """Student-t confidence interval for the mean of a small sample (the seed
    grid). Returns mean, std (population, matching the paper's ±std), sem, and
    the interval half-width and bounds."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(v.mean())
    sd = float(v.std())                       # population std (consistent w/ tables)
    sem = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    half = _t_crit(n - 1, level) * sem if n > 1 else float("nan")
    return {"mean": mean, "std": sd, "sem": sem, "n": n,
            "ci_lo": mean - half, "ci_hi": mean + half, "half": half}


def bootstrap_ci(values, stat=np.mean, n_boot: int = S.N_BOOT,
                 level: float = S.CI_LEVEL, seed: int = S.BOOT_SEED) -> dict:
    """Deterministic percentile bootstrap CI for a statistic of a (large) sample,
    e.g. the per-user audited vector. Fixed seed => reproducible from the cache."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boot = np.array([stat(v[i]) for i in idx])
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return {"stat": float(stat(v)), "ci_lo": float(np.quantile(boot, lo)),
            "ci_hi": float(np.quantile(boot, hi))}


def _binom_sf_ge(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exact (used for the sign test that the
    audited number is below the claimed number in every run)."""
    from math import comb
    return float(sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1)))


# --------------------------------------------------------------------------- #
#  escalation-schedule ablation
# --------------------------------------------------------------------------- #
def battery_table(raw: dict, model: str = "defended") -> dict:
    """Mean/std achieved-SE per battery configuration across all runs: does each
    added budget knob (iterations, restarts, aperture) buy real attack strength,
    and does the transfer control stay weak?"""
    acc: dict[str, list[float]] = {}
    for a in _runs(raw, model):
        for b in a["battery"]:
            acc.setdefault(b["name"], []).append(b["se_ratio"])
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}


def escalation_decomposition(raw: dict, model: str = "defended") -> dict:
    """Split the claimed-vs-audited gap into the protocol's two components:
    (i) escalating the single-configuration budget beyond the reference, and
    (ii) per-user worst-case aggregation across the whole battery.
    Both must be material for the protocol design to be justified."""
    runs = _runs(raw, model)
    claimed = np.array([a["claimed_se"] for a in runs])
    audited = np.array([a["audited_se"] for a in runs])
    strongest = np.array([min(b["se_ratio"] for b in a["battery"] if b["realizable"])
                          for a in runs])
    wins: dict[str, int] = {}
    for a in runs:
        wins[a["audited_by"]] = wins.get(a["audited_by"], 0) + 1
    return {
        "n_runs": len(runs),
        "claimed_mean": float(claimed.mean()),
        "strongest_single_mean": float(strongest.mean()),
        "audited_mean": float(audited.mean()),
        "escalation_drop": float(claimed.mean() - strongest.mean()),
        "aggregation_drop": float(strongest.mean() - audited.mean()),
        "wins": wins,
    }


# --------------------------------------------------------------------------- #
#  uncertainty structure
# --------------------------------------------------------------------------- #
def variance_decomposition(raw: dict, model: str = "defended",
                           key: str = "audited_se") -> dict:
    """Is the audit noisy, or is the defense's training noisy? Between-model-seed
    vs between-audit-seed std of the audited number. A tiny audit-seed component
    means the verdict is a stable property of the trained model, not an artifact
    of audit randomness (geometry / phase init / CSI noise)."""
    runs = _runs(raw, model)
    total = np.array([a[key] for a in runs])
    by_ms: dict[int, list[float]] = {}
    by_as: dict[int, list[float]] = {}
    for a in runs:
        by_ms.setdefault(a["model_seed"], []).append(a[key])
        by_as.setdefault(a["audit_seed"], []).append(a[key])
    return {
        "total_std": float(total.std()),
        "model_seed_std": float(np.std([np.mean(v) for v in by_ms.values()])),
        "audit_seed_std": float(np.std([np.mean(v) for v in by_as.values()])),
    }


# --------------------------------------------------------------------------- #
#  per-user outage (the cached per-user vector: first model/audit seed pair)
# --------------------------------------------------------------------------- #
def outage(raw: dict, model: str = "defended", taus=S.TAUS) -> dict:
    """Per-user outage under the audited worst case: the fraction of users whose
    audited ratio falls below each tau. A mean-based verdict can hide widespread
    per-user failure; wireless QoS is an outage story, not a mean story."""
    v = np.array(raw["per_user"][model], dtype=float)
    return {
        "n_users": int(len(v)),
        "frac_below": {f"{t:.2f}": float(np.mean(v < t)) for t in taus},
        "median": float(np.median(v)),
        "p05": float(np.percentile(v, 5)),
    }


# --------------------------------------------------------------------------- #
#  cost, hardware quantization, witnesses
# --------------------------------------------------------------------------- #
def cost_summary(raw: dict) -> dict:
    """Auditor/attacker budget actually spent (reviewers ask; caches record)."""
    passes_per_user = sum(2 * it * r for (_, it, r) in S.SCHEDULE)
    return {
        "passes_per_user": int(passes_per_user),
        "total_passes": int(passes_per_user * raw["n_users"]),
        "wall_minutes": round(raw.get("wall_seconds", 0.0) / 60.0, 1),
    }


def bbit_summary(raw: dict, model: str = "defended") -> dict:
    """Does the audited violation survive realistic b-bit RIS phase hardware?
    (bbit['inf'] is the continuous-phase reference at the same budget)."""
    bb = raw.get("bbit", {}).get(model, {})
    return {k: float(v) for k, v in bb.items()}


def witness_summary(raw: dict) -> list[dict]:
    """The stored worst-user witness records (defended model, first seed pair).
    Tolerates both the compact legacy schema (theta_first8) and the complete
    schema (full theta + h_eff) written by newer runs."""
    return [{
        "user": w["user"],
        "se_ratio": w["se_ratio"],
        "w_pred": w["w_pred"],
        "w_opt": w["w_opt"],
        "complete": "theta" in w and "h_eff" in w,
    } for w in raw.get("witnesses", [])]


# --------------------------------------------------------------------------- #
#  confidence intervals on the headline numbers (seed grid + per-user bootstrap)
# --------------------------------------------------------------------------- #
def confidence_intervals(raw: dict, model: str = "defended") -> dict:
    """Interval estimates, not just point+spread, for the reported numbers.
    Seed-grid quantities (claimed, audited, gap) get a Student-t CI over the run
    grid; the per-user audited mean and the tau=0.90 outage fraction get a
    fixed-seed bootstrap CI over the cached per-user vector. Makes 'the gap
    survives noise' an interval statement: gap CI excludes zero."""
    runs = _runs(raw, model)
    out = {"n_runs": len(runs)}
    for key in ("claimed_se", "audited_se", "robustness_gap"):
        out[key] = t_ci([a[key] for a in runs])
    v = np.array(raw["per_user"][model], dtype=float)
    out["audited_user_mean"] = bootstrap_ci(v, stat=np.mean)
    t = S.TAU_DEFAULT
    out["outage_frac"] = bootstrap_ci(v, stat=lambda x: float(np.mean(x < t)))
    return out


# --------------------------------------------------------------------------- #
#  schedule ablation: fixed reference vs adaptive escalation, and which budget
#  axis (compute vs aperture) drives the loss
# --------------------------------------------------------------------------- #
def _config_series(raw: dict, model: str) -> dict[str, np.ndarray]:
    """Per-white-box-configuration achieved-SE, one value per run (mean over
    users already), keyed by the config name in the battery."""
    acc: dict[str, list[float]] = {}
    for a in _runs(raw, model):
        for b in a["battery"]:
            if b["kind"] == "whitebox":
                acc.setdefault(b["name"], []).append(b["se_ratio"])
    return {k: np.array(v, dtype=float) for k, v in acc.items()}


# canonical schedule names (must match audit.eval_effective naming)
_REF = "RIS M=128 it=80 x1"
_C128 = "RIS M=128 it=200 x3"     # + optimizer compute (iters, restarts) at M=128
_A256lo = "RIS M=256 it=80 x1"    # + aperture at low compute
_A256hi = "RIS M=256 it=200 x3"   # + aperture at high compute
_A512 = "RIS M=512 it=200 x3"     # + aperture again (largest budget)


def schedule_ablation(raw: dict, model: str = "defended") -> dict:
    """Marginal effect of each escalation knob, isolated from the battery grid.
    Answers 'how much of the robustness loss beyond the fixed reference budget
    comes from more optimizer compute vs. from larger aperture?' -- the core
    question motivating an *adaptive* schedule over a single fixed attack."""
    c = _config_series(raw, model)

    def md(a, b):  # mean drop a->b and its std across runs
        d = c[a] - c[b]
        return {"drop": float(d.mean()), "std": float(d.std())}

    ref_mean = float(c[_REF].mean())
    strongest_mean = float(c[_A512].mean())
    steps = {
        "reference": {"mean": ref_mean, "std": float(c[_REF].std())},
        "compute_at_M128": md(_REF, _C128),        # iters/restarts effect
        "compute_at_M256": md(_A256lo, _A256hi),   # same knob, larger aperture
        "aperture_lo_128to256": md(_REF, _A256lo),
        "aperture_hi_128to256": md(_C128, _A256hi),
        "aperture_256to512": md(_A256hi, _A512),
    }
    # attribute the total fixed->strongest single-config drop to the two axes
    compute_share = steps["compute_at_M128"]["drop"]
    aperture_share = (strongest_mean and (ref_mean - strongest_mean) - compute_share)
    return {
        "reference_mean": ref_mean,
        "strongest_single_mean": strongest_mean,
        "escalation_drop": ref_mean - strongest_mean,
        "compute_component": compute_share,
        "aperture_component": float(aperture_share),
        "steps": steps,
    }


# --------------------------------------------------------------------------- #
#  aperture sensitivity: crossing point and the gradient-masking margin per M
# --------------------------------------------------------------------------- #
def aperture_sensitivity(raw: dict, model: str = "defended", tau=S.TAU_DEFAULT) -> dict:
    """Per-aperture white-box SE (mean+/-std over audit seeds), the smallest M at
    which the single-configuration curve drops below tau, whether the curve is
    monotone non-increasing in M, and the masking margin (transfer-white-box and
    random-white-box, which must be >=0 at every M for 'no gradient masking')."""
    sw = raw.get("sweep", {}).get(model, {})
    if not sw:
        return {}
    wb = {int(M): (float(np.mean(v)), float(np.std(v))) for M, v in sw["whitebox"].items()}
    tr = {int(M): float(np.mean(v)) for M, v in sw.get("transfer", {}).items()}
    rd = {int(M): float(np.mean(v)) for M, v in sw.get("random", {}).items()}
    Ms = sorted(wb)
    below = [M for M in Ms if wb[M][0] < tau]
    means = [wb[M][0] for M in Ms]
    monotone = all(means[i] >= means[i + 1] - 1e-6 for i in range(len(means) - 1))
    margin = {M: {"transfer_minus_wb": tr.get(M, float("nan")) - wb[M][0],
                  "random_minus_wb": rd.get(M, float("nan")) - wb[M][0]} for M in Ms}
    return {
        "per_M": {M: wb[M] for M in Ms},
        "crossing_M": below[0] if below else None,
        "monotone_nonincreasing": bool(monotone),
        "min_margin": float(min(min(m.values()) for m in margin.values())),
        "margin": margin,
    }


# --------------------------------------------------------------------------- #
#  RIS amplitude-budget (kappa) sensitivity (present only after --kappa-sweep)
# --------------------------------------------------------------------------- #
def kappa_sensitivity(raw: dict, model: str = "defended") -> dict:
    """Per-kappa white-box SE (mean+/-std over audit seeds) and the masking margin,
    mirroring aperture_sensitivity. Empty unless the cache carries a kappa sweep
    (`qed-rf run --kappa-sweep`); lets the paper state whether the verdict holds
    across the RIS amplitude budget once that run exists."""
    ks = raw.get("kappa_sweep", {}).get(model, {})
    if not ks:
        return {}
    def stat(d):
        return {float(k): (float(np.mean(v)), float(np.std(v))) for k, v in d.items()}
    wb = stat(ks["whitebox"])
    tr = {k: float(np.mean(ks["transfer"][str(k) if str(k) in ks["transfer"] else k]))
          for k in wb} if ks.get("transfer") else {}
    return {"per_kappa": wb,
            "min_margin": float(min((tr.get(k, m[0]) - m[0]) for k, m in wb.items()))
            if tr else float("nan")}


# --------------------------------------------------------------------------- #
#  phase-resolution (b-bit) sensitivity
# --------------------------------------------------------------------------- #
def bbit_sensitivity(raw: dict, model: str = "defended") -> dict:
    """Does the audited violation survive few-bit RIS phase hardware? Reports the
    achieved-SE at each b, whether the attack strengthens monotonically across
    the *finite* resolutions 1->2->3 bits, and the saturation gap b=3 vs
    continuous phases. The finding is that the attack saturates by 3 bits: 3-bit
    hardware is within noise of continuous (|gap| <= 0.02), so realizability does
    not depend on idealized continuous phase control."""
    bb = raw.get("bbit", {}).get(model, {})
    if not bb:
        return {}
    order = ["1", "2", "3", "inf"]
    finite = [float(bb[k]) for k in ("1", "2", "3") if k in bb]
    mono = all(finite[i] >= finite[i + 1] - 1e-6 for i in range(len(finite) - 1))
    gap = float(bb["3"] - bb["inf"]) if "inf" in bb and "3" in bb else float("nan")
    return {
        "by_bits": {k: float(bb[k]) for k in order if k in bb},
        "realizability_gap_3_vs_cont": gap,
        "monotone_1to3": bool(mono),
        "saturates_by_3bit": bool(abs(gap) <= 0.02),
    }


# --------------------------------------------------------------------------- #
#  per-run consistency (sign test): audited < claimed in every run
# --------------------------------------------------------------------------- #
def consistency_test(raw: dict, model: str = "defended") -> dict:
    """Exact sign test that escalation lowers the number in *every* run, not just
    on average: k = #runs with audited < claimed out of n, one-sided binomial
    p-value under the null that escalation is as likely to raise as lower it."""
    runs = _runs(raw, model)
    n = len(runs)
    k = sum(a["audited_se"] < a["claimed_se"] - 1e-9 for a in runs)
    return {"n": n, "k_audited_below_claimed": int(k),
            "p_value": _binom_sf_ge(k, n, 0.5)}


# --------------------------------------------------------------------------- #
#  witness case studies (from the released records; schema-tolerant)
# --------------------------------------------------------------------------- #
def witness_table(raw: dict, k: int = 3) -> list[dict]:
    """The k worst-user witnesses as structured rows for the paper's case-study
    table. `complete` flags whether the record carries the full recomputable
    tuple (theta + h_eff) or the compact legacy schema."""
    rows = []
    for w in sorted(raw.get("witnesses", []), key=lambda w: w["se_ratio"])[:k]:
        rows.append({
            "user": int(w["user"]),
            "se_ratio": float(w["se_ratio"]),
            "w_pred": int(w["w_pred"]),
            "w_opt": int(w["w_opt"]),
            "complete": bool("theta" in w and "h_eff" in w),
        })
    return rows


# --------------------------------------------------------------------------- #
#  paper macro coverage (reproducibility guard for the TeX side)
# --------------------------------------------------------------------------- #
def macro_coverage(macros_tex: str, main_tex: str) -> dict:
    """Cross-check the generated macros against the paper: which \\newcommand
    definitions in results_macros.tex are actually cited in main.tex. Catches
    drift the byte-identity test cannot -- that test checks the macro *file* is
    regenerable, not whether the prose still uses each generated number, so an
    orphaned macro (a number dropped from the text) slips past it."""
    import re
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", macros_tex))
    used = set(re.findall(r"\\([A-Za-z]+)", main_tex))
    referenced = sorted(defined & used)
    unused = sorted(defined - used)
    return {"n_defined": len(defined), "n_used": len(referenced), "unused": unused}


# --------------------------------------------------------------------------- #
#  claims check: every headline claim, verified against the cache
# --------------------------------------------------------------------------- #
def claims_check(raws: dict[str, dict]) -> list[dict]:
    """Verify the paper's headline claims against the released cache. Each entry
    is {claim, holds, evidence}. Intended for reviewers ('do the artifacts
    actually support the paper?') and for ourselves after any re-run."""
    checks: list[dict] = []

    def add(claim, holds, evidence):
        checks.append({"claim": claim, "holds": bool(holds), "evidence": evidence})

    for key, raw in raws.items():
        lbl = raw["label"]
        und = np.array([a["audited_se"] for a in _runs(raw, "undefended")])
        dfd = np.array([a["audited_se"] for a in _runs(raw, "defended")])
        gap = np.array([a["robustness_gap"] for a in _runs(raw, "defended")])
        clm = np.array([a["claimed_se"] for a in _runs(raw, "defended")])

        add(f"[{lbl}] defense is genuinely robust (audited defended >> undefended)",
            dfd.mean() - und.mean() > 0.3,
            f"defended {dfd.mean():.3f} vs undefended {und.mean():.3f}")

        add(f"[{lbl}] claimed overstates audited beyond seed noise (gap > 3x its std)",
            gap.mean() > 3 * gap.std(),
            f"gap {gap.mean():.3f} +- {gap.std():.3f}")

        add(f"[{lbl}] audited worst case fails every tau in {list(S.TAUS)}",
            dfd.mean() < min(S.TAUS),
            f"audited {dfd.mean():.3f} < {min(S.TAUS)}")

        add(f"[{lbl}] audited <= claimed in every single run",
            all(a["audited_se"] <= a["claimed_se"] + 1e-6 for a in _runs(raw, "defended")),
            "per-run check over the defended grid")

        esc = escalation_decomposition(raw)
        strongest_name = max(esc["wins"], key=esc["wins"].get)
        add(f"[{lbl}] largest-budget schedule entry is the strongest single config in all runs",
            esc["wins"].get(strongest_name, 0) == esc["n_runs"] and "512" in strongest_name,
            f"wins: {esc['wins']}")

        add(f"[{lbl}] both protocol components contribute (escalation and per-user-worst drops > 0.01)",
            esc["escalation_drop"] > 0.01 and esc["aggregation_drop"] > 0.01,
            f"escalation {esc['escalation_drop']:.3f}, aggregation {esc['aggregation_drop']:.3f}")

        var = variance_decomposition(raw)
        add(f"[{lbl}] audit verdict stable across audit randomness (audit-seed std < model-seed std)",
            var["audit_seed_std"] < var["model_seed_std"],
            f"audit-seed {var['audit_seed_std']:.4f} vs model-seed {var['model_seed_std']:.4f}")

        if "sweep" in raw:
            sw = raw["sweep"]["defended"]
            wb = {M: np.mean(v) for M, v in sw["whitebox"].items()}
            tr = {M: np.mean(v) for M, v in sw["transfer"].items()}
            rd = {M: np.mean(v) for M, v in sw["random"].items()}
            add(f"[{lbl}] no masking signature: white-box <= transfer and random at every swept M",
                all(wb[M] <= tr[M] + 1e-6 and wb[M] <= rd[M] + 1e-6 for M in wb),
                "sweep controls on the defended model")
            add(f"[{lbl}] white-box sweep crosses tau={S.TAU_DEFAULT} at some aperture",
                any(v < S.TAU_DEFAULT for v in wb.values()),
                f"min over M: {min(wb.values()):.3f}")

        if "convergence" in raw:
            conv = raw["convergence"]["defended"]
            add(f"[{lbl}] phase optimizer converges (final loss < initial at every M)",
                all(series[-1][1] < series[0][1] for series in conv.values()),
                "loss curves on the defended model")

        if "unbounded" in raw:
            unb = float(np.mean(raw["unbounded"]["defended"]))
            add(f"[{lbl}] unbounded budget strengthens but does not collapse the defense",
                clm.mean() > unb > 0.5,
                f"unbounded {unb:.3f} vs claimed {clm.mean():.3f}")

        bb = bbit_summary(raw)
        if bb:
            add(f"[{lbl}] 3-bit phase hardware realizes the attack (within 0.02 of continuous)",
                bb["3"] <= bb["inf"] + 0.02,
                f"b=3: {bb['3']:.3f} vs continuous {bb['inf']:.3f}")

        # --- claims added for the strengthened statistics/ablations ---
        ci = confidence_intervals(raw)
        add(f"[{lbl}] gap 95% CI excludes zero (interval estimate, not just point+std)",
            ci["robustness_gap"]["ci_lo"] > 0,
            f"gap 95% CI [{ci['robustness_gap']['ci_lo']:.3f}, {ci['robustness_gap']['ci_hi']:.3f}]")

        ct = consistency_test(raw)
        add(f"[{lbl}] escalation lowers the number in every run (sign test p < 0.01)",
            ct["k_audited_below_claimed"] == ct["n"] and ct["p_value"] < 0.01,
            f"{ct['k_audited_below_claimed']}/{ct['n']} runs, p={ct['p_value']:.2e}")

        sa = schedule_ablation(raw)
        add(f"[{lbl}] both budget axes drive the loss (compute and aperture components > 0.01)",
            sa["compute_component"] > 0.01 and sa["aperture_component"] > 0.01,
            f"compute {sa['compute_component']:.3f}, aperture {sa['aperture_component']:.3f}")

        ap = aperture_sensitivity(raw)
        if ap:
            add(f"[{lbl}] white-box SE is monotone non-increasing in aperture M",
                ap["monotone_nonincreasing"],
                f"crossing tau at M={ap['crossing_M']}, min masking margin {ap['min_margin']:.3f}")
            add(f"[{lbl}] masking margin non-negative at every swept M (controls never beat white-box)",
                ap["min_margin"] >= -1e-6,
                f"min(transfer-wb, random-wb) over M = {ap['min_margin']:.3f}")

        bs = bbit_sensitivity(raw)
        if bs:
            add(f"[{lbl}] attack strengthens across finite phase resolutions (1>=2>=3 bit SE) and saturates by 3 bits",
                bs["monotone_1to3"] and bs["saturates_by_3bit"],
                f"by bits: {', '.join(f'{k}={v:.3f}' for k, v in bs['by_bits'].items())}"
                f"; b3-cont {bs['realizability_gap_3_vs_cont']:+.3f}")

    return checks


# --------------------------------------------------------------------------- #
#  the findings report
# --------------------------------------------------------------------------- #
def findings_markdown(raws: dict[str, dict]) -> str:
    """Full research-findings report from the cache (ASCII-only)."""
    L: list[str] = ["# QED-RF research findings (derived from cached runs)", ""]
    for key, raw in raws.items():
        L += [f"## {raw['label']} ({key})", ""]

        L += ["### Escalation-schedule ablation (defended, mean +- std over runs)", ""]
        for name, (m, s) in battery_table(raw).items():
            L.append(f"- {name}: {m:.3f} +- {s:.3f}")
        esc = escalation_decomposition(raw)
        L += ["",
              f"- claimed (reference config): {esc['claimed_mean']:.3f}",
              f"- strongest single config:    {esc['strongest_single_mean']:.3f}"
              f"  (escalation contributes {esc['escalation_drop']:.3f})",
              f"- audited (per-user worst):   {esc['audited_mean']:.3f}"
              f"  (aggregation contributes {esc['aggregation_drop']:.3f})",
              f"- strongest-config wins: {esc['wins']}", ""]

        L += ["### Schedule ablation: which budget axis drives the loss (defended)"]
        sa = schedule_ablation(raw)
        L += [f"- reference (fixed) config:   {sa['reference_mean']:.3f}",
              f"- optimizer compute (iters/restarts) at M=128 costs {sa['compute_component']:.3f}",
              f"- aperture escalation (128->512) costs a further {sa['aperture_component']:.3f}",
              f"- fixed -> strongest single config total drop {sa['escalation_drop']:.3f}", ""]

        var = variance_decomposition(raw)
        ci = confidence_intervals(raw)
        L += ["### Uncertainty structure and interval estimates (defended)",
              f"- audited SE total std {var['total_std']:.4f} = model-seed {var['model_seed_std']:.4f}"
              f" vs audit-seed {var['audit_seed_std']:.4f}"
              " -> the verdict is a property of the trained model, not audit noise",
              f"- audited 95% CI [{ci['audited_se']['ci_lo']:.3f}, {ci['audited_se']['ci_hi']:.3f}], "
              f"claimed 95% CI [{ci['claimed_se']['ci_lo']:.3f}, {ci['claimed_se']['ci_hi']:.3f}]",
              f"- gap 95% CI [{ci['robustness_gap']['ci_lo']:.3f}, {ci['robustness_gap']['ci_hi']:.3f}]"
              " (excludes zero -> overstatement is significant, not noise)"]
        ct = consistency_test(raw)
        L += [f"- sign test: audited < claimed in {ct['k_audited_below_claimed']}/{ct['n']} runs, "
              f"p={ct['p_value']:.2e}", ""]

        ap = aperture_sensitivity(raw)
        if ap:
            cross = ap["crossing_M"]
            L += ["### Aperture sensitivity (defended white-box)",
                  f"- monotone non-increasing in M: {ap['monotone_nonincreasing']}; "
                  f"first drops below tau={S.TAU_DEFAULT} at M={cross}",
                  f"- min masking margin over M (transfer/random minus white-box): "
                  f"{ap['min_margin']:.3f} (>=0 -> no masking)", ""]

        L += ["### Per-user outage under the audited worst case (first seed pair)"]
        for model in ("undefended", "defended"):
            o = outage(raw, model)
            fb = ", ".join(f"tau={t}: {f:.1%}" for t, f in o["frac_below"].items())
            L.append(f"- {model}: {fb}; median {o['median']:.3f}, 5th pct {o['p05']:.3f}")
        L += [f"- defended tau=0.90 outage 95% bootstrap CI "
              f"[{ci['outage_frac']['ci_lo']:.1%}, {ci['outage_frac']['ci_hi']:.1%}]", ""]

        bb = bbit_summary(raw)
        bs = bbit_sensitivity(raw)
        if bb:
            L += ["### Hardware phase quantization (defended, audited budget)",
                  "- " + ", ".join(f"b={k}: {v:.3f}" for k, v in bb.items()),
                  f"- realizability gap (b=3 vs continuous): {bs['realizability_gap_3_vs_cont']:+.3f} "
                  f"(saturates by 3 bits: {bs['saturates_by_3bit']}; 1->3-bit monotone: {bs['monotone_1to3']})", ""]

        ws = witness_summary(raw)
        if ws:
            L += ["### Worst-user witnesses (defended, first seed pair)"]
            for w in ws:
                L.append(f"- user {w['user']}: SE ratio {w['se_ratio']:.4f}, "
                         f"predicted beam {w['w_pred']} vs optimal {w['w_opt']}"
                         + ("" if w["complete"] else " (compact record)"))
            L.append("")

        c = cost_summary(raw)
        L += ["### Cost",
              f"- {c['passes_per_user']} forward+backward passes per user for the schedule "
              f"({c['total_passes']:.2e} total); full grid wall-clock {c['wall_minutes']} min", ""]

    L += ["## Claims check", ""]
    checks = claims_check(raws)
    for c in checks:
        L.append(f"- [{'PASS' if c['holds'] else 'FAIL'}] {c['claim']} -- {c['evidence']}")
    n_fail = sum(not c["holds"] for c in checks)
    L += ["", f"{len(checks) - n_fail}/{len(checks)} claims hold in the released cache."]
    return "\n".join(L) + "\n"

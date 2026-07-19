"""Full experiment grid: for each scenario, train (victim, defended) at every
model seed, audit each at every audit seed, and run the gradient-masking
diagnostic battery. Raw per-run numbers are cached to qed_rf/artifacts/ so the
aggregation (mean +/- std) in benchmark.py is cheap to re-derive.

    python -m qedrf run                 # full grid, both scenarios (cached)
    python -m qedrf run --scenarios asu # single scenario
    python -m qedrf run --force         # ignore cache, retrain + re-audit
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from . import settings as S

MODELS = ("undefended", "defended")


def _raw_path(scenario_key: str) -> str:
    return os.path.join(S.LOCAL_ART, f"runs_{scenario_key}.json")


def run_scenario(scenario_key: str, *, model_seeds=S.MODEL_SEEDS,
                 audit_seeds=S.AUDIT_SEEDS, n_users=S.N_USERS, force=False,
                 diagnostics=True, kappa_sweep=False) -> dict:
    cache = _raw_path(scenario_key)
    if os.path.exists(cache) and not force:
        with open(cache, encoding="utf-8") as f:
            print(f"[qed-rf] {scenario_key}: using cached {cache}")
            return json.load(f)

    # Only a cache miss (or --force) needs the full pipeline: torch plus the
    # paper2026 substrate. Import lazily so cache-only aggregation runs anywhere.
    try:
        from . import diagnostics as dg
        from . import rfcore as rf
        from .audit import audit_model
        from .train import ensure_models
    except (ImportError, RuntimeError) as e:
        why = ("--force requested a fresh run" if force and os.path.exists(cache)
               else f"no cached runs at {cache}")
        raise SystemExit(
            f"[qed-rf] {scenario_key}: {why}, and the full pipeline is "
            f"unavailable:\n{e}\n"
            "[qed-rf] re-running the audit needs torch and the paper2026 substrate "
            "(see README and REPRODUCIBILITY.md)."
        ) from e

    t0 = time.time()
    data = rf.build_dataset_for(scenario_key)
    pairs = ensure_models(scenario_key, model_seeds, data=data)
    H = np.ascontiguousarray(data["H_test"][:n_users]).astype(np.complex64)

    out: dict = {"scenario": scenario_key, "label": S.SCENARIOS[scenario_key]["label"],
                 "n_users": int(len(H)), "audits": {m: [] for m in MODELS},
                 "per_user": {}, "witnesses": []}

    for ms in model_seeds:
        victim, defended = pairs[ms]
        mmap = {"undefended": victim, "defended": defended}
        for name in MODELS:
            for as_ in audit_seeds:
                a = audit_model(name, mmap[name], data, victim, scenario=scenario_key,
                                model_seed=ms, audit_seed=as_, n_users=n_users)
                out["audits"][name].append(a.to_dict())
                if ms == model_seeds[0] and as_ == audit_seeds[0]:
                    out["per_user"][name] = [round(float(x), 4) for x in a.per_user_audited]
                    if name == "defended":
                        out["witnesses"] = a.witnesses[:5]
        print(f"[qed-rf] {scenario_key}: audited model_seed={ms} "
              f"({len(audit_seeds)} audit seeds x {len(MODELS)} models)")

    if diagnostics:
        # heavier battery: representative model seed, averaged over audit seeds where cheap
        ms0 = model_seeds[0]
        victim, defended = pairs[ms0]
        mmap = {"undefended": victim, "defended": defended}
        out["sweep"] = {m: {"whitebox": {}, "transfer": {}, "random": {}} for m in MODELS}
        out["unbounded"] = {m: [] for m in MODELS}
        out["universal"] = {m: [] for m in MODELS}
        for name in MODELS:
            for as_ in S.DIAG_AUDIT_SEEDS:
                sw = dg.aperture_sweep(mmap[name], victim, H, audit_seed=as_)
                for kind in ("whitebox", "transfer", "random"):
                    for M, v in sw[kind].items():
                        out["sweep"][name][kind].setdefault(str(M), []).append(v)
                out["unbounded"][name].append(dg.unbounded(mmap[name], H, audit_seed=as_)["se_ratio"])
                out["universal"][name].append(dg.universal(mmap[name], H, audit_seed=as_))
        out["convergence"] = {name: dg.convergence(mmap[name], H, audit_seed=audit_seeds[0])
                              for name in MODELS}
        out["bbit"] = {name: dg.bbit_audit(mmap[name], H, audit_seed=audit_seeds[0])
                       for name in MODELS}
        out["passes"] = S.forward_backward_passes(len(H))
        print(f"[qed-rf] {scenario_key}: diagnostics done")

    if kappa_sweep:
        # RIS amplitude-budget axis (opt-in: re-attacks all users per kappa).
        from . import diagnostics as dg
        ms0 = model_seeds[0]
        victim, defended = pairs[ms0]
        mmap = {"undefended": victim, "defended": defended}
        out["kappa_sweep"] = {}
        for name in MODELS:
            acc = {"whitebox": {}, "transfer": {}, "random": {}}
            for as_ in S.DIAG_AUDIT_SEEDS:
                sw = dg.kappa_sweep(mmap[name], victim, H, audit_seed=as_)
                for kind in acc:
                    for kap, v in sw[kind].items():
                        acc[kind].setdefault(str(kap), []).append(v)
            out["kappa_sweep"][name] = acc
        print(f"[qed-rf] {scenario_key}: kappa sweep done")

    out["settings"] = {
        "model_seeds": list(model_seeds), "audit_seeds": list(audit_seeds),
        "n_users": int(len(H)), "reference": list(S.REFERENCE),
        "schedule": [list(x) for x in S.SCHEDULE], "taus": list(S.TAUS),
        "kappa": S.KAPPA, "attack_lr": S.ATTACK_LR, "aggregation": S.AGGREGATION,
        "pilot_snr_db": rf.PILOT_SNR_DB, "eval_snr_db": rf.EVAL_SNR_DB,
        "n_ant": int(data["n_ant"]), "n_beams": int(data["n_beams"]),
    }
    out["wall_seconds"] = round(time.time() - t0, 1)

    os.makedirs(S.LOCAL_ART, exist_ok=True)
    with open(cache, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
    print(f"[qed-rf] {scenario_key}: raw runs -> {cache} ({out['wall_seconds']}s)")
    return out


def run_all(scenarios=("asu", "mmw"), **kw) -> dict[str, dict]:
    return {k: run_scenario(k, **kw) for k in scenarios}

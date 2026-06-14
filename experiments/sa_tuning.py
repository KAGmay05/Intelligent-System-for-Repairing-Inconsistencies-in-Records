"""
experiments/sa_tuning.py
------------------------
Can Simulated Annealing escape the greedy trap?

hard_instances.py showed that SA (warm-started from greedy) stays stuck at the
greedy solution on the "trap" family: the only improving move — add the single
all-covering repair `big` and drop every singleton — requires a large uphill
cost step that the default temperature rejects.  This script studies HOW to make
SA escape, sweeping two levers:

  1. TEMPERATURE  — a hotter start (and slower cooling) accepts the uphill step,
     letting SA later cool down and prune the redundant singletons.
  2. NEIGHBOURHOOD OPERATOR — a "repair-and-clean" move that, after toggling a
     repair in, immediately removes any repair that became redundant.  Adding
     `big` then drops all singletons IN THE SAME MOVE, so the step is downhill
     and is accepted regardless of temperature.  This is the robust fix.

The takeaway for the report: a metaheuristic inherits the bias of its
initialisation; escaping a deep local optimum is a question of operator/temperature
design, not luck.  CP-SAT, being exact, is immune.

Usage:
    python experiments/sa_tuning.py
    python experiments/sa_tuning.py --trap-n 64 --seeds 1 2 3 4 5 --iters 8000
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector.repair_optimizer import (  # noqa: E402
    solve_minimum_repairs_greedy,
    solve_minimum_repairs_sa,
)
from hard_instances import make_greedy_trap, solve_cpsat_bench  # noqa: E402

DATA = ROOT / "data"


# ─────────────────────────────────────────────────────────────────
# SA VARIANT WITH A REPAIR-AND-CLEAN (REDUNDANCY-PRUNING) OPERATOR
# ─────────────────────────────────────────────────────────────────

def solve_sa_pruned(issues, repairs, temp0=25.0, cooling=0.997, iters=8000, seed=42):
    """SA identical in spirit to the production solver, but every neighbour is
    minimalised: redundant selected repairs (those whose covered issues are all
    covered by some OTHER selected repair) are removed before scoring.  This lets
    a single 'add big' move shed all the now-useless singletons at once."""
    rng = random.Random(seed)
    repair_ids = sorted(repairs.keys())

    issue_to_valid = {
        issue["issue_id"]: [rid for rid in issue["repair_ids"] if rid in repairs]
        for issue in issues
    }
    max_cost = max((rep.cost for rep in repairs.values()), default=1)
    penalty_uncovered = max(1000, len(repair_ids) * max_cost)

    def _uncovered(sel):
        return sum(
            1 for v in issue_to_valid.values()
            if v and not any(r in sel for r in v)
        )

    def _prune(sel: set) -> set:
        """Drop repairs that are redundant given the rest of the selection."""
        sel = set(sel)
        # coverage count per issue among the current selection
        cover_count = {}
        for iid, v in issue_to_valid.items():
            cover_count[iid] = sum(1 for r in v if r in sel)
        # Remove CHEAP, low-value coverers first and keep the heavy lifters: a
        # single high-coverage repair (e.g. `big`) should survive while the many
        # cheap singletons it makes redundant are shed.  (Removing expensive
        # repairs first would discard `big` and keep the costly singleton ladder.)
        for rid in sorted(sel, key=lambda r: repairs[r].cost):
            # issues where rid is the ONLY coverer cannot lose it
            only_coverer = any(
                rid in issue_to_valid[iid] and cover_count[iid] <= 1
                for iid in issue_to_valid
            )
            if not only_coverer:
                sel.discard(rid)
                for iid in issue_to_valid:
                    if rid in issue_to_valid[iid]:
                        cover_count[iid] -= 1
        return sel

    def _obj(sel):
        return sum(repairs[r].cost for r in sel) + penalty_uncovered * _uncovered(sel) + 0.1 * len(sel)

    greedy_sel, _, _, _ = solve_minimum_repairs_greedy(issues, repairs)
    current = _prune({r.repair_id for r in greedy_sel})
    cur_obj = _obj(current)
    best, best_obj = set(current), cur_obj

    temp = max(1e-6, float(temp0))
    cool = min(0.99999, max(0.90, float(cooling)))

    for _ in range(max(1, iters)):
        neighbor = set(current)
        if rng.random() < 0.65:
            rid = rng.choice(repair_ids)
            neighbor.discard(rid) if rid in neighbor else neighbor.add(rid)
        else:
            issue = rng.choice(issues)
            cands = issue_to_valid.get(issue["issue_id"], [])
            if cands:
                neighbor.add(rng.choice(cands))
        neighbor = _prune(neighbor)

        n_obj = _obj(neighbor)
        delta = n_obj - cur_obj
        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
            current, cur_obj = neighbor, n_obj
            if n_obj < best_obj:
                best, best_obj = set(neighbor), n_obj
        temp = max(1e-6, temp * cool)

    cost = sum(repairs[r].cost for r in best)
    return cost, _uncovered(best), len(best)


# ─────────────────────────────────────────────────────────────────
# RUNNERS
# ─────────────────────────────────────────────────────────────────

def _run_prod_sa(issues, repairs, temp0, cooling, iters, seed):
    t0 = time.perf_counter()
    selected, _c, uncovered, _d = solve_minimum_repairs_sa(
        issues, repairs, temp0=temp0, cooling=cooling, iters=iters, seed=seed)
    rt = time.perf_counter() - t0
    return sum(r.cost for r in selected), len(uncovered), rt


def _run_pruned_sa(issues, repairs, temp0, cooling, iters, seed):
    t0 = time.perf_counter()
    cost, uncovered, _n = solve_sa_pruned(
        issues, repairs, temp0=temp0, cooling=cooling, iters=iters, seed=seed)
    rt = time.perf_counter() - t0
    return cost, uncovered, rt


def main():
    ap = argparse.ArgumentParser(description="¿Puede SA escapar de la trampa-greedy?")
    ap.add_argument("--trap-n", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--output", default=str(DATA / "sa_tuning_report.json"))
    args = ap.parse_args()

    issues, repairs = make_greedy_trap(args.trap_n)
    opt = solve_cpsat_bench(issues, repairs, time_limit=20.0)["cost"]
    greedy_sel, _, _, _ = solve_minimum_repairs_greedy(issues, repairs)
    greedy_cost = sum(r.cost for r in greedy_sel)

    # (label, operator, temp0, cooling)
    configs = [
        ("SA default        (T0=25,   estándar)", "prod",   25.0,    0.997),
        ("SA caliente       (T0=2000, estándar)", "prod",   2000.0,  0.999),
        ("SA muy caliente   (T0=20000,estándar)", "prod",   20000.0, 0.9995),
        ("SA + poda redund. (T0=25)",             "pruned", 25.0,    0.997),
        ("SA + poda redund. (T0=2000)",           "pruned", 2000.0,  0.999),
    ]

    print(f"\n{'═'*100}")
    print(f"  ¿SA ESCAPA DE LA TRAMPA?   trap n={args.trap_n}   óptimo CP-SAT={opt}   greedy={greedy_cost} "
          f"(+{100*(greedy_cost-opt)/opt:.0f}%)")
    print(f"  (media ± peor caso sobre {len(args.seeds)} semillas, {args.iters} iteraciones)")
    print(f"{'═'*100}")

    results = []
    for label, op, t0, cool in configs:
        costs, uncs, rts = [], [], []
        for s in args.seeds:
            runner = _run_prod_sa if op == "prod" else _run_pruned_sa
            cost, unc, rt = runner(issues, repairs, t0, cool, args.iters, s)
            costs.append(cost); uncs.append(unc); rts.append(rt)
        avg_cost = sum(costs) / len(costs)
        best_cost = min(costs)
        avg_gap = 100 * (avg_cost - opt) / opt
        best_gap = 100 * (best_cost - opt) / opt
        escaped = sum(1 for c in costs if c <= opt * 1.01)   # within 1% of optimum
        results.append({
            "config": label, "operator": op, "temp0": t0, "cooling": cool,
            "avg_cost": round(avg_cost, 1), "best_cost": best_cost,
            "avg_gap_pct": round(avg_gap, 1), "best_gap_pct": round(best_gap, 1),
            "escaped_runs": f"{escaped}/{len(args.seeds)}",
            "avg_runtime_s": round(sum(rts) / len(rts), 3),
            "max_uncovered": max(uncs),
        })
        print(f"  {label:<42} coste medio={avg_cost:>8.0f} (+{avg_gap:>5.0f}%)  "
              f"mejor={best_cost:>6} (+{best_gap:>4.0f}%)  "
              f"escapó={escaped}/{len(args.seeds)}  {sum(rts)/len(rts):.2f}s")

    print(f"{'═'*100}\n")
    print("  Lectura: SA por defecto queda atrapado en la solución de greedy. Subir la")
    print("  temperatura ayuda pero es frágil; el operador de poda de redundancia escapa")
    print("  de forma robusta porque convierte 'añadir big' en un movimiento cuesta abajo.")
    print("  CP-SAT, exacto, encuentra el óptimo siempre sin necesidad de ajuste.\n")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump({
            "trap_n": args.trap_n, "optimum": opt, "greedy_cost": greedy_cost,
            "seeds": args.seeds, "iters": args.iters, "results": results,
        }, fh, indent=4, ensure_ascii=False)
    print(f"Reporte guardado en: {out}")


if __name__ == "__main__":
    main()

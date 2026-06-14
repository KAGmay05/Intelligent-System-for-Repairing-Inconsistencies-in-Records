"""
experiments/hard_instances.py
-----------------------------
Optimizer benchmark for the minimum-cost repair problem.

WHY THIS EXISTS
    On the real dataset almost every issue has a single candidate repair, so the
    minimum-cost hitting-set is trivial and CP-SAT / greedy / SA all return the
    SAME cost — the comparison carries no signal.  To compare the optimizers
    meaningfully we need instances where repairs COMPETE: one repair covers many
    issues, issues have several candidate repairs, and costs vary.  This script
    generates such instances in the exact (issues, repairs) format the pipeline
    solvers consume, then runs the SAME solver code used in production
    (detector/repair_optimizer.py) and reports cost, optimality gap and runtime.

TWO INSTANCE FAMILIES
    random  — weighted set-cover with tunable size/overlap (the typical case;
              greedy is usually near-optimal, SA sometimes closes the small gap).
    trap    — the classic worst case for greedy set-cover: a single cheap repair
              covers everything, but a ladder of singleton repairs lures greedy
              into a solution ~ln(n)x more expensive.  CP-SAT finds the optimum;
              this exposes the qualitative exact-vs-heuristic difference.

Usage:
    python experiments/hard_instances.py
    python experiments/hard_instances.py --sizes 50 100 200 --seeds 1 2 3
    python experiments/hard_instances.py --cpsat-time 30
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ortools.sat.python import cp_model  # noqa: E402

from detector.repair_optimizer import (  # noqa: E402
    RepairOption,
    solve_minimum_repairs_greedy,
    solve_minimum_repairs_sa,
)

DATA = ROOT / "data"


# ─────────────────────────────────────────────────────────────────
# INSTANCE GENERATORS  (output: issues list + repairs dict)
# ─────────────────────────────────────────────────────────────────

def _repair(rid: str, cost: int) -> RepairOption:
    """A minimal RepairOption — only repair_id and cost matter for the solvers."""
    return RepairOption(repair_id=rid, description=rid, cost=int(cost),
                        action="noop", target={})


def make_random_instance(n_issues: int, n_repairs: int, coverage_p: float,
                        cost_lo: int, cost_hi: int, seed: int):
    """Random weighted set-cover.

    Each repair covers each issue independently with probability coverage_p, so
    repairs overlap and most issues have several candidate repairs.  Any issue
    left uncovered gets one random repair assigned, guaranteeing feasibility.
    """
    rng = random.Random(seed)
    repairs = {f"r{j}": _repair(f"r{j}", rng.randint(cost_lo, cost_hi)) for j in range(n_repairs)}
    repair_ids = list(repairs.keys())

    issue_candidates = [[] for _ in range(n_issues)]
    for j, rid in enumerate(repair_ids):
        for i in range(n_issues):
            if rng.random() < coverage_p:
                issue_candidates[i].append(rid)

    for i in range(n_issues):
        if not issue_candidates[i]:                      # guarantee feasibility
            issue_candidates[i].append(rng.choice(repair_ids))

    issues = [
        {"issue_id": f"i{i}", "type": "synthetic", "repair_ids": issue_candidates[i]}
        for i in range(n_issues)
    ]
    return issues, repairs


def make_greedy_trap(n: int, scale: int = 10000, eps: float = 0.5):
    """Classic greedy set-cover worst case (~ln(n) approximation gap).

    - One 'big' repair covers ALL n issues at cost scale*(1+eps)  → the optimum.
    - Singleton repairs a_i cover issue i alone at cost scale/(n-i+1); their
      cost/coverage ratio always edges out 'big', so greedy keeps picking
      singletons and ends up paying ~scale*H(n) instead of scale*(1+eps).
    """
    issues, repairs = [], {}
    big_cost = int(round(scale * (1 + eps)))
    repairs["big"] = _repair("big", big_cost)

    for i in range(1, n + 1):
        rid = f"a{i}"
        repairs[rid] = _repair(rid, max(1, int(round(scale / (n - i + 1)))))
        issues.append({"issue_id": f"x{i}", "type": "trap", "repair_ids": [rid, "big"]})

    return issues, repairs


# ─────────────────────────────────────────────────────────────────
# SOLVERS
# ─────────────────────────────────────────────────────────────────

def solve_cpsat_bench(issues, repairs, time_limit: float):
    """Exact weighted hitting-set with optimality reporting (cost, proven-optimal, gap, runtime)."""
    model = cp_model.CpModel()
    var = {rid: model.NewBoolVar(rid) for rid in repairs}
    for issue in issues:
        valid = [rid for rid in issue["repair_ids"] if rid in var]
        if valid:
            model.Add(sum(var[rid] for rid in valid) >= 1)
    model.Minimize(sum(repairs[rid].cost * var[rid] for rid in var))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    t0 = time.perf_counter()
    status = solver.Solve(model)
    runtime = time.perf_counter() - t0

    proven = status == cp_model.OPTIMAL
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        cost = int(round(solver.ObjectiveValue()))
        bound = solver.BestObjectiveBound()
        n_sel = sum(1 for rid in var if solver.Value(var[rid]))
    else:
        cost, bound, n_sel = None, None, None
    return {"cost": cost, "proven_optimal": proven, "bound": bound,
            "n_selected": n_sel, "runtime_s": runtime}


def _run_heuristic(fn, issues, repairs, **kw):
    t0 = time.perf_counter()
    selected, _covered, uncovered, _dec = fn(issues, repairs, **kw)
    runtime = time.perf_counter() - t0
    cost = sum(r.cost for r in selected)
    return {"cost": cost, "n_selected": len(selected),
            "uncovered": len(uncovered), "runtime_s": runtime}


# ─────────────────────────────────────────────────────────────────
# COMPARISON
# ─────────────────────────────────────────────────────────────────

def compare_one(name: str, issues, repairs, cpsat_time: float, sa_iters: int):
    cp = solve_cpsat_bench(issues, repairs, cpsat_time)
    gr = _run_heuristic(solve_minimum_repairs_greedy, issues, repairs)
    sa = _run_heuristic(solve_minimum_repairs_sa, issues, repairs, iters=sa_iters)

    opt = cp["cost"]
    def gap(c):
        if opt is None or opt == 0 or c is None:
            return None
        return round(100.0 * (c - opt) / opt, 2)

    return {
        "instance": name,
        "n_issues": len(issues),
        "n_repairs": len(repairs),
        "cpsat":  {**cp, "gap_pct": 0.0 if cp["proven_optimal"] else None},
        "greedy": {**gr, "gap_pct": gap(gr["cost"])},
        "sa":     {**sa, "gap_pct": gap(sa["cost"])},
    }


def print_row(r):
    cp, gr, sa = r["cpsat"], r["greedy"], r["sa"]
    opt_flag = "✔" if cp["proven_optimal"] else "~"
    print(
        f"  {r['instance']:<22} I={r['n_issues']:>4} R={r['n_repairs']:>4} │ "
        f"CP-SAT {str(cp['cost']):>7}{opt_flag} ({cp['runtime_s']*1000:>6.0f}ms) │ "
        f"greedy {str(gr['cost']):>7} +{str(gr['gap_pct']):>6}% ({gr['runtime_s']*1000:>5.0f}ms) │ "
        f"SA {str(sa['cost']):>7} +{str(sa['gap_pct']):>6}% ({sa['runtime_s']*1000:>6.0f}ms)"
    )


def main():
    ap = argparse.ArgumentParser(description="Benchmark de optimizadores: CP-SAT vs greedy vs SA")
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200],
                    help="Tamaños (nº de issues) para la familia aleatoria")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                    help="Semillas por tamaño en la familia aleatoria")
    ap.add_argument("--coverage", type=float, default=0.08,
                    help="Prob. de que un repair cubra un issue (densidad de solapamiento)")
    ap.add_argument("--repairs-ratio", type=float, default=0.6,
                    help="nº de repairs = ratio * nº de issues")
    ap.add_argument("--cost-lo", type=int, default=1)
    ap.add_argument("--cost-hi", type=int, default=10)
    ap.add_argument("--trap-sizes", type=int, nargs="+", default=[16, 32, 64, 128],
                    help="Tamaños n para la familia trampa-greedy")
    ap.add_argument("--cpsat-time", type=float, default=20.0,
                    help="Límite de tiempo de CP-SAT por instancia (s)")
    ap.add_argument("--sa-iters", type=int, default=8000)
    ap.add_argument("--output", default=str(DATA / "hard_instances_report.json"))
    args = ap.parse_args()

    results = []

    print(f"\n{'═'*120}")
    print("  FAMILIA ALEATORIA  (set-cover ponderado con solapamiento)")
    print(f"{'═'*120}")
    for n in args.sizes:
        n_rep = max(2, int(round(n * args.repairs_ratio)))
        for s in args.seeds:
            issues, repairs = make_random_instance(
                n, n_rep, args.coverage, args.cost_lo, args.cost_hi, seed=s)
            r = compare_one(f"random n={n} seed={s}", issues, repairs,
                            args.cpsat_time, args.sa_iters)
            results.append(r)
            print_row(r)

    print(f"\n{'═'*120}")
    print("  FAMILIA TRAMPA-GREEDY  (un repair óptimo cubre todo; greedy cae en la escalera)")
    print(f"{'═'*120}")
    for n in args.trap_sizes:
        issues, repairs = make_greedy_trap(n)
        r = compare_one(f"trap n={n}", issues, repairs, args.cpsat_time, args.sa_iters)
        results.append(r)
        print_row(r)

    # ── Summary ───────────────────────────────────────────────────
    rnd = [r for r in results if r["instance"].startswith("random")]
    trp = [r for r in results if r["instance"].startswith("trap")]

    def _avg_gap(rows, key):
        vals = [r[key]["gap_pct"] for r in rows if r[key]["gap_pct"] is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    print(f"\n{'─'*120}")
    print("  RESUMEN")
    print(f"    Aleatorias → gap medio greedy: {_avg_gap(rnd,'greedy')}%   "
          f"gap medio SA: {_avg_gap(rnd,'sa')}%")
    print(f"    Trampa     → gap medio greedy: {_avg_gap(trp,'greedy')}%   "
          f"gap medio SA: {_avg_gap(trp,'sa')}%")
    all_opt = all(r["cpsat"]["proven_optimal"] for r in results)
    print(f"    CP-SAT demostró óptimo en todas las instancias: {all_opt}")
    print(f"{'─'*120}\n")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump({"config": vars(args), "results": results}, fh, indent=4, ensure_ascii=False)
    print(f"Reporte guardado en: {out}")


if __name__ == "__main__":
    main()

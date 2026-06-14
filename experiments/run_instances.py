"""
experiments/run_instances.py
-----------------------------
Runs the pipeline over multiple seeds to generate a set of test instances
and aggregates the results into a summary report.

Each seed produces an independent dataset with deterministically injected
inconsistencies, runs structural detection + CP-SAT repair, and evaluates
precision / recall / F1 / MCC.  Results are saved under:

    data/instances/seed_<N>/
        dataset.json
        inconsistencies.json
        repair_plan.json
        repaired_dataset.json
        evaluation_report.json

And a cross-seed summary at:

    data/instances/summary.json

Usage:
    python experiments/run_instances.py
    python experiments/run_instances.py --seeds 42 123 456 789 2024
    python experiments/run_instances.py --seeds 1 2 3 --skip-textual
"""

import argparse
import json
import statistics
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PY       = sys.executable
OLLAMA_URL = "http://localhost:11434"


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def run(cmd: list, cwd: Path = ROOT_DIR) -> bool:
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode == 0


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=4, ensure_ascii=False)


def _ollama_available() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


# ─────────────────────────────────────────────────────────────────
# SINGLE-SEED PIPELINE
# ─────────────────────────────────────────────────────────────────

def run_seed(seed: int, out_dir: Path, skip_textual: bool) -> dict | None:
    """
    Runs the full pipeline for one seed, storing all artifacts under out_dir.
    Returns the evaluation_report dict, or None on failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path   = out_dir / "dataset.json"
    incon_path     = out_dir / "inconsistencies.json"
    textual_path   = out_dir / "textual_inconsistencies.json"
    plan_path      = out_dir / "repair_plan.json"
    repaired_path  = out_dir / "repaired_dataset.json"
    eval_path      = out_dir / "evaluation_report.json"

    print(f"\n  ── Semilla {seed} ──────────────────────────────")

    steps = [
        ("Generar dataset",
         [PY, "data/seed.py", "--seed", str(seed), "--output", str(dataset_path)]),
        ("Detectar estructural",
         [PY, "detector/structural_rules.py",
          "--dataset", str(dataset_path), "--output", str(incon_path)]),
        ("Optimizar reparaciones",
         [PY, "detector/repair_optimizer.py",
          "--dataset", str(dataset_path),
          "--textual", str(textual_path),
          "--output-plan", str(plan_path),
          "--output-repaired", str(repaired_path)]),
        ("Evaluar",
         [PY, "evaluate.py",
          "--dataset", str(dataset_path),
          "--structural", str(incon_path),
          "--textual", str(textual_path),
          "--output", str(eval_path)]),
    ]

    if not skip_textual:
        steps.insert(2, (
            "Detectar textual",
            [PY, "detector/llm_detector.py",
             "--dataset", str(dataset_path),
             "--output", str(textual_path)],
        ))

    for label, cmd in steps:
        print(f"    {label}...", end=" ", flush=True)
        if run(cmd):
            print("OK")
        else:
            print("FAIL")
            return None

    if eval_path.exists():
        return _load(eval_path)
    return None


# ─────────────────────────────────────────────────────────────────
# CROSS-SEED AGGREGATION
# ─────────────────────────────────────────────────────────────────

def aggregate(results: list[dict]) -> dict:
    """
    Given a list of evaluation_report dicts (one per seed), produce
    a summary with mean ± std for every metric at global and per-type level.
    """
    def _stats(values: list[float]) -> dict:
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": round(statistics.mean(values), 4),
            "std":  round(statistics.stdev(values) if len(values) > 1 else 0.0, 4),
            "min":  round(min(values), 4),
            "max":  round(max(values), 4),
        }

    # Global metrics
    global_keys = ["precision", "recall", "f1", "mcc"]
    global_agg  = {
        k: _stats([r["global"][k] for r in results if k in r.get("global", {})])
        for k in global_keys
    }

    # Per-type metrics
    all_types: set = set()
    for r in results:
        all_types.update(r.get("by_type", {}).keys())

    per_type_agg: dict = {}
    per_type_keys = ["precision", "recall", "f1", "mcc", "tp", "fp", "fn"]
    for t in sorted(all_types):
        per_type_agg[t] = {
            k: _stats([r["by_type"][t][k] for r in results if t in r.get("by_type", {})])
            for k in per_type_keys
        }

    return {"n_seeds": len(results), "global": global_agg, "by_type": per_type_agg}


# ─────────────────────────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────

def print_summary(summary: dict, seeds: list[int]) -> None:
    n = summary["n_seeds"]
    print(f"\n{'═'*70}")
    print(f"  RESUMEN SOBRE {n} SEMILLAS: {seeds}")
    print(f"{'═'*70}")

    g = summary["global"]
    print(f"\n  Métricas globales (media ± std):")
    for k, v in g.items():
        print(f"    {k:<12}  {v['mean']:.4f} ± {v['std']:.4f}  "
              f"[{v['min']:.4f}, {v['max']:.4f}]")

    col = max(len(t) for t in summary["by_type"]) + 2
    header = f"  {'Tipo':<{col}} {'Prec(μ)':>8} {'Rec(μ)':>8} {'F1(μ)':>8} {'MCC(μ)':>8} {'TP(μ)':>7} {'FP(μ)':>7} {'FN(μ)':>7}"
    print(f"\n  Por tipo:\n  {'─'*(len(header)-2)}")
    print(header)
    print(f"  {'─'*(len(header)-2)}")
    for t, d in summary["by_type"].items():
        print(
            f"  {t:<{col}}"
            f" {d['precision']['mean']:>8.4f}"
            f" {d['recall']['mean']:>8.4f}"
            f" {d['f1']['mean']:>8.4f}"
            f" {d['mcc']['mean']:>8.4f}"
            f" {d['tp']['mean']:>7.1f}"
            f" {d['fp']['mean']:>7.1f}"
            f" {d['fn']['mean']:>7.1f}"
        )
    print(f"  {'─'*(len(header)-2)}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Genera instancias de prueba con múltiples semillas y agrega resultados"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2024],
                        help="Lista de semillas a ejecutar (default: 42 123 456 789 2024)")
    parser.add_argument("--skip-textual", action="store_true",
                        help="Omitir detector textual (usa solo reglas estructurales)")
    args = parser.parse_args()

    instances_dir = DATA_DIR / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ejecutando pipeline para {len(args.seeds)} semillas: {args.seeds}")
    if args.skip_textual:
        print("  [modo] Detector textual: omitido (--skip-textual)")
    else:
        print("  [modo] Detector textual: activo")
        if not _ollama_available():
            print(f"\nERROR: Ollama no disponible en {OLLAMA_URL}.")
            print("  Para ejecutar con detector textual, inicia Ollama (ollama serve)")
            print("  o usa --skip-textual para evaluar solo el componente estructural.")
            sys.exit(1)

    successful_results  = []
    per_seed_summary    = []

    for seed in args.seeds:
        out_dir = instances_dir / f"seed_{seed}"
        report  = run_seed(seed, out_dir, args.skip_textual)
        if report:
            successful_results.append(report)
            per_seed_summary.append({
                "seed":      seed,
                "out_dir":   str(out_dir),
                "global":    report.get("global", {}),
            })
            g = report.get("global", {})
            print(f"    → P={g.get('precision',0):.3f}  R={g.get('recall',0):.3f}  "
                  f"F1={g.get('f1',0):.3f}  MCC={g.get('mcc',0):.3f}")
        else:
            print(f"    → FALLÓ — semilla {seed} no incluida en el resumen")

    if not successful_results:
        print("\nNinguna semilla completó con éxito. Revisa los errores anteriores.")
        sys.exit(1)

    summary = aggregate(successful_results)
    summary["seeds_attempted"] = args.seeds
    summary["seeds_ok"]        = [p["seed"] for p in per_seed_summary]
    summary["per_seed"]        = per_seed_summary

    summary_path = instances_dir / "summary.json"
    _save(summary_path, summary)

    print_summary(summary, summary["seeds_ok"])
    print(f"\nResumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()

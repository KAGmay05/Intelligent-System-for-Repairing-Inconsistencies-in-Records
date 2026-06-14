"""
experiments/llm_robustness.py
-----------------------------
Does the LLM detector GENERALISE across instances, or did it just get lucky on
one seed?  This is the meaningful multi-seed experiment: it runs the LLM textual
detector on SEVERAL generated datasets (different seeds) and measures the
variance of its recall — overall and broken down by phrasing class
(canonical / paraphrased / global).

To stay tractable on local CPU inference (~1 min/report), each seed is evaluated
on a small SUBSET (all ground-truth reports + a few coherent ones).  Recall is
exact (every GT report is in the subset); precision is estimated on the sample.

Usage:
    python experiments/llm_robustness.py
    python experiments/llm_robustness.py --seeds 42 123 789 --subset 6 --model gemma4
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))   # import sibling textual_eval
from textual_eval import evaluate_textual                  # noqa: E402

DATA = ROOT / "data"
PY = sys.executable


def _run(cmd, env=None):
    merged = {**os.environ, **(env or {})}
    return subprocess.run(cmd, cwd=ROOT, env=merged).returncode == 0


def run_seed(seed: int, subset: int, model: str, timeout: int) -> dict | None:
    out_dir = DATA / "llm_robustness" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "dataset.json"
    textual_path = out_dir / "textual_llm.json"

    print(f"\n── Semilla {seed} ─────────────────────────────")
    print("  Generando dataset…", flush=True)
    if not _run([PY, "data/seed.py", "--seed", str(seed), "--output", str(dataset_path)]):
        print("  FALLÓ el seed"); return None

    print(f"  Detectando con LLM ({model}) sobre subconjunto…", flush=True)
    env = {"DETECTOR_MODE": "llm", "OLLAMA_MODEL": model, "PYTHONIOENCODING": "utf-8"}
    ok = _run([PY, "detector/llm_detector.py",
               "--dataset", str(dataset_path),
               "--output", str(textual_path),
               "--eval-subset", str(subset),
               "--timeout", str(timeout)], env=env)
    if not ok:
        print("  FALLÓ el detector LLM"); return None

    res = evaluate_textual(dataset_path, textual_path, f"seed {seed}")
    if res:
        res["seed"] = seed
    return res


def _agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {"media": round(sum(vals) / len(vals), 3),
            "min": round(min(vals), 3), "max": round(max(vals), 3)}


def main():
    ap = argparse.ArgumentParser(description="Robustez del LLM entre instancias (multi-semilla)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 789])
    ap.add_argument("--subset", type=int, default=6,
                    help="Reportes coherentes a muestrear además de los de ground truth")
    ap.add_argument("--model", default="gemma4")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--output", default=str(DATA / "llm_robustness_report.json"))
    args = ap.parse_args()

    results = []
    for s in args.seeds:
        r = run_seed(s, args.subset, args.model, args.timeout)
        if r:
            results.append(r)

    if not results:
        print("Ninguna semilla completó."); sys.exit(1)

    # ── Per-seed table ────────────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"  ROBUSTEZ DEL LLM ({args.model}) ENTRE {len(results)} SEMILLAS")
    print(f"{'='*78}")
    print(f"  {'seed':<6} {'Prec':>6} {'Rec':>6} {'F1':>6}   "
          f"{'canónica':>9} {'parafr.':>9} {'global':>8}")
    print(f"  {'-'*74}")
    for r in results:
        def pct(x): return f"{x:.0%}" if x is not None else "-"
        print(f"  {r['seed']:<6} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}   "
              f"{pct(r['recall_canonical']):>9} {pct(r['recall_paraphrased']):>9} "
              f"{pct(r['recall_global']):>8}")
    print(f"  {'-'*74}")

    # ── Aggregate (mean ± range) ──────────────────────────────────────────────
    agg = {
        "precision":          _agg([r["precision"] for r in results]),
        "recall":             _agg([r["recall"] for r in results]),
        "f1":                 _agg([r["f1"] for r in results]),
        "recall_canonical":   _agg([r["recall_canonical"] for r in results]),
        "recall_paraphrased": _agg([r["recall_paraphrased"] for r in results]),
        "recall_global":      _agg([r["recall_global"] for r in results]),
    }
    print("\n  Agregado (media [min–max] entre semillas):")
    for k, v in agg.items():
        if v:
            print(f"    {k:<20} {v['media']:.3f}  [{v['min']:.3f} – {v['max']:.3f}]")
    print(f"{'='*78}\n")

    out = Path(args.output)
    with out.open("w", encoding="utf-8") as fh:
        json.dump({"model": args.model, "subset": args.subset,
                   "seeds": args.seeds, "per_seed": results, "aggregate": agg}, fh,
                  indent=4, ensure_ascii=False)
    print(f"Reporte guardado en: {out}")


if __name__ == "__main__":
    main()

"""
experiments/compare_detectors.py
--------------------------------
Lightweight "comparison between configurations" for the TEXTUAL detector,
reusing already-computed outputs instead of re-invoking the (slow) local LLM.

It compares the three detector modes on the SAME evaluation subset:
    rule    — keyword rules only            (data/textual_inconsistencies.json)
    llm     — LLM only                       (data/textual_llm.json)
    hybrid  — rules ∪ LLM (rules catch the obvious, the LLM adds the rest)
              → derived synthetically as the union of the two verdict sets,
                which is exactly what detector/llm_detector.py does in hybrid mode.

Metrics are instance-level (by report_id) against metadata.textual_ground_truth,
broken down by phrasing class (canonical / paraphrased / global), which is where
the value of the LLM over the rules shows up.

Note on the optimizer axis: on the real dataset the three optimizers
(cpsat/greedy/sa) return the same minimum-cost repair (the instance is easy), so
the meaningful variant axis here is the DETECTOR.  For the optimizer comparison
see experiments/hard_instances.py.

Usage:
    python experiments/compare_detectors.py
    python experiments/compare_detectors.py --rule data/textual_inconsistencies.json --llm data/textual_llm.json
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _load(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _flagged_set(records, subset):
    """report_ids the detector marked inconsistent, restricted to `subset`."""
    return {
        r.get("report_id") for r in records
        if r.get("report_id") in subset
        and not r.get("llm_evaluation", {}).get("es_coherente", True)
    }


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser(description="Comparación ligera rule/hybrid/llm (detector textual)")
    ap.add_argument("--dataset", default=str(DATA / "dataset.json"))
    ap.add_argument("--rule",    default=str(DATA / "textual_inconsistencies.json"),
                    help="Salida del detector en modo rule")
    ap.add_argument("--llm",     default=str(DATA / "textual_llm.json"),
                    help="Salida del detector en modo llm")
    ap.add_argument("--output",  default=str(DATA / "config_comparison_light.json"))
    args = ap.parse_args()

    dataset = _load(Path(args.dataset))
    gt_list = dataset.get("metadata", {}).get("textual_ground_truth", [])
    phrasing_of = {g["report_id"]: g["phrasing"] for g in gt_list}

    rule_recs = _load(Path(args.rule))
    llm_recs  = _load(Path(args.llm))

    # Evaluation subset = the reports the LLM actually analysed (rule has all of them).
    subset = {r.get("report_id") for r in llm_recs}
    subset.discard(None); subset.discard("")
    gt_ids = {g["report_id"] for g in gt_list if g["report_id"] in subset}

    rule_flagged = _flagged_set(rule_recs, subset)
    llm_flagged  = _flagged_set(llm_recs,  subset)
    hybrid_flagged = rule_flagged | llm_flagged   # rules ∪ LLM

    modes = {"rule": rule_flagged, "hybrid": hybrid_flagged, "llm": llm_flagged}
    phrasings = ("canonical", "paraphrased", "global")

    print(f"\n{'='*86}")
    print(f"  COMPARACIÓN DE DETECTORES TEXTUALES  (subconjunto de {len(subset)} reportes, "
          f"{len(gt_ids)} con inconsistencia)")
    print(f"{'='*86}")
    print(f"  {'Modo':<8} {'TP':>3} {'FP':>3} {'FN':>3}  {'Prec':>6} {'Rec':>6} {'F1':>6}   "
          f"{'canónica':>9} {'parafr.':>9} {'global':>8}")
    print(f"  {'-'*82}")

    results = {}
    for mode, flagged in modes.items():
        tp = len(flagged & gt_ids); fp = len(flagged - gt_ids); fn = len(gt_ids - flagged)
        p, r, f = _prf(tp, fp, fn)

        by_phr = {}
        cells = []
        for phr in phrasings:
            ids = {i for i in gt_ids if phrasing_of.get(i) == phr}
            caught = len(ids & flagged)
            by_phr[phr] = {"caught": caught, "total": len(ids),
                           "recall": round(caught / len(ids), 4) if ids else None}
            cells.append(f"{caught}/{len(ids)}" if ids else "-")

        print(f"  {mode:<8} {tp:>3} {fp:>3} {fn:>3}  {p:>6.3f} {r:>6.3f} {f:>6.3f}   "
              f"{cells[0]:>9} {cells[1]:>9} {cells[2]:>8}")
        results[mode] = {"tp": tp, "fp": fp, "fn": fn,
                         "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                         "by_phrasing": by_phr}

    print(f"  {'-'*82}")
    best = max(results, key=lambda m: results[m]["f1"])
    print(f"  Mejor F1: {best}  (F1={results[best]['f1']:.3f})")
    print("  Nota: la detección estructural es 100% e idéntica en los 3 modos; el optimizador")
    print("  (cpsat/greedy/sa) no cambia en el dataset real — ver experiments/hard_instances.py.")
    print(f"{'='*86}\n")

    out = Path(args.output)
    with out.open("w", encoding="utf-8") as fh:
        json.dump({
            "subset_size": len(subset),
            "ground_truth_in_subset": len(gt_ids),
            "modes": results,
            "note": ("hybrid = unión de rule y llm; estructural 100% e independiente del modo; "
                     "optimizador comparado aparte en hard_instances.py"),
        }, fh, indent=4, ensure_ascii=False)
    print(f"Guardado en: {out}")


if __name__ == "__main__":
    main()

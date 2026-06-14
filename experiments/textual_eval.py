"""
experiments/textual_eval.py
---------------------------
Instance-level (by report_id) evaluation of the TEXTUAL detector against the
per-report ground truth stored in dataset.metadata.textual_ground_truth.

Unlike evaluate.py (which compares counts per type), this matches the EXACT
reports flagged, so a detection only counts if it landed on a truly inconsistent
report.  Results are broken down by phrasing class (canonical vs paraphrased) —
this is what shows whether the LLM adds real semantic value over the keyword rules.

Usage:
    python experiments/textual_eval.py --textual data/textual_llm.json --label llm
    python experiments/textual_eval.py --textual data/textual_rule.json --label rule
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _load(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def evaluate_textual(dataset_path: Path, textual_path: Path, label: str):
    dataset = _load(dataset_path)
    gt_list = dataset.get("metadata", {}).get("textual_ground_truth", [])
    if not gt_list:
        print("ERROR: el dataset no tiene 'textual_ground_truth'. Regenera con data/seed.py.")
        return None

    # Ground-truth inconsistent report_ids and their phrasing class
    gt_ids       = {g["report_id"] for g in gt_list}
    phrasing_of  = {g["report_id"]: g["phrasing"] for g in gt_list}
    gt_type_of   = {g["report_id"]: g["type"] for g in gt_list}

    # Detected inconsistent report_ids (es_coherente == False)
    records = _load(textual_path)
    detected_ids = {
        r.get("report_id")
        for r in records
        if not r.get("llm_evaluation", {}).get("es_coherente", True)
    }
    detected_ids.discard(None)
    detected_ids.discard("")

    # Overall confusion
    tp = len(gt_ids & detected_ids)
    fn = len(gt_ids - detected_ids)
    fp = len(detected_ids - gt_ids)
    p, r, f = _prf(tp, fp, fn)

    # Per-phrasing recall (of the truly inconsistent reports, how many caught)
    def recall_for(phrasing):
        ids = {i for i in gt_ids if phrasing_of.get(i) == phrasing}
        caught = len(ids & detected_ids)
        return caught, len(ids)

    can_caught, can_total = recall_for("canonical")
    par_caught, par_total = recall_for("paraphrased")
    glo_caught, glo_total = recall_for("global")

    print(f"\n══ Detector textual: {label}  ══")
    print(f"  TP={tp}  FP={fp}  FN={fn}   Precisión={p:.3f}  Recall={r:.3f}  F1={f:.3f}")
    print(f"  Recall por redacción:")
    print(f"    canónica         : {can_caught}/{can_total}"
          + (f"  ({can_caught/can_total:.0%})" if can_total else ""))
    print(f"    parafraseada     : {par_caught}/{par_total}"
          + (f"  ({par_caught/par_total:.0%})" if par_total else ""))
    print(f"    global(asignatura): {glo_caught}/{glo_total}"
          + (f"  ({glo_caught/glo_total:.0%})" if glo_total else ""))

    # Which paraphrased/global ones were missed (the cases that need real reasoning)
    missed_par = sorted(i for i in gt_ids - detected_ids
                        if phrasing_of.get(i) in ("paraphrased", "global"))
    if missed_par:
        print(f"  Parafraseadas NO detectadas: {', '.join(missed_par)}")
        for i in missed_par:
            print(f"      {i}  tipo={gt_type_of[i]}")

    return {
        "label": label, "tp": tp, "fp": fp, "fn": fn,
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
        "recall_canonical": round(can_caught / can_total, 4) if can_total else None,
        "recall_paraphrased": round(par_caught / par_total, 4) if par_total else None,
        "recall_global": round(glo_caught / glo_total, 4) if glo_total else None,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluación textual por report_id y por redacción")
    ap.add_argument("--dataset", default=str(DATA / "dataset.json"))
    ap.add_argument("--textual", required=True, help="Salida del detector textual a evaluar")
    ap.add_argument("--label", default="textual", help="Etiqueta para el reporte (rule/hybrid/llm)")
    args = ap.parse_args()
    evaluate_textual(Path(args.dataset), Path(args.textual), args.label)


if __name__ == "__main__":
    main()

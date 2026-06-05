"""
evaluate.py
-----------
Compares detector output against ground-truth injected inconsistencies.
Produces precision / recall / F1 per inconsistency type and overall.

Evaluation model (type-level, count-based):
  TP = min(injected, detected)   — detections that match real anomalies
  FP = max(0, detected - injected)  — detections beyond what was injected
  FN = max(0, injected - detected)  — injected anomalies that were missed

Usage:
    python evaluate.py
    python evaluate.py --dataset dataset.json
                       --structural inconsistencies.json
                       --textual textual_inconsistencies.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Mapping: ground-truth type name → structural_rules.py type name (where they differ)
GT_TO_DETECTOR_TYPE = {
    "professor_subject_mismatch": "invalid_professor_subject",
    "ghost_student":              "ghost_student",
    "ghost_exam":                 "ghost_exam",
}


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _metrics(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate(dataset_path: Path, structural_path: Path, textual_path: Path):
    dataset    = load_json(dataset_path)
    structural = load_json(structural_path)

    # ── Ground truth ────────────────────────────────────────────────────────
    ground_truth: dict = dataset.get("metadata", {}).get("injected_inconsistencies", {})
    if not ground_truth:
        print("ERROR: dataset.json tiene no tiene 'injected_inconsistencies' en metadata.")
        return

    # ── Structural detections ────────────────────────────────────────────────
    # inconsistencies.json format: {"inconsistencies": [...], "summary_by_type": {...}}
    inc_list = structural.get("inconsistencies", []) if isinstance(structural, dict) else structural
    detected_structural = Counter(inc.get("type") for inc in inc_list)

    # ── Textual detections ───────────────────────────────────────────────────
    detected_textual = 0
    if textual_path.exists():
        textual = load_json(textual_path)
        textual_list = textual if isinstance(textual, list) else []
        detected_textual = sum(
            1 for r in textual_list
            if not r.get("llm_evaluation", {}).get("es_coherente", True)
        )
    else:
        print(f"[warn] {textual_path.name} not found — textual recall will show as 0.")

    # ── Build detected dict using GT type names ──────────────────────────────
    detected: dict = {}
    for gt_type in ground_truth:
        if gt_type == "total":
            continue
        if gt_type == "textual_inconsistency":
            detected[gt_type] = detected_textual
        else:
            det_key = GT_TO_DETECTOR_TYPE.get(gt_type, gt_type)
            detected[gt_type] = detected_structural.get(det_key, 0)

    # ── Per-type table ───────────────────────────────────────────────────────
    col = 34
    header = f"{'Tipo':<{col}} {'GT':>4} {'Det':>4} {'TP':>4} {'FP':>4} {'FN':>4}  {'Prec':>6} {'Rec':>6} {'F1':>6}"
    sep    = "─" * len(header)
    print(sep)
    print(header)
    print(sep)

    total_tp = total_fp = total_fn = 0
    rows = []
    for gt_type in sorted(ground_truth):
        if gt_type == "total":
            continue
        injected = ground_truth[gt_type]
        det      = detected.get(gt_type, 0)
        tp       = min(injected, det)
        fp       = max(0, det - injected)
        fn       = max(0, injected - det)
        p, r, f  = _metrics(tp, fp, fn)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        rows.append((gt_type, injected, det, tp, fp, fn, p, r, f))

    for gt_type, injected, det, tp, fp, fn, p, r, f in rows:
        flag = ""
        if fn > 0: flag += " ← missed"
        if fp > 0: flag += " ← false+"
        print(f"{gt_type:<{col}} {injected:>4} {det:>4} {tp:>4} {fp:>4} {fn:>4}  {p:>6.1%} {r:>6.1%} {f:>6.1%}{flag}")

    print(sep)
    p_total, r_total, f_total = _metrics(total_tp, total_fp, total_fn)
    total_injected = ground_truth.get(
        "total", sum(v for k, v in ground_truth.items() if k != "total")
    )
    total_detected = sum(detected.values())
    print(
        f"{'TOTAL':<{col}} {total_injected:>4} {total_detected:>4} "
        f"{total_tp:>4} {total_fp:>4} {total_fn:>4}  "
        f"{p_total:>6.1%} {r_total:>6.1%} {f_total:>6.1%}"
    )
    print(sep)
    print(f"\nResumen global → Precisión: {p_total:.3f}  Recall: {r_total:.3f}  F1: {f_total:.3f}")

    # ── Undetected types (FN > 0) ────────────────────────────────────────────
    missed = [(t, inj, det, inj - det) for t, inj, det, *_ in rows
              if inj > det]
    if missed:
        print("\nTipos con inconsistencias no detectadas:")
        for t, inj, det, fn_n in missed:
            print(f"  {t}: inyectadas={inj}  detectadas={det}  perdidas={fn_n}")
    else:
        print("\n✔  Todos los tipos de inconsistencia fueron detectados.")

    return {"precision": p_total, "recall": r_total, "f1": f_total}


def main():
    parser = argparse.ArgumentParser(description="Evaluar precisión/recall del sistema de detección")
    parser.add_argument("--dataset",    default=str(BASE_DIR / "dataset.json"))
    parser.add_argument("--structural", default=str(BASE_DIR / "inconsistencies.json"))
    parser.add_argument("--textual",    default=str(BASE_DIR / "textual_inconsistencies.json"))
    args = parser.parse_args()

    evaluate(
        dataset_path    = Path(args.dataset),
        structural_path = Path(args.structural),
        textual_path    = Path(args.textual),
    )


if __name__ == "__main__":
    main()

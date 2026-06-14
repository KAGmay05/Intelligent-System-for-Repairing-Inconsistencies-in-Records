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
    python evaluate.py --dataset data/dataset.json
                       --structural data/inconsistencies.json
                       --textual data/textual_inconsistencies.json
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Mapping: ground-truth type name → structural_rules.py type name (where they differ)
GT_TO_DETECTOR_TYPE = {
    "professor_subject_mismatch": "invalid_professor_subject",
    "ghost_student":              "ghost_student",
    "ghost_exam":                 "ghost_exam",
    "orphan_regrade":             "orphan_regrade",
}
DETECTOR_TO_GT_TYPE = {v: k for k, v in GT_TO_DETECTOR_TYPE.items()}


def _detected_entity_id(gt_type: str, inc: dict):
    """Extract, from a detected structural inconsistency, the entity ID to match
    against metadata.structural_ground_truth (which stores the same ID scheme)."""
    e = inc.get("related_entities", {})
    if gt_type in ("age_birth_mismatch", "course_age_mismatch", "credits_mismatch"):
        return e.get("student_id")
    if gt_type in ("invalid_grade", "ghost_student", "ghost_exam", "duplicate_result"):
        return e.get("result_id")
    if gt_type in ("professor_subject_mismatch", "future_exam_date"):
        return e.get("exam_id")
    if gt_type == "orphan_regrade":
        return f"{e.get('student_id')}|{e.get('exam_id')}"
    return None


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


def _mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    """Matthews Correlation Coefficient — robust for imbalanced classes."""
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


def evaluate(dataset_path: Path, structural_path: Path, textual_path: Path,
             output_path: Path | None = None):
    dataset    = load_json(dataset_path)
    structural = load_json(structural_path)

    # ── Ground truth ────────────────────────────────────────────────────────
    ground_truth: dict = dataset.get("metadata", {}).get("injected_inconsistencies", {})
    if not ground_truth:
        print("ERROR: data/dataset.json no tiene 'injected_inconsistencies' en metadata.")
        return

    # ── Ground truth by ID (instance-level) ───────────────────────────────────
    meta = dataset.get("metadata", {})
    structural_gt_ids: dict = meta.get("structural_ground_truth", {})
    textual_gt_list: list   = meta.get("textual_ground_truth", [])
    id_level = bool(structural_gt_ids)
    if not id_level:
        print("[warn] dataset sin 'structural_ground_truth' — usando evaluación por conteo (legacy).")

    # ── Detected structural IDs per GT type ───────────────────────────────────
    inc_list = structural.get("inconsistencies", []) if isinstance(structural, dict) else structural
    detected_structural = Counter(inc.get("type") for inc in inc_list)   # legacy fallback
    detected_ids: dict = defaultdict(set)
    for inc in inc_list:
        gt_type = DETECTOR_TO_GT_TYPE.get(inc.get("type"), inc.get("type"))
        eid = _detected_entity_id(gt_type, inc)
        if eid is not None:
            detected_ids[gt_type].add(eid)

    # ── Detected textual IDs (by report_id) ───────────────────────────────────
    textual_list: list = []
    detected_textual_ids: set = set()
    if textual_path.exists():
        textual = load_json(textual_path)
        textual_list = textual if isinstance(textual, list) else []
        detected_textual_ids = {
            r.get("report_id") for r in textual_list
            if not r.get("llm_evaluation", {}).get("es_coherente", True)
        }
        detected_textual_ids.discard(None)
        detected_textual_ids.discard("")
    else:
        print(f"[warn] {textual_path.name} not found — textual recall will show as 0.")
    textual_gt_ids = {g["report_id"] for g in textual_gt_list}

    def _confusion(gt_type: str):
        """Return (injected, detected, tp, fp, fn) by matching entity ID sets."""
        if gt_type == "textual_inconsistency":
            gt_ids, det_set = textual_gt_ids, detected_textual_ids
        elif id_level:
            gt_ids  = set(structural_gt_ids.get(gt_type, []))
            det_set = detected_ids.get(gt_type, set())
        else:
            # Legacy count-based fallback (no IDs available)
            inj = ground_truth.get(gt_type, 0)
            det = detected_structural.get(GT_TO_DETECTOR_TYPE.get(gt_type, gt_type), 0)
            return inj, det, min(inj, det), max(0, det - inj), max(0, inj - det)
        tp = len(gt_ids & det_set)
        fp = len(det_set - gt_ids)
        fn = len(gt_ids - det_set)
        return len(gt_ids), len(det_set), tp, fp, fn

    # ── Total samples N (used for MCC TN estimate) ───────────────────────────
    total_samples = (
        len(dataset.get("students",  []))
        + len(dataset.get("results",  []))
        + len(dataset.get("exams",    []))
        + len(dataset.get("regrades", []))
        + len(dataset.get("teacher_reports", []))
    )

    # ── Per-type table ───────────────────────────────────────────────────────
    mode_label = "POR ID (instance-level)" if id_level else "POR CONTEO (legacy)"
    print(f"Modo de evaluación: {mode_label}\n")
    col = 34
    header = f"{'Tipo':<{col}} {'GT':>4} {'Det':>4} {'TP':>4} {'FP':>4} {'FN':>4}  {'Prec':>6} {'Rec':>6} {'F1':>6} {'MCC':>6}"
    sep    = "─" * len(header)
    print(sep)
    print(header)
    print(sep)

    total_tp = total_fp = total_fn = 0
    detected: dict = {}
    rows = []
    per_type_report = {}
    for gt_type in sorted(ground_truth):
        if gt_type == "total":
            continue
        injected, det, tp, fp, fn = _confusion(gt_type)
        detected[gt_type] = det
        tn       = max(0, total_samples - tp - fp - fn)
        p, r, f  = _metrics(tp, fp, fn)
        mcc      = _mcc(tp, fp, fn, tn)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        rows.append((gt_type, injected, det, tp, fp, fn, p, r, f, mcc))
        detector_kind = "textual" if gt_type == "textual_inconsistency" else "structural"
        per_type_report[gt_type] = {
            "detector": detector_kind,
            "injected": injected, "detected": det,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f, 4), "mcc": round(mcc, 4),
        }

    for gt_type, injected, det, tp, fp, fn, p, r, f, mcc in rows:
        flag = ""
        if fn > 0: flag += " ← missed"
        if fp > 0: flag += " ← false+"
        print(f"{gt_type:<{col}} {injected:>4} {det:>4} {tp:>4} {fp:>4} {fn:>4}  {p:>6.1%} {r:>6.1%} {f:>6.1%} {mcc:>6.3f}{flag}")

    print(sep)
    p_total, r_total, f_total = _metrics(total_tp, total_fp, total_fn)
    total_injected = ground_truth.get(
        "total", sum(v for k, v in ground_truth.items() if k != "total")
    )
    total_detected = sum(detected.values())
    total_tn = max(0, total_samples - total_tp - total_fp - total_fn)
    mcc_total = _mcc(total_tp, total_fp, total_fn, total_tn)
    print(
        f"{'TOTAL':<{col}} {total_injected:>4} {total_detected:>4} "
        f"{total_tp:>4} {total_fp:>4} {total_fn:>4}  "
        f"{p_total:>6.1%} {r_total:>6.1%} {f_total:>6.1%} {mcc_total:>6.3f}"
    )
    print(sep)
    print(f"\nResumen global → Precisión: {p_total:.3f}  Recall: {r_total:.3f}  F1: {f_total:.3f}  MCC: {mcc_total:.3f}")

    # ── Desglose structural vs textual ───────────────────────────────────────
    struct_rows = [(t, d) for t, d in per_type_report.items() if d["detector"] == "structural"]
    text_rows   = [(t, d) for t, d in per_type_report.items() if d["detector"] == "textual"]

    def _subtotal(rlist):
        stp = sum(d["tp"] for _, d in rlist)
        sfp = sum(d["fp"] for _, d in rlist)
        sfn = sum(d["fn"] for _, d in rlist)
        sp, sr, sf = _metrics(stp, sfp, sfn)
        stn = max(0, total_samples - stp - sfp - sfn)
        sm  = _mcc(stp, sfp, sfn, stn)
        return stp, sfp, sfn, sp, sr, sf, sm

    if struct_rows and text_rows:
        print(f"\n  {'Detector':<14} {'TP':>4} {'FP':>4} {'FN':>4}  {'Prec':>6} {'Rec':>6} {'F1':>6} {'MCC':>6}")
        print(f"  {'─'*58}")
        stp, sfp, sfn, sp, sr, sf, sm = _subtotal(struct_rows)
        print(f"  {'Estructural':<14} {stp:>4} {sfp:>4} {sfn:>4}  {sp:>6.1%} {sr:>6.1%} {sf:>6.1%} {sm:>6.3f}")
        ttp, tfp, tfn, tp2, tr2, tf2, tm2 = _subtotal(text_rows)
        print(f"  {'Textual':<14} {ttp:>4} {tfp:>4} {tfn:>4}  {tp2:>6.1%} {tr2:>6.1%} {tf2:>6.1%} {tm2:>6.3f}")

    # ── Textual recall by phrasing class (canonical vs paraphrased) ───────────
    # This is the key signal for the LLM's value: keyword rules catch canonical
    # wording but are blind to paraphrased wording, which needs semantic reasoning.
    textual_phrasing: dict = {}
    if textual_gt_list:
        for phrasing in ("canonical", "paraphrased", "global"):
            ids = {g["report_id"] for g in textual_gt_list if g["phrasing"] == phrasing}
            caught = len(ids & detected_textual_ids)
            textual_phrasing[phrasing] = {
                "caught": caught, "total": len(ids),
                "recall": round(caught / len(ids), 4) if ids else None,
            }
        print(f"\n  Recall textual por redacción:")
        for phrasing, d in textual_phrasing.items():
            pct = f"{d['recall']:.0%}" if d["recall"] is not None else "n/a"
            print(f"    {phrasing:<12} {d['caught']}/{d['total']}  ({pct})")

    # ── Undetected types (FN > 0) ────────────────────────────────────────────
    missed = [(t, inj, det, fn) for t, inj, det, tp, fp, fn, *_ in rows if fn > 0]
    if missed:
        print("\nTipos con inconsistencias no detectadas (FN > 0):")
        for t, inj, det, fn_n in missed:
            print(f"  {t}: inyectadas={inj}  detectadas={det}  no detectadas={fn_n}")
    else:
        print("\n✔  Todos los tipos de inconsistencia fueron detectados.")

    # ── Análisis de comportamiento ────────────────────────────────────────────
    behavior: dict = {}
    type_data = [(t, d) for t, d in per_type_report.items()]

    # Tipo con más FP (sobredetección)
    max_fp_type = max(type_data, key=lambda x: x[1]["fp"], default=None)
    if max_fp_type and max_fp_type[1]["fp"] > 0:
        behavior["max_false_positives"] = {
            "type": max_fp_type[0], "fp": max_fp_type[1]["fp"],
            "interpretation": "sobredetección — el detector genera falsas alarmas para este tipo"
        }

    # Tipo con más FN (baja detección)
    max_fn_type = max(type_data, key=lambda x: x[1]["fn"], default=None)
    if max_fn_type and max_fn_type[1]["fn"] > 0:
        behavior["max_false_negatives"] = {
            "type": max_fn_type[0], "fn": max_fn_type[1]["fn"],
            "interpretation": "infra-detección — el sistema omite inconsistencias de este tipo"
        }

    # Tipo con MCC más bajo (más difícil)
    min_mcc_type = min(type_data, key=lambda x: x[1]["mcc"], default=None)
    if min_mcc_type:
        behavior["lowest_mcc"] = {
            "type": min_mcc_type[0], "mcc": min_mcc_type[1]["mcc"],
            "interpretation": "tipo más difícil de detectar correctamente según MCC"
        }

    # Coste promedio por tipo de issue (de repair_plan.json si existe)
    repair_plan_path = dataset_path.parent / "repair_plan.json"
    avg_cost_by_type: dict = {}
    if repair_plan_path.exists():
        try:
            plan = load_json(repair_plan_path)
            summary = plan.get("summary", {}) if isinstance(plan, dict) else {}
            cost_by_type = summary.get("cost_by_issue_type", {})
            count_by_type = summary.get("issues_by_type", {})
            for it, total_cost in cost_by_type.items():
                cnt = int(count_by_type.get(it, 0) or 0)
                avg_cost_by_type[it] = round(total_cost / cnt, 2) if cnt else 0
            behavior["avg_repair_cost_by_type"] = avg_cost_by_type
        except (json.JSONDecodeError, OSError, ValueError, TypeError) as exc:
            print(f"[warn] No se pudo leer {repair_plan_path.name} para análisis de coste: {exc}")

    if behavior:
        print("\n  Análisis de comportamiento:")
        if "max_false_positives" in behavior:
            b = behavior["max_false_positives"]
            print(f"    Tipo con más FP   : {b['type']} (FP={b['fp']}) — {b['interpretation']}")
        if "max_false_negatives" in behavior:
            b = behavior["max_false_negatives"]
            print(f"    Tipo con más FN   : {b['type']} (FN={b['fn']}) — {b['interpretation']}")
        if "lowest_mcc" in behavior:
            b = behavior["lowest_mcc"]
            print(f"    Tipo con MCC más bajo: {b['type']} (MCC={b['mcc']:.3f}) — {b['interpretation']}")
        if avg_cost_by_type:
            print(f"    Coste medio por tipo: {avg_cost_by_type}")

    # ── Write evaluation report to disk ─────────────────────────────────────
    out = output_path or (dataset_path.parent / "evaluation_report.json")
    report = {
        "generated_at":  datetime.now().isoformat(),
        "dataset":       str(dataset_path),
        "structural":    str(structural_path),
        "textual":       str(textual_path),
        "evaluation_mode": "id_level" if id_level else "count_legacy",
        "global": {
            "precision": round(p_total, 4),
            "recall":    round(r_total, 4),
            "f1":        round(f_total, 4),
            "mcc":       round(mcc_total, 4),
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
        },
        "by_type": per_type_report,
        "textual_by_phrasing": textual_phrasing,
        "behavior_analysis": behavior,
    }
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=4, ensure_ascii=False)
    print(f"\nReporte guardado en: {out}")

    return {"precision": p_total, "recall": r_total, "f1": f_total, "mcc": mcc_total}


def main():
    parser = argparse.ArgumentParser(description="Evaluar precisión/recall del sistema de detección")
    parser.add_argument("--dataset",    default=str(DATA_DIR / "dataset.json"))
    parser.add_argument("--structural", default=str(DATA_DIR / "inconsistencies.json"))
    parser.add_argument("--textual",    default=str(DATA_DIR / "textual_inconsistencies.json"))
    parser.add_argument("--output",     default=str(DATA_DIR / "evaluation_report.json"),
                        help="Ruta donde guardar el reporte JSON (default: data/evaluation_report.json)")
    args = parser.parse_args()

    evaluate(
        dataset_path    = Path(args.dataset),
        structural_path = Path(args.structural),
        textual_path    = Path(args.textual),
        output_path     = Path(args.output),
    )


if __name__ == "__main__":
    main()

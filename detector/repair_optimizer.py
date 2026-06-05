"""
repair_optimizer.py
───────────────────
Minimum-cost repair planner for academic-record inconsistencies.

Pipeline:
  1. detect_structural_issues()   → issues + repairs from hard rules
  2. load_textual_issues()        → issues + repairs from textual_inconsistencies.json
                                    (pre-computed by llm_detector.py)
  3. solve_minimum_repairs()      → CP-SAT weighted minimum hitting-set
  4. apply_repairs()              → produce repaired_dataset.json
  5. main()                       → write repair_plan.json with full audit trail

Root-cause design
─────────────────
A single "root repair" (e.g. fix the professor-subject assignment of an exam)
can simultaneously satisfy:
  • the structural subject_mismatch issue on that exam
  • any suspicious_regrade issues caused by the bad assignment
  • any textual issues whose report is linked to that exam

The CP-SAT model expresses this naturally: every issue lists ALL repairs
(local OR root) that can resolve it. Selecting the root repair contributes
to ALL of those issues' constraints at once, letting the solver pick the
globally cheapest solution.
"""
import copy
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ortools.sat.python import cp_model


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH              = BASE_DIR / "dataset.json"
TEXTUAL_ISSUES_PATH       = BASE_DIR / "textual_inconsistencies.json"
OUTPUT_PLAN_PATH          = BASE_DIR / "repair_plan.json"
OUTPUT_REPAIRED_PATH      = BASE_DIR / "repaired_dataset.json"

COURSE_AGE = {
    "primary": (6, 11),
    "secondary": (12, 15),
    "highschool": (16, 19),
}

COURSE_CREDITS = {
    "primary": (5, 30),
    "secondary": (25, 55),
    "highschool": (50, 90),
}


@dataclass(frozen=True)
class RepairOption:
    repair_id:   str
    description: str
    cost:        int
    action:      str
    target:      dict


# ─────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────

def load_dataset() -> dict:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    with DATASET_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ─────────────────────────────────────────────
# STRUCTURAL ISSUE DETECTION
# ─────────────────────────────────────────────

def detect_structural_issues(dataset: dict):
    """
    Returns (issues, repairs, root_repair_map).

    root_repair_map: {exam_id → root_repair_id}
      Used downstream so textual issues can reference the same root repair,
      enabling CP-SAT to select one root fix that covers many issues.
    """
    issues:          list  = []
    repairs:         dict  = {}
    root_repair_map: dict  = {}   # exam_id → root_repair_id

    professors  = {p["id"]: p for p in dataset["professors"]}
    students    = {s["id"]: s for s in dataset["students"]}
    exams       = {e["id"]: e for e in dataset["exams"]}

    # ── Students ────────────────────────────────────────────────────────────

    for student in dataset["students"]:
        sid = student["id"]

        # 1. Age vs birth-year mismatch
        try:
            birth_year   = int(student["fecha_nacimiento"][:4])
            expected_age = datetime.now().year - birth_year
        except (KeyError, ValueError):
            expected_age = None

        if expected_age is not None and abs(expected_age - student["edad"]) > 1:
            rid = f"repair_student_age_{sid}"
            repairs[rid] = RepairOption(
                repair_id=rid,
                description=f"Ajustar edad de {sid} a {expected_age}",
                cost=1,
                action="modify_student",
                target={"student_id": sid, "field": "edad", "new_value": expected_age},
            )
            issues.append({"issue_id": f"age_mismatch_{sid}",
                           "type": "age_birth_mismatch",
                           "description": "Edad almacenada no coincide con la fecha de nacimiento",
                           "repair_ids": [rid]})

        # 2. Course vs age mismatch
        course = student.get("curso")
        age    = student.get("edad")
        if course in COURSE_AGE and isinstance(age, int):
            lo, hi = COURSE_AGE[course]
            if not (lo <= age <= hi):
                suggested = "primary"
                if 12 <= age <= 15:  suggested = "secondary"
                elif 16 <= age <= 19: suggested = "highschool"
                rid = f"repair_student_course_{sid}"
                repairs[rid] = RepairOption(
                    repair_id=rid,
                    description=f"Ajustar curso de {sid} a {suggested}",
                    cost=2,
                    action="modify_student",
                    target={"student_id": sid, "field": "curso", "new_value": suggested},
                )
                issues.append({"issue_id": f"course_age_mismatch_{sid}",
                               "type": "course_age_mismatch",
                               "description": "Edad incompatible con el nivel del curso",
                               "repair_ids": [rid]})

        # 3. Credits outside course range
        credits = student.get("creditos")
        if course in COURSE_CREDITS and isinstance(credits, int):
            c_lo, c_hi = COURSE_CREDITS[course]
            if not (c_lo <= credits <= c_hi):
                rid = f"repair_student_credits_{sid}"
                repairs[rid] = RepairOption(
                    repair_id=rid,
                    description=f"Ajustar créditos de {sid} al rango válido",
                    cost=2,
                    action="modify_student",
                    target={"student_id": sid, "field": "creditos",
                            "new_value": max(c_lo, min(c_hi, credits))},
                )
                issues.append({"issue_id": f"credits_mismatch_{sid}",
                               "type": "credits_mismatch",
                               "description": "Créditos fuera del rango del curso",
                               "repair_ids": [rid]})

    # ── Exams ────────────────────────────────────────────────────────────────

    for exam in dataset["exams"]:
        eid  = exam["id"]
        prof = professors.get(exam["profesor_id"])
        if not prof:
            continue

        # 4. Professor doesn't teach the exam subject  →  root-cause repair
        if exam["asignatura"] not in prof["asignaturas"]:
            local_rid = f"repair_exam_subject_{eid}"
            root_rid  = f"repair_exam_root_{eid}"
            correct   = prof["asignaturas"][0]

            repairs[local_rid] = RepairOption(
                repair_id=local_rid,
                description=f"Cambiar asignatura de {eid} a {correct}",
                cost=2,
                action="modify_exam",
                target={"exam_id": eid, "field": "asignatura", "new_value": correct},
            )
            # Root repair: fixing the exam at source cascades to regrades AND
            # textual comments, so it is offered as an alternative to every
            # downstream issue that references this exam.
            repairs[root_rid] = RepairOption(
                repair_id=root_rid,
                description=f"Reparación raíz: corregir asignatura de {eid} (cubre cascada)",
                cost=3,
                action="fix_exam_and_related",
                target={"exam_id": eid, "professor_id": exam["profesor_id"],
                        "new_value": correct},
            )
            root_repair_map[eid] = root_rid

            issues.append({"issue_id": f"subject_mismatch_{eid}",
                           "type": "professor_subject_mismatch",
                           "description": "Profesor no imparte la asignatura del examen",
                           "repair_ids": [local_rid, root_rid]})

        # 5. Future exam date
        try:
            exam_date = datetime.strptime(exam.get("fecha", ""), "%Y-%m-%d").date()
        except ValueError:
            exam_date = None

        if exam_date and exam_date > datetime.now().date():
            rid = f"repair_exam_date_{eid}"
            repairs[rid] = RepairOption(
                repair_id=rid,
                description=f"Corregir fecha futura del examen {eid}",
                cost=2,
                action="modify_exam",
                target={"exam_id": eid, "field": "fecha",
                        "new_value": datetime.now().date().isoformat()},
            )
            issues.append({"issue_id": f"future_exam_date_{eid}",
                           "type": "future_exam_date",
                           "description": "Fecha del examen está en el futuro",
                           "repair_ids": [rid]})

    # ── Results ──────────────────────────────────────────────────────────────

    valid_students = set(students.keys())
    valid_exams    = set(exams.keys())
    seen_pairs: set = set()

    for idx, result in enumerate(dataset["results"]):
        rid_list  = []
        result_id = result.get("id", f"idx_{idx}")
        key       = (result["estudiante_id"], result["examen_id"])

        # 6. Duplicate result
        if key in seen_pairs:
            rid = f"delete_duplicate_result_{result_id}"
            repairs[rid] = RepairOption(
                repair_id=rid,
                description=f"Eliminar resultado duplicado {result_id}",
                cost=3,
                action="delete_result_by_id",
                target={"result_id": result_id,
                        "student_id": result["estudiante_id"],
                        "exam_id":    result["examen_id"]},
            )
            issues.append({"issue_id": f"duplicate_result_{result_id}",
                           "type": "duplicate_result",
                           "description": "Resultado duplicado para el mismo par (estudiante, examen)",
                           "repair_ids": [rid]})
        else:
            seen_pairs.add(key)

        # 7. Ghost student
        if result["estudiante_id"] not in valid_students:
            rid = f"delete_result_student_{result['estudiante_id']}_{result['examen_id']}"
            repairs[rid] = RepairOption(
                repair_id=rid,
                description=f"Eliminar resultado con estudiante fantasma {result['estudiante_id']}",
                cost=5,
                action="delete_result",
                target={"student_id": result["estudiante_id"],
                        "exam_id":    result["examen_id"]},
            )
            rid_list.append(rid)

        # 8. Ghost exam
        if result["examen_id"] not in valid_exams:
            rid = f"delete_result_exam_{result['estudiante_id']}_{result['examen_id']}"
            repairs[rid] = RepairOption(
                repair_id=rid,
                description=f"Eliminar resultado con examen fantasma {result['examen_id']}",
                cost=5,
                action="delete_result",
                target={"student_id": result["estudiante_id"],
                        "exam_id":    result["examen_id"]},
            )
            rid_list.append(rid)

        if rid_list:
            issues.append({"issue_id": f"ghost_result_{result['estudiante_id']}_{result['examen_id']}",
                           "type": "ghost_reference",
                           "description": "Resultado referencia entidades inexistentes",
                           "repair_ids": rid_list})

        # 9. Invalid grade
        if not (0 <= result["nota"] <= 20):
            rid = f"repair_grade_{result['estudiante_id']}_{result['examen_id']}"
            repairs[rid] = RepairOption(
                repair_id=rid,
                description=f"Corregir nota inválida de {result['estudiante_id']} en {result['examen_id']}",
                cost=1,
                action="modify_result",
                target={"student_id": result["estudiante_id"],
                        "exam_id":    result["examen_id"],
                        "field":      "nota",
                        "new_value":  max(0, min(20, result["nota"]))},
            )
            issues.append({"issue_id": f"invalid_grade_{result['estudiante_id']}_{result['examen_id']}",
                           "type": "invalid_grade",
                           "description": "Nota fuera del rango [0, 20]",
                           "repair_ids": [rid]})

    # ── Regrades ─────────────────────────────────────────────────────────────

    # Build subject→professor map for finding replacement professors
    subject_to_profs: dict = {}
    for professor in dataset.get("professors", []):
        for subj in professor.get("asignaturas", []):
            subject_to_profs.setdefault(subj, []).append(professor["id"])

    for regrade in dataset.get("regrades", []):
        eid = regrade.get("examen_id")
        sid = regrade.get("estudiante_id")
        if not eid or eid not in exams:
            continue

        # 10. Regrade on a mismatched exam: root repair covers this too
        if eid in root_repair_map:
            bad_prof  = exams[eid]["profesor_id"]
            subj      = exams[eid].get("asignatura")
            candidates = [p for p in subject_to_profs.get(subj, []) if p != bad_prof]
            replacement = candidates[0] if candidates else bad_prof

            if regrade.get("profesor_id") == bad_prof:
                local_rid = f"repair_regrade_prof_{eid}_{sid}"
                repairs[local_rid] = RepairOption(
                    repair_id=local_rid,
                    description=f"Reasignar recalificación {eid}/{sid} a profesor válido",
                    cost=2,
                    action="modify_regrade",
                    target={"exam_id": eid, "student_id": sid,
                            "field": "profesor_id", "new_value": replacement},
                )
                issues.append({
                    "issue_id": f"suspicious_regrade_{sid}_{eid}",
                    "type": "suspicious_regrade",
                    "description": "Recalificación hecha por el mismo profesor que causó el examen erróneo",
                    # LOCAL fix (cost 2) OR ROOT fix (cost 3, but also resolves subject_mismatch)
                    "repair_ids": [local_rid, root_repair_map[eid]],
                })

    return issues, repairs, root_repair_map


# ─────────────────────────────────────────────
# TEXTUAL ISSUE LOADING  (from pre-computed file)
# ─────────────────────────────────────────────

def load_textual_issues(dataset: dict, root_repair_map: dict):
    """
    Reads textual_inconsistencies.json (produced by llm_detector.py) and
    converts it to issues + repairs compatible with the CP-SAT solver.

    Crucially: if a textual report is tied to an exam that already has a
    root structural repair, that repair is added as an ALTERNATIVE for the
    textual issue.  CP-SAT can then choose the root repair once and cover
    both the structural and textual problems simultaneously.
    """
    issues:  list = []
    repairs: dict = {}

    if not TEXTUAL_ISSUES_PATH.exists():
        print(f"[textual] {TEXTUAL_ISSUES_PATH.name} not found — skipping textual issues.")
        print("  Run `python detector/llm_detector.py` first.")
        return issues, repairs

    with TEXTUAL_ISSUES_PATH.open("r", encoding="utf-8") as fh:
        records = json.load(fh)

    students = {s["id"]: s for s in dataset["students"]}

    for record in records:
        evaluation = record.get("llm_evaluation", {})
        if evaluation.get("es_coherente", True):
            continue

        student_id = record.get("student_id")
        exam_id    = record.get("exam_id")
        issue_type = evaluation.get("tipo_inconsistencia", "otra")
        repaired   = evaluation.get("comentario_reparado") or "Comentario corregido."

        if student_id not in students:
            continue

        issue_repair_ids = []

        # Primary repair: rewrite the comment (cost 1 — minimal intervention)
        comment_rid = f"repair_comment_{student_id}_{exam_id}"
        repairs[comment_rid] = RepairOption(
            repair_id=comment_rid,
            description=f"Reescribir comentario de {student_id} en {exam_id}",
            cost=1,
            action="modify_report",
            target={"student_id": student_id, "exam_id": exam_id,
                    "field": "comment", "new_value": repaired},
        )
        issue_repair_ids.append(comment_rid)

        # If the issue is attendance-based, offer an attendance fix as alternative
        if issue_type == "asistencia_contradictoria":
            student  = students[student_id]
            grade    = record.get("grade", 10)
            new_att  = 95 if grade >= 15 else 65
            att_rid  = f"repair_attendance_{student_id}_{exam_id}"
            repairs[att_rid] = RepairOption(
                repair_id=att_rid,
                description=f"Ajustar asistencia de {student_id} para coherencia",
                cost=3,
                action="modify_student",
                target={"student_id": student_id, "field": "asistencia", "new_value": new_att},
            )
            issue_repair_ids.append(att_rid)

        # ROOT REPAIR CASCADE: if this textual report belongs to an exam that
        # has a structural root repair, selecting that root repair is also a
        # valid way to resolve this textual inconsistency (fixing the exam
        # assignment makes the context coherent).
        if exam_id in root_repair_map:
            issue_repair_ids.append(root_repair_map[exam_id])

        issues.append({
            "issue_id":    f"text_{student_id}_{exam_id}",
            "type":        "textual_inconsistency",
            "description": issue_type,
            "repair_ids":  issue_repair_ids,
        })

    return issues, repairs


# ─────────────────────────────────────────────
# CP-SAT MINIMUM-COST REPAIR SOLVER
# ─────────────────────────────────────────────

def solve_minimum_repairs(issues: list, repairs: dict):
    """
    Weighted minimum hitting-set via CP-SAT.

    For every issue at least one of its candidate repairs must be selected.
    The objective is to minimise the total cost of selected repairs.

    Returns (selected_repairs, covered_issue_ids, uncovered_issue_ids).
    """
    model     = cp_model.CpModel()
    variables = {rid: model.NewBoolVar(rid) for rid in repairs}

    for issue in issues:
        valid = [rid for rid in issue["repair_ids"] if rid in variables]
        if not valid:
            continue
        # At least one repair that can fix this issue must be selected
        model.Add(sum(variables[rid] for rid in valid) >= 1)

    # Minimise sum of (cost × selected)
    model.Minimize(
        sum(repairs[rid].cost * variables[rid] for rid in variables)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    status = solver.Solve(model)

    selected:     list = []
    selected_ids: set  = set()

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for rid, var in variables.items():
            if solver.Value(var):
                selected.append(repairs[rid])
                selected_ids.add(rid)

    # Compute coverage for the plan summary
    covered_issues   = []
    uncovered_issues = []
    for issue in issues:
        valid = [rid for rid in issue["repair_ids"] if rid in variables]
        if any(rid in selected_ids for rid in valid):
            covered_issues.append(issue["issue_id"])
        else:
            uncovered_issues.append(issue["issue_id"])

    return selected, covered_issues, uncovered_issues


# ─────────────────────────────────────────────
# APPLY REPAIRS
# ─────────────────────────────────────────────

def apply_repairs(dataset: dict, selected_repairs: list) -> dict:
    repaired = copy.deepcopy(dataset)

    students_map = {s["id"]: s for s in repaired["students"]}
    results_map  = {(r["estudiante_id"], r["examen_id"]): r for r in repaired["results"]}
    reports_map  = {(rp["student_id"], rp["exam_id"]): rp
                    for rp in repaired.get("teacher_reports", [])}

    for repair in selected_repairs:
        t = repair.target

        if repair.action == "modify_student":
            s = students_map.get(t["student_id"])
            if s:
                s[t["field"]] = t["new_value"]
                # When the course changes, credits valid for the old course may fall
                # outside the range of the new one — clamp them automatically so we
                # don't introduce a new inconsistency while fixing this one.
                if t["field"] == "curso":
                    new_course = t["new_value"]
                    if new_course in COURSE_CREDITS:
                        c_lo, c_hi = COURSE_CREDITS[new_course]
                        if not (c_lo <= s.get("creditos", c_lo) <= c_hi):
                            s["creditos"] = (c_lo + c_hi) // 2

        elif repair.action == "modify_exam":
            for exam in repaired["exams"]:
                if exam["id"] == t["exam_id"]:
                    exam[t["field"]] = t["new_value"]
                    break

        elif repair.action == "fix_exam_and_related":
            eid_target = t.get("exam_id")
            bad_prof   = t.get("professor_id")
            for exam in repaired["exams"]:
                if exam["id"] == eid_target:
                    exam["asignatura"] = t.get("new_value")
                    break
            # Also fix any regrades that reference the same bad professor on this exam
            for rg in repaired.get("regrades", []):
                if (rg.get("examen_id") == eid_target
                        and rg.get("profesor_id") == bad_prof):
                    rg["profesor_id"] = t.get("new_value", bad_prof)

        elif repair.action == "modify_result":
            r = results_map.get((t["student_id"], t["exam_id"]))
            if r:
                r[t["field"]] = t["new_value"]

        elif repair.action == "modify_report":
            rp = reports_map.get((t["student_id"], t["exam_id"]))
            if rp:
                rp[t["field"]] = t["new_value"]

        elif repair.action == "modify_regrade":
            for rg in repaired.get("regrades", []):
                if (rg.get("examen_id") == t.get("exam_id")
                        and rg.get("estudiante_id") == t.get("student_id")):
                    rg[t["field"]] = t["new_value"]
                    break

        elif repair.action == "delete_result":
            repaired["results"] = [
                r for r in repaired["results"]
                if not (r["estudiante_id"] == t["student_id"]
                        and r["examen_id"] == t["exam_id"])
            ]
            repaired["teacher_reports"] = [
                rp for rp in repaired.get("teacher_reports", [])
                if not (rp["student_id"] == t["student_id"]
                        and rp["exam_id"] == t["exam_id"])
            ]

        elif repair.action == "delete_result_by_id":
            rid_val    = t.get("result_id")
            sid, eid   = t.get("student_id"), t.get("exam_id")
            repaired["results"] = [r for r in repaired["results"]
                                   if r.get("id") != rid_val]
            removed = False
            kept    = []
            for rp in repaired.get("teacher_reports", []):
                if (not removed and rp.get("student_id") == sid
                        and rp.get("exam_id") == eid):
                    removed = True
                    continue
                kept.append(rp)
            repaired["teacher_reports"] = kept

        elif repair.action == "delete_report":
            repaired["teacher_reports"] = [
                rp for rp in repaired.get("teacher_reports", [])
                if not (rp["student_id"] == t["student_id"]
                        and rp["exam_id"] == t["exam_id"])
            ]

    # ── Post-repair consistency pass ─────────────────────────────────────────
    # Some repairs have cascading effects not visible to CP-SAT:
    # e.g. fixing a wrong stored age corrects the course-age mismatch too, but
    # if a course repair was ALSO selected (because the wrong age made the course
    # look invalid), it may have set the course to the wrong level for the now-
    # corrected age.  This pass enforces age ↔ course ↔ credits consistency after
    # all individual repairs have been applied.
    _enforce_student_consistency(repaired["students"])

    return repaired


def _enforce_student_consistency(students: list) -> None:
    """Ensure every student's course matches their age, and credits match their course.

    Handles two cascading cases:
    1. age repair was applied, making the previously-valid course invalid →
       pick the correct course for the corrected age, then clamp credits.
    2. course repair changed the course, then a credits repair (computed for
       the OLD course) overwrote with a value invalid for the new course →
       clamp credits to the current course's range regardless of how we got here.
    """
    for student in students:
        age    = student.get("edad")
        course = student.get("curso")

        # Step 1: fix course if it doesn't match age
        if isinstance(age, int) and course in COURSE_AGE:
            lo, hi = COURSE_AGE[course]
            if not (lo <= age <= hi):
                if   6  <= age <= 11: new_course = "primary"
                elif 12 <= age <= 15: new_course = "secondary"
                elif 16 <= age <= 19: new_course = "highschool"
                else:
                    new_course = None  # age outside all ranges — can't auto-assign
                if new_course:
                    student["curso"] = new_course
                    course = new_course  # use updated course for credits check

        # Step 2: clamp credits to the (possibly updated) course range
        credits = student.get("creditos")
        if isinstance(credits, int) and course in COURSE_CREDITS:
            c_lo, c_hi = COURSE_CREDITS[course]
            if not (c_lo <= credits <= c_hi):
                student["creditos"] = (c_lo + c_hi) // 2


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    dataset = load_dataset()

    structural_issues, structural_repairs, root_repair_map = detect_structural_issues(dataset)
    textual_issues,    textual_repairs                     = load_textual_issues(dataset, root_repair_map)

    all_issues  = structural_issues + textual_issues
    all_repairs = {**structural_repairs, **textual_repairs}

    print(f"Issues detectados  : {len(all_issues)}")
    print(f"  - estructurales  : {len(structural_issues)}")
    print(f"  - textuales      : {len(textual_issues)}")
    print(f"Reparaciones candidatas: {len(all_repairs)}")
    print(f"Reparaciones raíz  : {len(root_repair_map)}")
    print("Resolviendo con CP-SAT…")

    selected, covered, uncovered = solve_minimum_repairs(all_issues, all_repairs)
    repaired_dataset             = apply_repairs(dataset, selected)

    # ── Build audit trail: for each selected repair, list issues it covers ──
    selected_ids = {r.repair_id for r in selected}
    repair_audit: dict = {r.repair_id: [] for r in selected}
    for issue in all_issues:
        for rid in issue["repair_ids"]:
            if rid in selected_ids:
                repair_audit[rid].append(issue["issue_id"])
                break  # issue is covered; move on

    # ── Count issues covered by root repairs specifically ───────────────────
    root_covered = sum(
        1 for issue in all_issues
        if any(rid in root_repair_map.values() and rid in selected_ids
               for rid in issue["repair_ids"])
    )

    # ── Summary by type ─────────────────────────────────────────────────────
    from collections import Counter
    issues_by_type   = dict(Counter(i["type"] for i in all_issues))
    selected_by_type = dict(Counter(r.action for r in selected))

    plan = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_issues":          len(all_issues),
            "issues_by_type":        issues_by_type,
            "total_repairs_considered": len(all_repairs),
            "repairs_selected":      len(selected),
            "issues_covered":        len(covered),
            "issues_uncovered":      len(uncovered),
            "uncovered_issue_ids":   uncovered,
            "total_cost":            sum(r.cost for r in selected),
            "root_repairs_used":     sum(1 for r in selected if r.repair_id in root_repair_map.values()),
            "issues_resolved_by_root_repairs": root_covered,
            "selected_actions_by_type": selected_by_type,
        },
        "selected_repairs": [
            {
                "repair_id":    r.repair_id,
                "description":  r.description,
                "cost":         r.cost,
                "action":       r.action,
                "target":       r.target,
                "covers_issues": repair_audit.get(r.repair_id, []),
            }
            for r in selected
        ],
        "ground_truth": dataset.get("metadata", {}).get("injected_inconsistencies"),
    }

    with OUTPUT_PLAN_PATH.open("w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=4, ensure_ascii=False)

    with OUTPUT_REPAIRED_PATH.open("w", encoding="utf-8") as fh:
        json.dump(repaired_dataset, fh, indent=4, ensure_ascii=False)

    print(f"\nResultados:")
    print(f"  Issues totales      : {len(all_issues)}")
    print(f"  Reparaciones elegidas: {len(selected)}")
    print(f"  Issues cubiertos    : {len(covered)}")
    print(f"  Issues sin cubrir   : {len(uncovered)}")
    print(f"  Coste total         : {sum(r.cost for r in selected)}")
    root_count = sum(1 for r in selected if r.repair_id in root_repair_map.values())
    print(f"  Reparaciones raíz usadas: {root_count}  (cubren cascada de issues)")
    print(f"\nPlan guardado en      : {OUTPUT_PLAN_PATH.name}")
    print(f"Dataset reparado en   : {OUTPUT_REPAIRED_PATH.name}")


if __name__ == "__main__":
    main()

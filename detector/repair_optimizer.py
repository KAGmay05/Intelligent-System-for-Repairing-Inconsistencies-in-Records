"""
repair_optimizer.py
───────────────────
Minimum-cost repair planner for academic-record inconsistencies.

Pipeline:
  1. detect_structural_issues()   → issues + repairs from hard rules
    2. load_textual_issues()        → issues + repairs from data/textual_inconsistencies.json
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
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ortools.sat.python import cp_model

# Structural detector — reused to VERIFY global coherence on the repaired dataset.
# Works both when this file is run as a script (python detector/repair_optimizer.py)
# and when imported as a package module (from experiments/).
try:
    from detector.structural_rules import detect_all_structural
except ImportError:  # running as a script: detector/ is on sys.path
    from structural_rules import detect_all_structural


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET_PATH              = DATA_DIR / "dataset.json"
TEXTUAL_ISSUES_PATH       = DATA_DIR / "textual_inconsistencies.json"
OUTPUT_PLAN_PATH          = DATA_DIR / "repair_plan.json"
OUTPUT_REPAIRED_PATH      = DATA_DIR / "repaired_dataset.json"

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

def load_dataset(path: Path | None = None) -> dict:
    p = path or DATASET_PATH
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
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
            # Resolve the mismatch by correcting the BIRTH DATE to match the stored
            # age, NOT the age.  The age participates in the course-age and credits
            # constraints, so editing it can introduce a brand-new inconsistency
            # (e.g. push the student out of their course's age range); the birth
            # date participates in nothing else, so fixing it is side-effect free.
            corrected_birth_year = datetime.now().year - student["edad"]
            try:
                old_birth = datetime.strptime(student["fecha_nacimiento"], "%Y-%m-%d")
                try:
                    new_birth = old_birth.replace(year=corrected_birth_year)
                except ValueError:                       # Feb 29 → use 28
                    new_birth = old_birth.replace(year=corrected_birth_year, day=28)
                new_fecha = new_birth.strftime("%Y-%m-%d")
            except (KeyError, ValueError):
                new_fecha = f"{corrected_birth_year:04d}-01-01"

            rid = f"repair_student_age_{sid}"
            repairs[rid] = RepairOption(
                repair_id=rid,
                description=f"Ajustar fecha de nacimiento de {sid} para coincidir con edad {student['edad']}",
                cost=1,
                action="modify_student",
                target={"student_id": sid, "field": "fecha_nacimiento", "new_value": new_fecha},
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

    # 11. Orphan regrades (no matching base result)
    result_pairs = {
        (r["estudiante_id"], r["examen_id"]) for r in dataset.get("results", [])
    }
    for regrade in dataset.get("regrades", []):
        sid = regrade.get("estudiante_id")
        eid = regrade.get("examen_id")
        if (sid, eid) in result_pairs:
            continue
        rid = f"delete_orphan_regrade_{sid}_{eid}"
        repairs[rid] = RepairOption(
            repair_id=rid,
            description=f"Eliminar recalificación huérfana de {sid} en {eid} (sin resultado base)",
            cost=4,
            action="delete_regrade",
            target={"student_id": sid, "exam_id": eid},
        )
        issues.append({
            "issue_id":   f"orphan_regrade_{sid}_{eid}",
            "type":       "orphan_regrade",
            "description": "Recalificación sin resultado base",
            "repair_ids": [rid],
        })

    return issues, repairs, root_repair_map


# ─────────────────────────────────────────────
# TEXTUAL ISSUE LOADING  (from pre-computed file)
# ─────────────────────────────────────────────

def load_textual_issues(dataset: dict, root_repair_map: dict):
    """
    Reads data/textual_inconsistencies.json (produced by llm_detector.py) and
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
            grade    = record.get("grade")
            if not isinstance(grade, (int, float)):
                grade = 10
            # A strong grade suggests the student did attend → raise attendance;
            # otherwise the contradiction is resolved by lowering it.
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

def _build_issue_decisions(issues: list, repairs: dict, selected_ids: set, reason_mode: str = "solver"):
    """Build per-issue coverage + rationale from a selected repair set."""
    covered_issues   = []
    uncovered_issues = []
    issue_decision_map: dict = {}

    for issue in issues:
        valid = [rid for rid in issue["repair_ids"] if rid in repairs]
        unknowable = [rid for rid in issue["repair_ids"] if rid not in repairs]

        chosen_rid = None
        rejected   = []

        if any(rid in selected_ids for rid in valid):
            covered_issues.append(issue["issue_id"])
            for rid in valid:
                if rid in selected_ids:
                    chosen_rid = rid
                    break

            for rid in valid:
                if rid == chosen_rid:
                    continue
                if reason_mode == "solver":
                    chosen_cost    = repairs[chosen_rid].cost
                    candidate_cost = repairs[rid].cost
                    if candidate_cost > chosen_cost:
                        reason = "higher_cost"
                    elif candidate_cost == chosen_cost:
                        reason = "equivalent_cost_not_needed"
                    else:
                        reason = "lower_cost_but_not_needed"
                else:
                    reason = "not_selected_by_greedy"

                rejected.append({
                    "repair_id": rid,
                    "description": repairs[rid].description,
                    "cost": repairs[rid].cost,
                    "reason": reason,
                })

            coverable          = True
            uncoverable_reason = None
        else:
            uncovered_issues.append(issue["issue_id"])
            coverable = False
            if not valid and unknowable:
                uncoverable_reason = (
                    f"Todas las reparaciones candidatas ({', '.join(unknowable)}) "
                    "no llegaron al solver (posible conflicto de IDs)."
                )
            elif not valid:
                uncoverable_reason = "El issue no tiene reparaciones candidatas definidas."
            else:
                uncoverable_reason = (
                    f"No se seleccionó ninguna candidata para el issue "
                    f"({', '.join(valid)})."
                )

        issue_decision_map[issue["issue_id"]] = {
            "issue_type": issue["type"],
            "chosen": chosen_rid,
            "chosen_cost": repairs[chosen_rid].cost if chosen_rid else None,
            "rejected": rejected,
            "coverable": coverable,
            "uncoverable_reason": uncoverable_reason if not coverable else None,
        }

    return covered_issues, uncovered_issues, issue_decision_map


def solve_minimum_repairs(issues: list, repairs: dict):
    """
    Weighted minimum hitting-set via CP-SAT.

    For every issue at least one of its candidate repairs must be selected.
    The objective is to minimise the total cost of selected repairs.

    Returns (selected_repairs, covered_issue_ids, uncovered_issue_ids,
             issue_decision_map).

    issue_decision_map: issue_id → {
        "chosen":   repair_id | None,
        "rejected": [{"repair_id", "cost", "reason"}],
        "coverable": bool,
        "uncoverable_reason": str | None,
    }
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

    covered_issues, uncovered_issues, issue_decision_map = _build_issue_decisions(
        issues, repairs, selected_ids, reason_mode="solver"
    )

    return selected, covered_issues, uncovered_issues, issue_decision_map


def solve_minimum_repairs_greedy(issues: list, repairs: dict):
    """
    Greedy weighted set cover approximation:
    picks the repair with best (newly_covered_issues / cost) ratio each step.
    """
    issue_to_valid: dict = {}
    for issue in issues:
        issue_to_valid[issue["issue_id"]] = [rid for rid in issue["repair_ids"] if rid in repairs]

    uncovered = {
        issue["issue_id"]
        for issue in issues
        if issue_to_valid.get(issue["issue_id"])
    }

    selected_ids: set = set()

    while uncovered:
        best_rid = None
        best_score = -1.0
        best_gain = -1
        best_cost = 10**9

        for rid, rep in repairs.items():
            newly = 0
            for issue in issues:
                iid = issue["issue_id"]
                if iid in uncovered and rid in issue_to_valid.get(iid, []):
                    newly += 1
            if newly == 0:
                continue

            score = newly / max(1, rep.cost)
            if (
                score > best_score
                or (score == best_score and newly > best_gain)
                or (score == best_score and newly == best_gain and rep.cost < best_cost)
                or (score == best_score and newly == best_gain and rep.cost == best_cost and (best_rid is None or rid < best_rid))
            ):
                best_rid = rid
                best_score = score
                best_gain = newly
                best_cost = rep.cost

        if best_rid is None:
            break

        selected_ids.add(best_rid)

        to_remove = []
        for iid in uncovered:
            if best_rid in issue_to_valid.get(iid, []):
                to_remove.append(iid)
        for iid in to_remove:
            uncovered.remove(iid)

    selected = [repairs[rid] for rid in selected_ids]
    covered_issues, uncovered_issues, issue_decision_map = _build_issue_decisions(
        issues, repairs, selected_ids, reason_mode="greedy"
    )
    return selected, covered_issues, uncovered_issues, issue_decision_map


def solve_minimum_repairs_sa(
    issues: list,
    repairs: dict,
    temp0: float = 25.0,
    cooling: float = 0.995,
    iters: int = 5000,
    seed: int = 42,
):
    """
    Simulated annealing for weighted set-cover style repair selection.
    Objective:
      total_cost + penalty_uncovered * uncovered_issues + 0.1 * num_selected_repairs
    """
    rng = random.Random(seed)
    repair_ids = sorted(repairs.keys())

    issue_to_valid: dict[str, list[str]] = {}
    for issue in issues:
        issue_to_valid[issue["issue_id"]] = [rid for rid in issue["repair_ids"] if rid in repairs]

    max_cost = max((rep.cost for rep in repairs.values()), default=1)
    penalty_uncovered = max(1000, len(repair_ids) * max_cost)

    def _stats(selected_ids: set[str]):
        uncovered_count = 0
        for issue in issues:
            valid = issue_to_valid.get(issue["issue_id"], [])
            if valid and not any(rid in selected_ids for rid in valid):
                uncovered_count += 1
        total_cost = sum(repairs[rid].cost for rid in selected_ids)
        objective = total_cost + penalty_uncovered * uncovered_count + 0.1 * len(selected_ids)
        return objective, total_cost, uncovered_count

    # Start from greedy solution for a strong warm-start.
    greedy_selected, _, _, _ = solve_minimum_repairs_greedy(issues, repairs)
    current_ids = {r.repair_id for r in greedy_selected}
    if not current_ids and repair_ids:
        current_ids.add(rng.choice(repair_ids))

    current_obj, _, _ = _stats(current_ids)
    best_ids = set(current_ids)
    best_obj = current_obj

    temp = max(1e-6, float(temp0))
    n_iters = max(1, int(iters))
    cool = min(0.9999, max(0.90, float(cooling)))

    for _ in range(n_iters):
        neighbor = set(current_ids)

        if repair_ids:
            if rng.random() < 0.65:
                rid = rng.choice(repair_ids)
                if rid in neighbor:
                    neighbor.remove(rid)
                else:
                    neighbor.add(rid)
            else:
                issue = rng.choice(issues)
                candidates = issue_to_valid.get(issue["issue_id"], [])
                if candidates:
                    rid = rng.choice(candidates)
                    neighbor.add(rid)
                    if len(neighbor) > 1 and rng.random() < 0.4:
                        removable = [x for x in neighbor if x != rid]
                        if removable:
                            neighbor.remove(rng.choice(removable))

        n_obj, _, _ = _stats(neighbor)
        delta = n_obj - current_obj

        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
            current_ids = neighbor
            current_obj = n_obj
            if n_obj < best_obj:
                best_obj = n_obj
                best_ids = set(neighbor)

        temp *= cool
        if temp < 1e-6:
            temp = 1e-6

    selected = [repairs[rid] for rid in sorted(best_ids)]
    covered_issues, uncovered_issues, issue_decision_map = _build_issue_decisions(
        issues, repairs, best_ids, reason_mode="greedy"
    )
    return selected, covered_issues, uncovered_issues, issue_decision_map


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

        elif repair.action == "delete_regrade":
            repaired["regrades"] = [
                rg for rg in repaired.get("regrades", [])
                if not (rg.get("estudiante_id") == t["student_id"]
                        and rg.get("examen_id")    == t["exam_id"])
            ]

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
    import argparse as _ap, os as _os
    global DATASET_PATH, TEXTUAL_ISSUES_PATH, OUTPUT_PLAN_PATH, OUTPUT_REPAIRED_PATH
    _parser = _ap.ArgumentParser(description="Repair optimizer — CP-SAT minimum-cost hitting set")
    _parser.add_argument("--verbose", action="store_true",
                         help="Mostrar tabla de decisiones del solver (elegida/rechazadas por issue)")
    _parser.add_argument("--dataset",          default=None,
                         help="Ruta al dataset JSON (override de DATA_DIR/dataset.json)")
    _parser.add_argument("--textual",          default=None,
                         help="Ruta a textual_inconsistencies.json (override)")
    _parser.add_argument("--output-plan",      default=None,
                         help="Ruta de salida para repair_plan.json (override)")
    _parser.add_argument("--output-repaired",  default=None,
                         help="Ruta de salida para repaired_dataset.json (override)")
    _parser.add_argument("--optimizer", choices=["cpsat", "greedy", "sa"], default="cpsat",
                         help="Método de optimización: cpsat (exacto), greedy (heurístico) o sa (recocido simulado)")
    _parser.add_argument("--sa-temp0", type=float, default=25.0,
                         help="Temperatura inicial para SA")
    _parser.add_argument("--sa-cooling", type=float, default=0.995,
                         help="Factor de enfriamiento para SA (0.90-0.9999)")
    _parser.add_argument("--sa-iters", type=int, default=5000,
                         help="Número de iteraciones de SA")
    _parser.add_argument("--sa-seed", type=int, default=42,
                         help="Semilla aleatoria para SA")
    _args, _ = _parser.parse_known_args()
    verbose = _args.verbose or _os.environ.get("VERBOSE", "").strip() == "1"

    # Allow per-run path overrides (used by experiments/run_instances.py)
    if _args.dataset:
        DATASET_PATH = Path(_args.dataset)
    if _args.textual:
        TEXTUAL_ISSUES_PATH = Path(_args.textual)
    if _args.output_plan:
        OUTPUT_PLAN_PATH = Path(_args.output_plan)
    if _args.output_repaired:
        OUTPUT_REPAIRED_PATH = Path(_args.output_repaired)

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
    if _args.optimizer == "greedy":
        print("Resolviendo con Greedy set-cover (heurístico)…")
        selected, covered, uncovered, issue_decisions = solve_minimum_repairs_greedy(all_issues, all_repairs)
    elif _args.optimizer == "sa":
        print("Resolviendo con Recocido Simulado (metaheurístico)…")
        selected, covered, uncovered, issue_decisions = solve_minimum_repairs_sa(
            all_issues,
            all_repairs,
            temp0=_args.sa_temp0,
            cooling=_args.sa_cooling,
            iters=_args.sa_iters,
            seed=_args.sa_seed,
        )
    else:
        print("Resolviendo con CP-SAT…")
        selected, covered, uncovered, issue_decisions = solve_minimum_repairs(all_issues, all_repairs)
    repaired_dataset                               = apply_repairs(dataset, selected)

    # ── VERIFY global coherence ──────────────────────────────────────────────
    # The objective covers the DETECTED issues, but the goal is GLOBAL coherence:
    # the repaired dataset must contain no structural inconsistency at all — not
    # even one accidentally introduced by a repair (a cascade).  We re-run the
    # structural detector on the repaired dataset and treat any residual issue as
    # a coherence failure.  This makes "global coherence restored" a proven,
    # first-class property of the output rather than an assumption.
    from collections import Counter as _Counter
    residual = detect_all_structural(repaired_dataset)
    residual_by_type = dict(sorted(_Counter(i["type"] for i in residual).items()))
    global_coherence_verified = len(residual) == 0

    # ── Build audit trail: for each selected repair, list issues it covers ──
    selected_ids = {r.repair_id for r in selected}
    repair_audit: dict = {r.repair_id: [] for r in selected}
    for issue in all_issues:
        for rid in issue["repair_ids"]:
            if rid in selected_ids:
                repair_audit[rid].append(issue["issue_id"])
                break  # issue is covered; move on

    # ── Build rejected_alternatives per selected repair ──────────────────────
    # Gather all issues each repair covers so we can list what was competed against
    repair_to_issues: dict = {}  # repair_id → [issue_ids it was candidate for]
    for issue in all_issues:
        for rid in issue["repair_ids"]:
            repair_to_issues.setdefault(rid, []).append(issue["issue_id"])

    def _rejected_for_repair(repair_id: str) -> list:
        """For a selected repair, find sibling candidates that were NOT chosen."""
        rejected_alts = []
        my_issues = repair_to_issues.get(repair_id, [])
        seen: set = set()
        for iid in my_issues:
            dec = issue_decisions.get(iid, {})
            for alt in dec.get("rejected", []):
                if alt["repair_id"] not in seen:
                    seen.add(alt["repair_id"])
                    rejected_alts.append(alt)
        return rejected_alts

    # ── Count issues covered by root repairs specifically ───────────────────
    root_covered = sum(
        1 for issue in all_issues
        if any(rid in root_repair_map.values() and rid in selected_ids
               for rid in issue["repair_ids"])
    )

    # ── Cost breakdown by issue type ────────────────────────────────────────
    from collections import Counter, defaultdict
    issues_by_type   = dict(Counter(i["type"] for i in all_issues))
    selected_by_type = dict(Counter(r.action for r in selected))

    cost_by_type: dict = defaultdict(int)
    for issue in all_issues:
        dec = issue_decisions.get(issue["issue_id"], {})
        if dec.get("chosen"):
            cost_by_type[issue["type"]] += dec["chosen_cost"]

    # ── Uncovered issues with explanation ───────────────────────────────────
    uncovered_detail = [
        {
            "issue_id":  iid,
            "issue_type": issue_decisions.get(iid, {}).get("issue_type", "unknown"),
            "reason":    issue_decisions.get(iid, {}).get("uncoverable_reason", "unknown"),
        }
        for iid in uncovered
    ]

    plan = {
        "generated_at": datetime.now().isoformat(),
        "optimizer": _args.optimizer,
        "optimizer_params": {
            "sa_temp0": _args.sa_temp0,
            "sa_cooling": _args.sa_cooling,
            "sa_iters": _args.sa_iters,
            "sa_seed": _args.sa_seed,
        } if _args.optimizer == "sa" else {},
        # Proof that the repairs restored GLOBAL coherence (re-detected on the
        # repaired dataset): no residual structural inconsistencies, including none
        # introduced as a side-effect of the repairs themselves.
        "verification": {
            "global_coherence_verified": global_coherence_verified,
            "residual_structural_issues": len(residual),
            "residual_by_type": residual_by_type,
        },
        "summary": {
            "total_issues":             len(all_issues),
            "issues_by_type":           issues_by_type,
            "total_repairs_considered": len(all_repairs),
            "repairs_selected":         len(selected),
            "issues_covered":           len(covered),
            "issues_uncovered":         len(uncovered),
            "uncovered_issue_ids":      uncovered,
            "total_cost":               sum(r.cost for r in selected),
            "cost_by_issue_type":       dict(sorted(cost_by_type.items())),
            "root_repairs_used":        sum(1 for r in selected if r.repair_id in root_repair_map.values()),
            "issues_resolved_by_root_repairs": root_covered,
            "selected_actions_by_type": selected_by_type,
        },
        "selected_repairs": [
            {
                "repair_id":            r.repair_id,
                "description":          r.description,
                "cost":                 r.cost,
                "action":               r.action,
                "target":               r.target,
                "covers_issues":        repair_audit.get(r.repair_id, []),
                "rejected_alternatives": _rejected_for_repair(r.repair_id),
            }
            for r in selected
        ],
        "uncovered_issues":  uncovered_detail,
        "issue_decisions":   [
            {
                "issue_id":           iid,
                "issue_type":         dec["issue_type"],
                "coverable":          dec["coverable"],
                "chosen_repair":      dec["chosen"],
                "chosen_cost":        dec["chosen_cost"],
                "rejected_alternatives": dec["rejected"],
                "uncoverable_reason": dec["uncoverable_reason"],
            }
            for iid, dec in issue_decisions.items()
        ],
        "ground_truth": dataset.get("metadata", {}).get("injected_inconsistencies"),
    }

    with OUTPUT_PLAN_PATH.open("w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=4, ensure_ascii=False)

    with OUTPUT_REPAIRED_PATH.open("w", encoding="utf-8") as fh:
        json.dump(repaired_dataset, fh, indent=4, ensure_ascii=False)

    # ── Console output ───────────────────────────────────────────────────────
    total_cost = sum(r.cost for r in selected)
    root_count = sum(1 for r in selected if r.repair_id in root_repair_map.values())

    print(f"\nResultados:")
    print(f"  Issues totales       : {len(all_issues)}")
    print(f"  Reparaciones elegidas: {len(selected)}")
    print(f"  Issues cubiertos     : {len(covered)}")
    print(f"  Issues sin cubrir    : {len(uncovered)}")
    print(f"  Coste total          : {total_cost}")
    print(f"  Reparaciones raíz usadas: {root_count}  (cubren cascada de issues)")
    if global_coherence_verified:
        print(f"  Coherencia global    : ✔ VERIFICADA (0 inconsistencias estructurales residuales)")
    else:
        print(f"  Coherencia global    : ✘ FALLÓ — {len(residual)} issues residuales: {residual_by_type}")
        print(f"     (una reparación introdujo una inconsistencia nueva; revisar el modelo)")

    # ── Coste por tipo de issue ──────────────────────────────────────────────
    print(f"\n  Coste por tipo de issue:")
    col = max((len(k) for k in cost_by_type), default=10) + 2
    for itype in sorted(cost_by_type):
        count = issues_by_type.get(itype, 0)
        cost  = cost_by_type[itype]
        avg   = cost / count if count else 0
        print(f"    {itype:<{col}} issues={count:>3}  coste={cost:>4}  avg={avg:.1f}")

    # ── Detalle por issue: reparación elegida vs alternativas rechazadas ────
    if verbose:
        print(f"\n  Decisiones del solver (elegida ✔ / rechazadas ✘):")
        for issue in all_issues:
            dec  = issue_decisions[issue["issue_id"]]
            iid  = issue["issue_id"]
            if dec["coverable"]:
                chosen = all_repairs[dec["chosen"]]
                print(f"    [{issue['type']}] {iid}")
                print(f"      ✔ {chosen.repair_id}  coste={chosen.cost}  '{chosen.description}'")
                for alt in dec["rejected"]:
                    print(f"      ✘ {alt['repair_id']}  coste={alt['cost']}  razón={alt['reason']}  '{alt['description']}'")
            else:
                print(f"    [{issue['type']}] {iid}  ⚠ NO CUBIERTO")
                print(f"      Causa: {dec['uncoverable_reason']}")
    else:
        # Always show uncovered issues even without --verbose
        uncov_list = [d for d in issue_decisions.values() if not d["coverable"]]
        if uncov_list:
            print(f"\n  Issues NO cubiertos ({len(uncov_list)}):")
            for d in uncov_list:
                print(f"    [{d['issue_type']}]  Causa: {d['uncoverable_reason']}")

    print(f"\nPlan guardado en      : {OUTPUT_PLAN_PATH.name}")
    print(f"Dataset reparado en   : {OUTPUT_REPAIRED_PATH.name}")


if __name__ == "__main__":
    main()

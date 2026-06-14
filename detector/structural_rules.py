import argparse
import json
from collections import Counter
from datetime import datetime, date
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "dataset.json"
DEFAULT_OUTPUT_PATH = DATA_DIR / "inconsistencies.json"

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


def _new_issue(idx, issue_type, related_entities, field, current_value, expected_constraint, suggested_repairs, entity_type=None):
    issue = {
        "id": f"INC_{idx}",
        "type": issue_type,
        "related_entities": related_entities,
        "field": field,
        "current_value": current_value,
        "expected_constraint": expected_constraint,
        "suggested_repairs": suggested_repairs,
    }
    if entity_type:
        issue["entity_type"] = entity_type
    return issue


def detect_invalid_grades(dataset):
    issues = []
    for result in dataset.get("results", []):
        grade = result.get("nota")
        if isinstance(grade, (int, float)) and 0 <= grade <= 20:
            continue
        issues.append({
            "type": "invalid_grade",
            "related_entities": {
                "result_id": result.get("id"),
                "student_id": result.get("estudiante_id"),
                "exam_id": result.get("examen_id"),
            },
            "field": "nota",
            "current_value": grade,
            "expected_constraint": "0 <= nota <= 20",
            "suggested_repairs": [{
                "action": "modify",
                "field": "nota",
                "new_value": 0 if grade is None else max(0, min(20, int(grade))),
                "cost": 1,
            }],
            "entity_type": "result",
        })
    return issues


def detect_age_birth_mismatch(dataset):
    issues = []
    for student in dataset.get("students", []):
        birth_str = student.get("fecha_nacimiento")
        try:
            birth_year = datetime.strptime(birth_str, "%Y-%m-%d").year
        except (TypeError, ValueError):
            continue
        expected_age = date.today().year - birth_year
        stored_age = student.get("edad")
        if not isinstance(stored_age, int):
            continue
        if abs(expected_age - stored_age) <= 1:
            continue
        issues.append({
            "type": "age_birth_mismatch",
            "related_entities": {"student_id": student.get("id")},
            "field": "edad",
            "current_value": stored_age,
            "expected_constraint": f"edad should be approximately {expected_age}",
            "suggested_repairs": [{
                "action": "modify",
                "field": "edad",
                "new_value": expected_age,
                "cost": 1,
            }],
            "entity_type": "student",
        })
    return issues


def detect_course_age_mismatch(dataset):
    issues = []
    for student in dataset.get("students", []):
        course = student.get("curso")
        age = student.get("edad")
        if course not in COURSE_AGE or not isinstance(age, int):
            continue
        lo, hi = COURSE_AGE[course]
        if lo <= age <= hi:
            continue
        issues.append({
            "type": "course_age_mismatch",
            "related_entities": {"student_id": student.get("id")},
            "field": "curso",
            "current_value": course,
            "expected_constraint": f"age {age} must be in range [{lo}, {hi}] for course {course}",
            "suggested_repairs": [{
                "action": "modify",
                "field": "curso",
                "new_value": "secondary" if 12 <= age <= 15 else "highschool" if 16 <= age <= 19 else "primary",
                "cost": 2,
            }],
            "entity_type": "student",
        })
    return issues


def detect_credits_mismatch(dataset):
    issues = []
    for student in dataset.get("students", []):
        course = student.get("curso")
        credits = student.get("creditos")
        if course not in COURSE_CREDITS or not isinstance(credits, int):
            continue
        lo, hi = COURSE_CREDITS[course]
        if lo <= credits <= hi:
            continue
        issues.append({
            "type": "credits_mismatch",
            "related_entities": {"student_id": student.get("id")},
            "field": "creditos",
            "current_value": credits,
            "expected_constraint": f"creditos for {course} must be in range [{lo}, {hi}]",
            "suggested_repairs": [{
                "action": "modify",
                "field": "creditos",
                "new_value": max(lo, min(hi, credits)),
                "cost": 2,
            }],
            "entity_type": "student",
        })
    return issues


def detect_invalid_professor_subject(dataset):
    issues = []
    professors = {p.get("id"): p for p in dataset.get("professors", [])}
    for exam in dataset.get("exams", []):
        professor = professors.get(exam.get("profesor_id"))
        if not professor:
            continue
        subject = exam.get("asignatura")
        valid_subjects = professor.get("asignaturas", [])
        if subject in valid_subjects:
            continue
        issues.append({
            "type": "invalid_professor_subject",
            "related_entities": {
                "exam_id": exam.get("id"),
                "professor_id": professor.get("id"),
            },
            "field": "asignatura",
            "current_value": subject,
            "expected_constraint": "professor must teach subject",
            "suggested_repairs": [{
                "action": "modify",
                "field": "asignatura",
                "new_value": valid_subjects[0] if valid_subjects else subject,
                "cost": 2,
            }],
            "entity_type": "exam",
        })
    return issues


def detect_future_exam_dates(dataset):
    issues = []
    today = date.today()
    for exam in dataset.get("exams", []):
        exam_date_str = exam.get("fecha")
        try:
            exam_date = datetime.strptime(exam_date_str, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue
        if exam_date <= today:
            continue
        issues.append({
            "type": "future_exam_date",
            "related_entities": {"exam_id": exam.get("id")},
            "field": "fecha",
            "current_value": exam_date_str,
            "expected_constraint": f"exam date must be <= {today.isoformat()}",
            "suggested_repairs": [{
                "action": "modify",
                "field": "fecha",
                "new_value": today.isoformat(),
                "cost": 2,
            }],
            "entity_type": "exam",
        })
    return issues


def detect_ghost_students(dataset):
    issues = []
    valid_students = {student.get("id") for student in dataset.get("students", [])}
    for result in dataset.get("results", []):
        student_id = result.get("estudiante_id")
        if student_id in valid_students:
            continue
        issues.append({
            "type": "ghost_student",
            "related_entities": {
                "result_id": result.get("id"),
                "student_id": student_id,
                "exam_id": result.get("examen_id"),
            },
            "field": "estudiante_id",
            "current_value": student_id,
            "expected_constraint": "student_id must exist",
            "suggested_repairs": [{"action": "delete_record", "cost": 5}],
            "entity_type": "result",
        })
    return issues


def detect_ghost_exams(dataset):
    issues = []
    valid_exams = {exam.get("id") for exam in dataset.get("exams", [])}
    for result in dataset.get("results", []):
        exam_id = result.get("examen_id")
        if exam_id in valid_exams:
            continue
        issues.append({
            "type": "ghost_exam",
            "related_entities": {
                "result_id": result.get("id"),
                "student_id": result.get("estudiante_id"),
                "exam_id": exam_id,
            },
            "field": "examen_id",
            "current_value": exam_id,
            "expected_constraint": "exam_id must exist",
            "suggested_repairs": [{"action": "delete_record", "cost": 5}],
            "entity_type": "result",
        })
    return issues


def detect_duplicate_results(dataset):
    issues = []
    seen = {}
    for result in dataset.get("results", []):
        key = (result.get("estudiante_id"), result.get("examen_id"))
        if key not in seen:
            seen[key] = result
            continue

        issues.append({
            "type": "duplicate_result",
            "related_entities": {
                "result_id": result.get("id"),
                "student_id": result.get("estudiante_id"),
                "exam_id": result.get("examen_id"),
            },
            "field": "id",
            "current_value": result.get("id"),
            "expected_constraint": "(estudiante_id, examen_id) pair must be unique",
            "suggested_repairs": [{
                "action": "delete_record",
                "cost": 3,
            }],
            "entity_type": "result",
        })
    return issues


def detect_orphan_regrades(dataset):
    """Detect regrades that have no matching base result for the same (student, exam)."""
    issues = []
    result_pairs = {
        (r.get("estudiante_id"), r.get("examen_id"))
        for r in dataset.get("results", [])
    }
    for regrade in dataset.get("regrades", []):
        sid = regrade.get("estudiante_id")
        eid = regrade.get("examen_id")
        if (sid, eid) in result_pairs:
            continue
        issues.append({
            "type": "orphan_regrade",
            "related_entities": {
                "student_id": sid,
                "exam_id":    eid,
            },
            "field": "estudiante_id",
            "current_value": f"({sid}, {eid})",
            "expected_constraint": "regrade must have a matching base result",
            "suggested_repairs": [{"action": "delete_record", "cost": 4}],
            "entity_type": "regrade",
        })
    return issues


def detect_all_structural(dataset):
    detectors = [
        detect_invalid_grades,
        detect_age_birth_mismatch,
        detect_course_age_mismatch,
        detect_credits_mismatch,
        detect_invalid_professor_subject,
        detect_future_exam_dates,
        detect_ghost_students,
        detect_ghost_exams,
        detect_duplicate_results,
        detect_orphan_regrades,
    ]

    raw_issues = []
    for detector in detectors:
        raw_issues.extend(detector(dataset))

    inconsistencies = []
    for idx, raw in enumerate(raw_issues):
        inconsistencies.append(_new_issue(
            idx=idx,
            issue_type=raw["type"],
            related_entities=raw["related_entities"],
            field=raw["field"],
            current_value=raw["current_value"],
            expected_constraint=raw["expected_constraint"],
            suggested_repairs=raw["suggested_repairs"],
            entity_type=raw.get("entity_type"),
        ))
    return inconsistencies


def main():
    parser = argparse.ArgumentParser(description="Detect structural inconsistencies in data/dataset.json")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_PATH), help="Path to dataset JSON (default: data/dataset.json)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Path to output inconsistencies JSON (default: data/inconsistencies.json)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    inconsistencies = detect_all_structural(dataset)
    by_type = Counter(inc["type"] for inc in inconsistencies)

    output = {
        "dataset_path": str(dataset_path),
        "total_inconsistencies": len(inconsistencies),
        "summary_by_type": dict(sorted(by_type.items())),
        "inconsistencies": inconsistencies,
    }

    injected = dataset.get("metadata", {}).get("injected_inconsistencies")
    if isinstance(injected, dict):
        output["injected_ground_truth"] = injected

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=4, ensure_ascii=False)

    print("Dataset cargado correctamente ✔")
    print(f"Inconsistencias detectadas: {len(inconsistencies)}")
    for issue_type, count in sorted(by_type.items()):
        print(f"  - {issue_type}: {count}")
    print(f"Reporte guardado en: {output_path}")


if __name__ == "__main__":
    main()
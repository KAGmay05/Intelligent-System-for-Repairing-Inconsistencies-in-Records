import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

dataset_path = BASE_DIR / "data" / "dataset.json"

with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print("Dataset cargado correctamente ✔")
inconsistencies = []    

inc_counter = 0

def next_inc_id():
    global inc_counter
    inc_id = f"INC_{inc_counter}"
    inc_counter += 1
    return inc_id

def invalid_grades(dataset):
    for result in dataset["results"]:
        grade = result["nota"]
        if grade < 0 or grade > 20:
            inconsistency = {
                "id": next_inc_id(),
                "type": "Invalid Grade",
                "related_entities": {
                    "student_id": result["estudiante_id"],
                    "exam_id": result["examen_id"]
                },
                "field": "nota",

                "current_value": grade,

                "expected_constraint": "0 <= nota <= 20",

                "suggested_repairs": [
                    {
                        "action": "modify",
                        "field": "nota",

                        # reparar automáticamente
                        "new_value": max(0, min(20, grade)),

                        "cost": 1
                    }
                ]
            }
            inconsistencies.append(inconsistency)

def age_birth_mismatch(dataset):

    CURRENT_YEAR = 2026

    for student in dataset["students"]:

        birth_year = datetime.strptime(
            student["fecha_nacimiento"],
            "%Y-%m-%d"
        ).year

        real_age = CURRENT_YEAR - birth_year

        stored_age = student["edad"]

        if abs(real_age - stored_age) > 1:

            inconsistency = {
                "id": next_inc_id(),

                "type": "age_birth_mismatch",

                "entity_type": "student",

                "related_entities": {
                    "student_id": student["id"]
                },

                "field": "edad",

                "current_value": stored_age,

                "expected_constraint":
                    f"edad should be approximately {real_age}",

                "suggested_repairs": [
                    {
                        "action": "modify",
                        "field": "edad",
                        "new_value": real_age,
                        "cost": 1
                    }
                ]
            }

            inconsistencies.append(inconsistency)

def invalid_professor_subject(dataset):

    professors = {
        p["id"]: p
        for p in dataset["professors"]
    }

    for exam in dataset["exams"]:

        professor = professors[exam["profesor_id"]]

        subject = exam["asignatura"]

        if subject not in professor["asignaturas"]:

            inconsistency = {
                "id": next_inc_id(),

                "type": "invalid_professor_subject",

                "severity": "high",

                "entity_type": "exam",

                "related_entities": {
                    "exam_id": exam["id"],
                    "professor_id": professor["id"]
                },

                "field": "asignatura",

                "current_value": subject,

                "expected_constraint":
                    "professor must teach subject",

                "suggested_repairs": [
                    {
                        "action": "modify_subject",

                        # sugerimos la primera asignatura válida
                        "new_value": professor["asignaturas"][0],

                        "cost": 1
                    }
                ]
            }

            inconsistencies.append(inconsistency)            

def detect_ghost_students(dataset):

    # conjunto de ids válidos
    valid_students = {
        student["id"]
        for student in dataset["students"]
    }

    for result in dataset["results"]:

        student_id = result["estudiante_id"]

        if student_id not in valid_students:

            inconsistency = {
                "id": next_inc_id(),

                "type": "ghost_student",


                "entity_type": "result",

                "related_entities": {
                    "student_id": student_id,
                    "exam_id": result["examen_id"]
                },

                "field": "estudiante_id",

                "current_value": student_id,

                "expected_constraint": "student_id must exist",

                "suggested_repairs": [
                    {
                        "action": "delete_record",
                        "cost": 5
                    }
                ]
            }

            inconsistencies.append(inconsistency)

def detect_ghost_exams(dataset):

    valid_exams = {
        exam["id"]
        for exam in dataset["exams"]
    }

    for result in dataset["results"]:

        exam_id = result["examen_id"]

        if exam_id not in valid_exams:

            inconsistency = {
                "id": next_inc_id(),

                "type": "ghost_exam",


                "entity_type": "result",

                "related_entities": {
                    "student_id": result["estudiante_id"],
                    "exam_id": exam_id
                },

                "field": "examen_id",

                "current_value": exam_id,

                "expected_constraint": "exam_id must exist",

                "suggested_repairs": [
                    {
                        "action": "delete_record",
                        "cost": 5
                    }
                ]
            }

            inconsistencies.append(inconsistency)            

invalid_grades(dataset)
age_birth_mismatch(dataset)
invalid_professor_subject(dataset)

output = {
    "total_inconsistencies": len(inconsistencies),
    "inconsistencies": inconsistencies
}

with open("inconsistencies.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4)

print(f"{len(inconsistencies)} inconsistencias detectadas ✔")
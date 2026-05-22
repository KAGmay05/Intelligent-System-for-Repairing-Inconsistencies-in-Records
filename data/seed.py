import random
import json
from datetime import datetime, timedelta

# -------------------------
# CONFIGURACIÓN
# -------------------------
NUM_ESTUDIANTES = 10
NUM_PROFESORES = 3
NUM_EXAMENES = 15
NUM_RESULTADOS = 30

ASIGNATURAS = ["math", "physics", "history", "biology"]
CURSOS = ["primary", "secondary", "highschool"]

# Probabilidades para inconsistencias
P_INCONSISTENCIA_NOTA = 0.25
P_INCONSISTENCIA_EDAD = 0.12
P_INCONSISTENCIA_PROF_EXAMEN = 0.12
P_INCONSISTENCIA_RECALIFICACION = 0.2
P_ID_FANTASMA = 0.08

# -------------------------
# UTILIDADES
# -------------------------
def random_date(start_year=2000, end_year=2010):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)

# -------------------------
# GENERAR ESTUDIANTES
# -------------------------
def generate_students():
    students = []
    for i in range(NUM_ESTUDIANTES):
        birth = random_date(2005, 2012)
        age = 2026 - birth.year

        # Inconsistencia: edad que no concuerda con la fecha de nacimiento
        if random.random() < P_INCONSISTENCIA_EDAD:
            age += random.choice([-3, -2, 2, 3])

        students.append({
            "id": f"E{i}",
            "nombre": f"Student_{i}",
            "edad": age,
            "fecha_nacimiento": birth.strftime("%Y-%m-%d"),
            "curso": random.choice(CURSOS),
            "asistencia": random.randint(60, 100),
            "creditos": random.randint(10, 40)
        })
    return students

# -------------------------
# GENERAR PROFESORES
# -------------------------
def generate_professors():
    professors = []
    for i in range(NUM_PROFESORES):
        professors.append({
            "id": f"P{i}",
            "nombre": f"Professor_{i}",
            "edad": random.randint(30, 60),
            "asignaturas": random.sample(ASIGNATURAS, k=2)
        })
    return professors

# -------------------------
# GENERAR EXÁMENES
# -------------------------
def generate_exams(professors):
    exams = []
    for i in range(NUM_EXAMENES):
        prof = random.choice(professors)
        subject = random.choice(prof["asignaturas"])

        # Inconsistencia: asignatura no impartida por el profesor
        if random.random() < P_INCONSISTENCIA_PROF_EXAMEN:
            other_subjects = [s for s in ASIGNATURAS if s not in prof["asignaturas"]]
            if other_subjects:
                subject = random.choice(other_subjects)

        exams.append({
            "id": f"EX{i}",
            "asignatura": subject,
            "profesor_id": prof["id"]
        })
    return exams

# -------------------------
# RESULTADOS DE EXAMEN
# -------------------------
def generate_results(students, exams):
    results = []

    for _ in range(NUM_RESULTADOS):
        exam = random.choice(exams)
        student = random.choice(students)

        correct_grade = random.randint(10, 20)

        # Inconsistencia: nota fuera de rango o muy baja
        if random.random() < P_INCONSISTENCIA_NOTA:
            grade = random.choice([random.randint(0, 9), random.randint(21, 25)])
        else:
            grade = correct_grade

        # Inconsistencia: referencias a ids inexistentes
        if random.random() < P_ID_FANTASMA:
            student_id = f"E{NUM_ESTUDIANTES + random.randint(1, 5)}"
        else:
            student_id = student["id"]

        if random.random() < P_ID_FANTASMA:
            exam_id = f"EX{NUM_EXAMENES + random.randint(1, 5)}"
        else:
            exam_id = exam["id"]

        results.append({
            "examen_id": exam_id,
            "estudiante_id": student_id,
            "nota": grade
        })

    return results

# -------------------------
# RECALIFICACIONES
# -------------------------
def generate_regrading(results):
    regrades = []

    for r in results:
        if random.random() < 0.3:
            # Normalmente sube la nota, pero a veces se introduce inconsistencia
            if random.random() < P_INCONSISTENCIA_RECALIFICACION:
                new_grade = max(0, r["nota"] - random.randint(1, 5))
            else:
                new_grade = min(20, r["nota"] + random.randint(1, 5))

            regrades.append({
                "examen_id": r["examen_id"],
                "estudiante_id": r["estudiante_id"],
                "nota_final": new_grade,
                "profesor_id": f"P{random.randint(0, NUM_PROFESORES-1)}"
            })

    return regrades

# -------------------------
# DATASET COMPLETO
# -------------------------
def build_dataset():
    students = generate_students()
    professors = generate_professors()
    exams = generate_exams(professors)
    results = generate_results(students, exams)
    regrades = generate_regrading(results)

    dataset = {
        "students": students,
        "professors": professors,
        "exams": exams,
        "results": results,
        "regrades": regrades
    }

    return dataset

# -------------------------
# EXPORTAR
# -------------------------
if __name__ == "__main__":
    dataset = build_dataset()

    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

    print("Dataset generado correctamente ✔")
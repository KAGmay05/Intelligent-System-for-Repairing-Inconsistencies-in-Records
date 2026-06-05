import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# Output always goes to the project root, regardless of where the script is run from
ROOT_DIR    = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT_DIR / "dataset.json"

# =============================================================
# SCALE
# =============================================================
NUM_STUDENTS   = 50
NUM_PROFESSORS = 6
NUM_EXAMS      = 40
NUM_RESULTS    = 150
NUM_REGRADES   = 45   # max regrades to generate

# =============================================================
# DOMAIN
# =============================================================
SUBJECTS = ["math", "physics", "history", "biology", "chemistry", "literature"]
COURSES  = ["primary", "secondary", "highschool"]

# Expected age range per course level
COURSE_AGE = {
    "primary":    (6,  11),
    "secondary":  (12, 15),
    "highschool": (16, 19),
}

# Expected credit range per course level
COURSE_CREDITS = {
    "primary":    (5,  30),
    "secondary":  (25, 55),
    "highschool": (50, 90),
}

# =============================================================
# GUARANTEED INCONSISTENCY COUNTS
# Each type is injected deterministically so the count is exact.
# =============================================================
N_AGE_MISMATCH     = 5   # stored age differs from birth-year age by > 1
N_COURSE_AGE       = 4   # student age clearly incompatible with course level
N_CREDITS_MISMATCH = 4   # credits clearly outside the allowed range for the course
N_INVALID_GRADE    = 6   # grade outside the valid range [0, 20]
N_GHOST_STUDENT    = 3   # result references a non-existent student ID
N_GHOST_EXAM       = 3   # result references a non-existent exam ID
N_PROF_MISMATCH    = 4   # exam assigned to a professor who does not teach that subject
N_FUTURE_EXAM      = 3   # exam date set in the future (after 2026-06-05)
N_DUPLICATE_RESULT = 3   # same (student_id, exam_id) pair appears more than once
N_TEXT_INCON       = 15  # teacher comment semantically inconsistent with grade/attendance


# =============================================================
# NAME POOLS
# =============================================================
FIRST_NAMES = [
    "Alejandro", "María",     "Carlos",   "Lucía",     "Javier",  "Ana",
    "Miguel",    "Sara",      "David",    "Laura",     "Andrés",  "Sofía",
    "Pablo",     "Valentina", "Diego",    "Camila",    "Jorge",   "Isabella",
    "Fernando",  "Daniela",   "Sergio",   "Paula",     "Roberto", "Natalia",
    "Tomás",     "Mariana",   "Ricardo",  "Elena",     "Gustavo", "Claudia",
    "Héctor",    "Patricia",  "Emilio",   "Rosa",      "Alberto", "Carmen",
    "Ramón",     "Gloria",    "Felipe",   "Beatriz",   "Ignacio", "Verónica",
    "Marcos",    "Pilar",     "Gonzalo",  "Marta",     "Rafael",  "Silvia",
    "Ernesto",   "Raquel",
]

LAST_NAMES = [
    "García",    "Rodríguez", "Martínez", "López",    "González", "Pérez",
    "Sánchez",   "Ramírez",   "Torres",   "Flores",   "Rivera",   "Gómez",
    "Díaz",      "Reyes",     "Morales",  "Jiménez",  "Cruz",     "Herrera",
    "Medina",    "Aguilar",   "Vargas",   "Castillo", "Ramos",    "Ruiz",
    "Mendoza",   "Ortega",    "Delgado",  "Castro",   "Guerrero", "Vega",
]

PROF_FIRST = ["Elena", "Carlos", "Miguel", "Patricia", "Javier", "Beatriz"]
PROF_LAST  = ["Navarro", "Iglesias", "Fernández", "Molina", "Serrano", "Blanco"]

BIOS = [
    "Interés marcado en ciencias exactas y resolución lógica de problemas.",
    "Participa activamente en debates y actividades culturales del centro.",
    "Se destaca en trabajos colaborativos y proyectos de investigación grupal.",
    "Muestra un enfoque práctico y autodidacta en el aprendizaje.",
    "Habilidades destacadas en expresión oral y escritura académica.",
    "Rendimiento regular con áreas de mejora en materias cuantitativas.",
    "Asistencia constante y actitud positiva ante los retos académicos.",
    "Dificultades en la organización del tiempo y entrega de trabajos.",
    "Talento natural para el análisis crítico y la argumentación.",
    "Participa en competiciones académicas extracurriculares con buenos resultados.",
]

# =============================================================
# COMMENT POOLS  (used in teacher reports)
# =============================================================
COMMENTS_POSITIVE = [
    "Rendimiento excelente; supera todos los objetivos del curso.",
    "Comprensión profunda de los contenidos y participación muy destacada.",
    "Entrega puntual de tareas con calidad muy por encima de la media.",
    "Demuestra dominio completo de los conceptos evaluados.",
    "Progresión notable durante el trimestre; resultados muy satisfactorios.",
]

COMMENTS_NEGATIVE = [
    "No alcanza los objetivos mínimos establecidos para el curso.",
    "Rendimiento claramente insuficiente; necesita apoyo urgente.",
    "No entrega las tareas o las presenta con errores fundamentales.",
    "Dificultades graves para seguir el ritmo de la clase.",
    "Resultado muy por debajo de lo esperado para este nivel.",
]

COMMENTS_ATTENDANCE_HIGH = [
    "Asistencia perfecta y puntualidad ejemplar durante todo el periodo.",
    "Presencia constante en clase; nunca ha faltado sin justificación.",
]

COMMENTS_ATTENDANCE_LOW = [
    "Ausencias frecuentes y reiteradas a lo largo del trimestre.",
    "Asistencia muy baja que compromete el seguimiento del temario.",
]

# =============================================================
# UTILITIES
# =============================================================

def random_date(start_year: int, end_year: int) -> datetime:
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year,  12, 31)
    return start + timedelta(days=random.randint(0, (end - start).days))


def inject_typo(text: str) -> str:
    """Swap two adjacent characters at a random position."""
    if len(text) < 4:
        return text
    i = random.randint(0, len(text) - 2)
    return text[:i] + text[i + 1] + text[i] + text[i + 2:]


def unique_name(used: set, first_pool: list, last_pool: list) -> str:
    for _ in range(300):
        name = f"{random.choice(first_pool)} {random.choice(last_pool)}"
        if name not in used:
            used.add(name)
            return name
    # Fallback: append index to guarantee uniqueness
    name = f"{first_pool[len(used) % len(first_pool)]} {last_pool[len(used) % len(last_pool)]} {len(used)}"
    used.add(name)
    return name

# =============================================================
# GENERATORS
# =============================================================

def generate_students() -> list:
    students   = []
    used_names: set = set()

    # --- Base: fully consistent students ---
    for i in range(NUM_STUDENTS):
        course        = random.choice(COURSES)
        lo, hi        = COURSE_AGE[course]
        birth_year    = random.randint(2026 - hi, 2026 - lo)
        birth         = random_date(birth_year, birth_year)
        age           = 2026 - birth.year
        c_lo, c_hi    = COURSE_CREDITS[course]

        students.append({
            "id":               f"E{i}",
            "nombre":           unique_name(used_names, FIRST_NAMES, LAST_NAMES),
            "edad":             age,
            "fecha_nacimiento": birth.strftime("%Y-%m-%d"),
            "curso":            course,
            "asistencia":       random.randint(60, 100),
            "creditos":         random.randint(c_lo, c_hi),
            "biografia":        random.choice(BIOS),
        })

    # --- Inject: age-birth mismatches ---
    # stored 'edad' differs from (2026 - birth_year) by 3-5 years
    age_set = set(random.sample(range(NUM_STUDENTS), N_AGE_MISMATCH))
    for i in age_set:
        students[i]["edad"] += random.choice([-5, -4, -3, 3, 4, 5])

    # --- Inject: course-age mismatches ---
    # assign a course clearly incompatible with the student's age
    remaining = [i for i in range(NUM_STUDENTS) if i not in age_set]
    course_set = set(random.sample(remaining, N_COURSE_AGE))
    for i in course_set:
        age = students[i]["edad"]
        if age <= 11:
            students[i]["curso"] = "highschool"   # child in highschool
        else:
            students[i]["curso"] = "primary"      # teenager/adult in primary

    # --- Inject: credits mismatches ---
    # credits clearly outside the valid range for the student's course
    used_so_far = age_set | course_set
    remaining2  = [i for i in range(NUM_STUDENTS) if i not in used_so_far]
    cred_set    = set(random.sample(remaining2, N_CREDITS_MISMATCH))
    for i in cred_set:
        course      = students[i]["curso"]
        c_lo, c_hi  = COURSE_CREDITS[course]
        if random.random() < 0.5:
            students[i]["creditos"] = c_hi + random.randint(30, 60)   # too many
        else:
            students[i]["creditos"] = max(0, c_lo - random.randint(10, 25))  # too few

    return students


def generate_professors() -> list:
    professors = []
    used_names: set = set()

    for i in range(NUM_PROFESSORS):
        k = random.randint(2, 3)
        professors.append({
            "id":          f"P{i}",
            "nombre":      unique_name(used_names, PROF_FIRST, PROF_LAST),
            "edad":        random.randint(30, 60),
            "asignaturas": random.sample(SUBJECTS, k=k),
        })

    return professors


def generate_exams(professors: list) -> tuple:
    exams:      list = []
    mismatched: set  = set()

    future_idxs   = set(random.sample(range(NUM_EXAMS), N_FUTURE_EXAM))
    mismatch_idxs = set(random.sample(
        [i for i in range(NUM_EXAMS) if i not in future_idxs],
        N_PROF_MISMATCH,
    ))

    for i in range(NUM_EXAMS):
        prof    = random.choice(professors)
        subject = random.choice(prof["asignaturas"])

        # Inject professor-subject mismatch
        if i in mismatch_idxs:
            others = [s for s in SUBJECTS if s not in prof["asignaturas"]]
            if others:
                subject = random.choice(others)
                mismatched.add(f"EX{i}")

        # Inject future exam date
        date = random_date(2027, 2028) if i in future_idxs else random_date(2021, 2025)

        exams.append({
            "id":          f"EX{i}",
            "asignatura":  subject,
            "profesor_id": prof["id"],
            "fecha":       date.strftime("%Y-%m-%d"),
            "descripcion": f"Examen de {subject}.",
        })

    return exams, mismatched


def generate_results(students: list, exams: list) -> list:
    student_ids = [s["id"] for s in students]
    exam_ids    = [e["id"] for e in exams]
    results: list = []

    # --- Base: valid results ---
    for _ in range(NUM_RESULTS):
        results.append({
            "estudiante_id": random.choice(student_ids),
            "examen_id":     random.choice(exam_ids),
            "nota":          random.randint(10, 20),
        })

    # --- Inject: invalid grades (outside [0, 20]) ---
    grade_idxs = random.sample(range(NUM_RESULTS), N_INVALID_GRADE)
    for idx in grade_idxs:
        if random.random() < 0.5:
            results[idx]["nota"] = random.randint(21, 30)   # above maximum
        else:
            results[idx]["nota"] = random.randint(-10, -1)  # negative

    # --- Inject: ghost students (reference non-existent student IDs) ---
    for j in range(N_GHOST_STUDENT):
        results.append({
            "estudiante_id": f"E{NUM_STUDENTS + j + 1}",
            "examen_id":     random.choice(exam_ids),
            "nota":          random.randint(10, 20),
        })

    # --- Inject: ghost exams (reference non-existent exam IDs) ---
    for j in range(N_GHOST_EXAM):
        results.append({
            "estudiante_id": random.choice(student_ids),
            "examen_id":     f"EX{NUM_EXAMS + j + 1}",
            "nota":          random.randint(10, 20),
        })

    # --- Inject: duplicate results (same student+exam pair) ---
    clean_pool  = [r for r in results[:NUM_RESULTS] if 0 <= r["nota"] <= 20]
    dup_sources = random.sample(clean_pool, min(N_DUPLICATE_RESULT, len(clean_pool)))
    for r in dup_sources:
        results.append({
            "estudiante_id": r["estudiante_id"],
            "examen_id":     r["examen_id"],
            "nota":          random.randint(10, 20),  # may differ from original
        })

    # Add sequential result IDs
    for i, r in enumerate(results):
        r["id"] = f"R{i}"

    return results


def generate_regrades(results: list, professors: list, students: list, exams: list) -> list:
    valid_student_ids = {s["id"] for s in students}
    valid_exam_ids    = {e["id"] for e in exams}

    # Only regrade results that reference existing entities
    pool     = [r for r in results
                if r["estudiante_id"] in valid_student_ids
                and r["examen_id"] in valid_exam_ids
                and 0 <= r["nota"] <= 20]
    regrades = []

    for r in random.sample(pool, min(NUM_REGRADES, len(pool))):
        prof = random.choice(professors)
        regrades.append({
            "examen_id":     r["examen_id"],
            "estudiante_id": r["estudiante_id"],
            "nota_final":    min(20, r["nota"] + random.randint(0, 3)),
            "profesor_id":   prof["id"],
        })

    return regrades


def generate_teacher_reports(students: list, results: list) -> list:
    student_map = {s["id"]: s for s in students}

    # Reports are only generated for results with existing students
    valid_results = [r for r in results if r["estudiante_id"] in student_map]

    # Deterministically choose which reports will have textual inconsistencies
    incon_idxs = set(random.sample(
        range(len(valid_results)),
        min(N_TEXT_INCON, len(valid_results)),
    ))

    reports = []
    for i, result in enumerate(valid_results):
        student    = student_map[result["estudiante_id"]]
        grade      = result["nota"]
        attendance = student.get("asistencia", 80)
        age        = student.get("edad", 15)

        if i in incon_idxs:
            # Build list of applicable inconsistency types for this student/result
            # "contradictory" is always valid; the rest depend on actual data values
            options = ["contradictory"]

            if grade >= 16:
                options.append("negative_with_high_grade")
            if grade <= 8:
                options.append("positive_with_low_grade")
            if attendance >= 85:
                options.append("low_attendance_comment")
            if attendance <= 65:
                options.append("high_attendance_comment")
            if age <= 12:
                options.append("doctoral_comment_young_student")

            typ = random.choice(options)

            if typ == "negative_with_high_grade":
                comment = random.choice(COMMENTS_NEGATIVE)
            elif typ == "positive_with_low_grade":
                comment = random.choice(COMMENTS_POSITIVE)
            elif typ == "low_attendance_comment":
                comment = random.choice(COMMENTS_ATTENDANCE_LOW)
            elif typ == "high_attendance_comment":
                comment = random.choice(COMMENTS_ATTENDANCE_HIGH)
            elif typ == "doctoral_comment_young_student":
                comment = (
                    "El alumno presentó un trabajo de investigación doctoral de alto nivel, "
                    "demostrando un dominio muy avanzado de la materia."
                )
            else:  # contradictory
                comment = (
                    f"{random.choice(COMMENTS_POSITIVE)} "
                    f"Sin embargo, {random.choice(COMMENTS_NEGATIVE).lower()}"
                )

        else:
            # Coherent comment based on actual grade
            if not (0 <= grade <= 20):
                # Grade is invalid; use a neutral pending-review comment
                comment = "Nota registrada con error; pendiente de revisión por el equipo docente."
            elif grade >= 15:
                comment = random.choice(COMMENTS_POSITIVE)
            elif grade >= 10:
                comment = random.choice(COMMENTS_POSITIVE + COMMENTS_NEGATIVE)
            else:
                comment = random.choice(COMMENTS_NEGATIVE)

            # Add attendance note for very low attendance
            if attendance < 70:
                comment += " " + random.choice(COMMENTS_ATTENDANCE_LOW)

        reports.append({
            "student_id": result["estudiante_id"],
            "exam_id":    result["examen_id"],
            "comment":    comment,
            "grade":      grade,
            "student_snapshot": {
                "id":         student["id"],
                "nombre":     student["nombre"],
                "edad":       age,
                "asistencia": attendance,
                "curso":      student["curso"],
            },
        })

    return reports

# =============================================================
# BUILD & EXPORT
# =============================================================

def build_dataset(seed: int) -> dict:
    random.seed(seed)

    students   = generate_students()
    professors = generate_professors()
    exams, _mismatched = generate_exams(professors)
    results    = generate_results(students, exams)
    regrades   = generate_regrades(results, professors, students, exams)
    reports    = generate_teacher_reports(students, results)

    return {
        "metadata": {
            "seed":         seed,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "counts": {
                "students":        len(students),
                "professors":      len(professors),
                "exams":           len(exams),
                "results":         len(results),
                "regrades":        len(regrades),
                "teacher_reports": len(reports),
            },
            # Ground truth for evaluation: how many of each type were injected.
            # The detector/repair pipeline should find at least these counts.
            "injected_inconsistencies": {
                "age_birth_mismatch":         N_AGE_MISMATCH,
                "course_age_mismatch":        N_COURSE_AGE,
                "credits_mismatch":           N_CREDITS_MISMATCH,
                "invalid_grade":              N_INVALID_GRADE,
                "ghost_student":              N_GHOST_STUDENT,
                "ghost_exam":                 N_GHOST_EXAM,
                "professor_subject_mismatch": N_PROF_MISMATCH,
                "future_exam_date":           N_FUTURE_EXAM,
                "duplicate_result":           N_DUPLICATE_RESULT,
                "textual_inconsistency":      N_TEXT_INCON,
                "total": (
                    N_AGE_MISMATCH + N_COURSE_AGE + N_CREDITS_MISMATCH +
                    N_INVALID_GRADE + N_GHOST_STUDENT + N_GHOST_EXAM +
                    N_PROF_MISMATCH + N_FUTURE_EXAM + N_DUPLICATE_RESULT +
                    N_TEXT_INCON
                ),
            },
        },
        "students":        students,
        "professors":      professors,
        "exams":           exams,
        "results":         results,
        "regrades":        regrades,
        "teacher_reports": reports,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a synthetic academic dataset with injected inconsistencies.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for full reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_PATH),
        help=f"Output JSON file path (default: {OUTPUT_PATH})",
    )
    args = parser.parse_args()

    dataset = build_dataset(args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    meta = dataset["metadata"]
    print("Dataset generado correctamente ✔")
    print(f"  Seed:      {meta['seed']}")
    print(f"  Registros: {meta['counts']}")
    print("  Inconsistencias inyectadas:")
    for k, v in meta["injected_inconsistencies"].items():
        print(f"    {k:<32} {v}")
    print(f"  Guardado en: {out}")
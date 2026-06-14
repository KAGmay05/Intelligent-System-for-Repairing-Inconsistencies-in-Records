import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# Output always goes to the data folder, regardless of where the script is run from
DATA_DIR    = Path(__file__).resolve().parent
OUTPUT_PATH = DATA_DIR / "dataset.json"

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
N_ORPHAN_REGRADE   = 3   # regrade without a matching base result
N_TEXT_INCON       = 15  # teacher comment semantically inconsistent with grade/attendance
N_WRONG_SUBJECT    = 4   # comment discusses a subject different from the exam's (GLOBAL: report ↔ exam)


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
# PARAPHRASED COMMENT POOLS
# -------------------------------------------------------------
# Same MEANING as the canonical pools above, but worded with vocabulary that is
# deliberately ABSENT from the keyword markers used by the rule-based detector
# (detector/llm_detector.py).  A purely lexical detector cannot catch these, so
# they are the cases where the LLM must do real semantic reasoning.  Each
# injected textual inconsistency records whether it used "canonical" or
# "paraphrased" wording, so evaluation can measure rule-vs-LLM behaviour per
# phrasing class.
# =============================================================
PARA_POSITIVE = [
    "Trabaja con soltura y sus entregas destacan por su calidad.",
    "Asimila los contenidos con facilidad y aporta ideas valiosas en clase.",
    "Va muy por delante de lo exigido y resuelve los ejercicios con holgura.",
    "Maneja el temario con seguridad y muestra un esfuerzo sostenido.",
    "Su evolución a lo largo del curso ha sido magnífica.",
]

PARA_NEGATIVE = [
    "Su desempeño deja bastante que desear y arrastra carencias importantes.",
    "Le cuesta seguir el hilo de la asignatura y los trabajos llegan incompletos.",
    "Todavía está lejos de lo que se espera en este nivel.",
    "Muestra lagunas considerables que conviene atender cuanto antes.",
    "Le falta mucho para alcanzar un nivel aceptable en la materia.",
]

# Describes excellent attendance — injected when the REAL attendance is low.
PARA_ATTENDANCE_HIGH = [
    "Acudió a cada sesión sin excepción y siempre llegó a la hora.",
    "Su constancia en el aula fue total durante todo el periodo.",
    "No registró ni una sola falta en todo el trimestre.",
]

# Describes poor attendance — injected when the REAL attendance is high.
PARA_ATTENDANCE_LOW = [
    "Faltó a clase en numerosas ocasiones, lo que dificultó su seguimiento.",
    "Estuvo ausente buena parte del trimestre.",
    "Sus continuas inasistencias afectaron su evolución.",
]

PARA_DOCTORAL = [
    "Defendió un estudio de nivel posgrado con un manejo experto de la materia.",
    "Presentó un trabajo propio de un programa de máster, con notable rigor académico.",
    "Elaboró una indagación de alto nivel comparable a la de un estudiante de carrera.",
]

# Probability that an injected textual inconsistency uses paraphrased wording
# (the rest use canonical wording that the rule detector can catch).
PARAPHRASE_PROBABILITY = 0.6

# Subject-specific comments.  Used to inject GLOBAL inconsistencies: a comment that
# clearly discusses one subject placed on an exam of a DIFFERENT subject (report ↔
# exam contradiction).  This is invisible both to the keyword rules and to a local
# per-report check that never sees the exam's subject — only an LLM that compares
# the comment against the exam subject can catch it.
SUBJECT_COMMENTS = {
    "math":       "Resuelve ecuaciones y problemas de calculo con gran soltura.",
    "physics":    "Comprende muy bien las leyes del movimiento, las fuerzas y la energia.",
    "history":    "Analiza con rigor los procesos y acontecimientos historicos.",
    "biology":    "Domina los conceptos de celulas, genetica y ecosistemas.",
    "chemistry":  "Maneja con destreza las reacciones quimicas y la tabla periodica.",
    "literature": "Destaca en el analisis de textos literarios y la expresion escrita.",
}

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

def generate_students():
    """Return (students, ground_truth) where ground_truth maps each student-level
    inconsistency type to the exact list of student IDs that carry it.

    Injections are made NON-cascading on purpose so that injected == actual:
      - age-birth mismatch shifts the BIRTH DATE (not the stored age), keeping the
        age inside the course range so it does not also trip course_age_mismatch;
      - course-age mismatch resets credits into the NEW course's range so it does
        not also trip credits_mismatch.
    The three injected sets are disjoint, so each flagged student carries exactly
    one injected inconsistency.
    """
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

    gt = {"age_birth_mismatch": [], "course_age_mismatch": [], "credits_mismatch": []}

    # --- Inject: age-birth mismatches ---
    # Shift the birth YEAR so (2026 - birth_year) differs from the stored edad by
    # 3-5 years.  The stored edad is left untouched (still valid for the course).
    age_set = set(random.sample(range(NUM_STUDENTS), N_AGE_MISMATCH))
    for i in age_set:
        s     = students[i]
        delta = random.choice([-5, -4, -3, 3, 4, 5])
        new_birth_year = 2026 - (s["edad"] + delta)
        old = datetime.strptime(s["fecha_nacimiento"], "%Y-%m-%d")
        try:
            new_birth = old.replace(year=new_birth_year)
        except ValueError:                       # Feb 29 → use 28
            new_birth = old.replace(year=new_birth_year, day=28)
        s["fecha_nacimiento"] = new_birth.strftime("%Y-%m-%d")
        gt["age_birth_mismatch"].append(s["id"])

    # --- Inject: course-age mismatches ---
    # Assign a course clearly incompatible with the student's age, then re-roll
    # credits into the new course's valid range (so only course_age is broken).
    remaining = [i for i in range(NUM_STUDENTS) if i not in age_set]
    course_set = set(random.sample(remaining, N_COURSE_AGE))
    for i in course_set:
        s   = students[i]
        age = s["edad"]
        s["curso"] = "highschool" if age <= 11 else "primary"
        c_lo, c_hi = COURSE_CREDITS[s["curso"]]
        s["creditos"] = random.randint(c_lo, c_hi)
        gt["course_age_mismatch"].append(s["id"])

    # --- Inject: credits mismatches ---
    # credits clearly outside the valid range for the student's (unchanged) course
    used_so_far = age_set | course_set
    remaining2  = [i for i in range(NUM_STUDENTS) if i not in used_so_far]
    cred_set    = set(random.sample(remaining2, N_CREDITS_MISMATCH))
    for i in cred_set:
        s          = students[i]
        c_lo, c_hi = COURSE_CREDITS[s["curso"]]
        if random.random() < 0.5:
            s["creditos"] = c_hi + random.randint(30, 60)            # too many
        else:
            s["creditos"] = max(0, c_lo - random.randint(10, 25))    # too few
        gt["credits_mismatch"].append(s["id"])

    return students, gt


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


def generate_exams(professors: list):
    """Return (exams, ground_truth) with the exact exam IDs carrying each
    exam-level inconsistency.  Base exams use a subject the professor teaches and
    a past date, so the only flagged exams are the injected ones."""
    exams: list = []
    gt = {"professor_subject_mismatch": [], "future_exam_date": []}

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
                gt["professor_subject_mismatch"].append(f"EX{i}")

        # Inject future exam date
        if i in future_idxs:
            date = random_date(2027, 2028)
            gt["future_exam_date"].append(f"EX{i}")
        else:
            date = random_date(2021, 2025)

        exams.append({
            "id":          f"EX{i}",
            "asignatura":  subject,
            "profesor_id": prof["id"],
            "fecha":       date.strftime("%Y-%m-%d"),
            "descripcion": f"Examen de {subject}.",
        })

    return exams, gt


def generate_results(students: list, exams: list):
    """Return (results, ground_truth) with the exact result IDs carrying each
    result-level inconsistency.

    Base (student, exam) pairs are kept UNIQUE so the only duplicate results are
    the injected ones — previously random collisions created real-but-unintended
    duplicates that were unfairly counted as false positives."""
    student_ids = [s["id"] for s in students]
    exam_ids    = [e["id"] for e in exams]
    results: list = []
    gt = {"invalid_grade": [], "ghost_student": [], "ghost_exam": [], "duplicate_result": []}

    # --- Base: valid results with UNIQUE (student, exam) pairs ---
    used_pairs: set = set()
    for _ in range(NUM_RESULTS):
        for _try in range(500):
            sid = random.choice(student_ids)
            eid = random.choice(exam_ids)
            if (sid, eid) not in used_pairs:
                used_pairs.add((sid, eid))
                break
        results.append({"estudiante_id": sid, "examen_id": eid, "nota": random.randint(10, 20)})

    # --- Inject: invalid grades (outside [0, 20]) ---
    grade_idxs = random.sample(range(NUM_RESULTS), N_INVALID_GRADE)
    for idx in grade_idxs:
        if random.random() < 0.5:
            results[idx]["nota"] = random.randint(21, 30)   # above maximum
        else:
            results[idx]["nota"] = random.randint(-10, -1)  # negative

    # --- Inject: ghost students (reference non-existent student IDs) ---
    ghost_student_idxs = []
    for j in range(N_GHOST_STUDENT):
        results.append({
            "estudiante_id": f"E{NUM_STUDENTS + j + 1}",
            "examen_id":     random.choice(exam_ids),
            "nota":          random.randint(10, 20),
        })
        ghost_student_idxs.append(len(results) - 1)

    # --- Inject: ghost exams (reference non-existent exam IDs) ---
    ghost_exam_idxs = []
    for j in range(N_GHOST_EXAM):
        results.append({
            "estudiante_id": random.choice(student_ids),
            "examen_id":     f"EX{NUM_EXAMS + j + 1}",
            "nota":          random.randint(10, 20),
        })
        ghost_exam_idxs.append(len(results) - 1)

    # --- Inject: duplicate results (same student+exam pair) ---
    clean_pool  = [r for r in results[:NUM_RESULTS] if 0 <= r["nota"] <= 20]
    dup_sources = random.sample(clean_pool, min(N_DUPLICATE_RESULT, len(clean_pool)))
    dup_idxs = []
    for r in dup_sources:
        results.append({
            "estudiante_id": r["estudiante_id"],
            "examen_id":     r["examen_id"],
            "nota":          random.randint(10, 20),  # may differ from original
        })
        dup_idxs.append(len(results) - 1)

    # Add sequential result IDs, then record ground truth by ID
    for i, r in enumerate(results):
        r["id"] = f"R{i}"

    gt["invalid_grade"]     = [results[idx]["id"] for idx in grade_idxs]
    gt["ghost_student"]     = [results[idx]["id"] for idx in ghost_student_idxs]
    gt["ghost_exam"]        = [results[idx]["id"] for idx in ghost_exam_idxs]
    gt["duplicate_result"]  = [results[idx]["id"] for idx in dup_idxs]

    return results, gt


def generate_regrades(results: list, professors: list, students: list, exams: list):
    """Return (regrades, ground_truth).  Orphan regrades are identified by the
    "student_id|exam_id" key (regrades have no own ID).  Regular regrades always
    reference an existing base result, so the only orphans are the injected ones."""
    valid_student_ids = {s["id"] for s in students}
    valid_exam_ids    = {e["id"] for e in exams}

    # Only regrade results that reference existing entities
    pool     = [r for r in results
                if r["estudiante_id"] in valid_student_ids
                and r["examen_id"] in valid_exam_ids
                and 0 <= r["nota"] <= 20]
    regrades = []
    gt = {"orphan_regrade": []}

    for r in random.sample(pool, min(NUM_REGRADES, len(pool))):
        prof = random.choice(professors)
        regrades.append({
            "examen_id":     r["examen_id"],
            "estudiante_id": r["estudiante_id"],
            "nota_final":    min(20, r["nota"] + random.randint(0, 3)),
            "profesor_id":   prof["id"],
        })

    # --- Inject: orphan regrades (no matching base result) ---
    result_pairs = {(r["estudiante_id"], r["examen_id"]) for r in results}
    student_ids  = [s["id"] for s in students]
    exam_ids     = [e["id"] for e in exams]
    for _ in range(N_ORPHAN_REGRADE):
        # Keep trying until we find a pair that has no base result
        for _attempt in range(200):
            sid = random.choice(student_ids)
            eid = random.choice(exam_ids)
            if (sid, eid) not in result_pairs:
                regrades.append({
                    "examen_id":     eid,
                    "estudiante_id": sid,
                    "nota_final":    random.randint(10, 20),
                    "profesor_id":   random.choice(professors)["id"],
                })
                result_pairs.add((sid, eid))  # avoid injecting same pair twice
                gt["orphan_regrade"].append(f"{sid}|{eid}")
                break

    return regrades, gt


def _build_inconsistent_comment(typ: str, phrasing: str) -> str:
    """Return a comment of the given inconsistency `typ`.

    phrasing="canonical"   → wording the rule-based detector can match (keywords).
    phrasing="paraphrased" → semantically equivalent wording with vocabulary that
                             is NOT in the detector's markers (only an LLM that
                             reasons about meaning can catch it).
    """
    canonical = phrasing == "canonical"

    if typ == "negative_with_high_grade":
        return random.choice(COMMENTS_NEGATIVE if canonical else PARA_NEGATIVE)
    if typ == "positive_with_low_grade":
        return random.choice(COMMENTS_POSITIVE if canonical else PARA_POSITIVE)
    if typ == "low_attendance_comment":      # real attendance high, comment claims absences
        return random.choice(COMMENTS_ATTENDANCE_LOW if canonical else PARA_ATTENDANCE_LOW)
    if typ == "high_attendance_comment":     # real attendance low, comment claims perfect attendance
        return random.choice(COMMENTS_ATTENDANCE_HIGH if canonical else PARA_ATTENDANCE_HIGH)
    if typ == "doctoral_comment_young_student":
        if canonical:
            return (
                "El alumno presentó un trabajo de investigación doctoral de alto nivel, "
                "demostrando un dominio muy avanzado de la materia."
            )
        return random.choice(PARA_DOCTORAL)
    # contradictory: praise + criticism in the same comment
    if canonical:
        return (
            f"{random.choice(COMMENTS_POSITIVE)} "
            f"Sin embargo, {random.choice(COMMENTS_NEGATIVE).lower()}"
        )
    return (
        f"{random.choice(PARA_POSITIVE)} "
        f"Aun así, {random.choice(PARA_NEGATIVE).lower()}"
    )


def generate_teacher_reports(students: list, results: list, exams: list):
    """Return (reports, textual_ground_truth).

    textual_ground_truth: one entry per injected inconsistent report —
        {report_id, student_id, exam_id, type, phrasing}
    so evaluation can match detections by ID (not just by count) and break the
    rule-vs-LLM comparison down by phrasing class.

    Two families of textual inconsistency are injected:
      - LOCAL  (phrasing canonical/paraphrased): comment contradicts the student's
        own grade / attendance / age.
      - GLOBAL (phrasing "global"): comment discusses a subject different from the
        exam's — a report ↔ exam contradiction only visible by cross-checking
        elements of the history.
    """
    student_map  = {s["id"]: s for s in students}
    exam_subject = {e["id"]: e.get("asignatura") for e in exams}

    # Reports are only generated for results with existing students
    valid_results = [r for r in results if r["estudiante_id"] in student_map]
    n = len(valid_results)

    # Deterministically choose which reports get a LOCAL textual inconsistency
    incon_idxs = set(random.sample(range(n), min(N_TEXT_INCON, n)))

    # GLOBAL (wrong-subject) inconsistencies go on OTHER reports whose exam has a
    # known subject, so injected == actual and the sets stay disjoint.
    wrong_subject_eligible = [
        i for i in range(n)
        if i not in incon_idxs and valid_results[i]["examen_id"] in exam_subject
    ]
    random.shuffle(wrong_subject_eligible)
    wrong_subject_idxs = set(wrong_subject_eligible[:N_WRONG_SUBJECT])

    reports = []
    textual_gt = []
    for i, result in enumerate(valid_results):
        report_id  = f"TR{i}"
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

            typ      = random.choice(options)
            phrasing = "paraphrased" if random.random() < PARAPHRASE_PROBABILITY else "canonical"
            comment  = _build_inconsistent_comment(typ, phrasing)

            textual_gt.append({
                "report_id":  report_id,
                "student_id": result["estudiante_id"],
                "exam_id":    result["examen_id"],
                "type":       typ,
                "phrasing":   phrasing,
            })

        elif i in wrong_subject_idxs:
            # GLOBAL: comment about a subject different from the exam's actual one.
            actual_subject = exam_subject[result["examen_id"]]
            wrong_choices  = [s for s in SUBJECT_COMMENTS if s != actual_subject]
            wrong_subject  = random.choice(wrong_choices)
            comment = SUBJECT_COMMENTS[wrong_subject]
            textual_gt.append({
                "report_id":  report_id,
                "student_id": result["estudiante_id"],
                "exam_id":    result["examen_id"],
                "type":       "comentario_asignatura_incorrecta",
                "phrasing":   "global",
            })

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
            "report_id":  report_id,
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

    return reports, textual_gt

# =============================================================
# BUILD & EXPORT
# =============================================================

def build_dataset(seed: int) -> dict:
    random.seed(seed)

    students,  gt_students = generate_students()
    professors             = generate_professors()
    exams,     gt_exams    = generate_exams(professors)
    results,   gt_results  = generate_results(students, exams)
    regrades,  gt_regrades = generate_regrades(results, professors, students, exams)
    reports,   textual_gt  = generate_teacher_reports(students, results, exams)

    # Structural ground truth BY ID: {type: [entity_id, ...]}.  Because injections
    # are disjoint and non-cascading, this is the exact set of inconsistent
    # entities, enabling instance-level (not just count-based) evaluation.
    structural_gt = {**gt_students, **gt_exams, **gt_results, **gt_regrades}

    # Counts derived from the ID lists so they are always exact.
    counts_by_type = {k: len(v) for k, v in structural_gt.items()}
    counts_by_type["textual_inconsistency"] = len(textual_gt)
    counts_by_type["total"] = sum(counts_by_type.values())

    phrasing_counts = {
        "canonical":   sum(1 for g in textual_gt if g["phrasing"] == "canonical"),
        "paraphrased": sum(1 for g in textual_gt if g["phrasing"] == "paraphrased"),
        "global":      sum(1 for g in textual_gt if g["phrasing"] == "global"),
    }

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
            # Count-based ground truth (kept for quick inspection / legacy tools).
            "injected_inconsistencies": counts_by_type,
            # Instance-level ground truth: exact entity IDs per structural type.
            "structural_ground_truth": structural_gt,
            # Per-instance ground truth for the textual detector: which reports
            # are inconsistent, their type, and whether the wording is canonical
            # (rule-catchable) or paraphrased (needs semantic reasoning).
            "textual_ground_truth":   textual_gt,
            "textual_phrasing_counts": phrasing_counts,
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
    pc = meta["textual_phrasing_counts"]
    print(f"  Redacción textual → canónica: {pc['canonical']}  parafraseada: {pc['paraphrased']}  global(asignatura): {pc['global']}")
    print(f"  Guardado en: {out}")
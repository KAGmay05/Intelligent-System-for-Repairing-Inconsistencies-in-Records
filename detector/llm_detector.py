import json
import os
import urllib.error
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset.json"
OUTPUT_PATH = BASE_DIR / "textual_inconsistencies.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = os.getenv("OLLAMA_MODEL", "neural-chat")
MAX_REPORTS_TO_ANALYZE = int(os.getenv("MAX_REPORTS_TO_ANALYZE", "0"))
DETECTOR_MODE = os.getenv("DETECTOR_MODE", "rule").lower()  # rule | hybrid | llm
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

POSITIVE_MARKERS = [
    "rendimiento excelente",
    "supera todos los objetivos",
    "comprensión profunda",
    "participación muy destacada",
    "entrega puntual",
    "dominio completo",
    "resultados muy satisfactorios",
    "progresión notable",
]

NEGATIVE_MARKERS = [
    "no alcanza los objetivos",
    "rendimiento claramente insuficiente",
    "no entrega las tareas",
    "dificultades graves",
    "resultado muy por debajo",
]

LOW_ATTENDANCE_MARKERS = [
    "ausencias frecuentes",
    "asistencia muy baja",
]

HIGH_ATTENDANCE_MARKERS = [
    "asistencia perfecta",
    "nunca ha faltado",
    "presencia constante",
    "puntualidad ejemplar",
]

ADVANCED_YOUNG_MARKERS = [
    "investigación doctoral",
    "investigación universitaria",
    "dominio muy avanzado",
]


def _contains_any(text, markers):
    lowered = (text or "").lower()
    return any(m in lowered for m in markers)


def _normalize_llm_result(raw, fallback_comment):
    if not isinstance(raw, dict):
        return {
            "es_coherente": True,
            "tipo_inconsistencia": "coherente",
            "comentario_reparado": fallback_comment,
            "source": "llm_fallback",
        }

    coherent = bool(raw.get("es_coherente", True))
    issue_type = str(raw.get("tipo_inconsistencia", "coherente" if coherent else "otra"))
    repaired = raw.get("comentario_reparado")
    if repaired is None:
        repaired = fallback_comment

    return {
        "es_coherente": coherent,
        "tipo_inconsistencia": issue_type,
        "comentario_reparado": repaired,
        "source": "llm",
    }


def rule_based_analyze(student, grade, comment):
    age = student.get("edad")
    attendance = student.get("asistencia")
    txt = comment or ""

    has_positive = _contains_any(txt, POSITIVE_MARKERS)
    has_negative = _contains_any(txt, NEGATIVE_MARKERS)
    has_low_attendance = _contains_any(txt, LOW_ATTENDANCE_MARKERS)
    has_high_attendance = _contains_any(txt, HIGH_ATTENDANCE_MARKERS)
    has_advanced_young = _contains_any(txt, ADVANCED_YOUNG_MARKERS)
    has_contradiction = "sin embargo" in txt.lower() and has_positive and has_negative

    if has_contradiction:
        return {
            "es_coherente": False,
            "tipo_inconsistencia": "comentario_contradictorio",
            "comentario_reparado": "Comentario ajustado para mantener un solo criterio de evaluación coherente.",
            "source": "rules",
        }

    if isinstance(grade, (int, float)):
        if grade >= 16 and has_negative:
            return {
                "es_coherente": False,
                "tipo_inconsistencia": "nota_alta_con_comentario_negativo",
                "comentario_reparado": "Buen rendimiento general, con oportunidades de mejora puntuales.",
                "source": "rules",
            }

        if grade <= 10 and has_positive:
            return {
                "es_coherente": False,
                "tipo_inconsistencia": "nota_baja_con_comentario_positivo",
                "comentario_reparado": "El resultado requiere refuerzo y seguimiento académico adicional.",
                "source": "rules",
            }

    if isinstance(attendance, (int, float)):
        if attendance < 75 and has_high_attendance:
            return {
                "es_coherente": False,
                "tipo_inconsistencia": "asistencia_contradictoria",
                "comentario_reparado": "Presenta ausencias que afectan su continuidad en el aprendizaje.",
                "source": "rules",
            }

        if attendance >= 85 and has_low_attendance:
            return {
                "es_coherente": False,
                "tipo_inconsistencia": "asistencia_contradictoria",
                "comentario_reparado": "Registra buena asistencia y constancia durante el periodo.",
                "source": "rules",
            }

    if isinstance(age, int) and age < 13 and has_advanced_young:
        return {
            "es_coherente": False,
            "tipo_inconsistencia": "edad_comentario_avanzado",
            "comentario_reparado": "Muestra avances acordes a su etapa escolar.",
            "source": "rules",
        }

    return {
        "es_coherente": True,
        "tipo_inconsistencia": "coherente",
        "comentario_reparado": txt,
        "source": "rules",
    }


def llm_analyze(student, grade, comment):
    prompt = f"""Evalua coherencia de comentario academico.
Devuelve SOLO JSON estricto con esquema:
{{
  "es_coherente": true/false,
  "tipo_inconsistencia": "coherente|nota_alta_con_comentario_negativo|nota_baja_con_comentario_positivo|asistencia_contradictoria|edad_comentario_avanzado|comentario_contradictorio|otra",
  "comentario_reparado": "texto"
}}

Datos:
- Edad: {student.get('edad', 'Desconocida')}
- Asistencia: {student.get('asistencia', 'Desconocida')}
- Nota: {grade}
- Comentario: {comment}

Reglas:
1) Si nota >= 16, no debe ser negativo.
2) Si nota <= 10, no debe ser elogioso.
3) Si asistencia < 75, no debe afirmar asistencia perfecta.
4) Si edad < 13, no debe hablar de investigacion universitaria/doctoral.
5) No debe contener contradiccion interna positiva+negativa.
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "top_p": 0.1,
        },
    }

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_SECONDS) as response:
            response_json = json.loads(response.read().decode("utf-8"))
            llm_response = response_json.get("response", "{}")
            parsed = json.loads(llm_response)
            return _normalize_llm_result(parsed, fallback_comment=comment)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return {
            "es_coherente": True,
            "tipo_inconsistencia": "coherente",
            "comentario_reparado": comment,
            "source": "llm_error_fallback",
        }


def analyze_report(student, grade, comment):
    rule_result = rule_based_analyze(student, grade, comment)

    if DETECTOR_MODE == "rule":
        return rule_result

    if DETECTOR_MODE == "llm":
        return llm_analyze(student, grade, comment)

    # hybrid: trust hard-rule inconsistencies and only send coherent cases to LLM
    if not rule_result.get("es_coherente", True):
        return rule_result
    return llm_analyze(student, grade, comment)


def main():
    if not DATASET_PATH.exists():
        print(f"No se encuentra el dataset en {DATASET_PATH}. Ejecuta el seed primero.")
        return

    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    students_map = {s["id"]: s for s in data.get("students", [])}
    results_map = {(r["estudiante_id"], r["examen_id"]): r["nota"] for r in data.get("results", [])}

    reports = data.get("teacher_reports", [])
    if MAX_REPORTS_TO_ANALYZE > 0:
        reports = reports[:MAX_REPORTS_TO_ANALYZE]

    print(f"Analizando {len(reports)} reportes en modo={DETECTOR_MODE} con modelo={MODEL_NAME}...")

    analysis_results = []
    inconsistent_count = 0

    for idx, report in enumerate(reports, start=1):
        student_id = report.get("student_id")
        exam_id = report.get("exam_id")
        comment = report.get("comment", "")

        student = report.get("student_snapshot") or students_map.get(student_id, {})
        grade = report.get("grade")
        if grade is None:
            grade = results_map.get((student_id, exam_id))

        if grade is None:
            continue

        evaluation = analyze_report(student, grade, comment)
        is_coherent = bool(evaluation.get("es_coherente", True))
        if not is_coherent:
            inconsistent_count += 1

        if idx % 25 == 0:
            print(f"  Progreso: {idx}/{len(reports)}")

        analysis_results.append({
            "student_id": student_id,
            "exam_id": exam_id,
            "original_comment": comment,
            "llm_evaluation": evaluation,
        })

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(analysis_results, handle, indent=4, ensure_ascii=False)

    print("Proceso terminado.")
    print(f"  Reportes evaluados: {len(analysis_results)}")
    print(f"  Inconsistencias textuales detectadas: {inconsistent_count}")
    print(f"  Salida: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "dataset.json"
OUTPUT_PATH = DATA_DIR / "textual_inconsistencies.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
# gemma4 is the default: in our experiments it judged coherence accurately
# (neural-chat over-flagged coherent reports).  Override with OLLAMA_MODEL.
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma4")
MAX_REPORTS_TO_ANALYZE = int(os.getenv("MAX_REPORTS_TO_ANALYZE", "0"))
DETECTOR_MODE = os.getenv("DETECTOR_MODE", "llm").lower()   # llm (default) | hybrid | rule
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "180"))  # gemma4 cold start ≈150s
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_BACKOFF_SECONDS = float(os.getenv("LLM_RETRY_BACKOFF_SECONDS", "2.0"))
# Local CPU inference is generation-bound (~3 tok/s on a 7B model), so output
# length dominates latency.  By default we ask for a terse verdict (no free-text
# reasoning) and cap the generated tokens; set LLM_REASONING=1 to capture the
# model's reasoning (richer for analysis, but markedly slower).
LLM_REASONING = os.getenv("LLM_REASONING", "0") == "1"
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "0"))   # 0 → auto (cap chosen below)

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

VALID_INCONSISTENCY_TYPES = {
    "coherente",
    "nota_alta_con_comentario_negativo",
    "nota_baja_con_comentario_positivo",
    "asistencia_contradictoria",
    "edad_comentario_avanzado",
    "comentario_contradictorio",
    "comentario_asignatura_incorrecta",
    "otra",
}


class StudentSnapshot(BaseModel):
    id: str = "unknown"
    nombre: str = ""
    edad: int | None = None
    asistencia: int | None = None
    curso: str = ""


class LLMOutputRaw(BaseModel):
    es_coherente: bool = True
    tipo_inconsistencia: str = "coherente"
    comentario_reparado: str = ""
    razonamiento: str = ""

    @field_validator("tipo_inconsistencia", mode="before")
    @classmethod
    def _normalize_issue_type(cls, value: Any) -> str:
        return str(value or "").strip().lower()

    @field_validator("comentario_reparado", mode="before")
    @classmethod
    def _normalize_repair(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("razonamiento", mode="before")
    @classmethod
    def _normalize_reasoning(cls, value: Any) -> str:
        return str(value or "").strip()


class LLMOutputValidated(BaseModel):
    es_coherente: bool
    tipo_inconsistencia: Literal[
        "coherente",
        "nota_alta_con_comentario_negativo",
        "nota_baja_con_comentario_positivo",
        "asistencia_contradictoria",
        "edad_comentario_avanzado",
        "comentario_contradictorio",
        "comentario_asignatura_incorrecta",
        "otra",
    ]
    comentario_reparado: str = Field(min_length=1)
    razonamiento: str = ""
    source: str = "llm"
    semantic_check: Literal["pass", "corrected", "fallback"] = "pass"
    correction_flags: list[str] = Field(default_factory=list)


class TextualRecord(BaseModel):
    report_id: str = ""
    student_id: str
    exam_id: str
    grade: float | None = None
    original_comment: str = ""
    llm_evaluation: LLMOutputValidated


def _default_repair_for_type(issue_type, original_comment):
    templates = {
        "nota_alta_con_comentario_negativo": "Buen rendimiento general, con oportunidades de mejora puntuales.",
        "nota_baja_con_comentario_positivo": "El resultado requiere refuerzo y seguimiento académico adicional.",
        "asistencia_contradictoria": "El comentario se ajustó para que sea coherente con la asistencia registrada.",
        "edad_comentario_avanzado": "Muestra avances acordes a su etapa escolar.",
        "comentario_contradictorio": "Comentario ajustado para mantener un solo criterio de evaluación coherente.",
        "comentario_asignatura_incorrecta": "Comentario ajustado para referirse a la asignatura correcta del examen.",
        "otra": "Comentario revisado para mantener coherencia entre nota, asistencia y descripción académica.",
    }
    return templates.get(issue_type, original_comment or "Comentario revisado.")


def _finalize_llm_evaluation(comment, llm_result):
    """Validate the LLM's verdict for FORMAT only — never override the verdict itself.

    The LLM is the semantic judge: we keep its `es_coherente` / `tipo_inconsistencia`
    decision intact and only (a) coerce the label into the valid enum, (b) make sure
    an incoherent verdict carries a usable repaired comment, and (c) strip metadata
    noise from the repair.  No rule-based detector is consulted here, so the LLM's
    reasoning is what actually drives detection.
    """
    result = dict(llm_result or {})
    correction_flags: list[str] = []
    issue_type = str(result.get("tipo_inconsistencia", "")).strip().lower()
    is_coherent = bool(result.get("es_coherente", True))
    repaired = str(result.get("comentario_reparado", "") or "").strip()
    reasoning = str(result.get("razonamiento", "") or "").strip()
    original = comment or ""

    if issue_type not in VALID_INCONSISTENCY_TYPES:
        issue_type = "coherente" if is_coherent else "otra"
        correction_flags.append("invalid_issue_type")

    # Keep coherent outputs canonical.
    if is_coherent:
        issue_type = "coherente"
        if not repaired:
            repaired = original
            correction_flags.append("empty_repair_for_coherent")

    # If LLM says incoherent but gives no useful fix, enforce a deterministic repaired comment.
    if not is_coherent:
        if issue_type in ("", "coherente"):
            issue_type = "otra"
            correction_flags.append("incoherent_without_specific_type")

        repaired_lower = repaired.lower()
        has_meta_noise = any(
            token in repaired_lower
            for token in ("nota:", "asistencia corregida", "asistencia reajustada", "nota revisada")
        )
        if not repaired or repaired == original or has_meta_noise:
            repaired = _default_repair_for_type(issue_type, original)
            correction_flags.append("repaired_comment_enforced")

    semantic_check = "corrected" if correction_flags else "pass"
    try:
        validated = LLMOutputValidated(
            es_coherente=is_coherent,
            tipo_inconsistencia=issue_type,
            comentario_reparado=repaired or (original or "Comentario revisado."),
            razonamiento=reasoning,
            source=str(result.get("source", "llm") or "llm"),
            semantic_check=semantic_check,
            correction_flags=correction_flags,
        )
        return validated.model_dump()
    except ValidationError:
        fallback = LLMOutputValidated(
            es_coherente=True,
            tipo_inconsistencia="coherente",
            comentario_reparado=original or "Comentario revisado.",
            razonamiento=reasoning,
            source="llm_fallback",
            semantic_check="fallback",
            correction_flags=correction_flags + ["pydantic_validation_fallback"],
        )
        return fallback.model_dump()


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
            "semantic_check": "fallback",
            "correction_flags": ["non_dict_llm_response"],
        }

    try:
        parsed = LLMOutputRaw.model_validate(raw)
    except ValidationError:
        return {
            "es_coherente": True,
            "tipo_inconsistencia": "coherente",
            "comentario_reparado": fallback_comment,
            "source": "llm_fallback",
            "semantic_check": "fallback",
            "correction_flags": ["pydantic_raw_validation_error"],
        }

    coherent = bool(parsed.es_coherente)
    issue_type = parsed.tipo_inconsistencia or ("coherente" if coherent else "otra")
    repaired = parsed.comentario_reparado or fallback_comment

    return {
        "es_coherente": coherent,
        "tipo_inconsistencia": issue_type,
        "comentario_reparado": repaired,
        "razonamiento": parsed.razonamiento,
        "source": "llm",
        "semantic_check": "pass",
        "correction_flags": [],
    }


def rule_based_analyze(student, grade, comment, subject=None):
    # NB: the keyword rules deliberately ignore `subject` — detecting a
    # comment-vs-exam-subject mismatch requires semantic understanding the rules
    # do not have, so this GLOBAL inconsistency is left for the LLM to catch.
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


def llm_analyze(student, grade, comment, subject=None):
    # Generation is the bottleneck on local CPU inference, so the prompt asks for
    # the shortest output that still does the job: an empty repair when the comment
    # is coherent (the majority case), and reasoning only when explicitly enabled.
    reasoning_field = '  "razonamiento": "1 frase corta",\n' if LLM_REASONING else ""
    reasoning_hint  = 'Piensa en una frase ("razonamiento") y luego decide. ' if LLM_REASONING else ""

    prompt = f"""Eres un revisor academico. Decide si el COMENTARIO de un profesor es
coherente con los datos objetivos del estudiante: no debe contradecir la nota, la
asistencia ni la edad/nivel del alumno.

Significado de las escalas (NO son reglas de inconsistencia, solo el sentido de los numeros):
- nota: 0 a 20 (0-9 flojo, 10-13 aprobado, 14-16 notable, 17-20 sobresaliente).
- asistencia: 0 a 100 (100 = asistio siempre; bajo = muchas faltas).
- edad: en anios.

Datos:
- Edad: {student.get('edad', 'Desconocida')}
- Asistencia: {student.get('asistencia', 'Desconocida')}
- Curso: {student.get('curso', 'Desconocido')}
- Asignatura del examen: {subject or 'Desconocida'}
- Nota: {grade}
- Comentario: "{comment}"

Tipos (definicion, no umbral):
- coherente: el comentario encaja con los datos.
- nota_alta_con_comentario_negativo: nota buena pero comentario negativo.
- nota_baja_con_comentario_positivo: nota mala pero comentario elogioso.
- asistencia_contradictoria: el comentario sobre asistencia contradice la asistencia real.
- edad_comentario_avanzado: se atribuye un nivel impropio para la edad.
- comentario_contradictorio: el comentario elogia y critica a la vez.
- comentario_asignatura_incorrecta: el comentario describe una asignatura DISTINTA a la del examen (p.ej. habla de ecuaciones/calculo en un examen de historia). Un comentario generico sobre rendimiento/esfuerzo, sin mencionar otra materia, NO entra aqui.
- otra: incoherente pero no encaja arriba.

{reasoning_hint}Reglas de salida (IMPORTANTE para ser breve):
- Si es coherente: "comentario_reparado" debe ser "" (vacio).
- Si NO es coherente: reescribe el comentario corregido en 1 frase, sin metadatos como "(Nota: ...)".

Devuelve SOLO este JSON, sin texto extra:
{{
{reasoning_field}  "es_coherente": true/false,
  "tipo_inconsistencia": "coherente|nota_alta_con_comentario_negativo|nota_baja_con_comentario_positivo|asistencia_contradictoria|edad_comentario_avanzado|comentario_contradictorio|comentario_asignatura_incorrecta|otra",
  "comentario_reparado": ""
}}
"""

    # Token cap: enough for a terse verdict (+ a short repair), more when reasoning
    # is on.  Keeps a runaway generation from blowing the per-call budget.
    num_predict = LLM_NUM_PREDICT or (220 if LLM_REASONING else 120)
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "top_p": 0.1,
            "num_predict": num_predict,
        },
    }

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    attempts = max(1, LLM_MAX_RETRIES)
    last_exc = None

    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_SECONDS) as response:
                response_json = json.loads(response.read().decode("utf-8"))
                llm_response = response_json.get("response", "{}")
                parsed = json.loads(llm_response)
                return _normalize_llm_result(parsed, fallback_comment=comment)
        except (urllib.error.URLError, ConnectionError) as exc:
            # ConnectionError covers ConnectionResetError, which Ollama can raise
            # mid-stream (e.g. when the model is briefly unloaded) and which is
            # NOT wrapped in URLError — without this the whole run dies on one blip.
            last_exc = exc
            # Connection errors are usually transient during model startup; retry first.
            if attempt < attempts:
                wait_s = LLM_RETRY_BACKOFF_SECONDS * attempt
                print(f"[warn] Ollama no disponible (intento {attempt}/{attempts}). Reintentando en {wait_s:.1f}s...")
                time.sleep(wait_s)
                continue
            raise RuntimeError(
                f"No se puede conectar con Ollama en {OLLAMA_URL}.\n"
                f"  Asegúrate de que Ollama está activo: 'ollama serve'\n"
                f"  Modelo requerido: {MODEL_NAME} ('ollama pull {MODEL_NAME}')\n"
                f"  Timeout actual: {LLM_TIMEOUT_SECONDS}s  Reintentos: {attempts}\n"
                f"  Para usar solo reglas sin LLM: DETECTOR_MODE=rule python detector/llm_detector.py\n"
                f"  Detalle: {exc}"
            ) from exc
        except (json.JSONDecodeError, TimeoutError) as exc:
            last_exc = exc
            if attempt < attempts:
                wait_s = LLM_RETRY_BACKOFF_SECONDS * attempt
                print(f"[warn] Timeout/formato inesperado de Ollama (intento {attempt}/{attempts}). Reintentando en {wait_s:.1f}s...")
                time.sleep(wait_s)
                continue
            raise RuntimeError(
                f"Ollama respondió con timeout/formato inesperado tras {attempts} intentos.\n"
                f"  Timeout actual: {LLM_TIMEOUT_SECONDS}s (ajusta LLM_TIMEOUT_SECONDS)\n"
                f"  Modelo: {MODEL_NAME} (verifica/carga con 'ollama pull {MODEL_NAME}')\n"
                f"  Detalle: {exc}"
            ) from exc

    # Defensive fallback (should be unreachable because loop returns or raises)
    raise RuntimeError(f"Fallo inesperado en llm_analyze: {last_exc}")


def analyze_report(student, grade, comment, subject=None):
    # rule  : pure keyword baseline (no LLM).
    # llm   : the LLM is the sole semantic judge — its verdict is never overridden.
    # hybrid: the keyword baseline catches the obvious (canonical) cases; everything
    #         it considers coherent is escalated to the LLM, so the LLM ADDS recall
    #         on top of the rules (it can flag what the keywords miss).
    if DETECTOR_MODE == "rule":
        return rule_based_analyze(student, grade, comment, subject)

    if DETECTOR_MODE == "llm":
        llm_result = llm_analyze(student, grade, comment, subject)
        return _finalize_llm_evaluation(comment, llm_result)

    # hybrid
    rule_result = rule_based_analyze(student, grade, comment, subject)
    if not rule_result.get("es_coherente", True):
        return rule_result
    llm_result = llm_analyze(student, grade, comment, subject)
    return _finalize_llm_evaluation(comment, llm_result)


def main():
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Detector textual de inconsistencias")
    _parser.add_argument("--dataset", default=None, help="Override ruta dataset.json")
    _parser.add_argument("--output",  default=None, help="Override ruta textual_inconsistencies.json")
    _parser.add_argument("--timeout", type=int, default=None,
                         help="Timeout por request al LLM en segundos (override de LLM_TIMEOUT_SECONDS)")
    _parser.add_argument("--retries", type=int, default=None,
                         help="Número de reintentos por request LLM (override de LLM_MAX_RETRIES)")
    _parser.add_argument("--eval-subset", type=int, default=int(os.getenv("EVAL_SUBSET", "0")),
                         help="Si >0: analizar solo un subconjunto = TODOS los reportes con "
                              "ground truth textual + N coherentes muestreados. Acelera el run del "
                              "LLM manteniendo recall exacto (la precisión queda estimada sobre la muestra).")
    _parser.add_argument("--subset-seed", type=int, default=42,
                         help="Semilla para muestrear los reportes coherentes del subconjunto")
    _args, _ = _parser.parse_known_args()

    global DATASET_PATH, OUTPUT_PATH, LLM_TIMEOUT_SECONDS, LLM_MAX_RETRIES
    if _args.dataset:
        DATASET_PATH = Path(_args.dataset)
    if _args.output:
        OUTPUT_PATH = Path(_args.output)
    if _args.timeout is not None:
        LLM_TIMEOUT_SECONDS = max(1, int(_args.timeout))
    if _args.retries is not None:
        LLM_MAX_RETRIES = max(1, int(_args.retries))

    if not DATASET_PATH.exists():
        print(f"No se encuentra el dataset en {DATASET_PATH}. Ejecuta el seed primero.")
        return

    with DATASET_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    students_map = {s["id"]: s for s in data.get("students", [])}
    results_map = {(r["estudiante_id"], r["examen_id"]): r["nota"] for r in data.get("results", [])}
    exams_map = {e["id"]: e for e in data.get("exams", [])}   # for comment ↔ subject check

    reports = data.get("teacher_reports", [])

    # Evaluation subset: keep every report that carries a textual ground-truth
    # inconsistency (so recall is exact) plus a sampled set of the rest (so we can
    # still estimate precision/false positives) — used to run the slow local LLM
    # in minutes instead of hours.
    if _args.eval_subset and _args.eval_subset > 0:
        import random as _random
        gt_ids = {g["report_id"] for g in data.get("metadata", {}).get("textual_ground_truth", [])}
        gt_reports    = [r for r in reports if r.get("report_id") in gt_ids]
        other_reports = [r for r in reports if r.get("report_id") not in gt_ids]
        _random.Random(_args.subset_seed).shuffle(other_reports)
        reports = gt_reports + other_reports[:_args.eval_subset]
        print(f"[subset] {len(gt_reports)} con ground truth + {min(_args.eval_subset, len(other_reports))} "
              f"coherentes muestreados = {len(reports)} reportes")
    elif MAX_REPORTS_TO_ANALYZE > 0:
        reports = reports[:MAX_REPORTS_TO_ANALYZE]

    print(
        f"Analizando {len(reports)} reportes en modo={DETECTOR_MODE} con modelo={MODEL_NAME}...\n"
        f"  timeout={LLM_TIMEOUT_SECONDS}s  retries={LLM_MAX_RETRIES}"
    )

    analysis_results = []
    inconsistent_count = 0

    for idx, report in enumerate(reports, start=1):
        report_id = report.get("report_id", "")
        student_id = report.get("student_id")
        exam_id = report.get("exam_id")
        comment = report.get("comment", "")

        student_raw = report.get("student_snapshot") or students_map.get(student_id, {})
        try:
            student = StudentSnapshot.model_validate(student_raw).model_dump()
        except ValidationError:
            student = StudentSnapshot().model_dump()

        grade = report.get("grade")
        if grade is None:
            grade = results_map.get((student_id, exam_id))

        if grade is None:
            continue

        subject = exams_map.get(exam_id, {}).get("asignatura")
        evaluation = analyze_report(student, grade, comment, subject)
        is_coherent = bool(evaluation.get("es_coherente", True))
        if not is_coherent:
            inconsistent_count += 1

        if idx % 25 == 0:
            print(f"  Progreso: {idx}/{len(reports)}")

        try:
            rec = TextualRecord(
                report_id=str(report_id),
                student_id=str(student_id),
                exam_id=str(exam_id),
                grade=grade,
                original_comment=str(comment or ""),
                llm_evaluation=LLMOutputValidated.model_validate(evaluation),
            )
            analysis_results.append(rec.model_dump())
        except ValidationError:
            fallback_eval = LLMOutputValidated(
                es_coherente=True,
                tipo_inconsistencia="coherente",
                comentario_reparado=str(comment or "Comentario revisado."),
                source="llm_fallback",
                semantic_check="fallback",
                correction_flags=["record_validation_fallback"],
            )
            rec = TextualRecord(
                report_id=str(report_id),
                student_id=str(student_id),
                exam_id=str(exam_id),
                grade=grade,
                original_comment=str(comment or ""),
                llm_evaluation=fallback_eval,
            )
            analysis_results.append(rec.model_dump())

    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(analysis_results, handle, indent=4, ensure_ascii=False)

    print("Proceso terminado.")
    print(f"  Reportes evaluados: {len(analysis_results)}")
    print(f"  Inconsistencias textuales detectadas: {inconsistent_count}")
    print(f"  Salida: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

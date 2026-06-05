# Intelligent System for Repairing Inconsistencies in Records

Sistema de IA para detectar y reparar inconsistencias en registros académicos sintéticos.
Combina reglas estructurales deterministas, detección textual basada en reglas (o LLM opcional) y
optimización combinatoria CP-SAT para elegir el conjunto de mínimo coste que cubre todos los issues.

---

## Arquitectura

```
data/seed.py                 ← genera el dataset con inconsistencias inyectadas
detector/structural_rules.py ← detector estructural (9 tipos de reglas)
detector/llm_detector.py     ← detector textual (reglas deterministas + modo LLM opcional)
detector/repair_optimizer.py ← planificador de reparaciones (CP-SAT, OR-Tools)
evaluate.py                  ← métricas precisión / recall / F1 vs ground truth
validate.py                  ← verifica que el dataset reparado tiene 0 issues
run_pipeline.py              ← ejecuta todo el pipeline de una vez
```

---

## Requisitos

- Python 3.10+
- [OR-Tools](https://developers.google.com/optimization) (`pip install ortools`)
- [Ollama](https://ollama.ai) — **solo necesario en modo `llm` o `hybrid`** (opcional)

```bash
pip install -r requirements.txt
```

---

## Pipeline completo (una sola orden)

```bash
python run_pipeline.py
```

Opciones:

| Flag | Descripción |
|------|-------------|
| `--seed N` | Semilla del generador (default: 42) |
| `--skip-textual` | Reutiliza `textual_inconsistencies.json` existente (evita relanzar el detector) |

---

## Pipeline paso a paso

```bash
# 1. Generar dataset sintético (50 estudiantes, 10 tipos de inconsistencia)
python data/seed.py --seed 42

# 2. Detectar inconsistencias estructurales → inconsistencies.json
python detector/structural_rules.py

# 3. Detectar inconsistencias textuales → textual_inconsistencies.json
python detector/llm_detector.py          # modo rule (por defecto, sin Ollama)
DETECTOR_MODE=hybrid python detector/llm_detector.py  # reglas + LLM

# 4. Calcular reparaciones de mínimo coste → repair_plan.json + repaired_dataset.json
python detector/repair_optimizer.py

# 5. Validar que el dataset reparado tiene 0 inconsistencias estructurales
python validate.py

# 6. Evaluar precisión / recall / F1 del sistema de detección
python evaluate.py
```

---

## Variables de entorno (detector textual)

| Variable | Valores | Default |
|----------|---------|---------|
| `DETECTOR_MODE` | `rule` / `hybrid` / `llm` | `rule` |
| `OLLAMA_MODEL` | nombre del modelo Ollama | `neural-chat` |
| `LLM_TIMEOUT_SECONDS` | segundos de timeout por llamada | `30` |
| `MAX_REPORTS_TO_ANALYZE` | límite de reportes (`0` = todos) | `0` |

---

## Tipos de inconsistencia (10)

| # | Tipo | Entidad |
|---|------|---------|
| 1 | `age_birth_mismatch` | Estudiante |
| 2 | `course_age_mismatch` | Estudiante |
| 3 | `credits_mismatch` | Estudiante |
| 4 | `invalid_grade` | Resultado |
| 5 | `ghost_student` | Resultado |
| 6 | `ghost_exam` | Resultado |
| 7 | `professor_subject_mismatch` | Examen |
| 8 | `future_exam_date` | Examen |
| 9 | `duplicate_result` | Resultado |
| 10 | `textual_inconsistency` | Reporte del profesor |

---

## Salidas generadas

| Archivo | Descripción |
|---------|-------------|
| `dataset.json` | Dataset sintético con inconsistencias inyectadas |
| `inconsistencies.json` | Inconsistencias estructurales detectadas |
| `textual_inconsistencies.json` | Inconsistencias textuales detectadas |
| `repair_plan.json` | Plan de reparación con trazabilidad de cada repair |
| `repaired_dataset.json` | Dataset corregido |

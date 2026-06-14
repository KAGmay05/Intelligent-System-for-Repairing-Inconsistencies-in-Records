# Intelligent System for Repairing Inconsistencies in Records

Sistema de IA para detectar y reparar inconsistencias en registros académicos sintéticos.
Combina:
- **reglas estructurales deterministas** para las inconsistencias de datos (edad, créditos, fechas, referencias…),
- un **LLM como juez semántico** (vía Ollama) que evalúa la coherencia de los comentarios textuales
  razonando sobre nota/asistencia/edad, y
- **optimización combinatoria** (CP-SAT exacto, más baselines greedy y recocido simulado) para
  elegir el conjunto de reparaciones de **mínimo coste** que restaura la coherencia global.

El dataset inyecta inconsistencias textuales con redacción **canónica** (detectable por palabras clave)
y **parafraseada** (solo detectable razonando el significado), de modo que se puede medir el valor real
que aporta el LLM frente a un detector de reglas.

---

## Arquitectura

```
data/seed.py                 ← genera el dataset + ground truth por ID (textual canónico/parafraseado)
detector/structural_rules.py ← detector estructural (10 tipos de reglas)
detector/llm_detector.py     ← detector textual: LLM juez semántico (modos rule / hybrid / llm)
detector/repair_optimizer.py ← planificador de reparaciones (CP-SAT exacto + greedy + recocido simulado)
evaluate.py                  ← métricas precisión / recall / F1 / MCC vs ground truth
validate.py                  ← verifica que el dataset reparado tiene 0 issues estructurales
run_pipeline.py              ← ejecuta todo el pipeline de una vez

experiments/compare_configs.py ← compara modos de detección (rule/hybrid/llm) × optimizadores
experiments/run_instances.py   ← corre el pipeline sobre varias semillas
experiments/textual_eval.py    ← evalúa el detector textual POR ID y por redacción (canónica vs parafraseada)
experiments/hard_instances.py  ← banco de instancias difíciles: CP-SAT vs greedy vs SA (coste, gap, tiempo)
experiments/sa_tuning.py       ← estudio: ¿puede el recocido simulado escapar de la trampa de greedy?
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
| `--skip-textual` | Reutiliza `data/textual_inconsistencies.json` existente (evita relanzar el detector) |

---

## Pipeline paso a paso

```bash
# 1. Generar dataset sintético (50 estudiantes, 10 tipos de inconsistencia)
python data/seed.py --seed 42

# 2. Detectar inconsistencias estructurales → data/inconsistencies.json
python detector/structural_rules.py

# 3. Detectar inconsistencias textuales → data/textual_inconsistencies.json
python detector/llm_detector.py                       # modo llm (por defecto, requiere Ollama)
DETECTOR_MODE=rule   python detector/llm_detector.py  # solo reglas (baseline, sin Ollama)
DETECTOR_MODE=hybrid python detector/llm_detector.py  # reglas + LLM (el LLM añade recall)

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
| `DETECTOR_MODE` | `rule` / `hybrid` / `llm` | `llm` |
| `OLLAMA_MODEL` | nombre del modelo Ollama | `neural-chat` |
| `LLM_TIMEOUT_SECONDS` | segundos de timeout por llamada | `30` |
| `MAX_REPORTS_TO_ANALYZE` | límite de reportes (`0` = todos) | `0` |

**Modos de detección textual:**

| Modo | Qué hace | Uso |
|------|----------|-----|
| `rule` | Baseline de palabras clave (sin LLM). Caza la redacción canónica, **ciego** a la parafraseada. | Comparación / sin Ollama |
| `llm` | El **LLM es el único juez semántico**: se le dan los datos y razona la coherencia; su veredicto no se sobreescribe. | Producción |
| `hybrid` | Las reglas cazan lo obvio; lo que dan por coherente se escala al LLM, que **añade recall**. | Compromiso coste/calidad |

---

## Experimentos (comparación de variantes y análisis)

```bash
# Detección textual por ID y por redacción (canónica vs parafraseada) — mide el valor del LLM
python experiments/textual_eval.py --textual data/textual_llm.json  --label llm
python experiments/textual_eval.py --textual data/textual_rule.json --label rule

# Comparar modos de detección × optimizadores sobre el mismo dataset
python experiments/compare_configs.py --seed 42

# Banco de instancias DIFÍCILES para el optimizador: CP-SAT (exacto) vs greedy vs SA
#   → reporta coste, gap de optimalidad (%) y tiempo; familias "aleatoria" y "trampa-greedy"
python experiments/hard_instances.py

# ¿Puede el recocido simulado escapar de la trampa de greedy?  (temperatura vs operador de vecindad)
python experiments/sa_tuning.py
```

> Sobre el dataset real casi cada issue tiene una única reparación posible, así que CP-SAT, greedy y SA
> dan el mismo coste (la comparación no tiene señal). `hard_instances.py` genera instancias donde las
> reparaciones **compiten**, y ahí sí divergen: greedy se aleja del óptimo (hasta ~ln(n)× en la familia
> trampa) y SA cierra parte de la brecha según su configuración. Es la comparación que da contenido al
> análisis de optimizadores.

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
| `data/textual_inconsistencies.json` | Inconsistencias textuales detectadas |
| `repair_plan.json` | Plan de reparación con trazabilidad de cada repair |
| `repaired_dataset.json` | Dataset corregido |

#!/usr/bin/env bash
# =============================================================================
# exps.sh — конфиги запусков всех экспериментов SkipLayer.
#
# Все запуски делаются из директории SkipLayer/ командой
#   bash exps.sh <имя_эксперимента>
# либо `bash exps.sh all`, чтобы прогнать все по очереди.
#
# Требования окружения:
#   COMET_API_KEY  — ключ для CometML-логгера
#   COMET_WS       — workspace в CometML (по умолчанию ниже)
#   EVAL_LIMIT     — размер выборки для метрик (по умолчанию 200)
#
# Все эксперименты — eval-only (training_mode=False).
# Для каждого запуска метрики дублируются в локальные файлы:
#   outputs/<run_name>/metrics.jsonl       — все scalar-метрики
#   outputs/<run_name>/metrics_summary.json — последние значения
#   outputs/<run_name>/params.json          — resolved Hydra-конфиг
#   outputs/<run_name>/html/                — HTML-дашборды
#
# После всех запусков можно агрегировать командой:
#   python scripts/aggregate_results.py outputs --csv outputs/summary.csv
# =============================================================================

set -euo pipefail

COMET_WS="${COMET_WS:-your_workspace}"
EVAL_LIMIT="${EVAL_LIMIT:-200}"

# Общие override'ы, которые применяются ко всем запускам:
#   * локальный JSONL-логгер + CometML;
#   * единый размер выборки для оценочных метрик.
COMMON=(
  "logger=cometml_jsonl"
  "logger.workspace=${COMET_WS}"
  "dataset.generate.limit=${EVAL_LIMIT}"
)

# -----------------------------------------------------------------------------
# 0. Reference / sanity-check: «полная модель» — все слои честно считаются.
#    Достигается через baseline + p=0.0 (никаких пропусков).
#    Полезно как верхняя граница качества и проверка пайплайна.
# -----------------------------------------------------------------------------
exp_full_model() {
  python -m src.scripts.train --config-name=baseline \
    layer_skipper.p=0.0 \
    "logger.run_name=exp00-full-model" \
    "${COMMON[@]}"
}

# -----------------------------------------------------------------------------
# 1. БЕЙЗЛАЙН: random-skip без какой-либо компенсации скрытых состояний.
#    Aligner — identity, KV для пропущенного слоя — копируется из
#    последнего вычисленного (SimpleKVCachePropagate).
# -----------------------------------------------------------------------------

# 1a. Скип во второй половине сети с p=0.5 (дефолт baseline.yaml).
exp_baseline_p05_half() {
  python -m src.scripts.train --config-name=baseline \
    "logger.run_name=exp01a-baseline-p0.5-half" \
    "${COMMON[@]}"
}

# 1b. Более «лёгкий» скип — p=0.3 во второй половине.
exp_baseline_p03_half() {
  python -m src.scripts.train --config-name=baseline \
    layer_skipper.p=0.3 \
    "logger.run_name=exp01b-baseline-p0.3-half" \
    "${COMMON[@]}"
}

# 1c. Агрессивный скип — p=0.7 во второй половине.
exp_baseline_p07_half() {
  python -m src.scripts.train --config-name=baseline \
    layer_skipper.p=0.7 \
    "logger.run_name=exp01c-baseline-p0.7-half" \
    "${COMMON[@]}"
}

# 1d. Скип равномерно по всей сети (включая первые слои).
#     Должен ощутимо просесть — нужен как «нижняя граница».
exp_baseline_p05_all() {
  python -m src.scripts.train --config-name=baseline \
    layer_skipper.p=0.5 \
    "layer_skipper.skip_percentile_ranges=[[0.0, 1.0]]" \
    "logger.run_name=exp01d-baseline-p0.5-all" \
    "${COMMON[@]}"
}

# 1e. Скип только в средних слоях (≈ 25%–75%), p=0.5.
exp_baseline_p05_middle() {
  python -m src.scripts.train --config-name=baseline \
    layer_skipper.p=0.5 \
    "layer_skipper.skip_percentile_ranges=[[0.25, 0.75]]" \
    "logger.run_name=exp01e-baseline-p0.5-middle" \
    "${COMMON[@]}"
}

# -----------------------------------------------------------------------------
# 2. БЕЙЗЛАЙН-ВАРИАНТ: random-skip + KV проектируется через k_proj/v_proj
#    того же слоя (как у EMA-метода), но БЕЗ EMA-компенсации hidden state.
#    Изолирует вклад «честного» KV от вклада EMA-компенсации.
# -----------------------------------------------------------------------------
exp_baseline_p05_projkv() {
  python -m src.scripts.train --config-name=baseline \
    kv_cache_strategy=project_kv \
    "logger.run_name=exp02-baseline-p0.5-projkv" \
    "${COMMON[@]}"
}

# -----------------------------------------------------------------------------
# 3. ОСНОВНОЙ МЕТОД: EMA-компенсация (TODO.md, дефолтные гиперпараметры).
#    P=3, Q=2, skip_ratio=0.3, β=0.9, R=8, target_ratio=0.4, δ=0.03, M=16.
# -----------------------------------------------------------------------------
exp_ema_default() {
  python -m src.scripts.train --config-name=ema_skip \
    "logger.run_name=exp03-ema-default" \
    "${COMMON[@]}"
}

# -----------------------------------------------------------------------------
# 4. EMA-метод — ablation по доле кандидатов на пропуск (skip_ratio).
# -----------------------------------------------------------------------------
exp_ema_ratio_0_2() {
  python -m src.scripts.train --config-name=ema_skip \
    layer_skipper.skip_ratio=0.2 \
    "logger.run_name=exp04a-ema-ratio0.2" \
    "${COMMON[@]}"
}

exp_ema_ratio_0_4() {
  python -m src.scripts.train --config-name=ema_skip \
    layer_skipper.skip_ratio=0.4 \
    layer_skipper.target_ratio=0.5 \
    "logger.run_name=exp04b-ema-ratio0.4" \
    "${COMMON[@]}"
}

exp_ema_ratio_0_5() {
  python -m src.scripts.train --config-name=ema_skip \
    layer_skipper.skip_ratio=0.5 \
    layer_skipper.target_ratio=0.6 \
    layer_skipper.history_window=32 \
    "logger.run_name=exp04c-ema-ratio0.5" \
    "${COMMON[@]}"
}

# -----------------------------------------------------------------------------
# 5. EMA-метод — ablation по β (скорость EMA).
# -----------------------------------------------------------------------------
exp_ema_beta_0_5() {
  python -m src.scripts.train --config-name=ema_skip \
    layer_skipper.beta=0.5 \
    "logger.run_name=exp05a-ema-beta0.5" \
    "${COMMON[@]}"
}

exp_ema_beta_0_99() {
  python -m src.scripts.train --config-name=ema_skip \
    layer_skipper.beta=0.99 \
    "logger.run_name=exp05b-ema-beta0.99" \
    "${COMMON[@]}"
}

# -----------------------------------------------------------------------------
# 6. EMA-метод — ablation по periodic refresh (R).
# -----------------------------------------------------------------------------
exp_ema_refresh_4() {
  python -m src.scripts.train --config-name=ema_skip \
    layer_skipper.refresh_interval=4 \
    "logger.run_name=exp06a-ema-refresh4" \
    "${COMMON[@]}"
}

exp_ema_refresh_32() {
  python -m src.scripts.train --config-name=ema_skip \
    layer_skipper.refresh_interval=32 \
    "logger.run_name=exp06b-ema-refresh32" \
    "${COMMON[@]}"
}

# -----------------------------------------------------------------------------
# 7. EMA-метод — ablation по защищённым слоям (P, Q).
# -----------------------------------------------------------------------------
exp_ema_protected_strict() {
  python -m src.scripts.train --config-name=ema_skip \
    layer_skipper.protected_first=5 \
    layer_skipper.protected_last=4 \
    "logger.run_name=exp07a-ema-protected-strict" \
    "${COMMON[@]}"
}

exp_ema_protected_loose() {
  python -m src.scripts.train --config-name=ema_skip \
    layer_skipper.protected_first=1 \
    layer_skipper.protected_last=1 \
    "logger.run_name=exp07b-ema-protected-loose" \
    "${COMMON[@]}"
}

# -----------------------------------------------------------------------------
# 8. EMA-метод — ablation адаптации p_skip.
# -----------------------------------------------------------------------------
exp_ema_pfix_0_5() {
  python -m src.scripts.train --config-name=ema_skip \
    layer_skipper.p_init=0.5 \
    layer_skipper.p_min=0.5 \
    layer_skipper.p_max=0.5 \
    "logger.run_name=exp08a-ema-pfix0.5" \
    "${COMMON[@]}"
}

exp_ema_p_aggressive() {
  python -m src.scripts.train --config-name=ema_skip \
    layer_skipper.p_init=0.8 \
    layer_skipper.p_max=0.95 \
    layer_skipper.target_ratio=0.7 \
    layer_skipper.adaptation_rate=0.05 \
    "logger.run_name=exp08b-ema-p-aggressive" \
    "${COMMON[@]}"
}

# -----------------------------------------------------------------------------
# 9. EMA-метод без проектируемого KV (используем simple-propagate).
# -----------------------------------------------------------------------------
exp_ema_default_simplekv() {
  python -m src.scripts.train --config-name=ema_skip \
    kv_cache_strategy=default \
    "logger.run_name=exp09-ema-default-simplekv" \
    "${COMMON[@]}"
}

# -----------------------------------------------------------------------------
# 10. EMA-метод — расширенная выборка (×3) для самой стабильной оценки.
# -----------------------------------------------------------------------------
exp_ema_default_xl_eval() {
  python -m src.scripts.train --config-name=ema_skip \
    "dataset.generate.limit=$((EVAL_LIMIT * 3))" \
    "logger.run_name=exp10-ema-default-xl-eval" \
    "${COMMON[@]}"
}

# =============================================================================
# Раннер
# =============================================================================
ALL_EXPS=(
  exp_full_model
  exp_baseline_p05_half
  exp_baseline_p03_half
  exp_baseline_p07_half
  exp_baseline_p05_all
  exp_baseline_p05_middle
  exp_baseline_p05_projkv
  exp_ema_default
  exp_ema_ratio_0_2
  exp_ema_ratio_0_4
  exp_ema_ratio_0_5
  exp_ema_beta_0_5
  exp_ema_beta_0_99
  exp_ema_refresh_4
  exp_ema_refresh_32
  exp_ema_protected_strict
  exp_ema_protected_loose
  exp_ema_pfix_0_5
  exp_ema_p_aggressive
  exp_ema_default_simplekv
  exp_ema_default_xl_eval
)

usage() {
  echo "Usage: bash exps.sh <name|all|list|aggregate>"
  echo
  echo "Доступные эксперименты (override'ы — через env: COMET_WS, EVAL_LIMIT):"
  for name in "${ALL_EXPS[@]}"; do
    echo "  - ${name}"
  done
  echo
  echo "Спец-команды:"
  echo "  - list       — то же, что и без аргументов: показать список."
  echo "  - all        — выполнить все эксперименты по очереди."
  echo "  - aggregate  — собрать сводную таблицу по локальным метрикам."
}

aggregate() {
  python scripts/aggregate_results.py outputs --csv outputs/summary.csv
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi
  local target="$1"
  case "${target}" in
    list)
      usage
      exit 0
      ;;
    aggregate)
      aggregate
      exit 0
      ;;
    all)
      for name in "${ALL_EXPS[@]}"; do
        echo "=== Running ${name} ==="
        "${name}"
      done
      aggregate
      return
      ;;
  esac
  if declare -F "${target}" >/dev/null; then
    "${target}"
  else
    echo "Unknown experiment: ${target}"
    usage
    exit 1
  fi
}

main "$@"

#!/usr/bin/env bash
# =============================================================================
# exps.sh — конфиги запусков всех экспериментов SkipLayer.
#
# Все запуски делаются из директории SkipLayer/ командой
#   bash exps.sh <имя_эксперимента>
# либо `bash exps.sh all`, чтобы прогнать все по очереди.
#
# Матрица 3×3:
#   методы:  full / baseline / ema
#   режимы:  default / large-max-tokens / large-seq-len
#
# Метрики (см. configs/metrics/eval_only.yaml + src/metrics/skip_metrics.py):
#   accuracy/overall и accuracy/skip_{k}           — попадание argmax skip-модели
#                                                    в argmax полной модели;
#   avg_full_token_prob/overall и .../skip_{k}     — средняя вероятность,
#                                                    которую skip-модель ставит
#                                                    на «таргетный» токен полной
#                                                    модели (новая метрика);
#   avg_full_token_logprob/skip_{k}                — тот же сигнал в log-space.
#
# После всех запусков можно агрегировать командой:
#   python scripts/aggregate_results.py outputs --csv outputs/summary.csv
# =============================================================================

set -euo pipefail

# Размер выборки для метрик. Длинные контексты тяжёлые, поэтому отдельный
# дефолт для seq-len-режима (можно перекрыть через env).
EVAL_LIMIT="${EVAL_LIMIT:-50}"
EVAL_LIMIT_LONG="${EVAL_LIMIT_LONG:-20}"

# max_new_tokens для двух точек: default и large.
SKIP_MAX_TOKENS_DEFAULT="${SKIP_MAX_TOKENS_DEFAULT:-64}"
SKIP_MAX_TOKENS_LARGE="${SKIP_MAX_TOKENS_LARGE:-512}"

# Общие override'ы для всех запусков.
COMMON=(
  "logger=cometml_jsonl"
  "logger.workspace=serega-pirat"
)

# -----------------------------------------------------------------------------
# Режимные override'ы — три точки на оси «контекст / длина генерации».
# -----------------------------------------------------------------------------

# Режим 1: default — TriviaQA (короткие промпты), max_new_tokens=64.
mode_default() {
  echo \
    "dataset.generate.limit=${EVAL_LIMIT}" \
    "metrics.generate.1.max_new_tokens=${SKIP_MAX_TOKENS_DEFAULT}"
}

# Режим 2: large-max-tokens — TriviaQA, но SkipMetrics генерит длинные
# последовательности (нагружает компенсацию и стабильность EMA на длинном горизонте).
mode_large_max_tokens() {
  echo \
    "dataset.generate.limit=${EVAL_LIMIT}" \
    "metrics.generate.1.max_new_tokens=${SKIP_MAX_TOKENS_LARGE}"
}

# Режим 3: large-seq-len — переключаемся на Wikitext (длинные тексты-промпты),
# что увеличивает prefill и нагрузку на attention. Длина генерации обычная.
mode_large_seq_len() {
  echo \
    "dataset=eval_long" \
    "dataset.generate.limit=${EVAL_LIMIT_LONG}" \
    "metrics.generate.1.max_new_tokens=${SKIP_MAX_TOKENS_DEFAULT}"
}

# -----------------------------------------------------------------------------
# Метод 1: FULL — все слои честно считаются (baseline с p=0.0).
# -----------------------------------------------------------------------------
exp_full_default() {
  python -m src.scripts.train --config-name=baseline \
    layer_skipper.p=0.0 \
    "logger.run_name=exp-full-default" \
    $(mode_default) \
    "${COMMON[@]}"
}

exp_full_large_max_tokens() {
  python -m src.scripts.train --config-name=baseline \
    layer_skipper.p=0.0 \
    "logger.run_name=exp-full-large-max-tokens" \
    $(mode_large_max_tokens) \
    "${COMMON[@]}"
}

exp_full_large_seq_len() {
  python -m src.scripts.train --config-name=baseline \
    layer_skipper.p=0.0 \
    "logger.run_name=exp-full-large-seq-len" \
    $(mode_large_seq_len) \
    "${COMMON[@]}"
}

# -----------------------------------------------------------------------------
# Метод 2: BASELINE — random-skip без EMA-компенсации (p=0.5, half-net).
#          Aligner = identity, KV = SimpleKVCachePropagate (defaults baseline.yaml).
# -----------------------------------------------------------------------------
exp_baseline_default() {
  python -m src.scripts.train --config-name=baseline \
    "logger.run_name=exp-baseline-default" \
    $(mode_default) \
    "${COMMON[@]}"
}

exp_baseline_large_max_tokens() {
  python -m src.scripts.train --config-name=baseline \
    "logger.run_name=exp-baseline-large-max-tokens" \
    $(mode_large_max_tokens) \
    "${COMMON[@]}"
}

exp_baseline_large_seq_len() {
  python -m src.scripts.train --config-name=baseline \
    "logger.run_name=exp-baseline-large-seq-len" \
    $(mode_large_seq_len) \
    "${COMMON[@]}"
}

# -----------------------------------------------------------------------------
# Метод 2b: BASELINE с уменьшенной вдвое вероятностью пропуска (p=0.25).
#           Цель — выровнять wall-clock с EMA: baseline@p=0.5 «дешевле» EMA
#           на ~10%, но даёт почти в 2 раза меньше accuracy. При p=0.25
#           среднее число пропусков становится сравнимо с EMA (≈ 4-5 слоёв),
#           что даёт честное сравнение «за то же время».
# -----------------------------------------------------------------------------
exp_baseline_p025_default() {
  python -m src.scripts.train --config-name=baseline \
    layer_skipper.p=0.25 \
    "logger.run_name=exp-baseline-p025-default" \
    $(mode_default) \
    "${COMMON[@]}"
}

exp_baseline_p025_large_max_tokens() {
  python -m src.scripts.train --config-name=baseline \
    layer_skipper.p=0.25 \
    "logger.run_name=exp-baseline-p025-large-max-tokens" \
    $(mode_large_max_tokens) \
    "${COMMON[@]}"
}

exp_baseline_p025_large_seq_len() {
  python -m src.scripts.train --config-name=baseline \
    layer_skipper.p=0.25 \
    "logger.run_name=exp-baseline-p025-large-seq-len" \
    $(mode_large_seq_len) \
    "${COMMON[@]}"
}

# -----------------------------------------------------------------------------
# Метод 2c: BASELINE с пропуском по ВСЕМ слоям (skip_percentile_ranges=[[0,1]])
#           с вероятностью p=0.125. Среднее число пропусков на шаг
#           36 * 0.125 = 4.5 — сравнимо с EMA. Отличие от p025: пропуски
#           равномерно распределены по всей сети (включая «важные» нижние
#           и верхние слои), а не только в верхней половине.
# -----------------------------------------------------------------------------
exp_baseline_p0125_all_default() {
  python -m src.scripts.train --config-name=baseline \
    layer_skipper.p=0.125 \
    'layer_skipper.skip_percentile_ranges=[[0.0,1.0]]' \
    "logger.run_name=exp-baseline-p0125-all-default" \
    $(mode_default) \
    "${COMMON[@]}"
}

exp_baseline_p0125_all_large_max_tokens() {
  python -m src.scripts.train --config-name=baseline \
    layer_skipper.p=0.125 \
    'layer_skipper.skip_percentile_ranges=[[0.0,1.0]]' \
    "logger.run_name=exp-baseline-p0125-all-large-max-tokens" \
    $(mode_large_max_tokens) \
    "${COMMON[@]}"
}

exp_baseline_p0125_all_large_seq_len() {
  python -m src.scripts.train --config-name=baseline \
    layer_skipper.p=0.125 \
    'layer_skipper.skip_percentile_ranges=[[0.0,1.0]]' \
    "logger.run_name=exp-baseline-p0125-all-large-seq-len" \
    $(mode_large_seq_len) \
    "${COMMON[@]}"
}

# -----------------------------------------------------------------------------
# Метод 3: EMA — основной метод (EMA-компенсация + simple KV propagate).
#          ema_skip.yaml по умолчанию использует kv_cache_strategy=default
#          (SimpleKVCachePropagate), как и требует постановка TODO.
# -----------------------------------------------------------------------------
exp_ema_default() {
  python -m src.scripts.train --config-name=ema_skip \
    "logger.run_name=exp-ema-default" \
    $(mode_default) \
    "${COMMON[@]}"
}

exp_ema_large_max_tokens() {
  python -m src.scripts.train --config-name=ema_skip \
    "logger.run_name=exp-ema-large-max-tokens" \
    $(mode_large_max_tokens) \
    "${COMMON[@]}"
}

exp_ema_large_seq_len() {
  python -m src.scripts.train --config-name=ema_skip \
    "logger.run_name=exp-ema-large-seq-len" \
    $(mode_large_seq_len) \
    "${COMMON[@]}"
}

# -----------------------------------------------------------------------------
# ABLATION: EMA — доля кандидатов на пропуск (skip_ratio).
#           Запускается в default-режиме (TriviaQA, 64 max_new_tokens).
# -----------------------------------------------------------------------------
exp_ema_ratio_0_2() {
  python -m src.scripts.train --config-name=ema_skip \
    layer_skipper.skip_ratio=0.2 \
    "logger.run_name=exp-ema-ratio0.2" \
    $(mode_default) \
    "${COMMON[@]}"
}

exp_ema_ratio_0_4() {
  python -m src.scripts.train --config-name=ema_skip \
    layer_skipper.skip_ratio=0.4 \
    layer_skipper.target_ratio=0.5 \
    "logger.run_name=exp-ema-ratio0.4" \
    $(mode_default) \
    "${COMMON[@]}"
}

exp_ema_ratio_0_5() {
  python -m src.scripts.train --config-name=ema_skip \
    layer_skipper.skip_ratio=0.5 \
    layer_skipper.target_ratio=0.6 \
    layer_skipper.history_window=32 \
    "logger.run_name=exp-ema-ratio0.5" \
    $(mode_default) \
    "${COMMON[@]}"
}

# -----------------------------------------------------------------------------
# ABLATION: EMA — коэффициент сглаживания β.
# -----------------------------------------------------------------------------
exp_ema_beta_0_5() {
  python -m src.scripts.train --config-name=ema_skip \
    layer_skipper.beta=0.5 \
    "logger.run_name=exp-ema-beta0.5" \
    $(mode_default) \
    "${COMMON[@]}"
}

exp_ema_beta_0_99() {
  python -m src.scripts.train --config-name=ema_skip \
    layer_skipper.beta=0.99 \
    "logger.run_name=exp-ema-beta0.99" \
    $(mode_default) \
    "${COMMON[@]}"
}

# -----------------------------------------------------------------------------
# KV-CACHE ABLATION: baseline и ema с ProjectKVCacheStrategy.
#   - baseline+projkv  — изолирует вклад «честного» KV-проектора без EMA.
#   - ema+projkv       — основной метод, но с проективным KV (альтернатива
#                        simple-propagate, исследуется как ablation в Chapter 3).
# -----------------------------------------------------------------------------
exp_baseline_projkv() {
  python -m src.scripts.train --config-name=baseline \
    kv_cache_strategy=project_kv \
    "logger.run_name=exp-baseline-projkv" \
    $(mode_default) \
    "${COMMON[@]}"
}

exp_ema_projkv() {
  python -m src.scripts.train --config-name=ema_skip \
    kv_cache_strategy=project_kv \
    "logger.run_name=exp-ema-projkv" \
    $(mode_default) \
    "${COMMON[@]}"
}

# =============================================================================
# Trainable-aligner методы. Сравниваются с full/baseline/ema в default-режиме
# (TriviaQA, max_new_tokens=64). Конфигурация скиппера и KV-стратегии та же,
# что у `exp_baseline_default` — отличие только в обучаемом aligner'е вместо
# identity. Бюджет тренировки урезан (1 эпоха × 10k примеров).
#
# Переменные окружения для перекрытия:
#   TRAIN_EPOCHS / TRAIN_LIMIT / TRAIN_EVAL_STEPS / TRAIN_EVAL_MAX_BATCHES.
# =============================================================================
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
TRAIN_LIMIT="${TRAIN_LIMIT:-10000}"
TRAIN_EVAL_STEPS="${TRAIN_EVAL_STEPS:-50000}"
TRAIN_EVAL_MAX_BATCHES="${TRAIN_EVAL_MAX_BATCHES:-500}"

# device_map=cuda:0 обязателен для тренировки: дефолтный auto (accelerate hooks)
# работает только при inference и падает на обратном проходе.
TRAIN_OVERRIDES=(
  "training.num_epochs=${TRAIN_EPOCHS}"
  "+dataset.train.limit=${TRAIN_LIMIT}"
  "training.eval_steps=${TRAIN_EVAL_STEPS}"
  "training.eval_steps_max_batches=${TRAIN_EVAL_MAX_BATCHES}"
  "llm.model_loading.device_map=cuda:0"
  "dataset.generate.limit=${EVAL_LIMIT}"
)

# Общий MLP-aligner для всех пропущенных слоёв (самый простой обучаемый бейзлайн).
exp_mlp_aligner_default() {
  python -m src.scripts.train --config-name=layer_skip \
    "logger.run_name=exp-mlp-aligner-default" \
    "${TRAIN_OVERRIDES[@]}" \
    "${COMMON[@]}"
}

# Per-layer MLP — отдельный aligner на каждый слой; наиболее сильный
# обучаемый бейзлайн против EMA-компенсации.
exp_per_layer_mlp_default() {
  python -m src.scripts.train --config-name=layer_skip \
    aligner=per_layer_mlp \
    "logger.run_name=exp-per-layer-mlp-default" \
    "${TRAIN_OVERRIDES[@]}" \
    "${COMMON[@]}"
}

# Residual-MLP aligner: общий MLP с residual-связкой x + mlp(x).
# Эмпирически проще оптимизируется, т.к. MLP учится приращению Δ_l,
# а не реконструкции всего hidden state.
exp_residual_mlp_aligner_default() {
  python -m src.scripts.train --config-name=layer_skip \
    aligner=residual_mlp \
    "logger.run_name=exp-residual-mlp-aligner-default" \
    "${TRAIN_OVERRIDES[@]}" \
    "${COMMON[@]}"
}

# Residual per-layer MLP — самый «жирный» trainable-вариант:
# свой MLP на каждый слой + residual-связка.
exp_residual_per_layer_mlp_default() {
  python -m src.scripts.train --config-name=layer_skip \
    aligner=residual_per_layer_mlp \
    "logger.run_name=exp-residual-per-layer-mlp-default" \
    "${TRAIN_OVERRIDES[@]}" \
    "${COMMON[@]}"
}

# =============================================================================
# Раннер
# =============================================================================
ALL_EXPS=(
  # === FULL MODEL (reference) ===
  # exp_full_default
  # exp_full_large_max_tokens
  # exp_full_large_seq_len
  # === BASELINE (random-skip без EMA, p=0.5) ===
  # exp_baseline_default
  # exp_baseline_large_max_tokens
  # exp_baseline_large_seq_len
  # === BASELINE с p=0.25 (выровненный по wall-clock с EMA) ===
  exp_baseline_p025_default
  exp_baseline_p025_large_max_tokens
  exp_baseline_p025_large_seq_len
  # === BASELINE с пропуском по всем слоям, p=0.125 ===
  exp_baseline_p0125_all_default
  exp_baseline_p0125_all_large_max_tokens
  exp_baseline_p0125_all_large_seq_len
  # === EMA METHOD (основной, simple KV propagate) ===
  # exp_ema_default
  # exp_ema_large_max_tokens
  # exp_ema_large_seq_len
  # === EMA ablations: skip_ratio ===
  # exp_ema_ratio_0_2
  # exp_ema_ratio_0_4
  # exp_ema_ratio_0_5
  # === EMA ablations: beta ===
  # exp_ema_beta_0_5
  # exp_ema_beta_0_99
  # === KV-cache ablation: ProjectKVCacheStrategy ===
  # exp_baseline_projkv
  # exp_ema_projkv
  # === Trainable aligners (требуют тренировки) ===
  # exp_mlp_aligner_default
  # exp_per_layer_mlp_default
  exp_residual_mlp_aligner_default
  exp_residual_per_layer_mlp_default
)

usage() {
  echo "Usage: bash exps.sh <name|all|list|aggregate>"
  echo
  echo "Доступные эксперименты (override'ы — через env: COMET_WS, EVAL_LIMIT,"
  echo "EVAL_LIMIT_LONG, SKIP_MAX_TOKENS_DEFAULT, SKIP_MAX_TOKENS_LARGE):"
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

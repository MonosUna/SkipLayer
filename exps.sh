#!/usr/bin/env bash
set -euo pipefail

EVAL_LIMIT="${EVAL_LIMIT:-50}"
EVAL_LIMIT_LONG="${EVAL_LIMIT_LONG:-20}"

SKIP_MAX_TOKENS_DEFAULT="${SKIP_MAX_TOKENS_DEFAULT:-64}"
SKIP_MAX_TOKENS_LARGE="${SKIP_MAX_TOKENS_LARGE:-512}"

COMMON=(
  "logger=cometml_jsonl"
  "logger.workspace=serega-pirat"
)

mode_default() {
  echo \
    "dataset.generate.limit=${EVAL_LIMIT}" \
    "metrics.generate.1.max_new_tokens=${SKIP_MAX_TOKENS_DEFAULT}"
}

mode_large_max_tokens() {
  echo \
    "dataset.generate.limit=${EVAL_LIMIT}" \
    "metrics.generate.1.max_new_tokens=${SKIP_MAX_TOKENS_LARGE}"
}

mode_large_seq_len() {
  echo \
    "dataset=eval_long" \
    "dataset.generate.limit=${EVAL_LIMIT_LONG}" \
    "metrics.generate.1.max_new_tokens=${SKIP_MAX_TOKENS_DEFAULT}"
}

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

TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
TRAIN_LIMIT="${TRAIN_LIMIT:-10000}"
TRAIN_EVAL_STEPS="${TRAIN_EVAL_STEPS:-50000}"
TRAIN_EVAL_MAX_BATCHES="${TRAIN_EVAL_MAX_BATCHES:-500}"

TRAIN_OVERRIDES=(
  "training.num_epochs=${TRAIN_EPOCHS}"
  "+dataset.train.limit=${TRAIN_LIMIT}"
  "training.eval_steps=${TRAIN_EVAL_STEPS}"
  "training.eval_steps_max_batches=${TRAIN_EVAL_MAX_BATCHES}"
  "llm.model_loading.device_map=cuda:0"
  "dataset.generate.limit=${EVAL_LIMIT}"
)

exp_mlp_aligner_default() {
  python -m src.scripts.train --config-name=layer_skip \
    "logger.run_name=exp-mlp-aligner-default" \
    "${TRAIN_OVERRIDES[@]}" \
    "${COMMON[@]}"
}

exp_per_layer_mlp_default() {
  python -m src.scripts.train --config-name=layer_skip \
    aligner=per_layer_mlp \
    "logger.run_name=exp-per-layer-mlp-default" \
    "${TRAIN_OVERRIDES[@]}" \
    "${COMMON[@]}"
}

exp_residual_mlp_aligner_default() {
  python -m src.scripts.train --config-name=layer_skip \
    aligner=residual_mlp \
    "logger.run_name=exp-residual-mlp-aligner-default" \
    "${TRAIN_OVERRIDES[@]}" \
    "${COMMON[@]}"
}

exp_residual_per_layer_mlp_default() {
  python -m src.scripts.train --config-name=layer_skip \
    aligner=residual_per_layer_mlp \
    "logger.run_name=exp-residual-per-layer-mlp-default" \
    "${TRAIN_OVERRIDES[@]}" \
    "${COMMON[@]}"
}

ALL_EXPS=(
  exp_full_default
  exp_full_large_max_tokens
  exp_full_large_seq_len
  exp_baseline_default
  exp_baseline_large_max_tokens
  exp_baseline_large_seq_len
  exp_baseline_p025_default
  exp_baseline_p025_large_max_tokens
  exp_baseline_p025_large_seq_len
  exp_baseline_p0125_all_default
  exp_baseline_p0125_all_large_max_tokens
  exp_baseline_p0125_all_large_seq_len
  exp_ema_default
  exp_ema_large_max_tokens
  exp_ema_large_seq_len
  exp_ema_ratio_0_2
  exp_ema_ratio_0_4
  exp_ema_ratio_0_5
  exp_ema_beta_0_5
  exp_ema_beta_0_99
  exp_baseline_projkv
  exp_ema_projkv
  exp_mlp_aligner_default
  exp_per_layer_mlp_default
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

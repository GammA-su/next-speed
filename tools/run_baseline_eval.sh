#!/usr/bin/env bash
set -euo pipefail

: "${K:=4}"
: "${V:=256}"
: "${DECODER_PATH:=out/decoder/checkpoint-58352}"
: "${SENT_LM_PATH:=out/sentence_lm/sentence_lm.pt}"
: "${CODE_MODE:=special}"
: "${MAX_ROWS:=500}"

mkdir -p out/eval_runs

uv run python code_agreement_eval.py \
  MAX_ROWS=$MAX_ROWS K=$K V=$V \
  DECODER_PATH="$DECODER_PATH" SENT_LM_PATH="$SENT_LM_PATH" \
  CODE_MODE=$CODE_MODE RERANK=0 N_CAND=1 DO_SAMPLE=0 TOP_P=1.0

ts=$(date +"%Y%m%d_%H%M%S")
cp -f out/eval/code_agreement.json "out/eval_runs/code_agreement_${ts}.json"
echo "Saved baseline: out/eval_runs/code_agreement_${ts}.json"

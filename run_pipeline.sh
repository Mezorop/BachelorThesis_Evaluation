#!/bin/bash

set -euo pipefail

REPOSITORY_DIRECTORY="$DATA/translationEvaluation"
RESULT_TSV="$DATA/unchecked_model_results/hibiki_zero/output_results.tsv"
VENV_ROOT="$DATA/venvs"

"$VENV_ROOT/whisperx/bin/python" -u \
    "$REPOSITORY_DIRECTORY/whisperx_timing/evaluate.py" \
    "$RESULT_TSV"

"$VENV_ROOT/bleu/bin/python" -u \
    "$REPOSITORY_DIRECTORY/bleu/evaluate.py" \
    "$RESULT_TSV"

"$VENV_ROOT/asr-bleu/bin/python" -u \
    "$REPOSITORY_DIRECTORY/asr_bleu/evaluate.py" \
    "$RESULT_TSV"

"$VENV_ROOT/asr-comet/bin/python" -u \
    "$REPOSITORY_DIRECTORY/asr_comet/evaluate.py" \
    "$RESULT_TSV"

echo "Evaluation pipeline completed: $RESULT_TSV"

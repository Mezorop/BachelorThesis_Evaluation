#!/bin/bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/output_results.tsv" >&2
    exit 2
fi

REPOSITORY_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_TSV="$1"
VENV_ROOT="${EVALUATION_VENV_ROOT:-${DATA:?DATA must be set}/venvs/bachelor-thesis-evaluation}"

"$VENV_ROOT/whisperx/bin/python" -u \
    "$REPOSITORY_DIRECTORY/whisperx_timing/evaluate.py" \
    "$RESULT_TSV" \
    --source-language "${SOURCE_LANGUAGE:-de}" \
    --target-language "${TARGET_LANGUAGE:-en}" \
    --model "${WHISPERX_MODEL:-large-v3}" \
    --device "${EVALUATION_DEVICE:-cuda}" \
    --compute-type "${WHISPERX_COMPUTE_TYPE:-float16}" \
    --batch-size "${WHISPERX_BATCH_SIZE:-16}"

"$VENV_ROOT/bleu/bin/python" -u \
    "$REPOSITORY_DIRECTORY/bleu/evaluate.py" \
    "$RESULT_TSV"

"$VENV_ROOT/asr-bleu/bin/python" -u \
    "$REPOSITORY_DIRECTORY/asr_bleu/evaluate.py" \
    "$RESULT_TSV"

COMET_GPUS=1
if [[ "${EVALUATION_DEVICE:-cuda}" == "cpu" ]]; then
    COMET_GPUS=0
fi

"$VENV_ROOT/asr-comet/bin/python" -u \
    "$REPOSITORY_DIRECTORY/asr_comet/evaluate.py" \
    "$RESULT_TSV" \
    --model "${COMET_MODEL:-Unbabel/wmt22-comet-da}" \
    --batch-size "${COMET_BATCH_SIZE:-16}" \
    --gpus "$COMET_GPUS"

echo "Evaluation pipeline completed: $RESULT_TSV"

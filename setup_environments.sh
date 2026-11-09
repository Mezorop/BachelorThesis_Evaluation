#!/bin/bash

set -euo pipefail

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [venv-root]" >&2
    exit 2
fi

REPOSITORY_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ROOT="${1:-${EVALUATION_VENV_ROOT:-${DATA:?DATA must be set}/venvs/bachelor-thesis-evaluation}}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"

create_environment() {
    local environment_name="$1"
    local requirements_file="$2"
    local environment_path="$VENV_ROOT/$environment_name"

    "$PYTHON_EXECUTABLE" -m venv "$environment_path"
    "$environment_path/bin/python" -m pip install --upgrade pip
    "$environment_path/bin/python" -m pip install -r "$requirements_file"
}

mkdir -p "$VENV_ROOT"
create_environment whisperx "$REPOSITORY_DIRECTORY/whisperx_timing/requirements.txt"
create_environment bleu "$REPOSITORY_DIRECTORY/bleu/requirements.txt"
create_environment asr-bleu "$REPOSITORY_DIRECTORY/asr_bleu/requirements.txt"
create_environment asr-comet "$REPOSITORY_DIRECTORY/asr_comet/requirements.txt"

echo "Created four isolated environments below $VENV_ROOT"

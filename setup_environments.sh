#!/bin/bash

set -euo pipefail

REPOSITORY_DIRECTORY="$DATA/translationEvaluation"
VENV_ROOT="$DATA/venvs"

create_environment() {
    #whisperx || bleu || asr-bleu || asr-comet
    local environment_name="$1"
    #paths to requirement files
    local requirements_file="$2"
    #$DATA/venvs/whisperx etc.
    local environment_path="$VENV_ROOT/$environment_name"

    #Venv creation
    python -m venv "$environment_path"
    "$environment_path/bin/python" -m pip install --upgrade pip
    #Loading the requirements
    "$environment_path/bin/python" -m pip install -r "$requirements_file"
}

mkdir -p "$VENV_ROOT"
#Creates the 4 different environments for the 4 evaluations. (bleu and asr-bleu could actually be put together due to same requirements)
create_environment whisperx "$REPOSITORY_DIRECTORY/whisperx_timing/requirements.txt"
create_environment bleu "$REPOSITORY_DIRECTORY/bleu/requirements.txt"
create_environment asr-bleu "$REPOSITORY_DIRECTORY/asr_bleu/requirements.txt"
create_environment asr-comet "$REPOSITORY_DIRECTORY/asr_comet/requirements.txt"

echo "Created four isolated environments below $VENV_ROOT"

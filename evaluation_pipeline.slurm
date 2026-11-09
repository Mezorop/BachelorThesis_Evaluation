#!/bin/bash
#
#SBATCH -J thesis_evaluation
#SBATCH -t 08:00:00
#SBATCH --partition=zen3_0512_a100x2
#SBATCH --qos=zen3_0512_a100x2
#SBATCH --gres=gpu:1

set -euo pipefail

module purge
module load python/3.12.8-gcc-12.2.0-4y5tbpr
module load ffmpeg/4.4.1-gcc-12.2.0-d54gcg6

export XDG_CACHE_HOME="$DATA/.cache"
export HF_HOME="$DATA/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$DATA/.cache/torch"

CACHE_DIRECTORY="$DATA/.cache/bachelor_thesis_evaluation"
export TMPDIR="$CACHE_DIRECTORY/tmp/$SLURM_JOB_ID"
export TRITON_CACHE_DIR="$CACHE_DIRECTORY/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIRECTORY/torchinductor"
export CUDA_CACHE_PATH="$CACHE_DIRECTORY/cuda"

mkdir -p \
    "$HF_HUB_CACHE" \
    "$TORCH_HOME" \
    "$TMPDIR" \
    "$TRITON_CACHE_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" \
    "$CUDA_CACHE_PATH"

PROJECT_DIRECTORY="${EVALUATION_PROJECT_DIRECTORY:-$DATA/BachelorThesis-Evaluation}"
RESULT_TSV="${EVALUATION_RESULT_TSV:-$DATA/unchecked_model_results/hibiki_zero/output_results.tsv}"

"$PROJECT_DIRECTORY/run_pipeline.sh" "$RESULT_TSV"

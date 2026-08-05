#!/bin/bash
#SBATCH --job-name=yam_pi05_full
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=10:00:00
#SBATCH --output=logs/yam_train_%j.out
#SBATCH --error=logs/yam_train_%j.err

module load cuda

cd /gpfs/home/yuzhi/yuzhi/projects/yam_openpi
source .venv/bin/activate

export LD_LIBRARY_PATH=/gpfs/software/spack/1.0.1/opt/spack/linux-x86_64_v3/ffmpeg-7.1-jwmkhym7u5l3y7h5pgcjlpw446poeccg/lib:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export OPENPI_DATA_HOME=/gpfs/scrubbed/yuzhi/.cache/openpi

# Check if checkpoint exists to decide resume vs fresh start
CKPT_DIR="./checkpoints/pi05_yam/yam_full_v1_0415"
if [ -d "$CKPT_DIR" ] && [ "$(ls -A $CKPT_DIR 2>/dev/null)" ]; then
    echo "Checkpoint found at $CKPT_DIR, resuming training..."
    RESUME_FLAG="--resume"
else
    echo "No checkpoint found, starting fresh training..."
    RESUME_FLAG=""
fi

uv run scripts/train.py pi05_yam \
    --exp-name=yam_full_v1_0415 \
    --fsdp-devices 4 \
    --batch-size 64 \
    --num-train-steps 50000 \
    --save-interval 1000 \
    $RESUME_FLAG

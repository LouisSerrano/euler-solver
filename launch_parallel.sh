#!/bin/bash
# Launch parallel data generation across multiple GPUs using SLURM
#
# Usage:
#   ./launch_parallel.sh <num_gpus>
#   ./launch_parallel.sh 4   # Launch on 4 GPUs
#
# Each GPU will generate its slice of trajectories independently.
# After all jobs complete, run: python merge_h5.py --num_gpus 4

NUM_GPUS=${1:-4}
OUTPUT_DIR="/mnt/home/lserrano/ceph/data/euler_ns_short/"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Launching $NUM_GPUS parallel jobs..."
echo "Output directory: $OUTPUT_DIR"
echo ""

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Submitting job for GPU $GPU_ID..."
    
    sbatch --job-name="datagen_${GPU_ID}" \
           --output="${OUTPUT_DIR}/logs/gpu${GPU_ID}_%j.out" \
           --error="${OUTPUT_DIR}/logs/gpu${GPU_ID}_%j.err" \
           --partition=gpu \
           --gpus=1 \
           --cpus-per-gpu=16 \
           --time=12:00:00 \
           --mem=64G \
           --wrap="cd ${SCRIPT_DIR} && source .venv/bin/activate && python data_generator_parallel.py --gpu_id ${GPU_ID} --num_gpus ${NUM_GPUS} --output_dir ${OUTPUT_DIR}"
done

echo ""
echo "All jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "After completion, merge with: python merge_h5.py --input_dir ${OUTPUT_DIR} --num_gpus ${NUM_GPUS}"

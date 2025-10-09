#!/bin/bash
#BSUB -J video_classification
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -R "rusage[mem=2GB]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Default model parameters
MODEL="2D_CNN_early_fusion"
NUM_EPOCHS=5
BATCH_SIZE=8
IMAGE_SIZE=64
OUTPUT_DIR="./outputs"
SAVE_MODEL=""  # Empty means don't save
ROOT_DIR="/dtu/datasets1/02516/ufc10"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --num_epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --image_size)
      IMAGE_SIZE="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --save_model)
      SAVE_MODEL="--save_model"
      shift
      ;;
    --root_dir)
      ROOT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Activate Python environment (adjust the path to your environment)
# For DTU HPC systems, you might need to load modules instead
# module load python3/3.11.4

# If using a virtual environment
if [ -d "../.venv" ]; then
  source ../.venv/bin/activate
elif [ -d ".venv" ]; then
  source .venv/bin/activate
fi

echo "Running with parameters:"
echo "Model: $MODEL"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Image size: $IMAGE_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo "Save model: ${SAVE_MODEL:-No}"
echo "Dataset path: $ROOT_DIR"

# Run the main.py script with provided arguments
# Assuming main.py is in the current directory (Project_2)
python Project_2/main.py \
  --model $MODEL \
  --num_epochs $NUM_EPOCHS \
  --batch_size $BATCH_SIZE \
  --image_size $IMAGE_SIZE \
  --output_dir $OUTPUT_DIR \
  --root_dir $ROOT_DIR \
  $SAVE_MODEL

# Command to run (example):
# ./run_on_gpu.sh --model 2D_CNN_aggr --num_epochs 10 --save_model
# or submit to queue:
# bsub < run_on_gpu.sh

#!/bin/bash
#BSUB -J train_and_predict
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 5:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Parse arguments with defaults
DATASET=${1:-Ph2}
MODEL=${2:-EncDec}
LOSS=${3:-CrossEntropyLoss}
BATCH_SIZE=${4:-10} # batch size of 4 for Drive (small dataset)
EPOCHS=${5:-50}
SIZE=${6:-128}
LR=${7:-0.001}
POS_WEIGHT=${8:-} # Optional positive weight for WeightedCrossEntropyLoss (good value maybe around 10)

# Load CUDA module only (Python comes from virtual environment)
module load cuda/12.4

# Activate virtual environment (contains Python 3.13)
VENV_PATH="$HOME/Documents/env"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Virtual environment activated"
    echo "Python version: $(python --version)"
else
    echo "Warning: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Print job information
echo "========================================="
echo "Job ID: $LSB_JOBID"
echo "Job Name: $LSB_JOBNAME"
echo "Starting at: $(date)"
echo "Running on host: $(hostname)"
echo "========================================="
echo "Training Configuration:"
echo "  Dataset: $DATASET"
echo "  Model: $MODEL"
echo "  Loss: $LOSS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Image Size: ${SIZE}x${SIZE}"
echo "  Learning Rate: $LR"
if [ ! -z "$POS_WEIGHT" ]; then
    echo "  Positive Weight: $POS_WEIGHT"
fi
echo "========================================="
echo ""

# Change to project directory
cd $HOME/Documents/IntroDeepCompVision_projects/Project_3

# Build training command - with/out optional pos_weight parameter
CMD="python3 train.py --dataset $DATASET --model $MODEL --loss $LOSS --batch_size $BATCH_SIZE --epochs $EPOCHS --size $SIZE --lr $LR"
if [ ! -z "$POS_WEIGHT" ]; then
    CMD="$CMD --pos_weight $POS_WEIGHT"
fi

# Run training
echo "========================================="
echo "PHASE 1: TRAINING"
echo "========================================="
eval $CMD

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "Training failed! Exiting..."
    exit 1
fi

# Determine checkpoint path
CHECKPOINT="model/checkpoints/${DATASET,,}_${MODEL,,}_${LOSS,,}.pth"
OUTPUT_DIR="dataset/predictions/${DATASET,,}_${MODEL,,}_${LOSS,,}"

echo ""
echo "========================================="
echo "PHASE 2: PREDICTION ON TEST SET"
echo "========================================="
echo "  Dataset: $DATASET"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output: $OUTPUT_DIR"
echo "========================================="
echo ""

# Run prediction
python3 predict.py \
    --dataset $DATASET \
    --model $MODEL \
    --checkpoint $CHECKPOINT \
    --size $SIZE \
    --output_dir $OUTPUT_DIR

echo ""
echo "========================================="
echo "Job finished at: $(date)"
echo "========================================="

#!/bin/bash
#BSUB -J train_pothole
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o logs/train_%J.out
#BSUB -e logs/train_%J.err

# Change to project directory
cd $HOME/Documents/IntroDeepCompVision_projects/Project_4

# Create logs directory if it doesn't exist
mkdir -p logs

# Load CUDA module only (Python comes from virtual environment)
module load cuda/12.4

# Activate virtual environment (contains Python 3.13)
VENV_PATH="$HOME/Documents/IntroDeepCompVision_projects/.venv"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Virtual environment activated"
    echo "Python version: $(python --version)"
else
    echo "Warning: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Run training
python train.py --epochs 20 --batch_size 32 --lr 0.001 --output_dir checkpoints

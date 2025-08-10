#!/bin/bash

# Configuration
PYTHON_CMD="python"
SCRIPT="main.py"
BASE_ARGS="--tensorboard --save-checkpoint --epochs 10"

# Hidden features values to sweep
HIDDEN_FEATURES_VALUES=(64 128 256 512 1024)


for hidden_features in "${HIDDEN_FEATURES_VALUES[@]}"; do
    python main.py --hidden-features $hidden_features $BASE_ARGS
done


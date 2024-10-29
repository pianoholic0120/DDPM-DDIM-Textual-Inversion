#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: bash hw2_2.sh <noise_dir> <output_dir> <model_path>"
    exit 1
fi

NOISE_DIR=$1
OUTPUT_DIR=$2
MODEL_PATH=$3

python3 hw2_2_inference.py "$NOISE_DIR" "$OUTPUT_DIR" "$MODEL_PATH"

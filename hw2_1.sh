# !/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: bash hw2_1.sh <output_directory>"
    exit 1
fi

python3 hw2_1_inference.py "$1"
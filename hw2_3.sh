#!/bin/bash

json_file=$1
output_folder=$2
model_weights=$3

python3 hw2_3_inference.py "$json_file" "$output_folder" "$model_weights"
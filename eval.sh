#!/bin/bash

mkdir -p logs

timestamp=$(date +"%Y%m%d_%H%M%S")

log_file="logs/internVL3_$timestamp.log"

accelerate launch src/internVL3.py > "$log_file" 2>&1

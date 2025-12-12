#!/bin/bash
# AMix-specific directed evolution inference script
# This script performs directed evolution using AMix ProfileBFN model configuration
#
# Usage:
#   ./run_de_amix.sh <wt_sequence> <wt_fitness> <amix_ckpt> <decoder_ckpt> <save_name> [n_steps] [population] [devices] [seed]
#
# Example:
#   ./run_de_amix.sh "ACDEFGHIKLMNPQRSTVWY" 1.0 path/to/amix.pt path/to/decoder.pt results/output.csv 10 32 0 42
#
# Arguments:
#   wt_sequence    - Wild-type protein sequence
#   wt_fitness     - Wild-type fitness score
#   amix_ckpt      - Path to AMix model checkpoint
#   decoder_ckpt   - Path to trained decoder checkpoint
#   save_name      - Output CSV file path
#   n_steps        - Number of evolution steps (default: 10)
#   population     - Population size (default: 32)
#   devices        - GPU device ID (default: 0)
#   seed           - Random seed (default: 0)

wt_sequence=$1
wt_fitness=$2
amix_ckpt=$3
decoder_ckpt=$4
save_name=$5
n_steps=${6:-10}
population=${7:-32}
devices=${8:-0}
seed=${9:-0}

# AMix configuration file
amix_config="./configs/amix_config.yaml"

# Evolution parameters
num_proposes_per_var=4
beam_size=5
top_k=5
mask_ratio_low=0.1
mask_ratio_high=0.25
max_candidates_per_variant=200
population_ratio_per_mask="0.5 0.5"  # Equal split between low and high mask ratios

echo "==================================="
echo "AMix Directed Evolution Configuration"
echo "==================================="
echo "Wild-type sequence: $wt_sequence"
echo "Wild-type fitness: $wt_fitness"
echo "AMix checkpoint: $amix_ckpt"
echo "AMix config: $amix_config"
echo "Decoder checkpoint: $decoder_ckpt"
echo "Output file: $save_name"
echo "Evolution steps: $n_steps"
echo "Population size: $population"
echo "Device: $devices"
echo "Seed: $seed"
echo "==================================="

# Validate required files exist
if [ ! -f "$amix_ckpt" ]; then
    echo "Error: AMix checkpoint not found: $amix_ckpt"
    exit 1
fi

if [ ! -f "$decoder_ckpt" ]; then
    echo "Error: Decoder checkpoint not found: $decoder_ckpt"
    exit 1
fi

if [ ! -f "$amix_config" ]; then
    echo "Error: AMix config not found: $amix_config"
    exit 1
fi

# Create output directory if it doesn't exist
output_dir=$(dirname "$save_name")
if [ -n "$output_dir" ] && [ "$output_dir" != "." ]; then
    mkdir -p "$output_dir"
fi

# Run directed evolution with AMix configuration
python run_discrete_de_amix_new.py \
    --wt "$wt_sequence" \
    --wt_fitness "$wt_fitness" \
    --amix_ckpt "$amix_ckpt" \
    --amix_config "$amix_config" \
    --decoder_ckpt_path "$decoder_ckpt" \
    --save_name "$save_name" \
    --n_steps "$n_steps" \
    --population "$population" \
    --num_proposes_per_var "$num_proposes_per_var" \
    --beam_size "$beam_size" \
    --top_k "$top_k" \
    --mask_ratio_low "$mask_ratio_low" \
    --mask_ratio_high "$mask_ratio_high" \
    --population_ratio_per_mask $population_ratio_per_mask \
    --devices "$devices" \
    --seed "$seed" \
    --max_candidates_per_variant "$max_candidates_per_variant"

echo "==================================="
echo "Directed evolution complete!"
echo "Results saved to: $save_name"
echo "==================================="

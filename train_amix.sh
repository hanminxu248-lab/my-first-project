#!/bin/bash
# AMix-specific training script
# This script trains an AMix decoder regressor using the AMix ProfileBFN model configuration
# 
# Usage:
#   ./train_amix.sh <dataset> <devices> [batch_size] [ckpt_path]
# 
# Example:
#   ./train_amix.sh my_dataset 0 128
#   ./train_amix.sh my_dataset 0,1 64 path/to/pretrained_amix.pt

dataset=$1
devices=$2
batch_size=${3:-128}
ckpt_path=${4:-''}

# Set data file path (adjust as needed for your environment)
data_file="./data/${dataset}/${dataset}.csv"

# AMix configuration file
config_path="./configs/amix_config.yaml"

# AMix-specific parameters matching the ProfileBFN configuration
dec_hidden_dim=1680  # Must match hidden_dim in amix_config.yaml
lr=5e-5
num_epochs=10
num_ckpts=3
precision=32  # Use 32 for full precision, 16 for mixed precision

# Output directory
output_dir="./exps/${dataset}_amix"

echo "==================================="
echo "AMix Training Configuration"
echo "==================================="
echo "Dataset: $dataset"
echo "Data file: $data_file"
echo "Config: $config_path"
echo "Decoder hidden dim: $dec_hidden_dim"
echo "Batch size: $batch_size"
echo "Devices: $devices"
echo "Learning rate: $lr"
echo "Epochs: $num_epochs"
echo "Output dir: $output_dir"
if [ -n "$ckpt_path" ]; then
    echo "Checkpoint: $ckpt_path"
fi
echo "==================================="

# Run training with AMix configuration
python train_decoder_amix_fixed.py \
    --data_file "$data_file" \
    --dataset_name "$dataset" \
    --config_path "$config_path" \
    --dec_hidden_dim "$dec_hidden_dim" \
    --batch_size "$batch_size" \
    --devices "$devices" \
    --output_dir "$output_dir" \
    --lr "$lr" \
    --num_epochs "$num_epochs" \
    --num_ckpts "$num_ckpts" \
    --precision "$precision" \
    ${ckpt_path:+--ckpt_path "$ckpt_path"}

echo "==================================="
echo "Training complete!"
echo "Checkpoints saved to: $output_dir/checkpoints"
echo "==================================="

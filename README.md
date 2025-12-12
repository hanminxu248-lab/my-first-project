# AMix ProfileBFN Model - Training and Inference Pipeline

This repository contains training and inference pipelines for the AMix ProfileBFN model, a transformer-based architecture for protein fitness prediction and directed evolution.

## Table of Contents

- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Model Architecture

The AMix ProfileBFN model is based on the ESM-2 architecture with the following specifications:

- **Hidden dimension**: 1680
- **Number of layers**: 48
- **Number of attention heads**: 40
- **Intermediate (feedforward) size**: 6720
- **Vocabulary size**: 30 (20 amino acids + special tokens)

These parameters align with the `facebook/esm2_t30_150M_UR50D` pretrained model architecture.

## Configuration

### Configuration File

The model configuration is defined in `configs/amix_config.yaml`:

```yaml
# Model architecture parameters
hidden_dim: 1680                # Hidden dimension size (d_model)
num_layers: 48                  # Number of transformer encoder layers
num_attention_heads: 40         # Number of attention heads (nhead)
intermediate_size: 6720         # Feedforward network hidden dimension
vocab_size: 30                  # Vocabulary size

# Training parameters (optional)
batch_size: 128
learning_rate: 5.0e-5
num_epochs: 10
```

### Key Parameters

- **hidden_dim**: The dimensionality of the model's hidden states and embeddings
- **num_layers**: Number of transformer encoder layers in the stack
- **num_attention_heads**: Number of attention heads in multi-head attention (must divide hidden_dim evenly)
- **intermediate_size**: Size of the feedforward network's hidden layer (typically 4x hidden_dim)
- **vocab_size**: Size of the vocabulary (20 amino acids + padding/special tokens)

## Installation

### Requirements

```bash
# Python >= 3.8
pip install torch>=2.0.0
pip install lightning>=2.0.0
pip install transformers>=4.30.0
pip install pandas numpy pyyaml
```

### Repository Setup

```bash
git clone <repository-url>
cd my-first-project
```

## Training

### Basic Training

Train an AMix decoder regressor using the training script:

```bash
./train_amix.sh <dataset_name> <gpu_devices> [batch_size] [checkpoint_path]
```

**Example:**

```bash
# Train on a single GPU with default batch size
./train_amix.sh protein_dataset 0

# Train on multiple GPUs with custom batch size
./train_amix.sh protein_dataset 0,1 64

# Train with a pretrained checkpoint
./train_amix.sh protein_dataset 0 128 path/to/pretrained_amix.pt
```

### Training Script Parameters

- `dataset_name`: Name of your dataset (used for file paths and naming)
- `gpu_devices`: GPU device IDs (e.g., "0" for single GPU, "0,1" for multiple GPUs)
- `batch_size` (optional): Batch size for training (default: 128)
- `checkpoint_path` (optional): Path to pretrained AMix checkpoint

### Manual Training Command

For more control, use the Python script directly:

```bash
python train_decoder_amix_fixed.py \
    --data_file ./data/my_dataset/my_dataset.csv \
    --dataset_name my_dataset \
    --config_path ./configs/amix_config.yaml \
    --dec_hidden_dim 1680 \
    --batch_size 128 \
    --devices 0 \
    --output_dir ./exps/my_dataset_amix \
    --lr 5e-5 \
    --num_epochs 10 \
    --num_ckpts 3 \
    --precision 32
```

### Training Data Format

The training data should be a CSV file with two columns:

```csv
sequence,fitness
ACDEFGHIKLMNPQRSTVWY,1.0
ACDEFGHIKLMNPQRSTVWF,0.8
ACDEFGHIKLMNPQRSTVWG,1.2
```

### Training Outputs

Training produces the following outputs in `./exps/<dataset_name>_amix/`:

- **checkpoints/**: Model checkpoints saved during training
  - `AMix-dec_1680-<dataset_name>_epoch-XX-val_loss-X.XXX.ckpt`: Lightning checkpoints
  - `AMix-dec_1680-<dataset_name>.pt`: Final combined checkpoint for inference
- **lightning_logs/**: Training logs and metrics

## Inference

### Directed Evolution

Run directed evolution to optimize protein sequences:

```bash
./run_de_amix.sh <wt_sequence> <wt_fitness> <amix_ckpt> <decoder_ckpt> <output_file> [n_steps] [population] [gpu] [seed]
```

**Example:**

```bash
./run_de_amix.sh \
    "ACDEFGHIKLMNPQRSTVWY" \
    1.0 \
    path/to/amix.pt \
    path/to/AMix-dec_1680-dataset.pt \
    results/evolved_sequences.csv \
    10 \
    32 \
    0 \
    42
```

### Inference Script Parameters

- `wt_sequence`: Wild-type protein sequence (starting point)
- `wt_fitness`: Wild-type fitness score
- `amix_ckpt`: Path to AMix model checkpoint
- `decoder_ckpt`: Path to trained decoder checkpoint (from training)
- `output_file`: Path to save results (CSV format)
- `n_steps` (optional): Number of evolution steps (default: 10)
- `population` (optional): Population size per step (default: 32)
- `gpu` (optional): GPU device ID (default: 0)
- `seed` (optional): Random seed for reproducibility (default: 0)

### Manual Inference Command

For more control, use the Python script directly:

```bash
python run_discrete_de_amix_new.py \
    --wt "ACDEFGHIKLMNPQRSTVWY" \
    --wt_fitness 1.0 \
    --amix_ckpt path/to/amix.pt \
    --amix_config ./configs/amix_config.yaml \
    --decoder_ckpt_path path/to/decoder.pt \
    --save_name results/output.csv \
    --n_steps 10 \
    --population 32 \
    --beam_size 5 \
    --top_k 5 \
    --mask_ratio_low 0.1 \
    --mask_ratio_high 0.25 \
    --devices 0 \
    --seed 0
```

### Inference Outputs

The inference script produces a CSV file with the following columns:

```csv
WT,mutants,pred_score,sequence
ACDEFGHIKLMNPQRSTVWY,A1F:K9R,1.45,FCDEFGHIRRLMNPQRSTVWY
ACDEFGHIKLMNPQRSTVWY,K9R,1.32,ACDEFGHIRRLMNPQRSTVWY
...
```

- **WT**: Wild-type sequence
- **mutants**: Mutations in format `<original><position><mutant>` (e.g., "A1F:K9R")
- **pred_score**: Predicted fitness score
- **sequence**: Mutated sequence

## Troubleshooting

### Dimension Mismatch Errors

**Problem**: Error about dimension mismatch when loading checkpoints.

**Symptoms**:
```
RuntimeError: size mismatch for embedding.weight: copying a param with shape torch.Size([30, 1280]) from checkpoint, 
the shape in current model is torch.Size([30, 1680])
```

**Solution**:

1. **Check config file**: Ensure `configs/amix_config.yaml` has `hidden_dim: 1680`
2. **Match training dimensions**: Use `--dec_hidden_dim 1680` when training
3. **Verify checkpoint**: Check that checkpoint was trained with the same configuration

**Prevention**: The updated pipeline includes automatic dimension validation:
- Training: Validates `--dec_hidden_dim` matches `encoder.hidden_dim`
- Inference: Validates loaded checkpoint dimensions match current configuration

### Config File Not Found

**Problem**: Config file not loading properly.

**Symptoms**:
```
WARNING - [AMixEncoder] failed to load config configs/amix_config.yaml: [Errno 2] No such file or directory
```

**Solution**:

1. Verify the config file exists:
   ```bash
   ls -la configs/amix_config.yaml
   ```

2. Use absolute path if needed:
   ```bash
   --config_path $(pwd)/configs/amix_config.yaml
   ```

3. Check current working directory matches repository root

### Out of Memory Errors

**Problem**: GPU runs out of memory during training or inference.

**Solutions**:

1. **Reduce batch size**:
   ```bash
   ./train_amix.sh dataset 0 64  # Reduce from default 128 to 64
   ```

2. **Use gradient accumulation** (modify training script):
   ```python
   trainer = Trainer(..., accumulate_grad_batches=2)
   ```

3. **Use mixed precision training**:
   ```bash
   python train_decoder_amix_fixed.py ... --precision 16
   ```

4. **Reduce population size for inference**:
   ```bash
   ./run_de_amix.sh ... 10 16  # Reduce population from 32 to 16
   ```

### Checkpoint Loading Issues

**Problem**: Checkpoint fails to load or produces warnings.

**Common Issues**:

1. **Strict loading**: Change `strict=False` to `strict=True` if needed
2. **Key naming**: Ensure checkpoint keys match model structure
3. **Version compatibility**: Check PyTorch/Lightning versions match

**Debugging**:

```python
# Inspect checkpoint contents
import torch
ckpt = torch.load("path/to/checkpoint.pt", map_location="cpu")
print(ckpt.keys())  # See what's in the checkpoint
if "meta" in ckpt:
    print(ckpt["meta"])  # Check saved metadata
```

### Vocabulary Size Mismatch

**Problem**: Token IDs exceed vocabulary size.

**Symptoms**:
```
RuntimeError: index out of range in self
```

**Solution**:

1. Ensure config has `vocab_size: 30` (or appropriate size for your data)
2. Verify your data only uses the 20 standard amino acids
3. Check for unexpected characters in sequences

## Advanced Usage

### Custom Configuration

Create a custom config file for different model sizes:

```yaml
# configs/amix_config_small.yaml
hidden_dim: 640
num_layers: 12
num_attention_heads: 20
intermediate_size: 2560
vocab_size: 30
```

Use with:
```bash
python train_decoder_amix_fixed.py --config_path ./configs/amix_config_small.yaml ...
```

### Gradient Checkpointing

For training very large models, gradient checkpointing can reduce memory usage:

```python
# In AMixEncoder.__init__ after creating encoder_layers:
if config.get("gradient_checkpointing", False):
    self.encoder_layers.requires_grad_(True)
    # Enable gradient checkpointing if supported by PyTorch version
```

### Freezing Encoder

To train only the decoder head (useful for transfer learning):

```bash
python train_decoder_amix_fixed.py ... --freeze_encoder
```

### Custom Decoder Architecture

Use attention-based decoder instead of MLP:

```bash
python train_decoder_amix_fixed.py ... --use_attention_decoder
```

### Multiple Evolution Strategies

Adjust masking ratios for different exploration vs exploitation:

```bash
# More conservative (low masking)
python run_discrete_de_amix_new.py ... --mask_ratio_low 0.05 --mask_ratio_high 0.15

# More aggressive (high masking)
python run_discrete_de_amix_new.py ... --mask_ratio_low 0.15 --mask_ratio_high 0.35
```

## Model Checkpoints

### Checkpoint Structure

Training produces checkpoints with the following structure:

```python
{
    "encoder_state_dict": {...},      # Encoder model weights
    "regressor_state_dict": {...},    # Regressor model weights
    "meta": {                          # Metadata for compatibility checking
        "dec_hidden_dim": 1680,
        "encoder_hidden_dim": 1680,
        "num_layers": 48,
        "nhead": 40,
        "intermediate_size": 6720,
        "vocab_size": 30
    }
}
```

### Checkpoint Compatibility

The updated pipeline includes automatic compatibility checking:

- **Training**: Saves full architecture metadata in checkpoint
- **Inference**: Validates loaded checkpoint matches current configuration
- **Warnings**: Displays clear warnings for dimension mismatches

## Performance Tips

1. **Use appropriate batch size**: Start with 128 and adjust based on GPU memory
2. **Monitor validation loss**: Stop training if validation loss stops improving
3. **Use multiple GPUs**: Distribute batch across GPUs with `--devices 0,1,2,3`
4. **Optimize beam search**: Reduce `beam_size` and `top_k` for faster inference
5. **Cache embeddings**: For multiple runs on same sequences, cache encoder outputs

## Citation

If you use this code, please cite:

```bibtex
@article{amix_profilebfn,
  title={AMix ProfileBFN: Transformer-based Protein Fitness Prediction},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

[Add your license information here]

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]

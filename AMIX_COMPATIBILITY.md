# AMix Compatibility Fixes

## Overview

This document describes the fixes applied to make the AMix language model compatible with the Directed Evolution framework, which was originally designed for ESM2.

## Problem Statement

When replacing ESM2 with AMix in the Directed Evolution framework, several compatibility issues arose:

1. **Different Tokenization**: ESM2 uses HuggingFace's tokenizer with special tokens (`<cls>`, `<eos>`, `<pad>`, `<mask>`), while AMix uses a simple amino acid mapping.

2. **Different Output Formats**: 
   - ESM2's forward returns `MaskedLMOutput` with `.logits` and `.hidden_states` attributes
   - AMix's original forward returned mean-pooled vectors `[B, hidden_dim]`

3. **Missing Tokenizer Interface**: The framework expects models to have a `tokenizer` attribute with `tokenize()` and `decode()` methods.

4. **Incompatible Dimensions**: The mutation model and fitness predictor were using different tokenizers and model architectures.

## Solution

### 1. Created `AMixMutationModel` Class

A new wrapper class that provides ESM2-compatible interface for AMix:

```python
class AMixMutationModel(nn.Module):
    """
    AMix wrapper that provides ESM2-compatible interface for mutation model.
    """
    def __init__(self, ckpt_path, config_path=None, device=None):
        # ... initialization code ...
        self.tokenizer = self  # Self-referencing for framework compatibility
    
    def tokenize(self, inputs: List[str]):
        """Returns dict with 'input_ids' and 'attention_mask'"""
        # ... tokenization code ...
    
    def decode(self, tokens: torch.Tensor) -> List[str]:
        """Decodes token IDs back to sequences"""
        # ... decoding code ...
    
    def forward(self, inputs):
        """Returns object with .logits and .hidden_states attributes"""
        # ... forward pass ...
```

#### Key Features:

- **Tokenize Method**: Properly handles multi-character mask tokens like `<mask>`
- **Decode Method**: Converts token IDs back to amino acid sequences
- **Forward Method**: Returns an output object with `.logits` and `.hidden_states` attributes
- **Tokenizer Attribute**: Self-referencing to provide framework compatibility

### 2. Updated `initialize_mutation_model()` Function

Changed from ESM2 to AMixMutationModel:

```python
def initialize_mutation_model(args, device):
    # Use AMixMutationModel instead of ESM2 for compatibility
    encoder_config = getattr(args, "encoder_config", None)
    model = AMixMutationModel(
        ckpt_path=args.encoder_ckpt_path or args.decoder_ckpt_path,
        config_path=encoder_config,
        device=device
    )
    model.eval()
    tokenizer = model.tokenizer
    return model, tokenizer
```

### 3. Maintained Separation of Concerns

- **Mutation Model** (`AMixMutationModel`): Returns full hidden states `[B, L, D]` and logits for mutation sampling
- **Fitness Predictor** (`AMixEncoder`): Returns mean-pooled embeddings `[B, D]` for fitness prediction

This separation ensures each component has the appropriate output format for its use case.

## Files Modified

1. **`run_discrete_de_amix.py`**:
   - Added `AMixMutationModel` class
   - Updated `initialize_mutation_model()` function
   - Removed ESM2 import

## Files Verified (Already Compatible)

1. **`run_discrete_de_amix_beam.py`**: Already has compatible `AmixLMWrapper`
2. **`run_discrete_de_amix_new.py`**: Already has compatible `AmixLMWrapper`
3. **`train_decoder_amix.py`**: Correctly uses mean pooling for fitness prediction

## Compatibility Matrix

| Component | Tokenization | Forward Output | Usage |
|-----------|-------------|----------------|-------|
| ESM2 (original) | HuggingFace tokenizer | MaskedLMOutput with .logits and .hidden_states | Mutation model |
| AMixMutationModel | Custom amino acid mapping | Object with .logits and .hidden_states | Mutation model (ESM2 replacement) |
| AMixEncoder | Direct amino acid mapping | Mean-pooled embeddings [B, D] | Fitness predictor |
| AmixLMWrapper (beam/new files) | Custom with mask token parsing | Object with .logits and .hidden_states | Mutation model (beam search variants) |

## Testing

The implementation was validated with the following tests:

1. ✓ Tokenization produces correct format (dict with input_ids and attention_mask)
2. ✓ Decoding correctly reconstructs sequences
3. ✓ Forward pass produces ESM2-compatible output format
4. ✓ Mask token handling works correctly (multi-character tokens)
5. ✓ Tokenizer attribute is accessible and functional

## Usage

When running directed evolution with AMix:

```bash
python run_discrete_de_amix.py \
    --wt "ACDEFGHIKLMNPQRSTVWY" \
    --wt_fitness 1.0 \
    --task AAV \
    --decoder_ckpt_path path/to/decoder.ckpt \
    --encoder_ckpt_path path/to/encoder.ckpt \
    --encoder_config path/to/config.yaml \
    --save_name results.csv
```

The mutation model will now use `AMixMutationModel` instead of ESM2, providing full compatibility with the Directed Evolution framework.

## Benefits

1. **Full ESM2 Compatibility**: AMix can now be used as a drop-in replacement for ESM2
2. **Unified Interface**: Consistent tokenization and output format across all files
3. **Proper Mask Token Handling**: Multi-character mask tokens are correctly parsed
4. **Clear Separation**: Mutation model and fitness predictor have distinct, appropriate output formats
5. **No Breaking Changes**: Existing beam search and new variants remain compatible

## Future Improvements

Consider:
- Adding support for different mask token formats (e.g., `[MASK]`, `*`, etc.)
- Implementing attention mask usage in the encoder
- Adding batch inference optimization
- Supporting model ensembles

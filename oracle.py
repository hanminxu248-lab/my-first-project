#!/usr/bin/env python3
"""
oracle.py - AMix Fitness Oracle Module

Provides the AMixFitnessWrapper class for fitness prediction using AMix encoder + regressor.
This module delegates fitness inference to the wrapper and provides a clean interface
for the directed evolution pipeline.

Key features:
- Wrapper-based delegation pattern (no overwrites)
- Batch-safe infer_fitness returning list of floats
- Same seq->input_ids mapping as other modules
"""

import os
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class AttentionPool1D(nn.Module):
    """Attention-based pooling for sequence representations."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, seq_hidden: torch.Tensor) -> torch.Tensor:
        Q = self.q(seq_hidden)
        K = self.k(seq_hidden)
        V = self.v(seq_hidden)
        att = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        att = torch.nn.functional.softmax(att, dim=-1)
        pooled = torch.matmul(att, V)
        return pooled[:, 0, :]


class DecoderRegressor(nn.Module):
    """Decoder/regressor for fitness prediction."""
    def __init__(self, in_dim: int = 1680, hidden: int = 512):
        super().__init__()
        self.pool = AttentionPool1D(in_dim)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            p = self.pool(x)
        else:
            p = x
        return self.head(p)


class DummyEncoder(nn.Module):
    """Simple dummy encoder for testing purposes."""
    def __init__(self, vocab_size: int = 21, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        # Simple mean pooling
        x = x.mean(dim=1)
        return self.linear(x)


class AMixFitnessWrapper:
    """
    Wrapper for fitness prediction using AMix encoder + regressor.
    
    Uses a wrapper attribute pattern (not overwrites) and delegates infer_fitness to it.
    Provides batch-safe inference returning a plain list of floats.
    Uses the same seq->input_ids mapping as other modules.
    """
    
    def __init__(self, regressor: nn.Module, encoder_obj: Optional[nn.Module] = None, device: str = "cpu"):
        """
        Initialize the fitness wrapper.
        
        Args:
            regressor: The regressor/decoder model for fitness prediction
            encoder_obj: The encoder model for sequence embedding (optional)
            device: Device to run models on
        """
        self.device = torch.device(device)
        self._regressor = regressor.to(self.device)
        self._regressor.eval()
        self._encoder = encoder_obj.to(self.device) if encoder_obj else None
        
        # Same amino acid mapping as other modules
        self.amino2id = {a: i + 1 for i, a in enumerate(AMINO_ACIDS)}
        self.id2amino = {i + 1: a for i, a in enumerate(AMINO_ACIDS)}
        self.pad_id = 0

    @property
    def regressor(self) -> nn.Module:
        """Get the regressor model."""
        return self._regressor

    @property
    def encoder(self) -> Optional[nn.Module]:
        """Get the encoder model."""
        return self._encoder

    def seqs_to_input_ids(self, seqs: List[str]) -> torch.Tensor:
        """
        Convert sequences to input_ids tensor.
        Uses the same mapping as other modules for consistency.
        
        Args:
            seqs: List of protein sequences
            
        Returns:
            Tensor of input_ids with shape (batch_size, max_len)
        """
        if not seqs:
            return torch.zeros((0, 1), dtype=torch.long, device=self.device)
        max_len = max(len(s) for s in seqs)
        batch = torch.full((len(seqs), max_len), fill_value=self.pad_id, dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            ids = [self.amino2id.get(c, self.pad_id) for c in s]
            batch[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
        return batch

    @torch.inference_mode()
    def infer_fitness(self, inputs, batch_size: int = 64) -> List[float]:
        """
        Infer fitness for sequences or embeddings.
        
        Batch-safe implementation that processes sequences in batches
        and returns a plain list of floats.
        
        Args:
            inputs: List of sequences (str) or tensor of embeddings
            batch_size: Batch size for processing
            
        Returns:
            List of fitness values as floats
        """
        if isinstance(inputs, list):
            if len(inputs) == 0:
                return []
            if not self._encoder:
                raise RuntimeError("No encoder available for sequence -> embedding conversion")
            
            all_embs = []
            for i in range(0, len(inputs), batch_size):
                batch_seqs = inputs[i:i + batch_size]
                ids = self.seqs_to_input_ids(batch_seqs)
                emb = self._encoder(ids)
                all_embs.append(emb.detach().cpu())
            embs = torch.cat(all_embs, dim=0).to(self.device)
        else:
            embs = inputs.to(self.device)
        
        if embs.size(0) == 0:
            return []
        
        outs = []
        for i in range(0, embs.size(0), batch_size):
            out = self._regressor(embs[i:i + batch_size])
            outs.append(out.detach().cpu())
        
        result = torch.cat(outs, dim=0).squeeze(-1).numpy()
        # Ensure we return a list of floats
        if hasattr(result, 'tolist'):
            return result.tolist()
        return [float(result)] if np.isscalar(result) else list(result)

    def __call__(self, inputs, batch_size: int = 64) -> List[float]:
        """Alias for infer_fitness."""
        return self.infer_fitness(inputs, batch_size)


class AMixOracle:
    """
    High-level Oracle class that wraps the fitness predictor.
    
    Uses wrapper attribute (not overwrites) and delegates to AMixFitnessWrapper.
    This class provides the interface expected by the directed evolution pipeline.
    """
    
    def __init__(self, encoder_ckpt_path: Optional[str] = None,
                 decoder_ckpt_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 device: str = "cpu",
                 hidden_dim: int = 1680):
        """
        Initialize the AMix Oracle.
        
        Args:
            encoder_ckpt_path: Path to encoder checkpoint
            decoder_ckpt_path: Path to decoder/regressor checkpoint
            config_path: Path to config YAML
            device: Device to run models on
            hidden_dim: Hidden dimension for models
        """
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        
        # Create encoder (use dummy if no checkpoint)
        if encoder_ckpt_path and os.path.exists(encoder_ckpt_path):
            # Import AMixEncoder from run_discrete_de_amix_new if available
            try:
                from run_discrete_de_amix_new import AMixEncoder
                encoder = AMixEncoder(ckpt_path=encoder_ckpt_path, config_path=config_path, device=device)
            except ImportError:
                logging.warning("AMixEncoder not available, using DummyEncoder")
                encoder = DummyEncoder(hidden_dim=hidden_dim)
        else:
            encoder = DummyEncoder(hidden_dim=hidden_dim)
        
        # Create regressor
        regressor = DecoderRegressor(in_dim=hidden_dim)
        
        # Load decoder checkpoint if provided
        if decoder_ckpt_path and os.path.exists(decoder_ckpt_path):
            self._load_decoder_checkpoint(regressor, decoder_ckpt_path)
        
        # Create wrapper (use wrapper attribute, not overwrite self.model)
        self._wrapper = AMixFitnessWrapper(regressor, encoder, device)
        logging.info(f"[AMixOracle] initialized with device={device}, hidden_dim={hidden_dim}")

    def _load_decoder_checkpoint(self, regressor: nn.Module, ckpt_path: str):
        """Load decoder checkpoint with support for multiple formats."""
        try:
            ck = torch.load(ckpt_path, map_location="cpu")
            
            if isinstance(ck, dict):
                if "regressor_state_dict" in ck:
                    regressor.load_state_dict(ck["regressor_state_dict"], strict=False)
                    logging.info("[AMixOracle] Loaded regressor from 'regressor_state_dict'")
                elif "state_dict" in ck:
                    regressor.load_state_dict(ck["state_dict"], strict=False)
                    logging.info("[AMixOracle] Loaded regressor from 'state_dict'")
                else:
                    regressor.load_state_dict(ck, strict=False)
                    logging.info("[AMixOracle] Loaded regressor from raw dict")
            else:
                regressor.load_state_dict(ck, strict=False)
                logging.info("[AMixOracle] Loaded regressor from raw checkpoint")
        except Exception as e:
            logging.warning(f"[AMixOracle] Failed to load decoder checkpoint: {e}")

    @property
    def wrapper(self) -> AMixFitnessWrapper:
        """Get the fitness wrapper."""
        return self._wrapper

    def infer_fitness(self, inputs, batch_size: int = 64) -> List[float]:
        """
        Infer fitness values for inputs.
        Delegates to the wrapper's infer_fitness method.
        
        Args:
            inputs: List of sequences or tensor of embeddings
            batch_size: Batch size for processing
            
        Returns:
            List of fitness values as floats
        """
        return self._wrapper.infer_fitness(inputs, batch_size)

    def __call__(self, inputs, batch_size: int = 64) -> List[float]:
        """Alias for infer_fitness."""
        return self.infer_fitness(inputs, batch_size)


# -------------------- Smoke Test --------------------
def _smoke_test():
    """
    Smoke test to verify the module works correctly.
    Uses dummy encoder/decoder to ensure infer_fitness returns plausible outputs.
    """
    logging.info("Running oracle.py smoke test...")
    
    # Create dummy encoder and regressor
    hidden_dim = 64
    encoder = DummyEncoder(vocab_size=21, hidden_dim=hidden_dim)
    regressor = DecoderRegressor(in_dim=hidden_dim, hidden=32)
    
    # Create wrapper
    wrapper = AMixFitnessWrapper(regressor, encoder, device="cpu")
    
    # Test with sequences
    test_seqs = ["ACDEFGHIK", "LMNPQRSTV", "WYACDEFGH"]
    fitness = wrapper.infer_fitness(test_seqs)
    
    # Verify output
    assert isinstance(fitness, list), f"Expected list, got {type(fitness)}"
    assert len(fitness) == len(test_seqs), f"Expected {len(test_seqs)} values, got {len(fitness)}"
    assert all(isinstance(f, float) for f in fitness), "All values should be floats"
    logging.info(f"Wrapper test passed: fitness values = {fitness}")
    
    # Test with empty input
    empty_fitness = wrapper.infer_fitness([])
    assert empty_fitness == [], f"Expected empty list, got {empty_fitness}"
    logging.info("Empty input test passed")
    
    # Test Oracle class
    oracle = AMixOracle(device="cpu", hidden_dim=hidden_dim)
    oracle_fitness = oracle.infer_fitness(test_seqs)
    assert len(oracle_fitness) == len(test_seqs), f"Oracle returned wrong number of values"
    logging.info(f"Oracle test passed: fitness values = {oracle_fitness}")
    
    # Test batch processing
    large_batch = ["ACDEF"] * 100
    batch_fitness = wrapper.infer_fitness(large_batch, batch_size=32)
    assert len(batch_fitness) == 100, "Batch processing failed"
    logging.info("Batch processing test passed")
    
    logging.info("All oracle.py smoke tests passed!")
    return True


if __name__ == "__main__":
    _smoke_test()

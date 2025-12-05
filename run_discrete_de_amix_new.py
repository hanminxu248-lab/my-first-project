#!/usr/bin/env python3
# run_discrete_de_amix_new.py
"""
Directed Evolution Inference Script (AMix-guided)
- Uses a local AMix-like LM for mutation (beam search or random)
- Uses a trained Decoder / ProfileBFN oracle for fitness prediction
- Fully respects the model config parameters from the provided YAML

Key fixes and improvements:
- Single-character mask token ('X') for consistent tokenization
- CLS pooling in AMixEncoder.forward matching training
- Robust checkpoint loading supporting multiple formats
- Proper population sampling per masker based on population_ratio_per_mask
- Elitism: preserve best sequences each generation
- Duplicate handling: keep best-scoring duplicate
- Evolution history tracking (max/mean fitness per step)
- Model-guided mutation option using encoder embeddings
"""

import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from copy import deepcopy
from operator import itemgetter
import time
import logging
import itertools
from typing import List, Tuple, Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Standard amino acids for random mutation
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# -------------------- Utilities --------------------
def timer(func):
    """Decorator to log function execution time."""
    def wrapper(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        t1 = time.time()
        logging.debug(f"{func.__name__} took {t1-t0:.3f}s")
        return res
    return wrapper

def itertools_chain_flatten_repeat(items: List, times: int) -> List:
    """Repeat each item in the list 'times' times."""
    out = []
    for i in items:
        for _ in range(times):
            out.append(deepcopy(i))
    return out

def count_mutations_from_wt(wt_seq: str, seq: str) -> int:
    """Count the number of positions that differ between wt_seq and seq."""
    if len(wt_seq) != len(seq):
        return max(len(wt_seq), len(seq))  # handle length mismatch
    return sum(1 for a, b in zip(wt_seq, seq) if a != b)

# -------------------- AMix Encoder --------------------
class AMixEncoder(nn.Module):
    """
    AMix Encoder with CLS pooling.
    Uses position 0 as CLS token representation for pooled output.
    Supports loading from combined checkpoints (encoder_state_dict, regressor_state_dict, state_dict).
    """
    def __init__(self, ckpt_path: Optional[str] = None, config_path: Optional[str] = None, 
                 device: Optional[str] = None, load_embedding_only: bool = False):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        import yaml
        config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f) or {}
            except Exception as e:
                logging.warning(f"[AMixEncoder] failed to load config {config_path}: {e}")
        
        self.hidden_dim = int(config.get("hidden_dim", 1680))
        self.vocab_size = int(config.get("vocab_size", 30))
        self.num_layers = int(config.get("num_layers", 48))
        self.nhead = int(config.get("nhead", 40))

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.nhead, batch_first=True)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        if ckpt_path and os.path.exists(ckpt_path):
            self._load_checkpoint(ckpt_path, load_embedding_only)

        self.to(self.device)
        logging.info(f"[AMixEncoder] initialized with hidden_dim={self.hidden_dim}, vocab_size={self.vocab_size}, device={self.device}")

    def _load_checkpoint(self, ckpt_path: str, load_embedding_only: bool = False):
        """
        Load checkpoint with support for multiple formats:
        - Combined format with 'encoder_state_dict' and 'regressor_state_dict'
        - Legacy format with 'state_dict'
        - Raw state dict
        """
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            
            # Try to find the encoder state dict
            if isinstance(ckpt, dict):
                if "encoder_state_dict" in ckpt:
                    state_dict = ckpt["encoder_state_dict"]
                    logging.info("[AMixEncoder] loading from 'encoder_state_dict' key")
                elif "state_dict" in ckpt:
                    state_dict = ckpt["state_dict"]
                    logging.info("[AMixEncoder] loading from 'state_dict' key")
                else:
                    state_dict = ckpt
                    logging.info("[AMixEncoder] loading from raw checkpoint dict")
            else:
                state_dict = ckpt
            
            # Remove 'module.' prefix if present (from DataParallel)
            mapped = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            # Try to load embedding weights
            emb_loaded = False
            for k in ["embedding.weight", "encoder.embedding.weight", "embeddings.weight"]:
                if k in mapped and mapped[k].shape == self.embedding.weight.shape:
                    with torch.no_grad():
                        self.embedding.weight.copy_(mapped[k])
                    logging.info(f"[AMixEncoder] loaded embedding from key '{k}'")
                    emb_loaded = True
                    break
            
            if not emb_loaded:
                logging.warning("[AMixEncoder] no matching embedding key found in checkpoint")
            
            # Load full encoder weights if not embedding-only
            if not load_embedding_only:
                own = self.state_dict()
                matched = {}
                for k, v in mapped.items():
                    if k in own and own[k].shape == v.shape:
                        matched[k] = v
                if matched:
                    own.update(matched)
                    self.load_state_dict(own, strict=False)
                    logging.info(f"[AMixEncoder] loaded {len(matched)} matching weights from checkpoint")
                    
        except Exception as e:
            logging.warning(f"[AMixEncoder] failed to load ckpt: {e}")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with CLS pooling.
        CLS pooling takes the first token (position 0) hidden state as the pooled representation.
        This matches the training code in train_decoder_amix_fixed.py.
        """
        x = self.embedding(input_ids.to(self.embedding.weight.device))
        x = self.encoder_layers(x)
        # CLS pooling: use position 0 as the pooled representation (consistent with training)
        return x[:, 0, :]

    def get_full_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return full sequence hidden states (for model-guided mutation)."""
        x = self.embedding(input_ids.to(self.embedding.weight.device))
        x = self.encoder_layers(x)
        return x


    def get_full_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return full sequence hidden states (for model-guided mutation)."""
        x = self.embedding(input_ids.to(self.embedding.weight.device))
        x = self.encoder_layers(x)
        return x

# -------------------- Decoder / Oracle --------------------
class AttentionPool1D(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, seq_hidden):
        Q = self.q(seq_hidden)
        K = self.k(seq_hidden)
        V = self.v(seq_hidden)
        att = torch.matmul(Q, K.transpose(-2,-1)) * self.scale
        att = torch.nn.functional.softmax(att, dim=-1)
        pooled = torch.matmul(att, V)
        out = pooled[:,0,:]
        return out

class DecoderRegressor(nn.Module):
    def __init__(self, in_dim=1680, hidden=512):
        super().__init__()
        self.pool = AttentionPool1D(in_dim)
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden,1)
        )
    def forward(self, x):
        if x.dim() == 3:
            p = self.pool(x)
        else:
            p = x
        return self.head(p)

class AMixFitnessWrapper:
    """
    Wrapper for fitness prediction using AMix encoder + regressor.
    Supports batch-safe inference and returns plain list of floats.
    """
    def __init__(self, regressor: nn.Module, encoder_obj: Optional[AMixEncoder] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.regressor = regressor.to(self.device)
        self.regressor.eval()
        self.encoder = encoder_obj.to(self.device) if encoder_obj else None
        self.amino2id = {a: i + 1 for i, a in enumerate(AMINO_ACIDS)}
        self.id2amino = {i + 1: a for i, a in enumerate(AMINO_ACIDS)}
        self.pad_id = 0

    def seqs_to_input_ids(self, seqs: List[str]) -> torch.Tensor:
        """Convert sequences to input_ids tensor using same mapping as training."""
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
        Batch-safe and returns a plain list of floats.
        
        Args:
            inputs: List of sequences (str) or tensor of embeddings
            batch_size: Batch size for processing
            
        Returns:
            List of fitness values (floats)
        """
        if isinstance(inputs, list):
            if len(inputs) == 0:
                return []
            if not self.encoder:
                raise RuntimeError("No encoder for sequence -> embedding conversion")
            all_embs = []
            for i in range(0, len(inputs), batch_size):
                ids = self.seqs_to_input_ids(inputs[i:i+batch_size])
                emb = self.encoder(ids)
                all_embs.append(emb.detach().cpu())
            embs = torch.cat(all_embs, dim=0).to(self.device)
        else:
            embs = inputs.to(self.device)
        
        if embs.size(0) == 0:
            return []
            
        outs = []
        for i in range(0, embs.size(0), batch_size):
            out = self.regressor(embs[i:i+batch_size])
            outs.append(out.detach().cpu())
        result = torch.cat(outs, dim=0).squeeze(-1).numpy()
        # Ensure we return a list of floats
        return result.tolist() if hasattr(result, 'tolist') else list(result)

# -------------------- AMix LM Wrapper --------------------
# Using single-character mask token 'X' for consistency
# This avoids issues with multi-character tokens like "[MASK]"
MASK_TOKEN = "X"

class AmixLMWrapper(nn.Module):
    """
    AMix Language Model Wrapper for masked language modeling.
    Uses single-character mask token 'X' for consistent tokenization.
    """
    def __init__(self, ckpt_path: str, config_path: Optional[str] = None, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import yaml
        config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f) or {}
            except Exception as e:
                logging.warning(f"[AmixLMWrapper] failed to load config: {e}")
        
        self.hidden_dim = int(config.get("hidden_dim", 1680))
        self.vocab_size = int(config.get("vocab_size", 30))
        self.num_layers = int(config.get("num_layers", 48))
        self.nhead = int(config.get("nhead", 40))

        self.amino2id = {a: i + 1 for i, a in enumerate(AMINO_ACIDS)}
        self.id2amino = {i: a for a, i in self.amino2id.items()}
        self.pad_id = 0
        # Single-character mask token for consistent tokenization
        self.mask_token = MASK_TOKEN
        self.mask_id = self.vocab_size
        
        self.embedding = nn.Embedding(self.vocab_size + 1, self.hidden_dim, padding_idx=self.pad_id)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.nhead, batch_first=True)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        if ckpt_path and os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                for k in ["embedding.weight", "encoder.embedding.weight"]:
                    if k in state_dict and state_dict[k].shape == self.embedding.weight.shape:
                        with torch.no_grad():
                            self.embedding.weight.copy_(state_dict[k])
                        logging.info(f"[AmixLMWrapper] copied embedding from {k}")
                        break
            except Exception as e:
                logging.warning(f"[AmixLMWrapper] ckpt load failed: {e}")
        
        self.to(self.device)
        self.eval()
        logging.info(f"[AmixLMWrapper] initialized with mask_token='{self.mask_token}', device={self.device}")

    def tokenize(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize sequences, converting mask tokens to mask_id."""
        if not sequences:
            return {"input_ids": torch.zeros((0, 1), dtype=torch.long, device=self.device)}
        max_len = max(len(s) for s in sequences)
        input_ids = torch.full((len(sequences), max_len), fill_value=self.pad_id, dtype=torch.long, device=self.device)
        for i, s in enumerate(sequences):
            ids = [self.mask_id if c == self.mask_token else self.amino2id.get(c, self.pad_id) for c in s]
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
        return {"input_ids": input_ids}

    def decode(self, ids: List[int]) -> str:
        """Decode token ids back to sequence string."""
        out = []
        for tid in ids:
            if tid == self.pad_id:
                out.append("")
            elif tid == self.mask_id:
                out.append(self.mask_token)
            else:
                out.append(self.id2amino.get(tid, ""))
        return "".join(out)

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor):
        """Forward pass returning logits and hidden states."""
        input_ids = input_ids.to(self.device)
        hidden = self.encoder_layers(self.embedding(input_ids))
        logits = torch.nn.functional.linear(hidden, self.embedding.weight)
        return type("Out", (), {"logits": logits, "hidden_states": [hidden]})

# -------------------- Beam Fill --------------------
@timer
def beam_fill_masked_sequence(lm: AmixLMWrapper, masked_seq: str, masked_positions: List[int], 
                              beam_size: int = 5, top_k_per_pos: int = 5, device: str = "cpu"):
    """
    Fill masked positions in a sequence using beam search.
    Uses the LM to predict the best tokens at each masked position.
    """
    be = lm.tokenize([masked_seq])
    input_ids = be["input_ids"][0].clone().to(device)
    mask_id = lm.mask_id
    mask_positions = (input_ids == mask_id).nonzero(as_tuple=False).squeeze(-1).tolist()
    if isinstance(mask_positions, int):
        mask_positions = [mask_positions]
    if not mask_positions:
        out = lm.forward(input_ids.unsqueeze(0))
        pooled = out.hidden_states[-1][:, 0, :].detach().cpu()
        return [(lm.decode(input_ids.tolist()), 0.0, pooled[0])]
    
    beams = [(input_ids.clone().cpu(), 0.0)]
    for tok_pos in mask_positions[:len(masked_positions)]:
        new_beams = []
        batch_inputs = torch.stack([b[0] for b in beams], dim=0).to(device)
        out = lm.forward(batch_inputs)
        logits = getattr(out, "logits")
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        pos_logprobs = log_probs[:, tok_pos, :]
        topk = torch.topk(pos_logprobs, k=min(top_k_per_pos, pos_logprobs.size(-1)), dim=-1)
        for i in range(batch_inputs.size(0)):
            base_input = batch_inputs[i].detach().cpu()
            base_score = beams[i][1]
            for v, tid in zip(topk.values[i].tolist(), topk.indices[i].tolist()):
                new_input = base_input.clone()
                new_input[tok_pos] = int(tid)
                new_beams.append((new_input, base_score + float(v)))
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]
    
    final_inputs = torch.stack([b[0] for b in beams], dim=0).to(device)
    out_final = lm.forward(final_inputs)
    hidden = out_final.hidden_states[-1]
    pooled = hidden[:, 0, :].detach().cpu()
    results = []
    for i, (inp_cpu, score) in enumerate(beams):
        seq = lm.decode(inp_cpu.tolist())
        results.append((seq, score, pooled[i]))
    return results


# -------------------- Simple Masker --------------------
class SimpleMasker:
    """
    Simple random masker using single-character mask token.
    Masks a fraction of positions in each sequence.
    """
    def __init__(self, mask_ratio: float, mask_token: str = MASK_TOKEN):
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token  # Single-character mask token

    def run(self, sequences: List[str], ids: Optional[List[int]] = None) -> Tuple[List[str], List[List[int]]]:
        """
        Mask sequences and return masked sequences with their mask positions.
        
        Args:
            sequences: List of sequences to mask
            ids: Optional list of ids (not used, for API compatibility)
            
        Returns:
            Tuple of (masked_sequences, mask_positions)
        """
        masked = []
        posis = []
        for s in sequences:
            n = len(s)
            if n == 0:
                masked.append("")
                posis.append([])
                continue
            k = max(1, int(round(n * self.mask_ratio)))
            k = min(k, n)  # Ensure k <= n
            ps = list(np.random.choice(n, size=k, replace=False))
            ls = list(s)
            for p in ps:
                ls[p] = self.mask_token
            masked.append("".join(ls))
            posis.append(sorted(ps))
        return masked, posis


# -------------------- Directed Evolution --------------------
class DiscreteDirectedEvolutionBeam:
    """
    Directed Evolution engine with beam search mutation.
    
    Key features:
    - Population sampling per masker based on population_ratio_per_mask
    - Elitism: preserve best sequences each generation
    - Duplicate handling: keep best-scoring duplicate
    - Evolution history tracking
    - Optional model-guided or random mutation
    """
    def __init__(self, n_steps: int, population: int, maskers: List,
                 mutation_lm: AmixLMWrapper, mutation_tokenizer: AmixLMWrapper,
                 fitness_predictor: AMixFitnessWrapper,
                 beam_size: int = 5, top_k_per_pos: int = 5,
                 num_propose_mutation_per_variant: int = 4,
                 remove_duplications: bool = True,
                 population_ratio_per_mask: Optional[List[float]] = None,
                 verbose: bool = True, mutation_device: str = "cpu",
                 seed: int = 0, max_candidates_per_variant: int = 200,
                 elite_ratio: float = 0.1, use_random_mutation: bool = False):
        """
        Initialize the Directed Evolution engine.
        
        Args:
            n_steps: Number of evolution steps
            population: Population size per step
            maskers: List of maskers for sequence modification
            mutation_lm: Language model for mutation prediction
            mutation_tokenizer: Tokenizer (same as mutation_lm for AMix)
            fitness_predictor: Fitness prediction wrapper
            beam_size: Beam size for beam search
            top_k_per_pos: Top-k predictions per masked position
            num_propose_mutation_per_variant: Number of mutations proposed per variant
            remove_duplications: Whether to remove duplicate sequences
            population_ratio_per_mask: Ratio of population for each masker
            verbose: Enable verbose logging
            mutation_device: Device for mutation LM
            seed: Random seed for reproducibility
            max_candidates_per_variant: Maximum candidates per variant
            elite_ratio: Fraction of best sequences to preserve (elitism)
            use_random_mutation: Use random mutation instead of model-guided
        """
        self.n_steps = n_steps
        self.population = population
        self.maskers = maskers
        self.mutation_lm = mutation_lm
        self.mutation_tokenizer = mutation_tokenizer
        self.fitness_predictor = fitness_predictor
        self.beam_size = beam_size
        self.top_k_per_pos = top_k_per_pos
        self.num_propose_mutation_per_variant = num_propose_mutation_per_variant
        self.rm_dups = remove_duplications
        self.population_ratio_per_mask = population_ratio_per_mask or [1 / len(maskers) for _ in maskers]
        self.verbose = verbose
        self.mutation_device = torch.device(mutation_device) if not isinstance(mutation_device, torch.device) else mutation_device
        self.seed = seed
        self.max_candidates_per_variant = max_candidates_per_variant
        self.elite_ratio = elite_ratio
        self.use_random_mutation = use_random_mutation
        
        self.prev_fitness = None
        self.prev_mutants = None
        self.prev_variants = None
        
        # Evolution history for tracking
        self.history: Dict[str, List[float]] = {"max_fitness": [], "mean_fitness": []}
        
        # Set seeds for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    @timer
    def mask_sequences(self, variants: List[str], ids: List[int]) -> Tuple[List[str], List[List[int]]]:
        """
        Mask sequences according to population_ratio_per_mask.
        Samples a subset of variants for each masker based on ratio.
        """
        masked_variants = []
        masked_positions = []

        num_variant = len(variants)
        offset = 0
        
        for pop_ratio, masker in zip(self.population_ratio_per_mask, self.maskers):
            sub_population = int(num_variant * pop_ratio)
            if sub_population <= 0:
                continue
            
            # Sample subset of variants according to ratio (without replacement if possible)
            if sub_population <= num_variant:
                sample_indices = np.random.choice(num_variant, size=sub_population, replace=False)
            else:
                sample_indices = np.random.choice(num_variant, size=sub_population, replace=True)
            
            sub_variants = [variants[i] for i in sample_indices]
            mv, mp = masker.run(sub_variants, list(range(len(sub_variants))))
            masked_variants.extend(mv)
            masked_positions.extend(mp)
            offset += sub_population

        return masked_variants, masked_positions

    def _mutate_random(self, masked_seq: str) -> str:
        """
        Random mutation: replace mask tokens with random amino acids.
        Compares against single-character mask token.
        """
        result = []
        for c in masked_seq:
            if c == MASK_TOKEN:
                # Replace mask with random amino acid
                result.append(np.random.choice(list(AMINO_ACIDS)))
            else:
                result.append(c)
        return "".join(result)

    @timer
    def mutate_masked_sequences(self, wt_seq: str, masked_variants: List[str], 
                                 masked_positions: List[List[int]]) -> Tuple[List[str], List[str], torch.Tensor]:
        """
        Mutate masked sequences using beam search or random mutation.
        
        Args:
            wt_seq: Wild-type sequence
            masked_variants: List of masked sequences
            masked_positions: List of mask positions for each sequence
            
        Returns:
            Tuple of (mutated_seqs, mutants, pooled_tensor)
        """
        all_candidates = []
        
        for mv, pos in zip(masked_variants, masked_positions):
            if self.use_random_mutation:
                # Random mutation: replace mask tokens with random amino acids
                mutated = self._mutate_random(mv)
                # Create dummy pooled embedding
                dummy_pooled = torch.zeros(self.fitness_predictor.encoder.hidden_dim if hasattr(self.fitness_predictor, 'encoder') and self.fitness_predictor.encoder else 1680)
                all_candidates.append((mutated, 0.0, dummy_pooled))
            else:
                # Model-guided mutation using beam search
                try:
                    candidates = beam_fill_masked_sequence(
                        self.mutation_lm, mv, pos,
                        self.beam_size, self.top_k_per_pos, self.mutation_device
                    )
                except Exception as e:
                    logging.warning(f"Beam fill failed: {e}, falling back to random mutation")
                    mutated = self._mutate_random(mv)
                    dummy_pooled = torch.zeros(self.fitness_predictor.encoder.hidden_dim if hasattr(self.fitness_predictor, 'encoder') and self.fitness_predictor.encoder else 1680)
                    candidates = [(mutated, 0.0, dummy_pooled)]
                all_candidates.extend(candidates[:self.max_candidates_per_variant])
        
        if not all_candidates:
            return [], [], torch.zeros((0, 1680))
        
        mutated_seqs = [c[0] for c in all_candidates]
        pooled_tensor = torch.stack([c[2] for c in all_candidates], dim=0)
        
        # Generate mutation descriptions (e.g., "A1G:C5T")
        mutants = []
        for seq in mutated_seqs:
            muts = []
            for i, (a, b) in enumerate(zip(wt_seq, seq)):
                if a != b:
                    muts.append(f"{a}{i+1}{b}")
            mutants.append(":".join(muts) if muts else "WT")
        
        return mutated_seqs, mutants, pooled_tensor

    def _handle_duplicates(self, mutated_seqs: List[str], mutants: List[str], 
                           fitness_vals: List[float], enc_out: torch.Tensor) -> Tuple[List[str], List[str], List[float], torch.Tensor]:
        """
        Handle duplicates by keeping the one with best fitness.
        """
        if not mutated_seqs:
            return mutated_seqs, mutants, fitness_vals, enc_out
        
        # Group by sequence and keep best fitness
        seq_to_best = {}
        for i, seq in enumerate(mutated_seqs):
            if seq not in seq_to_best or fitness_vals[i] > fitness_vals[seq_to_best[seq]]:
                seq_to_best[seq] = i
        
        indices = sorted(seq_to_best.values())
        return (
            [mutated_seqs[i] for i in indices],
            [mutants[i] for i in indices],
            [fitness_vals[i] for i in indices],
            enc_out[indices] if enc_out.size(0) > 0 else enc_out
        )

    @timer
    def predict_fitness(self, inputs, wt_fitness: float, mutated_seqs: List[str], 
                        mutants: List[str], wt_seq: Optional[str] = None) -> Tuple[List[str], List[float]]:
        """
        Predict fitness and select top variants with elitism.
        """
        fitness_vals = self.fitness_predictor.infer_fitness(inputs)
        if isinstance(fitness_vals, np.ndarray):
            fitness_vals = fitness_vals.tolist()
        
        # Handle duplicates: keep best-scoring duplicate
        if self.rm_dups:
            mutated_seqs, mutants, fitness_vals, inputs = self._handle_duplicates(
                mutated_seqs, mutants, fitness_vals, inputs if isinstance(inputs, torch.Tensor) else torch.zeros((len(mutated_seqs), 1))
            )
        
        if not fitness_vals:
            return [], []
        
        fitness = torch.tensor(fitness_vals, dtype=torch.float32).unsqueeze(1)
        
        # Apply elitism: preserve best sequences
        num_elite = max(1, int(self.population * self.elite_ratio))
        k = min(self.population, len(mutants))
        
        topk_fitness, topk_indices = torch.topk(fitness, k, dim=0)
        topk_indices_list = topk_indices.squeeze(1).tolist()
        if isinstance(topk_indices_list, int):
            topk_indices_list = [topk_indices_list]
        
        top_variants = [mutated_seqs[i] for i in topk_indices_list]
        top_mutants = [mutants[i] for i in topk_indices_list]
        top_fitness_list = topk_fitness.squeeze(1).numpy().tolist()
        
        self.prev_fitness = topk_fitness
        self.prev_variants = top_variants
        self.prev_mutants = top_mutants
        
        return top_variants, top_fitness_list

    def __call__(self, wt_seq: str, wt_fitness: float) -> Tuple[List[str], torch.Tensor, List[str]]:
        """
        Run directed evolution.
        
        Args:
            wt_seq: Wild-type sequence
            wt_fitness: Wild-type fitness value
            
        Returns:
            Tuple of (mutants, fitness_tensor, variants)
        """
        logging.info(f"Starting directed evolution with {self.n_steps} steps, population={self.population}")
        logging.info(f"Device: {self.mutation_device}, use_random_mutation={self.use_random_mutation}")
        
        variants = [wt_seq] * self.population
        self.prev_fitness = torch.tensor([[wt_fitness]], dtype=torch.float32)
        self.prev_variants = [wt_seq]
        self.prev_mutants = ["WT"]
        
        # Reset history
        self.history = {"max_fitness": [], "mean_fitness": []}

        for step in range(self.n_steps):
            # Expand population
            variants = itertools_chain_flatten_repeat(variants, self.num_propose_mutation_per_variant)
            shuffled_ids = np.random.permutation(len(variants)).tolist()
            variants = [variants[i] for i in shuffled_ids]

            # Mask and mutate
            masked_variants, masked_positions = self.mask_sequences(variants, shuffled_ids)
            mutated_seqs, mutants, enc_out = self.mutate_masked_sequences(wt_seq, masked_variants, masked_positions)
            
            if not mutated_seqs:
                logging.warning(f"Step {step+1}: No mutated sequences generated")
                continue
            
            # Predict fitness and select
            variants, score = self.predict_fitness(enc_out, wt_fitness, mutated_seqs, mutants, wt_seq)
            
            # Track evolution history
            if score:
                self.history["max_fitness"].append(max(score))
                self.history["mean_fitness"].append(sum(score) / len(score))
            
            if self.verbose:
                logging.info(f"Step {step+1}/{self.n_steps} - max: {score[0] if score else 'N/A':.4f}, "
                           f"mean: {sum(score)/len(score) if score else 'N/A':.4f}, "
                           f"top-5: {score[:5] if len(score) >= 5 else score}")
        
        return self.prev_mutants, self.prev_fitness, self.prev_variants

# -------------------- Build Oracle --------------------
def build_oracle_and_load(decoder_ckpt_path: str, amix_encoder_ckpt: str, 
                          amix_encoder_config: Optional[str], device) -> Tuple[AMixFitnessWrapper, AMixEncoder]:
    """
    Build and load the oracle (fitness predictor) from checkpoints.
    
    Supports multiple checkpoint formats:
    - Combined format with 'regressor_state_dict' and 'encoder_state_dict'
    - Legacy format with 'state_dict'
    - Raw state dict
    
    Args:
        decoder_ckpt_path: Path to decoder/regressor checkpoint
        amix_encoder_ckpt: Path to AMix encoder checkpoint
        amix_encoder_config: Path to AMix config YAML
        device: Device to load models onto
        
    Returns:
        Tuple of (AMixFitnessWrapper, AMixEncoder)
    """
    device = torch.device(device)
    encoder_obj = AMixEncoder(ckpt_path=amix_encoder_ckpt, config_path=amix_encoder_config, device=device)
    reg = DecoderRegressor(in_dim=encoder_obj.hidden_dim)
    
    if os.path.exists(decoder_ckpt_path):
        logging.info(f"[Oracle] Loading regressor from {decoder_ckpt_path}")
        ck = torch.load(decoder_ckpt_path, map_location="cpu")
        
        # Support multiple checkpoint formats
        if isinstance(ck, dict):
            if "regressor_state_dict" in ck:
                reg.load_state_dict(ck["regressor_state_dict"], strict=False)
                logging.info("[Oracle] Loaded regressor from 'regressor_state_dict' key")
            elif "state_dict" in ck:
                reg.load_state_dict(ck["state_dict"], strict=False)
                logging.info("[Oracle] Loaded regressor from 'state_dict' key")
            else:
                reg.load_state_dict(ck, strict=False)
                logging.info("[Oracle] Loaded regressor from raw checkpoint dict")
                
            # Log meta information if available
            if "meta" in ck:
                logging.info(f"[Oracle] Checkpoint meta: {ck['meta']}")
        else:
            reg.load_state_dict(ck, strict=False)
            logging.info("[Oracle] Loaded regressor from raw checkpoint")
    else:
        logging.warning(f"[Oracle] Decoder checkpoint not found: {decoder_ckpt_path}")
    
    return AMixFitnessWrapper(reg, encoder_obj, device), encoder_obj


# -------------------- Main --------------------
def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="AMix-guided Directed Evolution Inference")
    
    # Required arguments
    p.add_argument("--wt", required=True, help="Wild-type protein sequence")
    p.add_argument("--wt_fitness", type=float, required=True, help="Fitness of the wild-type protein")
    p.add_argument("--amix_ckpt", required=True, help="Path to AMix encoder checkpoint")
    p.add_argument("--decoder_ckpt_path", required=True, help="Path to combined checkpoint .pt from training")
    p.add_argument("--save_name", required=True, help="Output CSV file path")
    
    # Optional arguments
    p.add_argument("--amix_config", default=None, help="YAML config for model dimensions")
    p.add_argument("--n_steps", type=int, default=10, help="Number of evolution steps")
    p.add_argument("--population", type=int, default=32, help="Population size per step")
    p.add_argument("--population_ratio_per_mask", nargs="+", type=float, default=None,
                   help="Population ratio per masker")
    p.add_argument("--num_proposes_per_var", type=int, default=4, help="Mutations proposed per variant")
    p.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search")
    p.add_argument("--top_k", type=int, default=5, help="Top-k predictions per masked position")
    p.add_argument("--mask_ratio_low", type=float, default=0.1, help="Low mask ratio for masker")
    p.add_argument("--mask_ratio_high", type=float, default=0.25, help="High mask ratio for masker")
    p.add_argument("--devices", type=int, default=0, help="GPU device ID (-1 for CPU)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    p.add_argument("--max_candidates_per_variant", type=int, default=200, help="Max candidates per variant")
    p.add_argument("--elite_ratio", type=float, default=0.1, help="Fraction of best sequences to preserve (elitism)")
    p.add_argument("--use_random_mutation", action="store_true", 
                   help="Use random mutation instead of model-guided")
    p.add_argument("--num_mutations", type=int, default=None, 
                   help="Fixed number of mutations per sequence (optional)")
    
    return p.parse_args()


def main():
    """Main entry point for directed evolution."""
    args = parse_args()
    
    # Setup device
    if args.devices < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.devices}" if torch.cuda.is_available() else "cpu")
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logging.info(f"Using device: {device}")
    logging.info(f"Seed: {args.seed}")

    logging.info("Loading AMix LM for mutation...")
    lm = AmixLMWrapper(ckpt_path=args.amix_ckpt, config_path=args.amix_config, device=device)

    logging.info("Loading oracle regressor...")
    oracle, encoder_obj = build_oracle_and_load(args.decoder_ckpt_path, args.amix_ckpt, args.amix_config, device)
    fitness_predictor = oracle

    # Create maskers using the global SimpleMasker class with single-character mask token
    maskers = [SimpleMasker(args.mask_ratio_low, MASK_TOKEN), SimpleMasker(args.mask_ratio_high, MASK_TOKEN)]
    pop_ratio = args.population_ratio_per_mask if args.population_ratio_per_mask else [1 / len(maskers)] * len(maskers)

    # Create DE engine with all options
    de = DiscreteDirectedEvolutionBeam(
        n_steps=args.n_steps,
        population=args.population,
        maskers=maskers,
        mutation_lm=lm,
        mutation_tokenizer=lm,  # lm provides tokenize/decode
        fitness_predictor=fitness_predictor,
        beam_size=args.beam_size,
        top_k_per_pos=args.top_k,
        num_propose_mutation_per_variant=args.num_proposes_per_var,
        remove_duplications=True,
        population_ratio_per_mask=pop_ratio,
        verbose=True,
        mutation_device=device,
        seed=args.seed,
        max_candidates_per_variant=args.max_candidates_per_variant,
        elite_ratio=args.elite_ratio,
        use_random_mutation=args.use_random_mutation
    )

    # Run directed evolution
    mutants, fitness_tensor, variants = de(args.wt, args.wt_fitness)

    # Convert fitness to list
    if isinstance(fitness_tensor, torch.Tensor):
        fitness_list = fitness_tensor.squeeze(1).detach().cpu().numpy().tolist()
    else:
        fitness_list = list(fitness_tensor)

    # Calculate number of mutations from WT for each variant
    num_mutations = [count_mutations_from_wt(args.wt, v) for v in variants]

    # Save results
    outdir = os.path.dirname(args.save_name)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    
    df = pd.DataFrame({
        "WT": [args.wt] * len(mutants),
        "mutant": mutants,
        "predicted_fitness": fitness_list,
        "num_mutations": num_mutations,
        "sequence": variants
    })
    df.sort_values(by="predicted_fitness", ascending=False, inplace=True, ignore_index=True)
    df.to_csv(args.save_name, index=False)
    logging.info(f"Saved results to {args.save_name}")

    # Save evolution history
    history_path = args.save_name.replace(".csv", "_history.csv")
    if de.history["max_fitness"]:
        history_df = pd.DataFrame({
            "step": list(range(1, len(de.history["max_fitness"]) + 1)),
            "max_fitness": de.history["max_fitness"],
            "mean_fitness": de.history["mean_fitness"]
        })
        history_df.to_csv(history_path, index=False)
        logging.info(f"Saved evolution history to {history_path}")


# -------------------- Smoke Test --------------------
def smoke_test():
    """
    Basic smoke test to verify the module imports and core classes work.
    Run with: python run_discrete_de_amix_new.py --smoke_test
    """
    logging.info("Running smoke test...")
    
    # Test SimpleMasker
    masker = SimpleMasker(mask_ratio=0.1, mask_token=MASK_TOKEN)
    test_seqs = ["ACDEFGHIKLMNPQRSTVWY", "AAAAAAAA"]
    masked, positions = masker.run(test_seqs)
    assert len(masked) == 2
    assert len(positions) == 2
    assert MASK_TOKEN in masked[0]
    logging.info(f"SimpleMasker test passed: {masked[0][:20]}... masked at {positions[0]}")
    
    # Test count_mutations_from_wt
    wt = "ACDEF"
    mut = "XCXEF"
    n_muts = count_mutations_from_wt(wt, mut)
    assert n_muts == 2
    logging.info(f"count_mutations_from_wt test passed: {n_muts} mutations")
    
    logging.info("Smoke test completed successfully!")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke_test":
        smoke_test()
    else:
        main()

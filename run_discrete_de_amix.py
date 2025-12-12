#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from typing import List, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from lightning.pytorch import LightningModule

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.append(_here)
repo_root = os.path.abspath(os.path.join(_here, ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from de.common.utils import set_seed, enable_full_deterministic
from de.directed_evolution import DiscreteDirectedEvolution2
from de.samplers.maskers import RandomMasker2, ImportanceMasker2
from de.predictors.oracle import ESM1b_Landscape

# =============================================
# Utility functions
# =============================================
def load_model_config(path):
    """Load model configuration from JSON or YAML file."""
    if path is None or not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            if path.endswith(".json"):
                import json
                return json.load(f)
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[Error] load_config failed: {e}")
        return None

# =============================================
# Output classes
# =============================================
class AMixOutput:
    """
    Output class compatible with ESM2's MaskedLMOutput.
    Contains logits and hidden_states attributes.
    """
    def __init__(self, logits, hidden_states):
        self.logits = logits
        self.hidden_states = [hidden_states]  # List format like ESM2

# =============================================
# 🧬 AMix Encoder
# =============================================
class AMixEncoder(nn.Module):
    def __init__(self, ckpt_path, config_path=None, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = ckpt_path

        # -------- Load configuration (supports JSON/YAML/no config) --------
        self.config = load_model_config(config_path)
        if self.config is None:
            print(f"[Warning] config not found or invalid at {config_path}, using defaults.")
            self.config = {"hidden_dim": 1280, "num_layers": 12, "vocab_size": 30, "dropout": 0.1}

        hidden_dim = self.config.get("hidden_dim", 1280)
        vocab_size = self.config.get("vocab_size", 30)
        num_layers = self.config.get("num_layers", 12)

        # -------- Model architecture --------
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=num_layers
        )

        self.to(self.device)

        # -------- Load checkpoint --------
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                state_dict = torch.load(ckpt_path, map_location=self.device)
                if "state_dict" in state_dict:
                    state_dict = {k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()}
                self.load_state_dict(state_dict, strict=False)
                print(f">> Loaded encoder weights from {ckpt_path}")
            except Exception as e:
                print(f"[Warning] Failed to load encoder checkpoint: {e}")
        else:
            print(f"[Warning] Encoder checkpoint not found: {ckpt_path}, using random init.")

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.encoder_layers(x)
        return x.mean(dim=1)

# =============================================
# 🧬 AMix Mutation Model (ESM2-compatible wrapper)
# =============================================
class AMixMutationModel(nn.Module):
    """
    AMix wrapper that provides ESM2-compatible interface for mutation model.
    Implements tokenize(), decode(), and forward() methods compatible with
    the Directed Evolution framework.
    """
    def __init__(self, ckpt_path, config_path=None, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # -------- Load configuration --------
        self.config = load_model_config(config_path)
        if self.config is None:
            print(f"[Warning] config not found or invalid at {config_path}, using defaults.")
            self.config = {"hidden_dim": 1280, "num_layers": 12, "vocab_size": 30, "dropout": 0.1}
        
        hidden_dim = self.config.get("hidden_dim", 1280)
        vocab_size = self.config.get("vocab_size", 30)
        num_layers = self.config.get("num_layers", 12)
        
        # -------- AMix amino acid mapping --------
        self.amino2id = {a: i + 1 for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        self.id2amino = {i: a for a, i in self.amino2id.items()}
        self.pad_id = 0
        self.mask_token = "<mask>"
        # Vocabulary structure:
        # - Position 0: padding token
        # - Positions 1-20: amino acids (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)
        # - Positions 21-29: reserved (from original vocab_size=30)
        # - Position 30 (vocab_size): mask token
        self.mask_id = vocab_size
        
        # -------- Model architecture --------
        # Total vocabulary size includes all positions from 0 to vocab_size (inclusive)
        self.total_vocab_size = vocab_size + 1
        self.embedding = nn.Embedding(self.total_vocab_size, hidden_dim, padding_idx=self.pad_id)
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=num_layers
        )
        
        # Linear layer for logits (tied to embedding weights)
        self.lm_head = nn.Linear(hidden_dim, self.total_vocab_size, bias=False)
        
        self.to(self.device)
        
        # -------- Load checkpoint --------
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                if "state_dict" in checkpoint:
                    processed_state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
                else:
                    processed_state_dict = checkpoint
                self.load_state_dict(processed_state_dict, strict=False)
                print(f">> Loaded AMixMutationModel weights from {ckpt_path}")
            except Exception as e:
                print(f"[Warning] Failed to load checkpoint: {e}")
        else:
            print(f"[Warning] Checkpoint not found: {ckpt_path}, using random init.")
        
        # Make self act as tokenizer for compatibility
        self.tokenizer = self
    
    def tokenize(self, inputs: List[str]):
        """
        Tokenize sequences to be compatible with ESM2 interface.
        Returns a dict with 'input_ids' and 'attention_mask'.
        """
        if not inputs:
            # Handle empty input list
            return {
                "input_ids": torch.empty((0, 0), dtype=torch.long, device=self.device),
                "attention_mask": torch.empty((0, 0), dtype=torch.long, device=self.device)
            }
        
        # First, we need to determine the max length in terms of tokens (not chars)
        token_sequences = []
        for seq in inputs:
            ids = []
            i = 0
            while i < len(seq):
                # Check if current position starts with mask token
                if seq[i:i+len(self.mask_token)] == self.mask_token:
                    ids.append(self.mask_id)
                    i += len(self.mask_token)
                else:
                    # Use pad_id for unknown amino acids (silent fallback for compatibility)
                    ids.append(self.amino2id.get(seq[i], self.pad_id))
                    i += 1
            token_sequences.append(ids)
        
        # Handle case where all sequences are empty
        max_len = max((len(ids) for ids in token_sequences), default=0)
        if max_len == 0:
            max_len = 1  # Ensure at least 1 position for empty sequences
        
        input_ids = torch.full((len(inputs), max_len), fill_value=self.pad_id, 
                              dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((len(inputs), max_len), dtype=torch.long, device=self.device)
        
        for i, ids in enumerate(token_sequences):
            if len(ids) > 0:
                input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
                attention_mask[i, :len(ids)] = 1
        
        # Return BatchEncoding-like dict
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def decode(self, tokens: torch.Tensor) -> List[str]:
        """
        Decode token IDs back to sequences.
        Compatible with ESM2's batch_decode interface.
        """
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        
        sequences = []
        for row in tokens:
            seq = []
            for token_id in row.tolist():
                if token_id == self.pad_id:
                    continue
                elif token_id == self.mask_id:
                    seq.append(self.mask_token)
                else:
                    seq.append(self.id2amino.get(token_id, ''))
            sequences.append(''.join(seq))
        
        return sequences
    
    def forward(self, inputs):
        """
        Forward pass returning ESM2-compatible output.
        Returns an object with 'logits' and 'hidden_states' attributes.
        """
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
        else:
            input_ids = inputs
        
        input_ids = input_ids.to(self.device)
        
        # Get embeddings and encode
        x = self.embedding(input_ids)  # [B, L, D]
        hidden_states = self.encoder_layers(x)  # [B, L, D]
        
        # Get logits for each position
        logits = self.lm_head(hidden_states)  # [B, L, V]
        
        # Return object with logits and hidden_states attributes (like MaskedLMOutput)
        return AMixOutput(logits, hidden_states)

# =============================================
# 🧬 AMix Decoder
# =============================================
class AMixDecoderModule(LightningModule):
    def __init__(self, encoder, dec_hidden_dim=1280, lr=5e-5):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Linear(dec_hidden_dim, 1)
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        emb = self.encoder(x)
        return self.decoder(emb)

    def training_step(self, batch, batch_idx):
        x, y = batch["input_ids"], batch["fitness"]
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["input_ids"], batch["fitness"]
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

# =============================================
# 🧬 Fitness Wrapper
# =============================================
class AMixFitnessWrapper:
    def __init__(self, amix_module: AMixDecoderModule, device="cpu"):
        self.model = amix_module.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.amino2id = {a: i + 1 for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}

    def seqs_to_input_ids(self, seqs: List[str]) -> torch.Tensor:
        max_len = max(len(s) for s in seqs)
        batch = torch.zeros((len(seqs), max_len), dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            ids = [self.amino2id.get(ch, 0) for ch in s]
            batch[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
        return batch

    @torch.inference_mode()
    def predict(self, seqs: List[str]) -> np.ndarray:
        if len(seqs) == 0:
            return np.array([])
        inputs = self.seqs_to_input_ids(seqs)
        out = self.model(inputs)
        out = out.detach().cpu().numpy().reshape(len(seqs), -1)
        return out[:, 0] if out.shape[1] == 1 else out

    # Make interface compatible with DE framework
    predict_fitness = predict
    infer_fitness = predict
    __call__ = predict

# =============================================
# Initialization functions
# =============================================
def initialize_mutation_model(args, device):
    # Use AMixMutationModel instead of ESM2 for compatibility
    encoder_config = getattr(args, "encoder_config", None)
    model = AMixMutationModel(
        ckpt_path=args.encoder_ckpt_path or args.decoder_ckpt_path,
        config_path=encoder_config,
        device=device
    )
    model.eval()
    tokenizer = model.tokenizer  # AMixMutationModel provides its own tokenizer interface
    return model, tokenizer

def initialize_maskers(args):
    imp_masker = ImportanceMasker2(args.k, max_subs=args.num_masked_tokens, low_importance_mask=not args.mask_high_importance)
    rand_masker = RandomMasker2(args.k, max_subs=args.num_masked_tokens)
    return [rand_masker, imp_masker]

def initialize_oracle(args, device):
    decoder_ckpt = args.decoder_ckpt_path

    # 用 decoder checkpoint 初始化 encoder（兼容 AMix）
    encoder_obj = AMixEncoder(ckpt_path=decoder_ckpt, device=device)

    # 用 encoder 初始化 AMixDecoderModule 并加载 checkpoint
    amix_model = AMixDecoderModule.load_from_checkpoint(
        decoder_ckpt,
        encoder=encoder_obj,
        dec_hidden_dim=args.dec_hidden_dim,
        map_location=device
    )

    # 返回包装类，保证 infer_fitness 方法存在
    return AMixFitnessWrapper(amix_model, device=device)

def initialize_fitness_predictor(args, device):
    decoder_ckpt = args.decoder_ckpt_path
    encoder_ckpt = args.encoder_ckpt_path or decoder_ckpt
    encoder_config = getattr(args, "encoder_config", None)

    # 初始化 Encoder
    encoder_obj = AMixEncoder(ckpt_path=encoder_ckpt, config_path=encoder_config, device=device)

    # 尝试加载 Decoder
    try:
        amix_module = AMixDecoderModule.load_from_checkpoint(
            decoder_ckpt, encoder=encoder_obj, dec_hidden_dim=args.dec_hidden_dim, map_location=device
        )
    except Exception as e:
        print(f"[Info] load_from_checkpoint failed with encoder: {e}\nRetrying without encoder...")
        amix_module = AMixDecoderModule.load_from_checkpoint(decoder_ckpt, map_location=device)

    print(">> AMix fitness predictor initialized successfully.")
    return AMixFitnessWrapper(amix_module, device=device)

# =============================================
# Save results to CSV
# =============================================
def save_results(wt_seqs, mutants, score, valid_score, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame({"WT": wt_seqs, "mutants": mutants, "score": score, "orc. score": valid_score})
    df.sort_values(by=["orc. score"], ascending=False, inplace=True, ignore_index=True)
    df.to_csv(output_path, index=False)

# =============================================
# Main logic
# =============================================
def main(args):
    set_seed(args.seed) if args.set_seed_only else enable_full_deterministic(args.seed)
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    device = torch.device("cpu" if args.devices == "-1" else f"cuda:{args.devices}")

    print(f">> Loading AMix decoder checkpoint from {args.decoder_ckpt_path}")
    mutation_model, mutation_tokenizer = initialize_mutation_model(args, device)
    fitness_predictor = initialize_fitness_predictor(args, device)
    oracle = initialize_oracle(args, device)
    maskers = initialize_maskers(args)

    result_dir = os.path.join(args.result_dir, args.task)
    log_dir = os.path.join(args.log_dir, args.task)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    direct_evo = DiscreteDirectedEvolution2(
        n_steps=args.n_steps,
        population=args.population,
        maskers=maskers,
        mutation_model=mutation_model,
        mutation_tokenizer=mutation_tokenizer,
        fitness_predictor=fitness_predictor,
        remove_duplications=args.rm_dups,
        k=args.k,
        population_ratio_per_mask=args.population_ratio_per_mask,
        num_propose_mutation_per_variant=args.num_proposes_per_var,
        verbose=args.verbose,
        mutation_device=device,
        log_dir=log_dir,
        seed=args.seed,
    )

    wt_seq = args.wt
    wt_fitness = args.wt_fitness
    mutants, pred_fitness_tensor, variants = direct_evo(wt_seq, wt_fitness)

    pred_fitness = np.asarray(pred_fitness_tensor).reshape(-1).tolist()
    valid_fitness = oracle.infer_fitness(variants)
    if isinstance(valid_fitness, torch.Tensor):
        valid_fitness = valid_fitness.detach().cpu().numpy().reshape(-1).tolist()

    filepath = os.path.join(result_dir, args.save_name)
    save_results([wt_seq] * len(mutants), mutants, pred_fitness, valid_fitness, filepath)
    print(f">> Results saved to {filepath}")

# =============================================
# Argument parsing
# =============================================
def parse_args():
    parser = argparse.ArgumentParser(description="Run Discrete Directed Evolution with AMix decoder checkpoint")
    parser.add_argument("--wt", type=str, required=True)
    parser.add_argument("--wt_fitness", type=float, required=True)
    parser.add_argument("--task", type=str, required=True,
                        choices=["AAV", "avGFP", "TEM", "E4B", "UBE2I", "LGK", "Pab1", "AMIE"])
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--num_proposes_per_var", type=int, default=4)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--rm_dups", action="store_true")
    parser.add_argument("--population_ratio_per_mask", nargs="+", type=float, default=None)
    parser.add_argument("--pretrained_mutation_name", type=str, default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--encoder_ckpt_path", type=str, default=None)
    parser.add_argument("--encoder_config", type=str, default=None)
    parser.add_argument("--decoder_ckpt_path", type=str, required=True)
    parser.add_argument("--dec_hidden_dim", type=int, default=1280)
    parser.add_argument("--num_masked_tokens", type=int, default=1)
    parser.add_argument("--mask_high_importance", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set_seed_only", action="store_true")
    parser.add_argument("--result_dir", type=str, default=os.path.abspath("./exps/results"))
    parser.add_argument("--log_dir", type=str, default=os.path.abspath("./exps/logs"))
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--devices", type=str, default="-1")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

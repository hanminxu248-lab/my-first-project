#!/usr/bin/env python3
# run_discrete_de_amix_beam.py
"""
Run Directed Evolution using a local AMix-like LM for mutation (with beam search)
and a trained decoder regressor as oracle.

This file is self-contained and uses a local LM wrapper (AmixLMWrapper) that:
 - tokenizes sequences as single-character tokens (one AA -> one token)
 - supports a mask token "[MASK]" mapped to a dedicated id
 - returns logits and hidden_states for beam search and pooling

It also loads a decoder checkpoint saved by your training script (combined .pt / .ckpt)
and uses AMixEncoder-compatible embedding loading for the oracle.
"""
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from copy import deepcopy
from operator import itemgetter
from typing import List, Union
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# -------------------- Utilities --------------------
def timer(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        t1 = time.time()
        logging.debug(f"{func.__name__} took {t1-t0:.3f}s")
        return res
    return wrapper


# -------------------- AMix Encoder (compat with train) --------------------
class AMixEncoder(nn.Module):
    def __init__(self, ckpt_path=None, config_path=None, device=None, load_embedding_only=False):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- load config ---
        import yaml
        config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

        self.hidden_dim = int(config.get("hidden_dim", 1280))
        self.vocab_size = int(config.get("vocab_size", 30))
        self.num_layers = int(config.get("num_layers", 12))
        self.nhead = int(config.get("nhead", 8))

        # --- model ---
        self.embedding = nn.Embedding(self.vocab_size + 1, self.hidden_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nhead,
            batch_first=True
        )
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # --- load full checkpoint ---
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                state = ckpt.get("state_dict", ckpt)

                new_state = {}
                for k, v in state.items():
                    k2 = k.replace("module.", "")
                    k2 = k2.replace("encoder.", "")
                    if k2.startswith("token_embedding"):
                        k2 = k2.replace("token_embedding", "embedding")
                    new_state[k2] = v

                self.load_state_dict(new_state, strict=False)
                logging.info("[AMixEncoder] fully loaded transformer weights.")
            except Exception as e:
                logging.warning(f"[AMixEncoder] failed full load: {e}")

        self.to(self.device)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.encoder_layers(x)
        cls_token = x[:, 0, :]   # CLS pooling
        return cls_token



# -------------------- Decoder Regressor (same as training) --------------------
class Attention1D(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x = x + h
        x = x + self.ff(x)
        return x


class DecoderRegressor(nn.Module):
    def __init__(self, in_dim=1280, hidden=512):
        super().__init__()
        self.att1d = Attention1D(in_dim, heads=8)
        self.reg = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        # x: [B, D]
        x = x.unsqueeze(1)        # -> [B, 1, D]
        x = self.att1d(x)         # -> [B, 1, D]
        x = x.squeeze(1)          # -> [B, D]
        return self.reg(x)


# -------------------- Fitness Wrapper (oracle) --------------------
class AMixFitnessWrapper:
    def __init__(self, regressor: nn.Module, encoder_obj: AMixEncoder = None, device="cpu"):
        self.device = torch.device(device)
        self.regressor = regressor.to(self.device)
        self.regressor.eval()
        self.encoder = encoder_obj.to(self.device) if encoder_obj is not None else None
        self.amino2id = {a: i + 1 for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        self.pad_id = 0

    def seqs_to_input_ids(self, seqs: List[str]) -> torch.Tensor:
        max_len = max(len(s) for s in seqs) if seqs else 0
        batch = torch.full((len(seqs), max_len), fill_value=self.pad_id, dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            ids = [self.amino2id.get(ch, self.pad_id) for ch in s]
            batch[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
        return batch

    @torch.inference_mode()
    def infer_fitness(self, inputs: Union[torch.Tensor, List[str]], batch_size: int = 64) -> np.ndarray:
        if isinstance(inputs, torch.Tensor):
            embs = inputs.to(self.device)
            outs = []
            for i in range(0, embs.size(0), batch_size):
                out = self.regressor(embs[i:i + batch_size])
                outs.append(out.detach().cpu())
            return torch.cat(outs, dim=0).squeeze(-1).numpy()
        if isinstance(inputs, list):
            if self.encoder is None:
                raise RuntimeError("No encoder to convert sequences to embeddings.")
            all_embs = []
            for i in range(0, len(inputs), batch_size):
                batch_seqs = inputs[i:i + batch_size]
                ids = self.seqs_to_input_ids(batch_seqs)
                emb = self.encoder(ids.to(self.encoder.device))
                all_embs.append(emb.detach().cpu())
            embs = torch.cat(all_embs, dim=0).to(self.device)
            outs = []
            for i in range(0, embs.size(0), batch_size):
                out = self.regressor(embs[i:i + batch_size])
                outs.append(out.detach().cpu())
            return torch.cat(outs, dim=0).squeeze(-1).numpy()
        raise ValueError("Unsupported input type for infer_fitness")


# -------------------- AMix LM Wrapper (local) --------------------
class AmixLMWrapper(nn.Module):
    """
    Local AMix LM: embedding + transformer encoder + linear tied to embedding
    - tokenization: single char -> single token
    - mask token: "[MASK]" -> mask_id
    - tokenize(sequences) -> {"input_ids": Tensor}
    - decode(ids_list) -> string sequence (ignores padding)
    - forward(input_ids) -> object with logits and hidden_states
    """
    def __init__(self, ckpt_path: str, config_path: str = None, device: torch.device = None):
        super().__init__()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        import yaml
        config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                logging.warning(f"[AmixLMWrapper] Failed to load config.yaml: {e}")
        self.hidden_dim = int(config.get("hidden_dim", 1280))
        self.vocab_size = int(config.get("vocab_size", 30))  # typical small protein vocab
        self.num_layers = int(config.get("num_layers", 12))
        self.nhead = int(config.get("nhead", 8))

        # amino mappings
        self.amino2id = {a: i + 1 for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        self.id2amino = {i: a for a, i in self.amino2id.items()}
        self.pad_id = 0
        self.mask_token = "[MASK]"
        # choose mask id beyond vocab (vocab_size reserved)
        self.mask_id = self.vocab_size
        # allocate embedding with capacity for pad + 1..vocab + mask token (hence +1)
        self.embedding = nn.Embedding(self.vocab_size + 1, self.hidden_dim, padding_idx=self.pad_id)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.nhead, batch_first=True)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # linear tied to embedding for logits
        # Note: using same weights as embedding for tied-weights (embedding: V+1 x D)
        # We'll compute logits via F.linear(hidden, embedding.weight)
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                mapped = {k.replace("module.", ""): v for k, v in state_dict.items()}
                for k in ["embedding.weight", "encoder.embedding.weight", "embeddings.weight", "token_embedding.weight"]:
                    if k in mapped and mapped[k].shape == self.embedding.weight.shape:
                        with torch.no_grad():
                            self.embedding.weight.copy_(mapped[k])
                        logging.info(f"[AmixLMWrapper] copied embedding from ckpt key {k}")
                        break
            except Exception as e:
                logging.warning(f"[AmixLMWrapper] failed to load ckpt: {e}")

        # move to device and set eval
        self.to(self.device)
        self.eval()

    def to(self, device):
        # override to move submodules
        self.device = device
        try:
            self.embedding.to(device)
            self.encoder_layers.to(device)
        except Exception:
            pass
        return self

    def eval(self):
        self.embedding.eval()
        self.encoder_layers.eval()

    def tokenize(self, sequences: List[str]) -> dict:
        """
        Tokenize a list of sequences into a dict with "input_ids" tensor on self.device.
        We map:
          - A/C/... to ids in amino2id
          - "[MASK]" (literal) to self.mask_id
        Each sequence is treated as a sequence of characters / mask tokens.
        """
        max_len = max(len(seq) for seq in sequences) if sequences else 0
        input_ids = torch.full((len(sequences), max_len), fill_value=self.pad_id, dtype=torch.long, device=self.device)
        for i, seq in enumerate(sequences):
            ids = []
            j = 0
            # simple parsing: if substring "[MASK]" occurs, treat as one token
            # otherwise single-char tokens
            while j < len(seq):
                if seq.startswith(self.mask_token, j):
                    ids.append(self.mask_id)
                    j += len(self.mask_token)
                else:
                    ch = seq[j]
                    ids.append(self.amino2id.get(ch, self.pad_id))
                    j += 1
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
        return {"input_ids": input_ids}

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token ids to sequence string.
        - mask_id -> "[MASK]"
        - pad (0) trimmed at end
        """
        out = []
        for tid in ids:
            if int(tid) == self.pad_id:
                # pad: skip (but continue; we'll strip trailing pads later)
                out.append("")  # placeholder
            elif int(tid) == self.mask_id:
                out.append(self.mask_token)
            else:
                out.append(self.id2amino.get(int(tid), ""))
        # join and strip trailing empty chars
        s = "".join([c for c in out])
        # remove trailing pad placeholders
        return s

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor):
        """
        input_ids: [B, L] torch.LongTensor on self.device
        returns object with:
          - logits: [B, L, V] where V == embedding.num_embeddings
          - hidden_states: list-like; here we'll return [hidden_states] for compatibility
        """
        input_ids = input_ids.to(self.device)
        x = self.embedding(input_ids)  # [B, L, D]
        hidden = self.encoder_layers(x)  # [B, L, D]
        # logits via tied weights: [B, L, V]
        logits = torch.nn.functional.linear(hidden, self.embedding.weight)  # uses embedding matrix [V, D]
        # wrap into simple object
        return type("Out", (), {"logits": logits, "hidden_states": [hidden]})


# -------------------- Beam-fill for one masked sequence --------------------
@timer
def beam_fill_masked_sequence(lm: AmixLMWrapper,
                              masked_seq: str,
                              masked_positions: List[int],
                              beam_size: int = 5,
                              top_k_per_pos: int = 5,
                              device: torch.device = torch.device("cpu")):
    """
    Fill multiple masked positions using beam search (left-to-right in masked_positions order).
    masked_seq: string where masked positions are represented by "[MASK]" tokens.
    masked_positions: list of positions (0-based character indices in the original sequence) to be filled,
                      but here we will rely on token-level mask occurrences so the caller must ensure
                      masks are inserted where intended.
    Returns: list of (filled_sequence_str, beam_score (logprob), pooled_hidden_tensor)
    """
    # Use lm.tokenize -> {"input_ids": Tensor}
    be = lm.tokenize([masked_seq])
    input_ids = be["input_ids"][0].clone().to(device)  # 1D tensor
    L = input_ids.size(0)

    # find token indices that equal mask_id
    mask_id = lm.mask_id
    token_mask_indices = (input_ids == mask_id).nonzero(as_tuple=False).squeeze(-1).tolist()
    if isinstance(token_mask_indices, int):
        token_mask_indices = [token_mask_indices]
    # If no explicit token masks found, try to identify based on mask substring (fallback)
    if len(token_mask_indices) == 0:
        # nothing to fill: return the original sequence's pooled state
        with torch.inference_mode():
            out = lm.forward(input_ids.unsqueeze(0))
            hidden = out.hidden_states[-1]  # [1,L,D]
            pooled = hidden[:, 0, :].detach().cpu()
        decoded = lm.decode(input_ids.tolist())
        return [(decoded, 0.0, pooled)]

    # Limit to the number of masked positions passed (defensive)
    token_mask_indices = token_mask_indices[:len(masked_positions)]

    # beams are (input_ids_cpu, score)
    beams = [(input_ids.clone().cpu(), 0.0)]

    for tok_pos in token_mask_indices:
        new_beams = []
        # prepare batch inputs
        batch_inputs = torch.stack([b[0] for b in beams], dim=0).to(device)  # [B_cur, L]
        # forward
        with torch.inference_mode():
            out = lm.forward(batch_inputs)
            logits = getattr(out, "logits", None)
            if logits is None:
                raise RuntimeError("LM did not return logits; cannot perform beam search.")
            # log probs at position
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [B_cur, L, V]
            pos_logprobs = log_probs[:, tok_pos, :]  # [B_cur, V]
            topk = torch.topk(pos_logprobs, k=min(top_k_per_pos, pos_logprobs.size(-1)), dim=-1)
            B_cur = batch_inputs.size(0)
            for i in range(B_cur):
                base_input = batch_inputs[i].detach().cpu()
                base_score = beams[i][1]
                vals = topk.values[i]   # [k]
                idxs = topk.indices[i]  # [k]
                for v, tid in zip(vals.tolist(), idxs.tolist()):
                    new_input = base_input.clone()
                    new_input[tok_pos] = int(tid)
                    new_score = base_score + float(v)
                    new_beams.append((new_input, new_score))
        # prune
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    # final forward to get pooled states
    final_inputs = torch.stack([b[0] for b in beams], dim=0).to(device)
    with torch.inference_mode():
        out_final = lm.forward(final_inputs)
        hidden = out_final.hidden_states[-1]  # [B, L, D]
        pooled = hidden[:, 0, :].detach().cpu()

    results = []
    for i, (inp_cpu, score) in enumerate(beams):
        seq = lm.decode(inp_cpu.tolist())
        results.append((seq, score, pooled[i]))
    return results


# -------------------- Directed Evolution Engine (uses beam-fill) --------------------
class DiscreteDirectedEvolutionBeam:
    def __init__(self,
                 n_steps: int,
                 population: int,
                 maskers: List[object],
                 mutation_lm: AmixLMWrapper,
                 mutation_tokenizer,
                 fitness_predictor: AMixFitnessWrapper,
                 beam_size: int = 5,
                 top_k_per_pos: int = 5,
                 num_propose_mutation_per_variant: int = 4,
                 remove_duplications: bool = True,
                 population_ratio_per_mask: List[float] = None,
                 verbose: bool = True,
                 mutation_device: Union[torch.device, str] = "cpu",
                 seed: int = 0,
                 max_candidates_per_variant: int = 200):
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

        self.mutation_logger = None
        self.prev_fitness = None
        self.prev_mutants = None
        self.prev_variants = None
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    @timer
    def mask_sequences(self, variants: List[str], ids: List[int]):
        num_variant = len(variants)
        masked_variants = []
        masked_positions = []
        begin_idx = 0
        for pop_ratio, masker in zip(self.population_ratio_per_mask, self.maskers):
            sub_population = int(num_variant * pop_ratio) if pop_ratio > 0 else 0
            sub_variants = variants[begin_idx: begin_idx + sub_population]
            sub_ids = ids[begin_idx: begin_idx + sub_population]
            begin_idx += sub_population
            if len(sub_variants) == 0:
                continue
            mv, mp = masker.run(sub_variants, sub_ids)
            masked_variants.extend(mv)
            masked_positions.extend(mp)
        if begin_idx < num_variant:
            rem = variants[begin_idx:]
            rem_mv, rem_mp = self.maskers[-1].run(rem, ids[begin_idx:])
            masked_variants.extend(rem_mv)
            masked_positions.extend(rem_mp)
        return masked_variants, masked_positions

    @timer
    def mutate_masked_sequences(self, wt_seq: str, masked_variants: List[str], masked_positions: List[List[int]]):
        assert len(masked_variants) == len(masked_positions)
        all_candidates = []
        for mv, pos in zip(masked_variants, masked_positions):
            try:
                candidates = beam_fill_masked_sequence(
                    lm=self.mutation_lm,
                    masked_seq=mv,
                    masked_positions=pos,
                    beam_size=self.beam_size,
                    top_k_per_pos=self.top_k_per_pos,
                    device=self.mutation_device
                )
            except Exception as e:
                logging.warning(f"Beam fill failed for {mv[:20]}... : {e}. Falling back to argmax.")
                # fallback to single forward + argmax
                be = self.mutation_lm.tokenize([mv])
                input_ids = be["input_ids"].to(self.mutation_device)
                with torch.inference_mode():
                    out = self.mutation_lm.forward(input_ids)
                logits = getattr(out, "logits")
                hidden = out.hidden_states[-1]
                # argmax per position
                predicted = torch.argmax(torch.nn.functional.log_softmax(logits, dim=-1), dim=-1)[0].detach().cpu()
                orig_ids = input_ids.detach().cpu()[0]
                mutated_ids = orig_ids.clone()
                # find mask token positions on orig_ids
                mask_positions = (orig_ids == self.mutation_lm.mask_id).nonzero(as_tuple=False).squeeze(-1).tolist()
                if isinstance(mask_positions, int):
                    mask_positions = [mask_positions]
                for i, p in enumerate(mask_positions):
                    mutated_ids[p] = int(predicted[p])
                seq = self.mutation_lm.decode(mutated_ids.tolist())
                pooled = hidden.mean(dim=1).detach().cpu()[0]
                candidates = [(seq, 0.0, pooled)]

            # append
            for seq, score, pooled in candidates:
                all_candidates.append((seq, score, pooled))
            if self.max_candidates_per_variant and len(candidates) > self.max_candidates_per_variant:
                all_candidates = all_candidates[: - (len(candidates) - self.max_candidates_per_variant)]

        if len(all_candidates) == 0:
            # fallback empty tensor
            return [], [], torch.zeros((0, self.fitness_predictor.regressor.net[0].in_features))

        mutated_seqs = [c[0] for c in all_candidates]
        # build mutants simple diff against wt_seq
        mutants = []
        for seq in mutated_seqs:
            muts = []
            # handle possibly different lengths: iterate up to min
            for i, (a, b) in enumerate(zip(wt_seq, seq)):
                if a != b:
                    muts.append(f"{a}{i+1}{b}")
            mutants.append(":".join(muts))
        pooled_list = [c[2] for c in all_candidates]
        pooled_tensor = torch.stack(pooled_list, dim=0) if len(pooled_list) > 0 else torch.zeros((0, pooled_list[0].shape[0] if pooled_list else 0))
        return mutated_seqs, mutants, pooled_tensor

    @timer
    def predict_fitness(self,
                        inputs: torch.Tensor,
                        wt_fitness: float,
                        mutated_seqs: List[str],
                        mutants: List[str],
                        wt_seq: str = None):
        fitness_vals = self.fitness_predictor.infer_fitness(inputs)  # np.ndarray
        fitness = torch.tensor(fitness_vals, dtype=torch.float32).unsqueeze(1)
        fitness = torch.concat([fitness, self.prev_fitness], dim=0)
        mutants = mutants + self.prev_mutants
        mutated_seqs = mutated_seqs + self.prev_variants

        k = self.population if len(mutants) >= self.population else len(mutants)
        topk_fitness, topk_indices = torch.topk(fitness, k, dim=0)
        top_fitness_score = topk_fitness.squeeze(1).numpy().tolist()
        top_indices = topk_indices.squeeze(1).numpy().tolist()

        n = 0
        if len(top_fitness_score) < self.population:
            n = self.population - len(top_fitness_score)
            top_fitness_score = [top_fitness_score[0] for _ in range(n)] + top_fitness_score
            top_indices = [top_indices[0] for _ in range(n)] + top_indices

        retriever = itemgetter(*top_indices) if len(top_indices) > 1 else (lambda x: x[top_indices[0]])
        top_variants = list(retriever(mutated_seqs))
        top_mutants = list(retriever(mutants))
        self.mutation_logger = self.mutants2logger(top_mutants)
        self.prev_fitness = topk_fitness
        self.prev_mutants = top_mutants[n:]
        self.prev_variants = top_variants[n:]
        return top_variants, top_fitness_score

    def __call__(self, wt_seq: str, wt_fitness: float):
        logging.info(f"Starting DE with WT length {len(wt_seq)}")
        variants = [wt_seq for _ in range(self.population)]
        self.mutation_logger = [{} for _ in range(self.population)]
        self.prev_fitness = torch.tensor([[wt_fitness]], dtype=torch.float32)
        self.prev_mutants = [""]
        self.prev_variants = [wt_seq]

        for step in range(self.n_steps):
            logging.info(f"===== Step {step + 1}/{self.n_steps} =====")
            variants = list(itertools_chain_flatten_repeat(variants, self.num_propose_mutation_per_variant))
            self.mutation_logger = list(itertools_chain_flatten_repeat(self.mutation_logger, self.num_propose_mutation_per_variant))
            shuffled_ids = np.random.permutation(len(variants)).tolist()
            retriever = itemgetter(*shuffled_ids) if len(shuffled_ids) > 1 else (lambda x: x[shuffled_ids[0]])
            shuffled_variants = list(retriever(variants))
            if step != 0:
                self.mutation_logger = list(retriever(self.mutation_logger))
            del retriever

            masked_variants, masked_positions = self.mask_sequences(shuffled_variants, shuffled_ids)
            mutated_seqs, mutants, enc_out = self.mutate_masked_sequences(wt_seq, masked_variants, masked_positions)

            if self.rm_dups:
                candidate_array = np.array(mutated_seqs)
                unique_cand, indices = np.unique(candidate_array, return_index=True)
                mutated_seqs = unique_cand.tolist()
                mutants = [mutants[i] for i in indices.tolist()]
                enc_out = enc_out[indices.tolist()]

            variants, score = self.predict_fitness(enc_out, wt_fitness, mutated_seqs, mutants, wt_seq)
            logging.info("Top fitness (sample): %s", score[:5])

        return self.prev_mutants, self.prev_fitness, variants

    def logger2mutants(self, num2convert: int):
        mutants = []
        for i in range(num2convert):
            mutant = ''
            for k, v in (self.mutation_logger[i].items() if i < len(self.mutation_logger) else []):
                mutant += v[0] + k + v[1] + ":"
            mutants.append(mutant[:-1])
        return mutants

    def mutants2logger(self, mutants: List[str]):
        logger = [{} for _ in range(len(mutants))]
        for idx, mutant in enumerate(mutants):
            if len(mutant) == 0:
                continue
            for m in mutant.split(":"):
                if len(m) < 3:
                    continue
                before, pos, after = m[0], m[1:-1], m[-1]
                logger[idx][pos] = [before, after]
        return logger


# small util used in above
def itertools_chain_flatten_repeat(items, times):
    out = []
    for i in items:
        for _ in range(times):
            out.append(deepcopy(i))
    return out


import itertools as _it
itertools_chain = _it.chain


# -------------------- Checkpoint loader for oracle --------------------
def build_oracle_and_load(decoder_ckpt_path, amix_encoder_ckpt, amix_encoder_config, device):
    device = torch.device(device)

    # ---- load full AMix encoder ----
    encoder_obj = AMixEncoder(
        ckpt_path=amix_encoder_ckpt,
        config_path=amix_encoder_config,
        device=device,
        load_embedding_only=False
    )

    # ---- load decoder regressor ----
    reg = DecoderRegressor(in_dim=encoder_obj.hidden_dim)

    if os.path.exists(decoder_ckpt_path):
        ck = torch.load(decoder_ckpt_path, map_location="cpu")
        if "regressor_state_dict" in ck:
            reg.load_state_dict(ck["regressor_state_dict"], strict=False)
        elif "state_dict" in ck:
            reg.load_state_dict(ck["state_dict"], strict=False)
        else:
            reg.load_state_dict(ck, strict=False)
        logging.info("[Oracle] regressor loaded.")
    else:
        logging.warning("[Oracle] decoder ckpt not found, using random weights.")

    oracle = AMixFitnessWrapper(
        regressor=reg,
        encoder_obj=encoder_obj,
        device=device
    )

    return oracle, encoder_obj



# -------------------- Main / CLI --------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wt", required=True)
    p.add_argument("--wt_fitness", type=float, required=True)
    p.add_argument("--amix_ckpt", required=True, help="local ckpt for embeddings (optional; used to populate embedding weights)")
    p.add_argument("--amix_config", default=None, help="yaml config for model dims")
    p.add_argument("--decoder_ckpt_path", required=True, help="path to combined checkpoint .pt produced by training script")
    p.add_argument("--save_name", required=True)
    p.add_argument("--n_steps", type=int, default=10)
    p.add_argument("--population", type=int, default=32)
    p.add_argument("--population_ratio_per_mask", nargs="+", type=float, default=None)
    p.add_argument("--num_proposes_per_var", type=int, default=4)
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--mask_ratio_low", type=float, default=0.1)
    p.add_argument("--mask_ratio_high", type=float, default=0.25)
    p.add_argument("--devices", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_candidates_per_variant", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:%d" % args.devices if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.info("Loading AMix LM for mutation...")
    lm = AmixLMWrapper(ckpt_path=args.amix_ckpt, config_path=args.amix_config, device=device)

    logging.info("Loading oracle regressor...")
    oracle, encoder_obj = build_oracle_and_load(args.decoder_ckpt_path, args.amix_ckpt, args.amix_config, device)
    fitness_predictor = oracle

    # maskers
    class SimpleMaskerLocal:
        def __init__(self, mask_ratio):
            self.mask_ratio = mask_ratio
            self.mask_token = "[MASK]"

        def run(self, sequences, ids=None):
            masked = []
            posis = []
            for s in sequences:
                n = len(s)
                k = max(1, int(round(n * self.mask_ratio)))
                ps = list(np.random.choice(n, size=k, replace=False))
                ls = list(s)
                # replace by literal "[MASK]" at chosen positions (this increases char-length)
                for p in ps:
                    ls[p] = self.mask_token
                masked.append("".join(ls))
                posis.append(sorted(ps))
            return masked, posis

    maskers = [SimpleMaskerLocal(args.mask_ratio_low), SimpleMaskerLocal(args.mask_ratio_high)]
    pop_ratio = args.population_ratio_per_mask if args.population_ratio_per_mask else [1 / len(maskers)] * len(maskers)

    # DE engine
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
        max_candidates_per_variant=args.max_candidates_per_variant
    )

    mutants, fitness_tensor, variants = de(args.wt, args.wt_fitness)

    if isinstance(fitness_tensor, torch.Tensor):
        fitness_list = fitness_tensor.squeeze(1).detach().cpu().numpy().tolist()
    else:
        fitness_list = list(fitness_tensor)

    outdir = os.path.dirname(args.save_name)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame({"WT": [args.wt] * len(mutants), "mutants": mutants, "pred_score": fitness_list, "sequence": variants})
    df.sort_values(by="pred_score", ascending=False, inplace=True, ignore_index=True)
    df.to_csv(args.save_name, index=False)
    logging.info(f"Saved results to {args.save_name}")


if __name__ == "__main__":
    main()

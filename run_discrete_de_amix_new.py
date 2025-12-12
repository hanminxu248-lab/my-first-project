#!/usr/bin/env python3
# run_discrete_de_amix_bfn.py
"""
Directed Evolution Inference Script
- Uses a local AMix-like LM for mutation (beam search)
- Uses a trained Decoder / ProfileBFN oracle for fitness prediction
- Fully respects the model config parameters from the provided YAML
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

def itertools_chain_flatten_repeat(items, times):
    out = []
    for i in items:
        for _ in range(times):
            out.append(deepcopy(i))
    return out

# -------------------- AMix Encoder --------------------
class AMixEncoder(nn.Module):
    def __init__(self, ckpt_path=None, config_path=None, device=None, load_embedding_only=False):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        import yaml
        config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        self.hidden_dim = int(config.get("hidden_dim", 1680))
        self.vocab_size = int(config.get("vocab_size", 30))
        self.num_layers = int(config.get("num_layers", 48))
        self.nhead = int(config.get("nhead", 40))
        self.intermediate_size = int(config.get("intermediate_size", 6720))

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, 
            nhead=self.nhead, 
            dim_feedforward=self.intermediate_size,
            batch_first=True
        )
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        if ckpt_path and os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                mapped = {k.replace("module.", ""): v for k, v in state_dict.items()}
                # try load embedding
                for k in ["embedding.weight", "encoder.embedding.weight"]:
                    if k in mapped and mapped[k].shape == self.embedding.weight.shape:
                        with torch.no_grad():
                            self.embedding.weight.copy_(mapped[k])
                        logging.info(f"[AMixEncoder] loaded embedding from {k}")
                        break
            except Exception as e:
                logging.warning(f"[AMixEncoder] failed to load ckpt: {e}")

        self.to(self.device)
        logging.info(f"[AMixEncoder] Configuration: hidden_dim={self.hidden_dim}, num_layers={self.num_layers}, "
                    f"nhead={self.nhead}, intermediate_size={self.intermediate_size}, vocab_size={self.vocab_size}")

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.encoder_layers(x)
        return x[:, 0, :]  # CLS pooling，与训练一致

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
    def __init__(self, regressor: nn.Module, encoder_obj: AMixEncoder=None, device="cpu"):
        self.device = torch.device(device)
        self.regressor = regressor.to(self.device)
        self.regressor.eval()
        self.encoder = encoder_obj.to(self.device) if encoder_obj else None
        self.amino2id = {a:i+1 for i,a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        self.pad_id = 0

    def seqs_to_input_ids(self, seqs):
        max_len = max(len(s) for s in seqs) if seqs else 0
        batch = torch.full((len(seqs), max_len), fill_value=self.pad_id, dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            batch[i,:len(s)] = torch.tensor([self.amino2id.get(c,self.pad_id) for c in s], device=self.device)
        return batch

    @torch.inference_mode()
    def infer_fitness(self, inputs, batch_size=64):
        if isinstance(inputs, list):
            if not self.encoder:
                raise RuntimeError("No encoder for sequence -> embedding conversion")
            all_embs = []
            for i in range(0,len(inputs), batch_size):
                ids = self.seqs_to_input_ids(inputs[i:i+batch_size])
                emb = self.encoder(ids)
                all_embs.append(emb.detach().cpu())
            embs = torch.cat(all_embs, dim=0).to(self.device)
        else:
            embs = inputs.to(self.device)
        outs = []
        for i in range(0, embs.size(0), batch_size):
            out = self.regressor(embs[i:i+batch_size])
            outs.append(out.detach().cpu())
        return torch.cat(outs, dim=0).squeeze(-1).numpy()

# -------------------- AMix LM Wrapper --------------------
class AmixLMWrapper(nn.Module):
    def __init__(self, ckpt_path:str, config_path:str=None, device:torch.device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import yaml
        config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path,"r") as f:
                config = yaml.safe_load(f)
        self.hidden_dim = int(config.get("hidden_dim",1680))
        self.vocab_size = int(config.get("vocab_size",30))
        self.num_layers = int(config.get("num_layers",48))
        self.nhead = int(config.get("nhead",40))
        self.intermediate_size = int(config.get("intermediate_size",6720))

        self.amino2id = {a:i+1 for i,a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        self.id2amino = {i:a for a,i in self.amino2id.items()}
        self.pad_id = 0
        self.mask_token = "*"
        self.mask_id = self.vocab_size
        self.embedding = nn.Embedding(self.vocab_size+1,self.hidden_dim,padding_idx=self.pad_id)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nhead,
            dim_feedforward=self.intermediate_size,
            batch_first=True
        )
        self.encoder_layers = nn.TransformerEncoder(encoder_layer,num_layers=self.num_layers)

        if ckpt_path and os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path,map_location="cpu")
                state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                for k in ["embedding.weight","encoder.embedding.weight"]:
                    if k in state_dict and state_dict[k].shape==self.embedding.weight.shape:
                        with torch.no_grad(): self.embedding.weight.copy_(state_dict[k])
                        logging.info(f"[LM] copied embedding {k}")
                        break
            except Exception as e:
                logging.warning(f"[LM] ckpt load failed: {e}")
        self.to(self.device)
        self.eval()
        logging.info(f"[AmixLMWrapper] Configuration: hidden_dim={self.hidden_dim}, num_layers={self.num_layers}, "
                    f"nhead={self.nhead}, intermediate_size={self.intermediate_size}, vocab_size={self.vocab_size}")

    def tokenize(self,sequences):
        max_len = max(len(s) for s in sequences)
        input_ids = torch.full((len(sequences),max_len),fill_value=self.pad_id,dtype=torch.long,device=self.device)
        for i,s in enumerate(sequences):
            ids = [self.mask_id if c==self.mask_token else self.amino2id.get(c,self.pad_id) for c in s]
            input_ids[i,:len(ids)] = torch.tensor(ids,device=self.device)
        return {"input_ids": input_ids}

    def decode(self,ids):
        out = []
        for tid in ids:
            if tid==self.pad_id: out.append("")
            elif tid==self.mask_id: out.append(self.mask_token)
            else: out.append(self.id2amino.get(tid,""))
        return "".join([c for c in out])

    @torch.inference_mode()
    def forward(self,input_ids:torch.Tensor):
        input_ids = input_ids.to(self.device)
        hidden = self.encoder_layers(self.embedding(input_ids))
        logits = torch.nn.functional.linear(hidden,self.embedding.weight)
        return type("Out",(),{"logits":logits,"hidden_states":[hidden]})

# -------------------- Beam Fill --------------------
@timer
def beam_fill_masked_sequence(lm, masked_seq, masked_positions, beam_size=5, top_k_per_pos=5, device="cpu"):
    be = lm.tokenize([masked_seq])
    input_ids = be["input_ids"][0].clone().to(device)
    mask_id = lm.mask_id
    mask_positions = (input_ids==mask_id).nonzero(as_tuple=False).squeeze(-1).tolist()
    if isinstance(mask_positions,int): mask_positions=[mask_positions]
    if not mask_positions:
        out = lm.forward(input_ids.unsqueeze(0))
        pooled = out.hidden_states[-1][:,0,:].detach().cpu()
        return [(lm.decode(input_ids.tolist()),0.0,pooled[0])]
    beams = [(input_ids.clone().cpu(),0.0)]
    for tok_pos in mask_positions[:len(masked_positions)]:
        new_beams=[]
        batch_inputs = torch.stack([b[0] for b in beams],dim=0).to(device)
        out = lm.forward(batch_inputs)
        logits = getattr(out,"logits")
        log_probs = torch.nn.functional.log_softmax(logits,dim=-1)
        pos_logprobs = log_probs[:,tok_pos,:]
        topk = torch.topk(pos_logprobs,k=min(top_k_per_pos,pos_logprobs.size(-1)),dim=-1)
        for i in range(batch_inputs.size(0)):
            base_input = batch_inputs[i].detach().cpu()
            base_score = beams[i][1]
            for v,tid in zip(topk.values[i].tolist(),topk.indices[i].tolist()):
                new_input = base_input.clone()
                new_input[tok_pos] = int(tid)
                new_beams.append((new_input,base_score+float(v)))
        new_beams.sort(key=lambda x:x[1],reverse=True)
        beams = new_beams[:beam_size]
    final_inputs = torch.stack([b[0] for b in beams],dim=0).to(device)
    out_final = lm.forward(final_inputs)
    hidden = out_final.hidden_states[-1]
    pooled = hidden[:,0,:].detach().cpu()
    results=[]
    for i,(inp_cpu,score) in enumerate(beams):
        seq = lm.decode(inp_cpu.tolist())
        results.append((seq,score,pooled[i]))
    return results

# -------------------- Directed Evolution --------------------
class DiscreteDirectedEvolutionBeam:
    def __init__(self,n_steps,population,maskers,mutation_lm,mutation_tokenizer,fitness_predictor,
                 beam_size=5,top_k_per_pos=5,num_propose_mutation_per_variant=4,remove_duplications=True,
                 population_ratio_per_mask=None,verbose=True,mutation_device="cpu",seed=0,max_candidates_per_variant=200):
        self.n_steps=n_steps
        self.population=population
        self.maskers=maskers
        self.mutation_lm=mutation_lm
        self.mutation_tokenizer=mutation_tokenizer
        self.fitness_predictor=fitness_predictor
        self.beam_size=beam_size
        self.top_k_per_pos=top_k_per_pos
        self.num_propose_mutation_per_variant=num_propose_mutation_per_variant
        self.rm_dups=remove_duplications
        self.population_ratio_per_mask=population_ratio_per_mask or [1/len(maskers) for _ in maskers]
        self.verbose=verbose
        self.mutation_device=torch.device(mutation_device) if not isinstance(mutation_device,torch.device) else mutation_device
        self.seed=seed
        self.max_candidates_per_variant=max_candidates_per_variant
        self.prev_fitness=None
        self.prev_mutants=None
        self.prev_variants=None
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    @timer
    def mask_sequences(self,variants,ids):
        masked_variants = []
        masked_positions = []

        num_variant = len(variants)
        for pop_ratio, masker in zip(self.population_ratio_per_mask, self.maskers):
            sub_population = int(num_variant * pop_ratio)  # 先定义 sub_population
            if sub_population <= 0:
                continue  # 如果分配到的子population为0就跳过

            sub_variants = variants[:sub_population]
            ids = list(range(sub_population))
            mv, mp = masker.run(sub_variants, ids)
            masked_variants.extend(mv)
            masked_positions.extend(mp)

        return masked_variants,masked_positions

    @timer
    def mutate_masked_sequences(self,wt_seq,masked_variants,masked_positions):
        all_candidates=[]
        for mv,pos in zip(masked_variants,masked_positions):
            try:
                candidates=beam_fill_masked_sequence(self.mutation_lm,mv,pos,self.beam_size,self.top_k_per_pos,self.mutation_device)
            except:
                candidates=[(mv,0.0,torch.zeros(self.fitness_predictor.encoder.hidden_dim))]
            all_candidates.extend(candidates[:self.max_candidates_per_variant])
        mutated_seqs=[c[0] for c in all_candidates]
        pooled_tensor=torch.stack([c[2] for c in all_candidates],dim=0)
        mutants=[]
        for seq in mutated_seqs:
            muts=[]
            for i,(a,b) in enumerate(zip(wt_seq,seq)):
                if a!=b: muts.append(f"{a}{i+1}{b}")
            mutants.append(":".join(muts))
        return mutated_seqs,mutants,pooled_tensor

    @timer
    def predict_fitness(self,inputs,wt_fitness,mutated_seqs,mutants,wt_seq=None):
        fitness_vals = self.fitness_predictor.infer_fitness(inputs)
        fitness = torch.tensor(fitness_vals,dtype=torch.float32).unsqueeze(1)
        k = self.population if len(mutants)>=self.population else len(mutants)
        topk_fitness,topk_indices = torch.topk(fitness,k,dim=0)
        top_variants=[mutated_seqs[i] for i in topk_indices.squeeze(1).tolist()]
        top_mutants=[mutants[i] for i in topk_indices.squeeze(1).tolist()]
        self.prev_fitness=topk_fitness
        self.prev_variants=top_variants
        self.prev_mutants=top_mutants
        return top_variants, topk_fitness.squeeze(1).numpy().tolist()

    def __call__(self,wt_seq,wt_fitness):
        variants=[wt_seq]*self.population
        self.prev_fitness=torch.tensor([[wt_fitness]],dtype=torch.float32)
        self.prev_variants=[wt_seq]
        self.prev_mutants=[""]

        for step in range(self.n_steps):
            variants = itertools_chain_flatten_repeat(variants,self.num_propose_mutation_per_variant)
            shuffled_ids = np.random.permutation(len(variants)).tolist()
            variants = [variants[i] for i in shuffled_ids]

            masked_variants, masked_positions = self.mask_sequences(variants, shuffled_ids)
            mutated_seqs, mutants, enc_out = self.mutate_masked_sequences(wt_seq, masked_variants, masked_positions)
            if self.rm_dups:
                _, idx = np.unique(mutated_seqs, return_index=True)
                mutated_seqs = [mutated_seqs[i] for i in idx]
                mutants = [mutants[i] for i in idx]
                enc_out = enc_out[idx]
            variants, score = self.predict_fitness(enc_out, wt_fitness, mutated_seqs, mutants, wt_seq)
            logging.info(f"Step {step+1}/{self.n_steps} top fitness sample: {score[:5]}")
        return self.prev_mutants,self.prev_fitness,self.prev_variants

# -------------------- Build Oracle --------------------
def build_oracle_and_load(decoder_ckpt_path, amix_encoder_ckpt, amix_encoder_config, device):
    device = torch.device(device)
    encoder_obj = AMixEncoder(ckpt_path=amix_encoder_ckpt, config_path=amix_encoder_config, device=device)
    reg = DecoderRegressor(in_dim=encoder_obj.hidden_dim)
    if os.path.exists(decoder_ckpt_path):
        ck = torch.load(decoder_ckpt_path,map_location="cpu")
        
        # Validate dimension compatibility
        if "meta" in ck:
            meta = ck["meta"]
            expected_dim = meta.get("encoder_hidden_dim", meta.get("dec_hidden_dim"))
            if expected_dim and expected_dim != encoder_obj.hidden_dim:
                logging.warning(f"[Oracle] ⚠️  Dimension mismatch: checkpoint expects hidden_dim={expected_dim} "
                              f"but current encoder has hidden_dim={encoder_obj.hidden_dim}. "
                              f"This may cause errors. Please check your config file.")
            else:
                logging.info(f"[Oracle] ✓ Dimension compatibility validated: hidden_dim={encoder_obj.hidden_dim}")
        
        if "regressor_state_dict" in ck:
            reg.load_state_dict(ck["regressor_state_dict"],strict=False)
        elif "state_dict" in ck:
            reg.load_state_dict(ck["state_dict"],strict=False)
        else:
            reg.load_state_dict(ck,strict=False)
        logging.info("[Oracle] regressor loaded")
    return AMixFitnessWrapper(reg, encoder_obj, device), encoder_obj

# -------------------- Main --------------------
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
            self.mask_token = "#"  # single-character mask token

        def run(self, sequences, ids=None):
            masked = []
            posis = []
            for s in sequences:
                n = len(s)
                if n == 0:
                    masked.append("")
                    posis.append([])
                    continue
                k = max(1, int(round(n * self.mask_ratio)))
                k = min(k, n)  # 防止 k > n
                ps = list(np.random.choice(n, size=k, replace=False))
                ls = list(s)
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

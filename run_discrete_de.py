import argparse
import numpy as np
import os
import pandas as pd
import torch
from typing import List, Union, Tuple

from de.common.utils import set_seed, enable_full_deterministic
from de.directed_evolution import DiscreteDirectedEvolution2
from de.samplers.maskers import RandomMasker2, ImportanceMasker2
from de.samplers.models.esm import ESM2
from de.predictors.attention.module import ESM2DecoderModule, ESM2_Attention
from de.predictors.oracle import ESM1b_Landscape, ESM1v
from amix_encoder import AMixEncoder   # 新增：AMix 支持


def parse_args():
    parser = argparse.ArgumentParser()

    # Wild-type
    parser.add_argument("--wt", type=str, help="Wild-type protein sequence.")
    parser.add_argument("--wt_fitness", type=float, help="Fitness of the wild-type protein.")

    # General task
    parser.add_argument(
        "--task",
        type=str,
        choices=["AAV", "avGFP", "TEM", "E4B", "UBE2I", "LGK", "Pab1", "AMIE"],
        help="Benchmark task.",
    )
    parser.add_argument("--n_steps", type=int, default=100, help="No. steps to run DE.")
    parser.add_argument("--population", type=int, default=128, help="Population per step.")
    parser.add_argument("--num_proposes_per_var", type=int, default=4, help="Proposed mutations per variant.")
    parser.add_argument("--k", type=int, default=1, help="Token length k.")
    parser.add_argument("--rm_dups", action="store_true", help="Remove duplicated variants.")
    parser.add_argument(
        "--population_ratio_per_mask",
        nargs="+",
        type=float,
        help="Population ratio per masker.",
    )

    # Encoder
    parser.add_argument(
        "--pretrained_mutation_name",
        type=str,
        default="facebook/esm2_t12_35M_UR50D",
        help="Mutation model (ESM name or 'amix').",
    )
    parser.add_argument("--dec_hidden_size", type=int, default=512, help="Decoder hidden size.")
    parser.add_argument("--predictor_ckpt_path", type=str, help="Fitness predictor checkpoint path.")

    # Mask
    parser.add_argument("--num_masked_tokens", type=int, default=1, help="Masked tokens per step.")
    parser.add_argument("--mask_high_importance", action="store_true", help="Mask high-importance tokens.")

    # Env
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set_seed_only", action="store_true")
    parser.add_argument("--result_dir", type=str, default=os.path.abspath("./exps/results"))
    parser.add_argument("--log_dir", type=str, default=os.path.abspath("./exps/logs"))
    parser.add_argument("--save_name", type=str, help="Result csv file name.")
    parser.add_argument("--devices", type=str, default="-1", help="Devices, comma separated.")
    parser.add_argument("--esm1v_seed", type=int, choices=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    return args


def extract_from_csv(csv_file: str, top_k: int = -1) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(csv_file)
    if top_k != -1:
        df = df.nlargest(top_k, columns="fitness")
    targets = df["fitness"].to_list()
    seqs = df.sequence.tolist()
    return seqs, targets


def initialize_mutation_model(args, device: torch.device):
    if args.pretrained_mutation_name.lower() == "amix":
        print(">> 使用 AMix Encoder for mutation")
        model = AMixEncoder(
            ckpt_path="./AMix-1-main/checkpoints/amix.ckpt",
            hidden_dim=args.dec_hidden_size,
        )
        tokenizer = None
    else:
        print(">> 使用 ESM2 Encoder for mutation:", args.pretrained_mutation_name)
        model = ESM2(pretrained_model_name_or_path=args.pretrained_mutation_name)
        tokenizer = model.tokenizer
        model.to(device)
        model.eval()
    return model, tokenizer


def initialize_maskers(args):
    imp_masker = ImportanceMasker2(
        args.k, max_subs=args.num_masked_tokens, low_importance_mask=not args.mask_high_importance
    )
    rand_masker = RandomMasker2(args.k, max_subs=args.num_masked_tokens)
    return [rand_masker, imp_masker]


def initialize_oracle(args, device: Union[str, torch.device]):
    return ESM1b_Landscape(args.task, device)


def initialize_oracle2(args, device):
    return ESM1v(f"esm1v_t33_650M_UR90S_{args.esm1v_seed}", device, "masked", 1)


def initialize_fitness_predictor(args, device: Union[str, torch.device]):
    if args.pretrained_mutation_name.lower() == "amix":
        net = AMixEncoder(ckpt_path="./AMix-1-main/checkpoints/amix.ckpt", hidden_dim=args.dec_hidden_size)
    else:
        net = ESM2_Attention(args.pretrained_mutation_name, hidden_dim=args.dec_hidden_size)

    model = ESM2DecoderModule.load_from_checkpoint(args.predictor_ckpt_path, map_location=device, net=net)
    model.eval()
    return model


def save_results(wt_seqs: List[str], mutants, score, valid_score, output_path: str):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame.from_dict(
        {"WT": wt_seqs, "mutants": mutants, "score": score, "orc. score": valid_score}
    )
    df.sort_values(by=["orc. score"], ascending=False, inplace=True, ignore_index=True)
    df.to_csv(output_path, index=False)


def main(args):
    set_seed(args.seed) if args.set_seed_only else enable_full_deterministic(args.seed)
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    device = torch.device("cpu" if args.devices == "-1" else f"cuda:{args.devices}")

    # Mutation model
    mutation_model, mutation_tokenizer = initialize_mutation_model(args, device)
    # Fitness predictor
    fitness_predictor = initialize_fitness_predictor(args, device)
    # Oracle
    oracle = initialize_oracle(args, device)
    # Maskers
    maskers = initialize_maskers(args)

    # Create folders
    result_dir = os.path.join(args.result_dir, args.task)
    log_dir = os.path.join(args.log_dir, args.task)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Evolution procedure
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

    wt_seq, wt_fitness = args.wt, args.wt_fitness
    mutants, pred_fitness, variants = direct_evo(wt_seq, wt_fitness)
    pred_fitness = pred_fitness.squeeze(1).numpy().tolist()

    valid_fitness = oracle.infer_fitness(variants)

    filepath = os.path.join(result_dir, args.save_name)
    save_results([wt_seq] * len(mutants), mutants, pred_fitness, valid_fitness, filepath)


if __name__ == "__main__":
    args = parse_args()
    main(args)

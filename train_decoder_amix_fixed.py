#!/usr/bin/env python3
# Train an AMix decoder regressor using AMixEncoder CLS outputs.
# Outputs a combined state_dict .pt that run_discrete_de_amix_beam.py can load.

import argparse
import os
import logging
import yaml
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from lightning.pytorch import Trainer, seed_everything, loggers, callbacks
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch import LightningModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------- amino acid mapping ----------------
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO2ID = {a: i + 1 for i, a in enumerate(AMINO_ACIDS)}  # 1..20
PAD_ID = 0
VOCAB_SIZE = max(AMINO2ID.values()) + 1  # 21


# ---------------- Dataset ----------------
class ProteinsDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        # allow files without headers
        if 'sequence' not in df.columns or 'fitness' not in df.columns:
            df.rename(columns={df.columns[0]: 'sequence', df.columns[1]: 'fitness'}, inplace=True)
        self.seqs = df['sequence'].astype(str).tolist()
        self.fitness = df['fitness'].astype(float).tolist()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {'sequence': self.seqs[idx], 'fitness': self.fitness[idx]}


def collate_batch(batch):
    seqs = [b['sequence'] for b in batch]
    fitness = torch.tensor([b['fitness'] for b in batch], dtype=torch.float32).unsqueeze(1)
    max_len = max(len(s) for s in seqs)
    input_ids = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids = [AMINO2ID.get(ch, PAD_ID) for ch in s]
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return {'input_ids': input_ids, 'fitness': fitness}


class ProteinsDataModuleNoTokenizer:
    def __init__(self, csv_file, batch_size=128, num_workers=0):
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = ProteinsDataset(self.csv_file)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=collate_batch, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=collate_batch, num_workers=self.num_workers)


# ---------------- AMix Encoder (CLS pooling + robust ckpt load) ----------------
class AMixEncoder(nn.Module):
    def __init__(self, ckpt_path=None, config_path=None, device=None, load_embedding_only=False):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load config if provided
        config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f) or {}
            except Exception as e:
                logging.warning(f"[AMixEncoder] failed to load config {config_path}: {e}")
                config = {}

        self.hidden_dim = int(config.get("hidden_dim", 1280))
        self.vocab_size = int(config.get("vocab_size", VOCAB_SIZE))
        self.num_layers = int(config.get("num_layers", 12))
        self.nhead = int(config.get("nhead", 8))

        # Embedding and transformer encoder
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=PAD_ID)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.nhead, batch_first=True)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Track what keys were loaded for debugging
        self._loaded_ckpt_keys = []

        # try to load weights from checkpoint
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                mapped = {k.replace("module.", ""): v for k, v in state_dict.items()}

                # 1) try embedding keys
                emb_loaded = False
                for k in ["embedding.weight", "encoder.embedding.weight", "embeddings.weight", "token_embedding.weight"]:
                    if k in mapped and mapped[k].shape == self.embedding.weight.shape:
                        with torch.no_grad():
                            self.embedding.weight.copy_(mapped[k])
                        emb_loaded = True
                        self._loaded_ckpt_keys.append(k)
                        logging.info(f"[AMixEncoder] copied embedding from ckpt key: {k}")
                        break
                if not emb_loaded:
                    logging.info("[AMixEncoder] no matching embedding key found in ckpt (or shape mismatch).")

                # 2) if requested, try to load full encoder weights (transformer & layernorm etc.)
                if not load_embedding_only:
                    own = self.state_dict()
                    matched = {}
                    for k, v in mapped.items():
                        if k in own and own[k].shape == v.shape:
                            matched[k] = v
                            self._loaded_ckpt_keys.append(k)
                    if matched:
                        # update only matched keys to avoid missing keys errors
                        own.update(matched)
                        self.load_state_dict(own)
                        logging.info(f"[AMixEncoder] loaded {len(matched)} matching weights from checkpoint (including possible transformer weights).")
                    else:
                        logging.info("[AMixEncoder] no transformer weights matched by shape; encoder remains with default init (embedding may still be loaded).")
            except Exception as e:
                logging.warning(f"[AMixEncoder] failed to load ckpt {ckpt_path}: {e}")

        self.to(self.device)

    def forward(self, input_ids: torch.Tensor):
        """
        CLS pooling: take token 0 hidden as pooled representation.
        Expectation: the LM/encoder that produced ckpt used position 0 as CLS.
        """
        x = self.embedding(input_ids.to(self.embedding.weight.device))
        x = self.encoder_layers(x)
        pooled = x[:, 0, :]   # CLS pooling
        return pooled


# ---------------- Decoder regressor (or Attention1D alternative) ----------------
class DecoderRegressor(nn.Module):
    def __init__(self, in_dim=1280, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)


# Optional: a small 1D self-attention pooling decoder (Attention1D) you can swap in.
# If you prefer attention pooling instead of simple MLP, uncomment/use this class.
class Attention1DDecoder(nn.Module):
    def __init__(self, in_dim=1280, attn_dim=256, out_dim=1):
        super().__init__()
        self.q = nn.Linear(in_dim, attn_dim)
        self.k = nn.Linear(in_dim, attn_dim)
        self.v = nn.Linear(in_dim, attn_dim)
        self.fc = nn.Linear(attn_dim, out_dim)

    def forward(self, x):
        # x: [B, D]  (assuming CLS already pooled). To use attention over sequence, you must pass sequence.
        # For compatibility we accept [B, D] and do self-attention on the vector (trivial), then map to score.
        # If you want true seq-attention, modify encoder to return full hidden states here.
        q = self.q(x).unsqueeze(1)  # [B,1,A]
        k = self.k(x).unsqueeze(1)
        v = self.v(x).unsqueeze(1)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        attn = torch.softmax(attn_scores, dim=-1)
        out = (attn @ v).squeeze(1)  # [B,A]
        out = self.fc(out)  # [B, out_dim]
        return out


# ---------------- Lightning Module ----------------
class AMixDecoderTrainer(LightningModule):
    def __init__(self, encoder: AMixEncoder, dec_hidden_dim=None,
                 lr=5e-5, freeze_encoder=False, use_attention_decoder=False):
        super().__init__()
        self.encoder = encoder
        # if dec_hidden_dim not specified, use encoder.hidden_dim
        self.dec_hidden_dim = dec_hidden_dim if dec_hidden_dim is not None else getattr(encoder, "hidden_dim", 1280)
        self.use_attention_decoder = use_attention_decoder
        if use_attention_decoder:
            self.regressor = Attention1DDecoder(in_dim=self.dec_hidden_dim, attn_dim=min(256, self.dec_hidden_dim // 2), out_dim=1)
        else:
            self.regressor = DecoderRegressor(in_dim=self.dec_hidden_dim)
        self.criterion = nn.MSELoss()
        self.lr = lr

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids):
        emb = self.encoder(input_ids)
        return self.regressor(emb)

    def training_step(self, batch, batch_idx):
        x, y = batch['input_ids'], batch['fitness']
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['input_ids'], batch['fitness']
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


# ---------------- Argparse ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", type=str, required=True)
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, default=None, help="AMix ckpt (will try to load embedding and optionally transformer weights)")
    p.add_argument("--config_path", type=str, default=None)
    p.add_argument("--dec_hidden_dim", type=int, default=None, help="regressor input dim; if unset uses encoder.hidden_dim")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--devices", type=str, default="0")
    p.add_argument("--output_dir", type=str, default="./exps")
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", type=int, choices=[16, 32], default=32)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--use_attention_decoder", action="store_true", help="use Attention1D decoder instead of simple MLP")
    p.add_argument("--num_ckpts", type=int, default=3)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


# ---------------- Main ----------------
def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)

    # device string: keep compatibility with original script (cpu/-1 or cuda:N)
    device = "cpu" if args.devices == "-1" else f"cuda:{int(args.devices)}" if "," not in args.devices else "cuda"

    encoder = AMixEncoder(
        ckpt_path=args.ckpt_path,
        config_path=args.config_path,
        device=device,
        load_embedding_only=False  # try to load full transformer weights if available
    )

    # Log what keys were loaded (if any)
    if hasattr(encoder, "_loaded_ckpt_keys") and encoder._loaded_ckpt_keys:
        logging.info(f"[Main] AMixEncoder loaded keys from ckpt sample: {encoder._loaded_ckpt_keys[:10]} (total {len(encoder._loaded_ckpt_keys)})")
    else:
        logging.info("[Main] AMixEncoder did not detect loaded checkpoint keys (embedding/transformer). Check ckpt content if this is unexpected.")

    module = AMixDecoderTrainer(
        encoder,
        dec_hidden_dim=args.dec_hidden_dim,
        lr=args.lr,
        freeze_encoder=args.freeze_encoder,
        use_attention_decoder=args.use_attention_decoder
    )

    dm = ProteinsDataModuleNoTokenizer(
        csv_file=args.data_file,
        batch_size=args.batch_size,
        num_workers=0
    )
    dm.setup()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    csv_logger = loggers.CSVLogger(save_dir=args.output_dir, name=args.dataset_name)
    logger_list = [csv_logger]

    if not args.no_wandb:
        wandb_logger = loggers.WandbLogger(save_dir=args.output_dir, project="directed_evolution", mode="offline")
        logger_list.append(wandb_logger)

    checkpoint_cb = callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"AMix-dec_{module.dec_hidden_dim}-{args.dataset_name}" + "_{epoch:02d}-{val_loss:.3f}",
        monitor="val_loss",
        save_top_k=args.num_ckpts,
        save_weights_only=False,
        every_n_epochs=1
    )

    trainer = Trainer(
        accelerator="cpu" if args.devices == "-1" else "gpu",
        devices=[int(d) for d in args.devices.split(",")] if args.devices != "-1" else None,
        max_epochs=args.num_epochs,
        logger=logger_list,
        callbacks=[checkpoint_cb, TQDMProgressBar(refresh_rate=20)],
        precision=args.precision,
        deterministic=True
    )

    trainer.fit(module, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

    # Save combined checkpoint compatible with inference loader
    out_path = os.path.join(ckpt_dir, f"AMix-dec_{module.dec_hidden_dim}-{args.dataset_name}.pt")
    combined = {
        "encoder_state_dict": module.encoder.state_dict(),
        "regressor_state_dict": module.regressor.state_dict(),
        "meta": {"dec_hidden_dim": module.dec_hidden_dim, "encoder_hidden_dim": module.encoder.hidden_dim}
    }
    torch.save(combined, out_path)
    logging.info(f"[Saved] Combined state saved to {out_path}")


if __name__ == "__main__":
    main()

import argparse
import os
import torch
from functools import partial
from lightning import Trainer, seed_everything
from lightning.pytorch import loggers, callbacks
from torch.optim import Adam
from torch.utils.data import DataLoader
import pandas as pd
from lightning.pytorch import LightningDataModule, LightningModule
import yaml
import torch.nn as nn
from lightning.pytorch.callbacks import TQDMProgressBar
# ----------------- DataModule -----------------
class ProteinsDataModuleNoTokenizer(LightningDataModule):
    def __init__(self, csv_file, train_batch_size=128, val_batch_size=None, test_batch_size=None):
        super().__init__()
        self.csv_file = csv_file
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size or train_batch_size
        self.test_batch_size = test_batch_size or train_batch_size
        self.amino2id = {a: i+1 for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}  # 0 留给 padding

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_file)
        df.rename(columns={df.columns[0]: 'sequence', df.columns[1]: 'fitness'}, inplace=True)
        df['input_ids'] = df['sequence'].apply(lambda seq: [self.amino2id.get(s, 0) for s in seq])
        self.samples = df[['input_ids', 'fitness']].to_dict(orient='records')
        self.train_data = self.samples
        self.val_data = self.samples
        self.test_data = self.samples

    def collate_fn(self, batch):
        max_len = max(len(b['input_ids']) for b in batch)
        input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, b in enumerate(batch):
            seq = torch.tensor(b['input_ids'], dtype=torch.long)
            input_ids[i, :len(seq)] = seq
        fitness = torch.tensor([b['fitness'] for b in batch], dtype=torch.float).unsqueeze(1)  # [B,1]
        return {'input_ids': input_ids, 'fitness': fitness}

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.train_batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.val_batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.test_batch_size, collate_fn=self.collate_fn)

# ----------------- AMix Encoder -----------------
class AMixEncoder(nn.Module):
    def __init__(self, ckpt_path, config_path=None, device=None, freeze_encoder=False):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ckpt file not found: {ckpt_path}")
        self.ckpt_path = ckpt_path

        # Load config from yaml
        self.config = {}
        if config_path is not None:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"config file not found: {config_path}")
            with open(config_path, "r") as f:
                full_config = yaml.safe_load(f)
                # Extract model config from nested structure
                if 'model' in full_config and 'bfn' in full_config['model']:
                    bfn_config = full_config['model']['bfn']
                    if 'net' in bfn_config and 'config' in bfn_config['net']:
                        self.config = bfn_config['net']['config']
                else:
                    self.config = full_config

        # Get model dimensions from config
        hidden_dim = self.config.get("hidden_size", self.config.get("hidden_dim", 1280))
        num_layers = self.config.get("num_hidden_layers", self.config.get("num_layers", 12))
        num_heads = self.config.get("num_attention_heads", self.config.get("nhead", 8))
        vocab_size = self.config.get("vocab_size", 30)
        
        print(f"[AMixEncoder] Initializing with hidden_dim={hidden_dim}, num_layers={num_layers}, num_heads={num_heads}")

        # Model architecture
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )

        self.to(self.device)

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Handle different checkpoint formats
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove module prefix if exists
            k = k.replace("module.", "")
            # Try to map keys
            if "token_embedding" in k or "embeddings.word_embeddings" in k:
                new_state_dict["embedding.weight"] = v
            elif k.startswith("encoder."):
                new_state_dict[k] = v
            elif "embedding" in k:
                new_state_dict[k] = v
        
        # Load weights with strict=False to allow partial loading
        self.load_state_dict(new_state_dict, strict=False)
        print(f"[AMixEncoder] Loaded {len(new_state_dict)} parameters from checkpoint")
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.parameters():
                param.requires_grad = False
            print("[AMixEncoder] Encoder parameters frozen")

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # (B,L,H)
        x = self.encoder_layers(x)
        return x.mean(dim=1)  # [B,H]

# ----------------- Decoder Module -----------------
class AMixDecoderModule(LightningModule):
    def __init__(self, encoder, dec_hidden_dim=1280, lr=5e-5, weight_decay=1e-4, use_scheduler=True):
        super().__init__()
        self.encoder = encoder
        self.dec_hidden_dim = dec_hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        
        # Improved decoder architecture with attention
        self.attention = nn.MultiheadAttention(dec_hidden_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(dec_hidden_dim, dec_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dec_hidden_dim // 2, 1)
        )
        self.criterion = nn.MSELoss()
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['encoder'])

    def forward(self, x):
        emb = self.encoder(x)  # [B,H]
        # Add sequence dimension for attention
        emb = emb.unsqueeze(1)  # [B,1,H]
        attn_out, _ = self.attention(emb, emb, emb)
        attn_out = attn_out.squeeze(1)  # [B,H]
        return self.decoder(attn_out)  # [B,1]

    def training_step(self, batch, batch_idx):
        x, y = batch["input_ids"], batch["fitness"]
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["input_ids"], batch["fitness"]
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        # Log correlation
        with torch.no_grad():
            corr = torch.corrcoef(torch.stack([pred.squeeze(), y.squeeze()]))[0, 1]
            self.log("val_corr", corr, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        if self.use_scheduler:
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5, verbose=True
                ),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return optimizer

# ----------------- Argument Parser -----------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train decoder with AMix encoder.")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to AMix checkpoint (.ckpt)")
    parser.add_argument("--config_path", type=str, required=True, help="Path to AMix config.yaml")
    parser.add_argument("--dec_hidden_dim", type=int, default=None, help="Decoder hidden dim (default: use encoder hidden_dim)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--output_dir", type=str, default="./exps")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--wandb_project", type=str, default="directed_evolution")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set_seed_only", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_ckpts", type=int, default=5)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--precision", type=str, choices=["highest", "high", "medium"], default="highest")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder parameters during training")
    parser.add_argument("--use_scheduler", action="store_true", help="Use learning rate scheduler")
    return parser.parse_args()

# ----------------- Training -----------------
def train(args):
    seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(args.precision)

    accelerator = "cpu" if args.devices == "-1" else "gpu"
    devices = [int(d) for d in args.devices.split(",")] if accelerator=="gpu" else None

    # ====== Model & Optimizer ======
    encoder = AMixEncoder(args.ckpt_path, args.config_path, freeze_encoder=args.freeze_encoder)
    
    # Use encoder's hidden_dim if dec_hidden_dim not specified
    dec_hidden_dim = args.dec_hidden_dim if args.dec_hidden_dim is not None else encoder.hidden_dim
    print(f"[Training] Using dec_hidden_dim={dec_hidden_dim}")
    
    module = AMixDecoderModule(
        encoder, 
        dec_hidden_dim=dec_hidden_dim, 
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_scheduler=args.use_scheduler
    )

    # ====== Data ======
    datamodule = ProteinsDataModuleNoTokenizer(
        csv_file=args.data_file,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        test_batch_size=args.batch_size
    )

    # ====== Logging ======
    os.makedirs(args.output_dir, exist_ok=True)
    logger_list = [
        loggers.CSVLogger(args.output_dir),
        loggers.WandbLogger(save_dir=args.output_dir, project=args.wandb_project, mode="offline")
    ]

    # ====== Callbacks ======
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callback_list = [
        callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"AMix-dec_{dec_hidden_dim}-{args.dataset_name}_{{epoch:02d}}-{{val_loss:.3f}}",
            monitor="val_loss",
            mode="min",
            verbose=True,
            save_top_k=args.num_ckpts,
            save_weights_only=False,
            every_n_epochs=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
            verbose=True
        ),
        TQDMProgressBar(refresh_rate=20)
    ]

    # ====== Trainer ======
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=args.num_epochs,
        log_every_n_steps=args.log_interval,
        accumulate_grad_batches=args.grad_accum_steps,
        deterministic=not args.set_seed_only,
        default_root_dir=args.output_dir,
        logger=logger_list,
        callbacks=callback_list,
        enable_progress_bar=True,
        gradient_clip_val=1.0  # Add gradient clipping for stability
    )

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Encoder hidden_dim: {encoder.hidden_dim}")
    print(f"  Decoder hidden_dim: {dec_hidden_dim}")
    print(f"  Encoder frozen: {args.freeze_encoder}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Scheduler: {args.use_scheduler}")
    print(f"{'='*60}\n")

    trainer.fit(module, datamodule=datamodule, ckpt_path=None)


if __name__ == "__main__":
    args = parse_args()
    train(args)

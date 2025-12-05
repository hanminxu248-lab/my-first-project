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
    def __init__(self, ckpt_path, config_path=None, device=None):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ckpt 文件不存在: {ckpt_path}")
        self.ckpt_path = ckpt_path

        self.config = None
        if config_path is not None:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"config 文件不存在: {config_path}")
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

        state_dict = torch.load(ckpt_path, map_location=self.device)
        hidden_dim = self.config.get("hidden_dim", 1280)
        vocab_size = self.config.get("vocab_size", 30)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=self.config.get("num_layers", 12)
        )

        self.to(self.device)

        if "embedding.weight" in state_dict:
            self.embedding.weight.data.copy_(state_dict["embedding.weight"])

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # (B,L,H)
        x = self.encoder_layers(x)
        return x.mean(dim=1)  # [B,H]

# ----------------- Decoder Module -----------------
class AMixDecoderModule(LightningModule):
    def __init__(self, encoder, dec_hidden_dim=1280, lr=5e-5):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Linear(dec_hidden_dim, 1)  # 输出 [B,1]
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        emb = self.encoder(x)
        return self.decoder(emb)  # [B,1]

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

# ----------------- Argument Parser -----------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train decoder with AMix encoder.")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--dec_hidden_dim", type=int, default=1280)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--output_dir", type=str, default="./exps")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--wandb_project", type=str, default="directed_evolution")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set_seed_only", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_ckpts", type=int, default=5)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--precision", type=str, choices=["highest", "high", "medium"], default="highest")
    return parser.parse_args()

# ----------------- Training -----------------
def train(args):
    seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(args.precision)

    accelerator = "cpu" if args.devices == "-1" else "gpu"
    devices = [int(d) for d in args.devices.split(",")] if accelerator=="gpu" else None

    # ====== Model & Optimizer ======
    encoder = AMixEncoder(args.ckpt_path, args.config_path)
    module = AMixDecoderModule(encoder, dec_hidden_dim=args.dec_hidden_dim, lr=args.lr)

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
    callback_list = [
        callbacks.ModelCheckpoint(
            dirpath="/root/data1/Directed_Evolution-main/exps/checkpoints",  # 固定保存目录
            filename=f"AMix-dec_{args.dec_hidden_dim}-{args.dataset_name}_{{epoch:02d}}-{{train_loss:.3f}}-{{val_loss:.3f}}",
            monitor="val_loss",
            verbose=True,
            save_top_k=args.num_ckpts,
            save_weights_only=False,
            every_n_epochs=1,
        ),
        TQDMProgressBar(refresh_rate=20)  # 简单进度条
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
        enable_progress_bar=True  # 开启进度条
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=None)


if __name__ == "__main__":
    args = parse_args()
    train(args)

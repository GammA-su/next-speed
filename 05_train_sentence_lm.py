import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import seed_all, save_json

TRAIN_PATH = os.environ.get("TRAIN_PATH", "data/sent_lm_train.jsonl")
OUT_DIR = os.environ.get("OUT_DIR", "out/sentence_lm")

K = int(os.environ.get("K", "8"))
V = int(os.environ.get("V", "1024"))
CONTEXT_SENTS = int(os.environ.get("CONTEXT_SENTS", "6"))

# ctx_codes length = CONTEXT_SENTS*K
CTX_LEN = CONTEXT_SENTS * K

D_MODEL = int(os.environ.get("D_MODEL", "512"))
N_LAYERS = int(os.environ.get("N_LAYERS", "6"))
N_HEADS = int(os.environ.get("N_HEADS", "8"))
DROP = float(os.environ.get("DROP", "0.1"))

BS = int(os.environ.get("BS", "256"))
LR = float(os.environ.get("LR", "2e-4"))
STEPS = int(os.environ.get("STEPS", "2000"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "20"))
SEED = int(os.environ.get("SEED", "0"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "1") == "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CodeDataset(Dataset):
    def __init__(self, path):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if len(r["ctx_codes"]) != CTX_LEN or len(r["tgt_codes"]) != K:
                    continue
                self.rows.append(r)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return (
            torch.tensor(r["ctx_codes"], dtype=torch.long),
            torch.tensor(r["tgt_codes"], dtype=torch.long),
        )


class SentenceLM(nn.Module):
    def __init__(self):
        super().__init__()
        # token ids are 0..V-1 but each position corresponds to a codebook slot in context
        # simplest: share vocab embedding across slots
        self.tok_emb = nn.Embedding(V, D_MODEL)
        self.pos_emb = nn.Embedding(CTX_LEN, D_MODEL)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=4 * D_MODEL,
            dropout=DROP,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)

        # K heads, each predicts one codebook index
        self.heads = nn.ModuleList([nn.Linear(D_MODEL, V) for _ in range(K)])

    def forward(self, ctx_codes):
        # ctx_codes: (B, CTX_LEN) ints in [0..V-1]
        b, t = ctx_codes.shape
        pos = torch.arange(t, device=ctx_codes.device)
        x = self.tok_emb(ctx_codes) + self.pos_emb(pos)[None, :, :]
        x = self.enc(x)
        x = self.norm(x)
        # pool: last position representation
        h = x[:, -1, :]  # (B, D)
        logits = [head(h) for head in self.heads]  # list of (B,V)
        return logits


def main():
    seed_all(SEED, deterministic=DETERMINISTIC)
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = CodeDataset(TRAIN_PATH)
    g = torch.Generator()
    g.manual_seed(SEED)
    dl = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=0, drop_last=True, generator=g)

    model = SentenceLM().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    it = iter(dl)
    pbar = tqdm(range(STEPS), desc="train_sentence_lm")
    for step in pbar:
        try:
            ctx, tgt = next(it)
        except StopIteration:
            it = iter(dl)
            ctx, tgt = next(it)

        ctx = ctx.to(DEVICE)
        tgt = tgt.to(DEVICE)  # (B,K)

        logits = model(ctx)

        # sentence-level loss = sum over K heads (one per codebook)
        loss = 0.0
        for j in range(K):
            loss = loss + nn.functional.cross_entropy(logits[j], tgt[:, j])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % LOG_EVERY == 0:
            # quick top-1 accuracy per codebook
            accs = []
            for j in range(K):
                pred = logits[j].argmax(dim=-1)
                accs.append((pred == tgt[:, j]).float().mean().item())
            pbar.set_postfix(loss=float(loss.item()), acc=float(np.mean(accs)))

    ckpt_path = os.path.join(OUT_DIR, "sentence_lm.pt")
    cfg = {
        "K": K,
        "V": V,
        "CONTEXT_SENTS": CONTEXT_SENTS,
        "D_MODEL": D_MODEL,
        "N_LAYERS": N_LAYERS,
        "N_HEADS": N_HEADS,
        "DROP": DROP,
        "CTX_LEN": CTX_LEN,
    }
    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, ckpt_path)

    meta = {
        "train_path": TRAIN_PATH,
        "out_dir": OUT_DIR,
        "device": DEVICE,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
        "steps": STEPS,
        "batch_size": BS,
        "lr": LR,
        "dataset_size": len(ds),
        "cfg": cfg,
    }
    save_json(os.path.join(OUT_DIR, "config.json"), meta)

    print("Saved:", ckpt_path)
    print("Meta:", os.path.join(OUT_DIR, "config.json"))


if __name__ == "__main__":
    main()

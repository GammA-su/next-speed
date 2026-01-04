import os
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import seed_all, save_json

VAL_PATH = os.environ.get("VAL_PATH", "data/sent_lm_val.jsonl")
SENT_LM_PATH = os.environ.get("SENT_LM_PATH", "out/sentence_lm/sentence_lm.pt")
OUT_PATH = os.environ.get("OUT_PATH", "out/eval/sent_lm_eval.json")
BS = int(os.environ.get("BS", "256"))
MAX_ROWS = int(os.environ.get("MAX_ROWS", "-1"))
SEED = int(os.environ.get("SEED", "0"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "1") == "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CodeDataset(Dataset):
    def __init__(self, path, ctx_len, k, max_rows=-1):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if len(r["ctx_codes"]) != ctx_len or len(r["tgt_codes"]) != k:
                    continue
                self.rows.append(r)
                if max_rows > 0 and len(self.rows) >= max_rows:
                    break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return (
            torch.tensor(r["ctx_codes"], dtype=torch.long),
            torch.tensor(r["tgt_codes"], dtype=torch.long),
        )


class SentenceLM(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.K = cfg["K"]
        self.V = cfg["V"]
        ctx_len = cfg["CONTEXT_SENTS"] * cfg["K"]
        d_model = cfg["D_MODEL"]
        n_layers = cfg["N_LAYERS"]
        n_heads = cfg["N_HEADS"]
        self.ctx_len = ctx_len
        self.tok_emb = torch.nn.Embedding(self.V, d_model)
        self.pos_emb = torch.nn.Embedding(self.ctx_len, d_model)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.enc = torch.nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = torch.nn.LayerNorm(d_model)
        self.heads = torch.nn.ModuleList([torch.nn.Linear(d_model, self.V) for _ in range(self.K)])

    def forward(self, ctx_codes):
        b, t = ctx_codes.shape
        pos = torch.arange(t, device=ctx_codes.device)
        x = self.tok_emb(ctx_codes) + self.pos_emb(pos)[None, :, :]
        x = self.enc(x)
        x = self.norm(x)
        h = x[:, -1, :]
        return [head(h) for head in self.heads]


def main():
    seed_all(SEED, deterministic=DETERMINISTIC)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    ckpt = torch.load(SENT_LM_PATH, map_location="cpu")
    model = SentenceLM(ckpt["cfg"]).to(DEVICE).eval()
    model.load_state_dict(ckpt["state_dict"])

    ctx_len = ckpt["cfg"]["CONTEXT_SENTS"] * ckpt["cfg"]["K"]
    k = ckpt["cfg"]["K"]

    ds = CodeDataset(VAL_PATH, ctx_len, k, max_rows=MAX_ROWS)
    dl = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=0)

    total = 0
    head_correct = np.zeros(k, dtype=np.int64)
    tuple_correct = 0
    loss_sum = 0.0

    with torch.no_grad():
        for ctx, tgt in dl:
            ctx = ctx.to(DEVICE)
            tgt = tgt.to(DEVICE)
            logits = model(ctx)

            batch = tgt.shape[0]
            total += batch

            for j in range(k):
                pred = logits[j].argmax(dim=-1)
                head_correct[j] += int((pred == tgt[:, j]).sum().item())
                loss_sum += torch.nn.functional.cross_entropy(
                    logits[j], tgt[:, j], reduction="sum"
                ).item()

            pred_all = torch.stack([logits[j].argmax(dim=-1) for j in range(k)], dim=1)
            tuple_correct += int((pred_all == tgt).all(dim=1).sum().item())

    head_acc = (head_correct / max(1, total)).tolist()
    tuple_acc = float(tuple_correct / max(1, total))
    avg_ce = float(loss_sum / max(1, total * k))
    ppl = float(math.exp(avg_ce)) if avg_ce < 50 else float("inf")

    out = {
        "val_path": VAL_PATH,
        "sent_lm_path": SENT_LM_PATH,
        "total": total,
        "head_acc": head_acc,
        "tuple_acc": tuple_acc,
        "avg_ce": avg_ce,
        "perplexity": ppl,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
    }
    save_json(OUT_PATH, out)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()

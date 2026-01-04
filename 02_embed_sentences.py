import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from utils import seed_all, save_json

SENT_PATH = os.environ.get("SENT_PATH", "data/sentences.jsonl")
OUT_EMB = os.environ.get("OUT_EMB", "data/embeddings.npy")
OUT_META = os.environ.get("OUT_META", "data/emb_meta.json")
EMB_MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BS = int(os.environ.get("BS", "128"))
MAXLEN = int(os.environ.get("MAXLEN", "256"))
SEED = int(os.environ.get("SEED", "0"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "1") == "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mean_pool(last_hidden, attn_mask):
    mask = attn_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def main():
    seed_all(SEED, deterministic=DETERMINISTIC)
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    enc = AutoModel.from_pretrained(EMB_MODEL).to(DEVICE).eval()

    texts = []
    doc_ids = []
    with open(SENT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
            doc_ids.append(r["doc_id"])

    all_emb = np.zeros((len(texts), enc.config.hidden_size), dtype=np.float32)

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BS), desc="embed"):
            batch = texts[i : i + BS]
            x = tok(batch, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
            x = {k: v.to(DEVICE) for k, v in x.items()}
            out = enc(**x)
            e = mean_pool(out.last_hidden_state, x["attention_mask"])
            e = F.normalize(e, p=2, dim=-1)
            all_emb[i : i + len(batch)] = e.cpu().numpy().astype(np.float32)

    np.save(OUT_EMB, all_emb)
    meta = {
        "emb_model": EMB_MODEL,
        "n": len(texts),
        "d": int(all_emb.shape[1]),
        "sent_path": SENT_PATH,
        "out_emb": OUT_EMB,
        "maxlen": MAXLEN,
        "batch_size": BS,
        "device": DEVICE,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
    }
    save_json(OUT_META, meta)

    print("Saved:", OUT_EMB, "shape:", all_emb.shape)
    print("Meta:", OUT_META)


if __name__ == "__main__":
    main()

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from contextlib import nullcontext

from utils import seed_all, save_json

SENT_PATH = os.environ.get("SENT_PATH", "data/sentences.jsonl")
OUT_EMB = os.environ.get("OUT_EMB", "data/embeddings.npy")
OUT_META = os.environ.get("OUT_META", "data/emb_meta.json")
EMB_MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BS = int(os.environ.get("BS", "128"))
MAXLEN = int(os.environ.get("MAXLEN", "256"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
PIN_MEMORY = os.environ.get("PIN_MEMORY", "1") == "1"
COMPILE = os.environ.get("COMPILE", "0") == "1"
SEED = int(os.environ.get("SEED", "0"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "1") == "1"

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = os.environ.get("DEVICE", DEFAULT_DEVICE)
if DEVICE != "cpu" and not torch.cuda.is_available():
    print(f"DEVICE={DEVICE} but cuda not available; falling back to cpu.")
    DEVICE = "cpu"

AMP_DEFAULT = DEVICE == "cuda"
AMP_ENV = os.environ.get("AMP")
AMP = AMP_DEFAULT if AMP_ENV is None else AMP_ENV == "1"
DTYPE = (os.environ.get("DTYPE") or ("fp16" if DEVICE == "cuda" else "fp32")).lower()


def mean_pool(last_hidden, attn_mask):
    mask = attn_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


class SentDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return idx, self.texts[idx]


def main():
    seed_all(SEED, deterministic=DETERMINISTIC)
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    enc = AutoModel.from_pretrained(EMB_MODEL).to(DEVICE).eval()
    if COMPILE and hasattr(torch, "compile"):
        try:
            enc = torch.compile(enc)
        except Exception as exc:
            print(f"torch.compile failed, continuing without it: {exc}")

    texts = []
    doc_ids = []
    with open(SENT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
            doc_ids.append(r["doc_id"])

    if DEVICE != "cuda":
        AMP = False
    if AMP and DTYPE == "fp32":
        print("AMP=1 with DTYPE=fp32; disabling AMP.")
        AMP = False

    amp_dtype = None
    if AMP and DEVICE == "cuda":
        if DTYPE == "bf16":
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                print("bf16 not supported; falling back to fp16.")
                amp_dtype = torch.float16
                DTYPE = "fp16"
        elif DTYPE == "fp16":
            amp_dtype = torch.float16
        else:
            amp_dtype = None

    gpu_name = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "none"
    print(
        "embed: device={device} cuda={cuda} gpu={gpu} amp={amp} dtype={dtype} "
        "num_workers={num_workers} pin_memory={pin_memory} compile={compile}".format(
            device=DEVICE,
            cuda=torch.cuda.is_available(),
            gpu=gpu_name,
            amp=AMP,
            dtype=DTYPE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            compile=COMPILE,
        )
    )

    def autocast_context():
        if AMP and DEVICE == "cuda":
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        return nullcontext()

    dataset = SentDataset(texts)
    loader = DataLoader(
        dataset,
        batch_size=BS,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and DEVICE == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        collate_fn=lambda batch: (
            torch.tensor([b[0] for b in batch], dtype=torch.long),
            tok(
                [b[1] for b in batch],
                padding=True,
                truncation=True,
                max_length=MAXLEN,
                return_tensors="pt",
            ),
        ),
    )

    all_emb = np.zeros((len(texts), enc.config.hidden_size), dtype=np.float32)

    processed = 0
    start = time.time()
    with torch.no_grad():
        for idxs, x in tqdm(loader, desc="embed"):
            x = {k: v.to(DEVICE, non_blocking=True) for k, v in x.items()}
            with autocast_context():
                out = enc(**x)
            e = mean_pool(out.last_hidden_state, x["attention_mask"])
            e = F.normalize(e, p=2, dim=-1)
            all_emb[idxs.numpy()] = e.cpu().numpy().astype(np.float32)
            processed += len(idxs)

    elapsed = max(1e-6, time.time() - start)
    examples_per_sec = processed / elapsed
    print(f"examples_per_sec={examples_per_sec:.2f}")

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
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "amp": AMP,
        "dtype": DTYPE,
        "amp_dtype": str(amp_dtype) if amp_dtype is not None else None,
        "compile": COMPILE,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
    }
    save_json(OUT_META, meta)

    print("Saved:", OUT_EMB, "shape:", all_emb.shape)
    print("Meta:", OUT_META)


if __name__ == "__main__":
    main()

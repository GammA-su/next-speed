import os
import json
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import pysbd
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

from utils import seed_all, save_json

DEC_VAL_PATH = os.environ.get("DEC_VAL_PATH", "data/decoder_val.jsonl")
SENT_LM_PATH = os.environ.get("SENT_LM_PATH", "out/sentence_lm/sentence_lm.pt")
RVQ_PATH = os.environ.get("RVQ_PATH", "out/rvq.npz")
DECODER_PATH = os.environ.get("DECODER_PATH", "out/decoder")
EMB_MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OUT_PATH = os.environ.get("OUT_PATH", "out/eval/code_agreement.json")

CODE_MODE = os.environ.get("CODE_MODE", "special")  # special | text | none
MAX_ROWS = int(os.environ.get("MAX_ROWS", "100"))
DO_SAMPLE = os.environ.get("DO_SAMPLE", "0") == "1"
TOP_P = float(os.environ.get("TOP_P", "0.95"))
TEMP = float(os.environ.get("TEMP", "0.9"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "80"))
MAX_INPUT = int(os.environ.get("MAX_INPUT", "512"))

SEED = int(os.environ.get("SEED", "0"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "1") == "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

segmenter = pysbd.Segmenter(language="en", clean=True)


def split_sents(text: str):
    s = [x.strip() for x in segmenter.segment(text) if x.strip()]
    return [x for x in s if len(x) >= 5]


def mean_pool(last_hidden, attn_mask):
    mask = attn_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


@torch.no_grad()
def embed_sentence(enc_tok, enc_model, sent: str):
    x = enc_tok([sent], padding=True, truncation=True, max_length=256, return_tensors="pt")
    x = {k: v.to(DEVICE) for k, v in x.items()}
    out = enc_model(**x)
    e = mean_pool(out.last_hidden_state, x["attention_mask"])
    e = F.normalize(e, p=2, dim=-1)
    return e[0].cpu().numpy().astype("float32")


def encode_rvq_one(e, centroids):
    d = e.shape[0]
    residual = e[None, :].copy()
    codes = []
    for k in range(centroids.shape[0]):
        C = centroids[k].astype("float32")
        faiss.normalize_L2(C)
        index = faiss.IndexFlatIP(d)
        index.add(C)
        _, ids = index.search(residual, 1)
        idx = int(ids[0, 0])
        codes.append(idx)
        residual = residual - C[idx][None, :]
        rn = np.linalg.norm(residual, axis=1, keepdims=True) + 1e-8
        residual = residual / rn
    return codes


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


def code_token(k: int, v: int, width: int) -> str:
    return f"<c{k}_{v:0{width}d}>"


def codes_to_special(codes: List[int], width: int) -> str:
    return " ".join([code_token(i, c, width) for i, c in enumerate(codes)])


def codes_to_text(codes: List[int]) -> str:
    return " ".join([f"c{i}={c}" for i, c in enumerate(codes)])


def format_prompt(ctx_text: str, codes: List[int], v: int) -> str:
    if CODE_MODE == "none":
        return f"CTX: {ctx_text}\nWrite exactly one next sentence:"
    if CODE_MODE == "text":
        return f"CTX: {ctx_text}\nCODES: {codes_to_text(codes)}\nWrite exactly one next sentence:"
    if CODE_MODE == "special":
        width = len(str(v - 1))
        return f"CTX: {ctx_text}\nCODES: {codes_to_special(codes, width)}\nWrite exactly one next sentence:"
    raise ValueError(f"Unknown CODE_MODE: {CODE_MODE}")


def main():
    seed_all(SEED, deterministic=DETERMINISTIC)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    centroids = np.load(RVQ_PATH)["centroids"].astype("float32")

    ckpt = torch.load(SENT_LM_PATH, map_location="cpu")
    lm = SentenceLM(ckpt["cfg"]).to(DEVICE).eval()
    lm.load_state_dict(ckpt["state_dict"])

    enc_tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    enc_model = AutoModel.from_pretrained(EMB_MODEL).to(DEVICE).eval()

    dec_tok = AutoTokenizer.from_pretrained(DECODER_PATH)
    dec = AutoModelForSeq2SeqLM.from_pretrained(DECODER_PATH).to(DEVICE).eval()

    total = 0
    head_match = np.zeros(ckpt["cfg"]["K"], dtype=np.int64)
    tuple_match = 0
    skipped = 0

    with open(DEC_VAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ctx_text = r["ctx_text"]
            ctx_codes = r.get("ctx_codes")
            if ctx_codes is None:
                skipped += 1
                continue
            if len(ctx_codes) != ckpt["cfg"]["CONTEXT_SENTS"] * ckpt["cfg"]["K"]:
                skipped += 1
                continue

            ctx = torch.tensor([ctx_codes], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                logits = lm(ctx)
            pred_codes = [int(l.argmax(dim=-1).item()) for l in logits]

            prompt = format_prompt(ctx_text, pred_codes, ckpt["cfg"]["V"])
            inp = dec_tok([prompt], return_tensors="pt", truncation=True, max_length=MAX_INPUT).to(DEVICE)
            out = dec.generate(
                **inp,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                top_p=TOP_P,
                temperature=TEMP,
            )
            gen = dec_tok.decode(out[0], skip_special_tokens=True).strip()
            fs = split_sents(gen)
            fs = fs[0] if fs else gen

            e2 = embed_sentence(enc_tok, enc_model, fs)
            gen_codes = encode_rvq_one(e2, centroids)

            total += 1
            for j, (a, b) in enumerate(zip(pred_codes, gen_codes)):
                if a == b:
                    head_match[j] += 1
            if all(a == b for a, b in zip(pred_codes, gen_codes)):
                tuple_match += 1

            if MAX_ROWS > 0 and total >= MAX_ROWS:
                break

    head_acc = (head_match / max(1, total)).tolist()
    tuple_acc = float(tuple_match / max(1, total))

    out = {
        "dec_val_path": DEC_VAL_PATH,
        "sent_lm_path": SENT_LM_PATH,
        "rvq_path": RVQ_PATH,
        "decoder_path": DECODER_PATH,
        "emb_model": EMB_MODEL,
        "total": total,
        "skipped": skipped,
        "head_match": head_acc,
        "tuple_match": tuple_acc,
        "code_mode": CODE_MODE,
        "do_sample": DO_SAMPLE,
        "top_p": TOP_P,
        "temperature": TEMP,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
    }
    save_json(OUT_PATH, out)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()

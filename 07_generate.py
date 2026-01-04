import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import pysbd
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

from utils import seed_all, save_json, now_id

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# paths
RVQ_PATH = os.environ.get("RVQ_PATH", "out/rvq.npz")
SENT_LM_PATH = os.environ.get("SENT_LM_PATH", "out/sentence_lm/sentence_lm.pt")
DECODER_PATH = os.environ.get("DECODER_PATH", "out/decoder")

# embedder for coding prompt sentences
EMB_MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

K = int(os.environ.get("K", "8"))
V = int(os.environ.get("V", "1024"))
CONTEXT_SENTS = int(os.environ.get("CONTEXT_SENTS", "6"))
GEN_STEPS = int(os.environ.get("GEN_STEPS", "6"))
N_CAND = int(os.environ.get("N_CAND", "4"))  # candidate rerank

CODE_MODE = os.environ.get("CODE_MODE", "special")  # special | text | none
RERANK = os.environ.get("RERANK", "1") == "1"
DO_SAMPLE = os.environ.get("DO_SAMPLE", "1") == "1"
TOP_P = float(os.environ.get("TOP_P", "0.95"))
TEMP = float(os.environ.get("TEMP", "0.9"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "80"))
MAX_INPUT = int(os.environ.get("MAX_INPUT", "512"))
SAVE_CANDIDATES = os.environ.get("SAVE_CANDIDATES", "0") == "1"

SEED = int(os.environ.get("SEED", "0"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "1") == "1"

RUN_ID = os.environ.get("RUN_ID", now_id())
OUT_PATH = os.environ.get("OUT_PATH", f"out/generate/{RUN_ID}.json")

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
    # e: (D,)
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


def format_prompt(ctx_text: str, codes: List[int]) -> str:
    if CODE_MODE == "none":
        return f"CTX: {ctx_text}\nWrite exactly one next sentence:"
    if CODE_MODE == "text":
        return f"CTX: {ctx_text}\nCODES: {codes_to_text(codes)}\nWrite exactly one next sentence:"
    if CODE_MODE == "special":
        width = len(str(V - 1))
        return f"CTX: {ctx_text}\nCODES: {codes_to_special(codes, width)}\nWrite exactly one next sentence:"
    raise ValueError(f"Unknown CODE_MODE: {CODE_MODE}")


@torch.no_grad()
def main():
    seed_all(SEED, deterministic=DETERMINISTIC)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    centroids = np.load(RVQ_PATH)["centroids"].astype("float32")

    # load sentence LM
    ckpt = torch.load(SENT_LM_PATH, map_location="cpu")
    lm = SentenceLM(ckpt["cfg"]).to(DEVICE).eval()
    lm.load_state_dict(ckpt["state_dict"])

    # embedder for coding prompt sentences
    enc_tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    enc_model = AutoModel.from_pretrained(EMB_MODEL).to(DEVICE).eval()

    # decoder
    dec_tok = AutoTokenizer.from_pretrained(DECODER_PATH)
    dec = AutoModelForSeq2SeqLM.from_pretrained(DECODER_PATH).to(DEVICE).eval()

    context_text = (
        "Paris is the capital of France. It is known for art and architecture. "
        "The Eiffel Tower is a famous landmark."
    )

    steps = []
    for step in range(GEN_STEPS):
        sents = split_sents(context_text)
        sents = sents[-CONTEXT_SENTS:]
        # if not enough, pad by repeating first
        while len(sents) < CONTEXT_SENTS:
            sents = [sents[0]] + sents

        # convert context sentences -> context codes
        ctx_codes = []
        for s in sents:
            e = embed_sentence(enc_tok, enc_model, s)
            c = encode_rvq_one(e, centroids)
            ctx_codes.extend(c)  # flatten

        ctx = torch.tensor([ctx_codes], dtype=torch.long, device=DEVICE)
        logits = lm(ctx)
        pred_codes = [int(l.argmax(dim=-1).item()) for l in logits]

        prompt = format_prompt(context_text, pred_codes)
        inp = dec_tok([prompt], return_tensors="pt", truncation=True, max_length=MAX_INPUT).to(DEVICE)

        best_sent = None
        best_score = None
        cand_records = []

        cand_count = N_CAND if RERANK else 1
        for _ in range(cand_count):
            out = dec.generate(
                **inp,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                top_p=TOP_P,
                temperature=TEMP,
            )
            gen = dec_tok.decode(out[0], skip_special_tokens=True).strip()

            # take first sentence only
            fs = split_sents(gen)
            fs = fs[0] if fs else gen

            score = None
            if RERANK:
                e2 = embed_sentence(enc_tok, enc_model, fs)
                c2 = encode_rvq_one(e2, centroids)
                score = sum(int(a == b) for a, b in zip(pred_codes, c2))

            if SAVE_CANDIDATES:
                cand_records.append({"text": fs, "score": score})

            if best_sent is None:
                best_sent = fs
                best_score = score
                continue

            if RERANK and score is not None and (best_score is None or score > best_score):
                best_sent = fs
                best_score = score

        step_rec = {
            "step": step + 1,
            "pred_codes": pred_codes,
            "best_score": best_score,
            "selected": best_sent,
        }
        if SAVE_CANDIDATES:
            step_rec["candidates"] = cand_records
        steps.append(step_rec)

        print(f"[{step + 1}] match={best_score}/{K}" if RERANK else f"[{step + 1}]")
        print(best_sent)
        context_text = context_text.rstrip() + " " + best_sent

    meta = {
        "rvq_path": RVQ_PATH,
        "sent_lm_path": SENT_LM_PATH,
        "decoder_path": DECODER_PATH,
        "emb_model": EMB_MODEL,
        "K": K,
        "V": V,
        "context_sents": CONTEXT_SENTS,
        "gen_steps": GEN_STEPS,
        "n_cand": N_CAND,
        "code_mode": CODE_MODE,
        "rerank": RERANK,
        "do_sample": DO_SAMPLE,
        "top_p": TOP_P,
        "temperature": TEMP,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
        "steps": steps,
    }
    save_json(OUT_PATH, meta)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()

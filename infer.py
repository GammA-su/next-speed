import os
import re
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import pysbd
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

from utils import seed_all

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

segmenter = pysbd.Segmenter(language="en", clean=True)


def _strip_quotes(s: str) -> str:
    s = (s or "").strip()
    if len(s) >= 2 and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        return s[1:-1]
    return s


def _clean_path(s: str) -> str:
    s = _strip_quotes(s)
    # Strip ANSI escape sequences and control chars that can sneak in from env/terminal.
    s = re.sub(r"\x1b\[[0-9;]*m", "", s)
    s = re.sub(r"[\x00-\x1f\x7f]", "", s)
    # If ESC was stripped already, remove leftover CSI fragments like "[36m".
    s = re.sub(r"\[[0-9;]*m", "", s)
    return s.strip()


def _resolve_local_model_path(p: str) -> str | None:
    clean = _clean_path(p)
    if not clean:
        return None
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if os.path.isabs(clean):
        candidates.append(clean)
    else:
        candidates.append(os.path.abspath(clean))
        candidates.append(os.path.abspath(os.path.join(base_dir, clean)))
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    return None


def split_sents(text: str) -> List[str]:
    s = [x.strip() for x in segmenter.segment(text) if x.strip()]
    return [x for x in s if len(x) >= 5]


def mean_pool(last_hidden, attn_mask):
    mask = attn_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


@torch.no_grad()
def embed_sentence(enc_tok, enc_model, sent: str, device: str = DEVICE):
    x = enc_tok([sent], padding=True, truncation=True, max_length=256, return_tensors="pt")
    x = {k: v.to(device) for k, v in x.items()}
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


def format_prompt(ctx_text: str, codes: List[int], code_mode: str, v: int) -> str:
    if code_mode == "none":
        return f"CTX: {ctx_text}\nWrite exactly one next sentence:"
    if code_mode == "text":
        return f"CTX: {ctx_text}\nCODES: {codes_to_text(codes)}\nWrite exactly one next sentence:"
    if code_mode == "special":
        width = len(str(v - 1))
        return f"CTX: {ctx_text}\nCODES: {codes_to_special(codes, width)}\nWrite exactly one next sentence:"
    raise ValueError(f"Unknown CODE_MODE: {code_mode}")


class SentenceGenerator:
    def __init__(
        self,
        rvq_path: str = RVQ_PATH,
        sent_lm_path: str = SENT_LM_PATH,
        decoder_path: str = DECODER_PATH,
        emb_model: str = EMB_MODEL,
        device: str = DEVICE,
        k: int = K,
        v: int = V,
        context_sents: int = CONTEXT_SENTS,
        n_cand: int = N_CAND,
        code_mode: str = CODE_MODE,
        rerank: bool = RERANK,
        do_sample: bool = DO_SAMPLE,
        top_p: float = TOP_P,
        temp: float = TEMP,
        max_new_tokens: int = MAX_NEW_TOKENS,
        max_input: int = MAX_INPUT,
        save_candidates: bool = SAVE_CANDIDATES,
        seed: int = SEED,
        deterministic: bool = DETERMINISTIC,
    ) -> None:
        self.device = device
        self.rvq_path = rvq_path
        self.sent_lm_path = sent_lm_path
        self.decoder_path = decoder_path
        self.emb_model = emb_model
        self.K = k
        self.V = v
        self.context_sents = context_sents
        self.n_cand = n_cand
        self.code_mode = code_mode
        self.rerank = rerank
        self.do_sample = do_sample
        self.top_p = top_p
        self.temp = temp
        self.max_new_tokens = max_new_tokens
        self.max_input = max_input
        self.save_candidates = save_candidates
        self.seed = seed
        self.deterministic = deterministic

        seed_all(self.seed, deterministic=self.deterministic)

        self.centroids = np.load(self.rvq_path)["centroids"].astype("float32")

        ckpt = torch.load(self.sent_lm_path, map_location="cpu")
        self.lm = SentenceLM(ckpt["cfg"]).to(self.device).eval()
        self.lm.load_state_dict(ckpt["state_dict"])

        self.enc_tok = AutoTokenizer.from_pretrained(self.emb_model)
        self.enc_model = AutoModel.from_pretrained(self.emb_model).to(self.device).eval()

        dec_raw = _clean_path(os.path.expanduser(os.path.expandvars(self.decoder_path)))
        dec_path = _resolve_local_model_path(dec_raw)
        if dec_path is not None:
            self.decoder_path = dec_path
            self.dec_tok = AutoTokenizer.from_pretrained(dec_path, local_files_only=True)
            self.dec = AutoModelForSeq2SeqLM.from_pretrained(dec_path, local_files_only=True).to(self.device).eval()
        else:
            self.decoder_path = dec_raw
            self.dec_tok = AutoTokenizer.from_pretrained(dec_raw)
            self.dec = AutoModelForSeq2SeqLM.from_pretrained(dec_raw).to(self.device).eval()

    def _score_match(self, pred_codes: List[int], sent: str) -> int:
        e2 = embed_sentence(self.enc_tok, self.enc_model, sent, self.device)
        c2 = encode_rvq_one(e2, self.centroids)
        return sum(int(a == b) for a, b in zip(pred_codes, c2))

    @torch.no_grad()
    def generate(self, prompt: str, max_sentences: int, **kwargs: Any) -> Dict[str, Any]:
        seed = kwargs.pop("seed", self.seed)
        deterministic = kwargs.pop("deterministic", self.deterministic)
        if seed is not None:
            seed_all(seed, deterministic=deterministic)

        code_mode = kwargs.pop("code_mode", self.code_mode)
        rerank = kwargs.pop("rerank", self.rerank)
        n_cand = kwargs.pop("n_cand", self.n_cand)
        temp = kwargs.pop("temp", self.temp)
        top_p = kwargs.pop("top_p", self.top_p)
        do_sample = kwargs.pop("do_sample", self.do_sample)
        max_new_tokens = kwargs.pop("max_new_tokens", self.max_new_tokens)
        max_input = kwargs.pop("max_input", self.max_input)
        context_sents = kwargs.pop("context_sents", self.context_sents)
        save_candidates = kwargs.pop("save_candidates", self.save_candidates)

        if kwargs:
            raise ValueError(f"Unknown generate() params: {', '.join(sorted(kwargs.keys()))}")

        context_text = prompt.strip()
        steps = []
        matches = []
        generated = []

        for step in range(max_sentences):
            sents = split_sents(context_text)
            if not sents:
                raise ValueError("Prompt must contain at least one sentence with 5+ characters.")
            sents = sents[-context_sents:]
            while len(sents) < context_sents:
                sents = [sents[0]] + sents

            ctx_codes = []
            for s in sents:
                e = embed_sentence(self.enc_tok, self.enc_model, s, self.device)
                c = encode_rvq_one(e, self.centroids)
                ctx_codes.extend(c)

            ctx = torch.tensor([ctx_codes], dtype=torch.long, device=self.device)
            logits = self.lm(ctx)
            pred_codes = [int(l.argmax(dim=-1).item()) for l in logits]

            prompt_text = format_prompt(context_text, pred_codes, code_mode, self.V)
            inp = self.dec_tok(
                [prompt_text], return_tensors="pt", truncation=True, max_length=max_input
            ).to(self.device)

            best_sent = None
            best_score = None
            cand_records = []

            cand_count = n_cand if rerank else 1
            for _ in range(cand_count):
                out = self.dec.generate(
                    **inp,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temp,
                )
                gen = self.dec_tok.decode(out[0], skip_special_tokens=True).strip()

                fs = split_sents(gen)
                fs = fs[0] if fs else gen

                score = None
                if rerank:
                    score = self._score_match(pred_codes, fs)

                if save_candidates:
                    cand_records.append({"text": fs, "score": score})

                if best_sent is None:
                    best_sent = fs
                    best_score = score
                    continue

                if rerank and score is not None and (best_score is None or score > best_score):
                    best_sent = fs
                    best_score = score

            match_score = best_score
            if match_score is None:
                match_score = self._score_match(pred_codes, best_sent)

            step_rec = {
                "step": step + 1,
                "pred_codes": pred_codes,
                "selected": best_sent,
                "match": match_score,
                "match_str": f"{match_score}/{self.K}",
            }
            if save_candidates:
                step_rec["candidates"] = cand_records
            steps.append(step_rec)
            matches.append(step_rec["match_str"])
            generated.append(best_sent)

            context_text = context_text.rstrip() + " " + best_sent

        meta = {
            "seed": seed,
            "deterministic": deterministic,
            "K": self.K,
            "V": self.V,
            "code_mode": code_mode,
            "rerank": rerank,
            "n_cand": n_cand,
            "temp": temp,
            "top_p": top_p,
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "max_input": max_input,
            "context_sents": context_sents,
        }

        return {
            "text": context_text,
            "sentences": generated,
            "matches": matches,
            "steps": steps,
            "meta": meta,
        }

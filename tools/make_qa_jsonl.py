import json
import os
import re
import zlib

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

QA_IN = os.environ.get("QA_IN", "data/qa/train.jsonl")
OUT_TRAIN = os.environ.get("OUT_TRAIN", "data/decoder_train_qa.jsonl")
OUT_VAL = os.environ.get("OUT_VAL", "data/decoder_val_qa.jsonl")
RVQ_PATH = os.environ.get("RVQ_PATH", "out/rvq.npz")
EMB_MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

K = int(os.environ.get("K", "4"))
V = int(os.environ.get("V", "256"))
SEED = int(os.environ.get("SEED", "0"))
VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.02"))
BS = int(os.environ.get("BS", "512"))

MIN_LEN = 5
MAX_LEN = 220
MAX_SYMBOL_RATIO = 0.35

_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MARKUP_PATTERNS = [
    "{|",
    "|}",
    "align=",
    "data-sort-value",
    "{{",
    "}}",
    "<ref",
    "</ref>",
    "[[",
    "]]",
]
_CRC32_SCALE = 1 << 32


def clean_text(text):
    text = (text or "").replace("\u00a0", " ")
    return _WS.sub(" ", text).strip()


def first_sentence(text):
    parts = _SENT_SPLIT.split(text, maxsplit=1)
    return parts[0].strip() if parts else ""


def has_markup(text):
    low = text.lower()
    for pat in _MARKUP_PATTERNS:
        if pat in low:
            return True
    return False


def symbol_ratio(text):
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 1.0
    symbols = sum(1 for c in chars if not c.isalnum())
    return symbols / len(chars)


def is_val(question):
    if VAL_RATIO <= 0:
        return False
    if VAL_RATIO >= 1:
        return True
    key = f"{SEED}|{question}".encode("utf-8")
    h = zlib.crc32(key) & 0xFFFFFFFF
    return h < int(VAL_RATIO * _CRC32_SCALE)


def rvq_encode_batch(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    r = x.copy()
    codes = np.zeros((x.shape[0], centroids.shape[0]), dtype=np.int32)
    for k in range(centroids.shape[0]):
        C = centroids[k]  # [V,d]
        dots = r @ C.T  # [n,V]
        idx = np.argmax(dots, axis=1).astype(np.int32)
        codes[:, k] = idx
        r = r - C[idx]
    return codes


def main():
    if not os.path.isfile(QA_IN):
        raise FileNotFoundError(f"Missing QA_IN: {QA_IN}")
    if not os.path.isfile(RVQ_PATH):
        raise FileNotFoundError(f"Missing RVQ_PATH: {RVQ_PATH}")

    out_dir = os.path.dirname(OUT_TRAIN)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_dir = os.path.dirname(OUT_VAL)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rvq = np.load(RVQ_PATH)
    centroids = rvq["centroids"].astype("float32")
    if centroids.shape[0] != K or centroids.shape[1] != V:
        raise ValueError(f"centroids shape {centroids.shape} != (K={K}, V={V}, d)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer(EMB_MODEL, device=device)

    counts = {
        "bad_json": 0,
        "missing_fields": 0,
        "no_sentence": 0,
        "markup": 0,
        "pipe_eq": 0,
        "symbol_ratio": 0,
        "too_short": 0,
        "too_long": 0,
    }
    total_in = 0
    kept = 0
    kept_train = 0
    kept_val = 0
    samples = []

    def flush(buf, fout_train, fout_val):
        nonlocal kept, kept_train, kept_val
        if not buf:
            return
        answers = [row["tgt_text"] for row in buf]
        try:
            embs = enc.encode(
                answers,
                normalize_embeddings=True,
                batch_size=BS,
                convert_to_numpy=True,
            ).astype(np.float32)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg and device == "cuda":
                print("CUDA OOM during encode; lower BS to reduce memory.")
            raise

        codes = rvq_encode_batch(embs, centroids)
        for row, code_vec in zip(buf, codes):
            tgt_codes_str = " ".join([f"c{i}={int(c)}" for i, c in enumerate(code_vec)])
            out = {
                "ctx_text": row["ctx_text"],
                "tgt_text": row["tgt_text"],
                "tgt_codes_str": tgt_codes_str,
            }
            if row["is_val"]:
                fout_val.write(json.dumps(out, ensure_ascii=False) + "\n")
                kept_val += 1
            else:
                fout_train.write(json.dumps(out, ensure_ascii=False) + "\n")
                kept_train += 1
            kept += 1
            if len(samples) < 3:
                samples.append(out)

    with open(QA_IN, "r", encoding="utf-8") as fin, open(
        OUT_TRAIN, "w", encoding="utf-8"
    ) as ftrain, open(OUT_VAL, "w", encoding="utf-8") as fval:
        buf = []
        for line in fin:
            total_in += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                counts["bad_json"] += 1
                continue

            q = clean_text(rec.get("question", ""))
            a = clean_text(rec.get("answer", ""))
            if not q or not a:
                counts["missing_fields"] += 1
                continue

            sent = first_sentence(a)
            if not sent:
                counts["no_sentence"] += 1
                continue
            if has_markup(sent):
                counts["markup"] += 1
                continue
            if sent.count("|") >= 4 or sent.count("=") >= 6:
                counts["pipe_eq"] += 1
                continue
            if symbol_ratio(sent) > MAX_SYMBOL_RATIO:
                counts["symbol_ratio"] += 1
                continue
            if len(sent) < MIN_LEN:
                counts["too_short"] += 1
                continue
            if len(sent) > MAX_LEN:
                counts["too_long"] += 1
                continue

            ctx_text = f"### Instruction:\n{q}\n\n### Response:\n"
            buf.append({"ctx_text": ctx_text, "tgt_text": sent, "is_val": is_val(q)})
            if len(buf) >= BS:
                flush(buf, ftrain, fval)
                buf = []

        if buf:
            flush(buf, ftrain, fval)

    skipped = total_in - kept
    print(
        json.dumps(
            {
                "total_in": total_in,
                "kept": kept,
                "kept_train": kept_train,
                "kept_val": kept_val,
                "skipped": skipped,
                "skipped_by_reason": counts,
                "sample": samples,
            },
            indent=2,
        )
    )
    print(f"OUT_TRAIN={OUT_TRAIN}")
    print(f"OUT_VAL={OUT_VAL}")


if __name__ == "__main__":
    main()

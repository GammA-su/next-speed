import json
import os
import re
import zlib

import numpy as np
import torch
import pysbd
from sentence_transformers import SentenceTransformer

QA_IN = os.environ.get("QA_IN", "data/instruct.jsonl")
OUT_TRAIN = os.environ.get("OUT_TRAIN", "data/decoder_train_instruct.jsonl")
OUT_VAL = os.environ.get("OUT_VAL", "data/decoder_val_instruct.jsonl")
RVQ_PATH = os.environ.get("RVQ_PATH", "out/rvq.npz")
EMB_MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

K = int(os.environ.get("K", "4"))
V = int(os.environ.get("V", "256"))
SEED = int(os.environ.get("SEED", "0"))
VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.02"))
BS = int(os.environ.get("BS", "512"))
MIN_LEN = int(os.environ.get("MIN_LEN", "5"))
MAX_LEN = int(os.environ.get("MAX_LEN", "220"))

_WS = re.compile(r"\s+")
_CTRL_TOK = re.compile(r"(?:<c\d+_\d+>|\bc\d+_\d+\b)")
_CRC32_SCALE = 1 << 32

segmenter = pysbd.Segmenter(language="en", clean=True)


def clean_text(text):
    text = (text or "").replace("\u00a0", " ")
    return _WS.sub(" ", text).strip()


def split_sents(text):
    sents = [s.strip() for s in segmenter.segment(text) if s.strip()]
    return sents


def strip_control_tokens(text):
    text = _CTRL_TOK.sub("", text)
    return _WS.sub(" ", text).strip()


def normalize_role(role):
    role = (role or "").strip().lower()
    if role == "system":
        return "System"
    if role == "user":
        return "User"
    if role == "assistant":
        return "Assistant"
    return role[:1].upper() + role[1:] if role else "Unknown"


def serialize_history(history):
    out = []
    for role, content in history:
        out.append(f"### {role}:\n{content}\n\n")
    return "".join(out)


def is_val(row_id):
    if VAL_RATIO <= 0:
        return False
    if VAL_RATIO >= 1:
        return True
    key = f"{SEED}|{row_id}".encode("utf-8")
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
        rn = np.linalg.norm(r, axis=1, keepdims=True) + 1e-8
        r = r / rn
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
    centroids = np.ascontiguousarray(rvq["centroids"], dtype=np.float32)
    if centroids.shape[0] != K or centroids.shape[1] != V:
        raise ValueError(f"centroids shape {centroids.shape} != (K={K}, V={V}, d)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer(EMB_MODEL, device=device)

    counts = {
        "bad_json": 0,
        "missing_messages": 0,
        "missing_content": 0,
        "no_assistant": 0,
        "control_stripped_empty": 0,
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
        for line_idx, line in enumerate(fin):
            total_in += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                counts["bad_json"] += 1
                continue

            messages = rec.get("messages")
            if not isinstance(messages, list):
                counts["missing_messages"] += 1
                continue

            row_id = (
                rec.get("id")
                or rec.get("conversation_id")
                or rec.get("doc_id")
                or rec.get("uid")
                or str(line_idx)
            )
            row_is_val = is_val(row_id)

            history = []
            saw_assistant = False
            for msg in messages:
                role = normalize_role(msg.get("role"))
                content = clean_text(msg.get("content", ""))
                if not content:
                    counts["missing_content"] += 1
                    continue

                if role != "Assistant":
                    history.append((role, content))
                    continue

                saw_assistant = True
                sents = split_sents(content)
                prefix = ""
                for sent in sents:
                    tgt = strip_control_tokens(sent)
                    if not tgt:
                        counts["control_stripped_empty"] += 1
                        prefix = (prefix + " " + sent).strip()
                        continue
                    if len(tgt) < MIN_LEN:
                        counts["too_short"] += 1
                        prefix = (prefix + " " + sent).strip()
                        continue
                    if len(tgt) > MAX_LEN:
                        counts["too_long"] += 1
                        prefix = (prefix + " " + sent).strip()
                        continue

                    ctx = serialize_history(history) + f"### Assistant:\n{prefix}\n"
                    buf.append({"ctx_text": ctx, "tgt_text": tgt, "is_val": row_is_val})
                    if len(buf) >= BS:
                        flush(buf, ftrain, fval)
                        buf = []

                    prefix = (prefix + " " + sent).strip()

                history.append((role, content))

            if not saw_assistant:
                counts["no_assistant"] += 1

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

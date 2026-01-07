import json
import os
import time
import zlib
from collections import deque

import numpy as np

from utils import seed_all

SENTENCES_JSONL = os.environ.get("SENTENCES_JSONL", "data/sentences.jsonl")
RVQ_PATH = os.environ.get("RVQ_PATH", "out/rvq.npz")
EMB_PATH = os.environ.get("EMB_PATH", "data/embeddings.npy")

OUT_DEC = os.environ.get("OUT_DEC", "data/decoder_train.jsonl")
OUT_DEC_VAL = os.environ.get("OUT_DEC_VAL", "data/decoder_val.jsonl")
OUT_LM = os.environ.get("OUT_LM", "data/sent_lm_train.jsonl")
OUT_LM_VAL = os.environ.get("OUT_LM_VAL", "data/sent_lm_val.jsonl")

K = int(os.environ.get("K", "4"))
V = int(os.environ.get("V", "256"))
CONTEXT_SENTS = int(os.environ.get("CONTEXT_SENTS", "6"))
VAL_RATIO = float(os.environ.get("VAL_RATIO", os.environ.get("VAL_FRAC", "0.05")))
SEED = int(os.environ.get("SEED", "0"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "1") == "1"

_CRC32_SCALE = 1 << 32


def _is_val_doc(doc_id):
    if VAL_RATIO <= 0:
        return False
    if VAL_RATIO >= 1:
        return True
    key = f"{SEED}|{doc_id}".encode("utf-8")
    h = zlib.crc32(key) & 0xFFFFFFFF
    return h < int(VAL_RATIO * _CRC32_SCALE)


def _rvq_encode_one(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    r = np.asarray(x, dtype=np.float32).copy()
    rn = np.linalg.norm(r) + 1e-8
    r = r / rn
    codes = np.zeros((centroids.shape[0],), dtype=np.int32)
    for k in range(centroids.shape[0]):
        C = centroids[k]
        dots = C @ r
        idx = int(np.argmax(dots))
        codes[k] = idx
        r = r - C[idx]
        rn = np.linalg.norm(r) + 1e-8
        r = r / rn
    return codes


def main():
    seed_all(SEED, deterministic=DETERMINISTIC)

    if not os.path.isfile(SENTENCES_JSONL):
        raise FileNotFoundError(f"Missing sentences: {SENTENCES_JSONL}")
    if not os.path.isfile(RVQ_PATH):
        raise FileNotFoundError(f"Missing RVQ: {RVQ_PATH}")
    if not os.path.isfile(EMB_PATH):
        raise FileNotFoundError(f"Missing embeddings: {EMB_PATH} (run 02_embed_sentences.py)")

    rvq = np.load(RVQ_PATH)
    centroids = rvq["centroids"].astype("float32")
    if centroids.ndim != 3:
        raise ValueError(f"Expected centroids to be 3D, got shape {centroids.shape}")
    if centroids.shape[0] != K or centroids.shape[1] != V:
        raise ValueError(f"Centroids shape mismatch: {centroids.shape} vs K={K} V={V}")

    norms = np.linalg.norm(centroids, axis=2, keepdims=True) + 1e-8
    centroids = centroids / norms

    emb = np.load(EMB_PATH, mmap_mode="r")
    if emb.ndim != 2:
        raise ValueError(f"Expected embeddings to be 2D, got shape {emb.shape}")
    if emb.shape[1] != centroids.shape[2]:
        raise ValueError(
            f"Embedding dim {emb.shape[1]} != centroid dim {centroids.shape[2]}"
        )

    for path in [OUT_DEC, OUT_DEC_VAL, OUT_LM, OUT_LM_VAL]:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    counts = {
        "lm_train": 0,
        "lm_val": 0,
        "dec_train": 0,
        "dec_val": 0,
    }
    docs_seen = 0
    sents_seen = 0
    total_rows = 0
    ctx_chars_sum = 0
    tgt_chars_sum = 0

    idx = 0
    current_doc_id = None
    is_val = False
    context = deque(maxlen=CONTEXT_SENTS)
    start = time.time()

    with open(SENTENCES_JSONL, "r", encoding="utf-8") as fin, \
        open(OUT_LM, "w", encoding="utf-8") as flm_train, \
        open(OUT_LM_VAL, "w", encoding="utf-8") as flm_val, \
        open(OUT_DEC, "w", encoding="utf-8") as fdec_train, \
        open(OUT_DEC_VAL, "w", encoding="utf-8") as fdec_val:
        flm = flm_train
        fdec = fdec_train

        for line in fin:
            rec = json.loads(line)
            text = rec.get("text")
            if text is None:
                text = rec.get("content", "")
            text = str(text or "").strip()
            doc_id = rec.get("doc_id")
            if doc_id is None:
                doc_id = current_doc_id if current_doc_id is not None else 0

            if doc_id != current_doc_id:
                current_doc_id = doc_id
                is_val = _is_val_doc(doc_id)
                flm = flm_val if is_val else flm_train
                fdec = fdec_val if is_val else fdec_train
                context.clear()
                docs_seen += 1

            if not text:
                idx += 1
                sents_seen += 1
                continue

            if idx >= emb.shape[0]:
                raise RuntimeError(
                    f"Embeddings shorter than sentences: idx={idx} emb_rows={emb.shape[0]}"
                )
            codes = _rvq_encode_one(emb[idx], centroids)
            idx += 1
            sents_seen += 1

            if len(context) == CONTEXT_SENTS:
                ctx_text = " ".join([t[0] for t in context])
                ctx_codes = [int(c) for t in context for c in t[1]]
                tgt_codes = [int(c) for c in codes.tolist()]

                flm.write(json.dumps({"ctx_codes": ctx_codes, "tgt_codes": tgt_codes}))
                flm.write("\n")
                if is_val:
                    counts["lm_val"] += 1
                else:
                    counts["lm_train"] += 1

                tgt_codes_str = " ".join([f"c{j}={tgt_codes[j]}" for j in range(len(tgt_codes))])
                out = {
                    "ctx_text": ctx_text,
                    "tgt_text": text,
                    "tgt_codes_str": tgt_codes_str,
                }
                if "doc_id" in rec:
                    out["doc_id"] = rec["doc_id"]
                if "sent_id" in rec:
                    out["sent_id"] = rec["sent_id"]
                fdec.write(json.dumps(out, ensure_ascii=False))
                fdec.write("\n")
                if is_val:
                    counts["dec_val"] += 1
                else:
                    counts["dec_train"] += 1

                total_rows += 1
                ctx_chars_sum += len(ctx_text)
                tgt_chars_sum += len(text)

            context.append((text, codes))

    if idx < emb.shape[0]:
        print(
            f"Warning: embeddings longer than sentences (emb_rows={emb.shape[0]} sent_rows={idx})"
        )

    elapsed = max(1e-6, time.time() - start)
    avg_ctx_chars = ctx_chars_sum / max(1, total_rows)
    avg_tgt_chars = tgt_chars_sum / max(1, total_rows)
    rows_per_sec = total_rows / elapsed

    print(
        "docs_seen={docs_seen} sents_seen={sents_seen} rows_written={rows_written} rows/sec={rows_per_sec:.2f}".format(
            docs_seen=docs_seen,
            sents_seen=sents_seen,
            rows_written=total_rows,
            rows_per_sec=rows_per_sec,
        )
    )
    print(
        "lm_train={lm_train} lm_val={lm_val} dec_train={dec_train} dec_val={dec_val}".format(
            **counts
        )
    )
    print(
        "avg_ctx_chars={avg_ctx_chars:.1f} avg_tgt_chars={avg_tgt_chars:.1f} context_sents={context_sents}".format(
            avg_ctx_chars=avg_ctx_chars,
            avg_tgt_chars=avg_tgt_chars,
            context_sents=CONTEXT_SENTS,
        )
    )
    print("Wrote:", OUT_LM)
    print("Wrote:", OUT_LM_VAL)
    print("Wrote:", OUT_DEC)
    print("Wrote:", OUT_DEC_VAL)


if __name__ == "__main__":
    main()

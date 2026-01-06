import os
import json
import time
import zlib
from collections import deque

import numpy as np

from utils import seed_all, save_json

try:
    import orjson as _orjson
except ImportError:
    _orjson = None

SENT_PATH = os.environ.get("SENT_PATH", "data/sentences.jsonl")
EMB_PATH = os.environ.get("EMB_PATH", "data/embeddings.npy")
RVQ_PATH = os.environ.get("RVQ_PATH", "out/rvq.npz")

OUT_CODES = os.environ.get("OUT_CODES", "data/codes.npy")
OUT_LM = os.environ.get("OUT_LM", "data/sent_lm_train.jsonl")
OUT_LM_VAL = os.environ.get("OUT_LM_VAL", "data/sent_lm_val.jsonl")
OUT_DEC = os.environ.get("OUT_DEC", "data/decoder_train.jsonl")
OUT_DEC_VAL = os.environ.get("OUT_DEC_VAL", "data/decoder_val.jsonl")
META = os.environ.get("TRAIN_META", "data/training_meta.json")

K = int(os.environ.get("K", "8"))
V = int(os.environ.get("V", "1024"))
CONTEXT_SENTS = int(os.environ.get("CONTEXT_SENTS", "6"))
VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.05"))
SEED = int(os.environ.get("SEED", "0"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "1") == "1"
MAX_ROWS = int(os.environ.get("MAX_ROWS", "-1"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "200000"))
REUSE_CODES = os.environ.get("REUSE_CODES", "1") == "1"

_IO_BUFFER = 1024 * 1024
_CRC32_SCALE = 1 << 32


if _orjson is not None:
    def _json_loads(line):
        return _orjson.loads(line)

    def _dumps_lm(obj):
        if hasattr(_orjson, "OPT_ESCAPE_UNICODE"):
            return _orjson.dumps(obj, option=_orjson.OPT_ESCAPE_UNICODE)
        return _orjson.dumps(obj)

    def _dumps_dec(obj):
        return _orjson.dumps(obj)
else:
    def _json_loads(line):
        return json.loads(line.decode("utf-8"))

    def _dumps_lm(obj):
        return json.dumps(obj, ensure_ascii=True).encode("utf-8")

    def _dumps_dec(obj):
        return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _is_val_doc(doc_id):
    if VAL_FRAC <= 0:
        return False
    if VAL_FRAC >= 1:
        return True
    key = f"{SEED}|{doc_id}".encode("utf-8")
    h = zlib.crc32(key) & 0xFFFFFFFF
    return h < int(VAL_FRAC * _CRC32_SCALE)


def main():
    seed_all(SEED, deterministic=DETERMINISTIC)
    os.makedirs(os.path.dirname(OUT_CODES), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_LM), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_LM_VAL), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_DEC), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_DEC_VAL), exist_ok=True)

    if not REUSE_CODES:
        raise ValueError("REUSE_CODES=0 is not supported; run step 3 to generate codes.")
    if not os.path.isfile(OUT_CODES):
        raise FileNotFoundError(f"Missing codes file: {OUT_CODES}")
    if not os.path.isfile(SENT_PATH):
        raise FileNotFoundError(f"Missing sentences file: {SENT_PATH}")

    print("Phase: load codes")
    codes = np.load(OUT_CODES, mmap_mode="r")
    if codes.ndim != 2:
        raise ValueError(f"Expected codes to be 2D, got shape {codes.shape}")
    if codes.shape[1] != K:
        raise ValueError(f"codes K mismatch: codes.shape[1]={codes.shape[1]} K={K}")
    print("Loaded codes:", OUT_CODES, codes.shape)

    counts = {
        "lm_train": 0,
        "lm_val": 0,
        "dec_train": 0,
        "dec_val": 0,
    }
    total_rows = 0
    sent_rows = 0
    docs_seen = 0
    start = time.time()
    next_log_rows = LOG_EVERY if LOG_EVERY > 0 else None
    next_log_seen = LOG_EVERY if LOG_EVERY > 0 else None

    def maybe_log_rows():
        nonlocal next_log_rows
        if next_log_rows is None:
            return
        if total_rows >= next_log_rows:
            elapsed = time.time() - start
            rate = total_rows / elapsed if elapsed > 0 else 0.0
            print(
                f"rows_written={total_rows} sent_rows={sent_rows} "
                f"docs_seen={docs_seen} elapsed={elapsed:.1f}s rows/sec={rate:.2f}",
                flush=True,
            )
            next_log_rows += LOG_EVERY

    def maybe_log_seen():
        nonlocal next_log_seen
        if next_log_seen is None:
            return
        if sent_rows >= next_log_seen:
            elapsed = time.time() - start
            rate = sent_rows / elapsed if elapsed > 0 else 0.0
            print(
                f"sent_rows={sent_rows} docs_seen={docs_seen} "
                f"elapsed={elapsed:.1f}s sents/sec={rate:.2f}",
                flush=True,
            )
            next_log_seen += LOG_EVERY

    print("Phase: stream sentences + build outputs")
    stop_early = False
    with open(SENT_PATH, "rb", buffering=_IO_BUFFER) as fin, \
        open(OUT_LM, "wb", buffering=_IO_BUFFER) as flm_train, \
        open(OUT_LM_VAL, "wb", buffering=_IO_BUFFER) as flm_val, \
        open(OUT_DEC, "wb", buffering=_IO_BUFFER) as fdec_train, \
        open(OUT_DEC_VAL, "wb", buffering=_IO_BUFFER) as fdec_val:
        idx = 0
        context = deque(maxlen=CONTEXT_SENTS)
        current_doc_id = None
        is_val = False
        flm = flm_train
        fdec = fdec_train

        for line in fin:
            rec = _json_loads(line)
            doc_id = rec["doc_id"]
            if doc_id != current_doc_id:
                current_doc_id = doc_id
                is_val = _is_val_doc(doc_id)
                flm = flm_val if is_val else flm_train
                fdec = fdec_val if is_val else fdec_train
                context.clear()
                docs_seen += 1

            if idx >= codes.shape[0]:
                raise RuntimeError(
                    f"Codes shorter than sentences: idx={idx} codes_rows={codes.shape[0]}"
                )

            text = rec["text"]
            sent_id = rec["sent_id"]

            if len(context) == CONTEXT_SENTS and not stop_early:
                ctx_idxs = [t[0] for t in context]
                ctx_codes = codes[ctx_idxs].reshape(-1).tolist()
                tgt_codes = codes[idx].tolist()
                ctx_text = " ".join([t[1] for t in context])

                flm.write(_dumps_lm({"ctx_codes": ctx_codes, "tgt_codes": tgt_codes}))
                flm.write(b"\n")
                if is_val:
                    counts["lm_val"] += 1
                else:
                    counts["lm_train"] += 1

                code_str = " ".join(
                    [f"c{j}={tgt_codes[j]}" for j in range(len(tgt_codes))]
                )
                fdec.write(
                    _dumps_dec(
                        {
                            "ctx_text": ctx_text,
                            "ctx_codes": ctx_codes,
                            "tgt_codes": tgt_codes,
                            "tgt_codes_str": code_str,
                            "tgt_text": text,
                            "doc_id": doc_id,
                            "sent_id": sent_id,
                        }
                    )
                )
                fdec.write(b"\n")
                if is_val:
                    counts["dec_val"] += 1
                else:
                    counts["dec_train"] += 1

                total_rows += 1
                maybe_log_rows()
                if MAX_ROWS > 0 and total_rows >= MAX_ROWS:
                    stop_early = True

            context.append((idx, text))
            idx += 1
            sent_rows = idx
            maybe_log_seen()

        if idx != codes.shape[0]:
            raise RuntimeError(
                f"Sentences rows ({idx}) != codes rows ({codes.shape[0]})"
            )

    meta = {
        "sent_path": SENT_PATH,
        "emb_path": EMB_PATH,
        "rvq_path": RVQ_PATH,
        "out_codes": OUT_CODES,
        "out_lm_train": OUT_LM,
        "out_lm_val": OUT_LM_VAL,
        "out_dec_train": OUT_DEC,
        "out_dec_val": OUT_DEC_VAL,
        "K": K,
        "V": V,
        "context_sents": CONTEXT_SENTS,
        "val_frac": VAL_FRAC,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
        "counts": counts,
    }
    save_json(META, meta)

    elapsed = time.time() - start
    rate = total_rows / elapsed if elapsed > 0 else 0.0
    print(f"Rows written: {total_rows} rows/sec={rate:.2f}")
    print("Wrote:", OUT_LM)
    print("Wrote:", OUT_LM_VAL)
    print("Wrote:", OUT_DEC)
    print("Wrote:", OUT_DEC_VAL)
    print("Meta:", META)


if __name__ == "__main__":
    main()

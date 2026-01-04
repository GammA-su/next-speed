import os
import json
import sys

os.environ.setdefault("USE_TORCH", "0")

from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import seed_all, save_json

DATASET = os.environ.get("HF_DATASET", "wikimedia/wikipedia")
CONFIG = os.environ.get("HF_CONFIG", "20231101.en")
SPLIT = os.environ.get("HF_SPLIT", "train")

OUT_PATH = os.environ.get("OUT_PATH", "data/raw/docs.jsonl")
OUT_META = os.environ.get("OUT_META", "data/raw/docs.meta.json")

STREAMING = os.environ.get("STREAMING", "1") == "1"
SHUFFLE_BUFFER = int(os.environ.get("SHUFFLE_BUFFER", "10000"))
MIN_CHARS = int(os.environ.get("MIN_CHARS", "200"))
MAX_DOCS = int(os.environ.get("MAX_DOCS", "-1"))

SEED = int(os.environ.get("SEED", "0"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "1") == "1"


def main():
    seed_all(SEED, deterministic=DETERMINISTIC)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    ds = load_dataset(DATASET, CONFIG, split=SPLIT, streaming=STREAMING)
    if SHUFFLE_BUFFER > 0:
        if STREAMING:
            ds = ds.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)
        else:
            ds = ds.shuffle(seed=SEED)

    docs_seen = 0
    docs_written = 0
    skipped_empty = 0
    skipped_short = 0

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in ds:
            docs_seen += 1
            text = ex.get("text", "")
            if not text or not text.strip():
                skipped_empty += 1
                continue
            if MIN_CHARS > 0 and len(text) < MIN_CHARS:
                skipped_short += 1
                continue

            rec = {
                "doc_id": docs_written,
                "title": ex.get("title"),
                "url": ex.get("url"),
                "text": text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            docs_written += 1

            if MAX_DOCS > 0 and docs_written >= MAX_DOCS:
                break

    meta = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "streaming": STREAMING,
        "shuffle_buffer": SHUFFLE_BUFFER,
        "min_chars": MIN_CHARS,
        "max_docs": MAX_DOCS,
        "docs_seen": docs_seen,
        "docs_written": docs_written,
        "skipped_empty": skipped_empty,
        "skipped_short": skipped_short,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
        "out_path": OUT_PATH,
    }
    save_json(OUT_META, meta)

    print("Wrote:", OUT_PATH)
    print("Meta:", OUT_META)


if __name__ == "__main__":
    main()

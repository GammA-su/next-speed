import os
import re
import json
import time
from multiprocessing import Pool

import pysbd
from datasets import load_dataset

from utils import seed_all, save_json

OUT = os.environ.get("SENT_PATH", "data/sentences.jsonl")
META = os.environ.get("SENT_META", "data/sentences.meta.json")
DATASET = os.environ.get("HF_DATASET", "wikitext")
CONFIG = os.environ.get("HF_CONFIG", "wikitext-2-raw-v1")
SPLIT = os.environ.get("HF_SPLIT", "train")
DOCS_JSONL = os.environ.get("DOCS_JSONL", "")
LANG = os.environ.get("LANG", "en")
MIN_LEN = int(os.environ.get("MIN_LEN", "15"))
MAX_LEN = int(os.environ.get("MAX_LEN", "300"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "1000"))
MAX_DOCS = int(os.environ.get("MAX_DOCS", "-1"))
MAX_DOC_CHARS = int(os.environ.get("MAX_DOC_CHARS", "200000"))
SEED = int(os.environ.get("SEED", "0"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "1") == "1"

segmenter = pysbd.Segmenter(language=LANG, clean=True)
segmenter_raw = pysbd.Segmenter(language=LANG, clean=False)
_worker_segmenter = None
_worker_segmenter_raw = None
_worker_min_len = None
_worker_max_len = None


def _segment_text(segmenter_clean, segmenter_fallback, text: str):
    try:
        return [s.strip() for s in segmenter_clean.segment(text) if s.strip()]
    except re.error:
        # pysbd cleaner can choke on bad escapes; retry without cleaning.
        try:
            return [s.strip() for s in segmenter_fallback.segment(text) if s.strip()]
        except re.error:
            return []


def segment(text: str):
    sents = _segment_text(segmenter, segmenter_raw, text)
    # basic filters (tune)
    sents = [s for s in sents if MIN_LEN <= len(s) <= MAX_LEN]
    return sents


def _init_worker(lang: str, min_len: int, max_len: int) -> None:
    global _worker_segmenter, _worker_segmenter_raw, _worker_min_len, _worker_max_len
    _worker_segmenter = pysbd.Segmenter(language=lang, clean=True)
    _worker_segmenter_raw = pysbd.Segmenter(language=lang, clean=False)
    _worker_min_len = min_len
    _worker_max_len = max_len


def _segment_worker(task):
    doc_id, extra, text = task
    sents = _segment_text(_worker_segmenter, _worker_segmenter_raw, text)
    sents = [s for s in sents if _worker_min_len <= len(s) <= _worker_max_len]
    return doc_id, extra, sents


def main():
    seed_all(SEED, deterministic=DETERMINISTIC)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    sent_id = 0
    num_docs = 0

    if DOCS_JSONL:
        docs_seen = 0
        docs_invalid = 0
        docs_skipped = 0
        docs_written = 0
        sents_written = 0
        docs_processed = 0
        start = time.time()
        next_log = LOG_EVERY if LOG_EVERY > 0 else None

        def log_progress():
            elapsed = time.time() - start
            rate = docs_seen / elapsed if elapsed > 0 else 0.0
            print(
                f"docs_seen={docs_seen} docs_written={docs_written} "
                f"sents_written={sents_written} docs_skipped={docs_skipped} "
                f"docs_invalid={docs_invalid} elapsed={elapsed:.1f}s "
                f"docs/sec={rate:.2f}",
                flush=True,
            )

        def maybe_log():
            nonlocal next_log
            if next_log is None:
                return
            if docs_processed >= next_log:
                log_progress()
                next_log += LOG_EVERY

        def iter_docs(fin):
            nonlocal docs_seen, docs_invalid, docs_skipped, docs_processed
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                if MAX_DOCS > 0 and docs_seen >= MAX_DOCS:
                    break
                docs_seen += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    docs_invalid += 1
                    docs_processed += 1
                    maybe_log()
                    continue

                text = rec.get("text", "")
                if not text or not text.strip():
                    docs_skipped += 1
                    docs_processed += 1
                    maybe_log()
                    continue
                if MAX_DOC_CHARS > 0 and len(text) > MAX_DOC_CHARS:
                    docs_skipped += 1
                    docs_processed += 1
                    maybe_log()
                    continue

                fallback_id = docs_seen - 1
                doc_val = rec.get("doc_id", fallback_id)
                try:
                    doc_val = int(doc_val)
                except (TypeError, ValueError):
                    doc_val = fallback_id

                extra = {}
                if "title" in rec:
                    extra["title"] = rec.get("title")
                if "url" in rec:
                    extra["url"] = rec.get("url")

                yield doc_val, extra, text

        with open(DOCS_JSONL, "r", encoding="utf-8") as fin, open(
            OUT, "w", encoding="utf-8"
        ) as fout:
            if NUM_WORKERS > 0:
                with Pool(
                    processes=NUM_WORKERS,
                    initializer=_init_worker,
                    initargs=(LANG, MIN_LEN, MAX_LEN),
                ) as pool:
                    for doc_val, extra, sents in pool.imap(_segment_worker, iter_docs(fin), chunksize=16):
                        if not sents:
                            docs_skipped += 1
                            docs_processed += 1
                            maybe_log()
                            continue
                        for s in sents:
                            out_rec = {"doc_id": doc_val, "sent_id": sent_id, "text": s}
                            if extra:
                                out_rec.update(extra)
                            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                            sent_id += 1
                            sents_written += 1
                        num_docs += 1
                        docs_written += 1
                        docs_processed += 1
                        maybe_log()
            else:
                for doc_val, extra, text in iter_docs(fin):
                    sents = segment(text)
                    if not sents:
                        docs_skipped += 1
                        docs_processed += 1
                        maybe_log()
                        continue
                    for s in sents:
                        out_rec = {"doc_id": doc_val, "sent_id": sent_id, "text": s}
                        if extra:
                            out_rec.update(extra)
                        fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                        sent_id += 1
                        sents_written += 1
                    num_docs += 1
                    docs_written += 1
                    docs_processed += 1
                    maybe_log()

        meta = {
            "mode": "docs_jsonl",
            "docs_jsonl": DOCS_JSONL,
            "lang": LANG,
            "min_len": MIN_LEN,
            "max_len": MAX_LEN,
            "sent_path": OUT,
            "num_docs": num_docs,
            "num_sents": sent_id,
            "docs_seen": docs_seen,
            "docs_written": docs_written,
            "sents_written": sents_written,
            "docs_invalid": docs_invalid,
            "docs_skipped": docs_skipped,
            "num_workers": NUM_WORKERS,
            "log_every": LOG_EVERY,
            "max_docs": MAX_DOCS,
            "max_doc_chars": MAX_DOC_CHARS,
            "seed": SEED,
            "deterministic": DETERMINISTIC,
        }
        save_json(META, meta)

        print("Wrote:", OUT)
        print("Meta:", META)
        return

    ds = load_dataset(DATASET, CONFIG, split=SPLIT)
    doc_id = 0
    buf = []

    def flush():
        nonlocal doc_id, sent_id, buf, num_docs
        if not buf:
            return
        text = " ".join(buf)
        sents = segment(text)
        if sents:
            for s in sents:
                rec = {"doc_id": doc_id, "sent_id": sent_id, "text": s}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sent_id += 1
            doc_id += 1
            num_docs += 1
        buf = []

    with open(OUT, "w", encoding="utf-8") as f:
        for ex in ds:
            t = ex.get("text", "")
            if not t.strip():
                flush()
                continue
            # wikitext has short lines; we buffer them into a doc until blank line
            buf.append(t.strip())
        flush()

    meta = {
        "mode": "hf",
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "lang": LANG,
        "min_len": MIN_LEN,
        "max_len": MAX_LEN,
        "sent_path": OUT,
        "num_docs": num_docs,
        "num_sents": sent_id,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
    }
    save_json(META, meta)

    print("Wrote:", OUT)
    print("Meta:", META)


if __name__ == "__main__":
    main()

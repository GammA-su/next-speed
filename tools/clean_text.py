import json
import os
import re

DOCS_JSONL = os.environ.get("DOCS_JSONL", "data/raw/docs.jsonl")
OUT_JSONL = os.environ.get("OUT_JSONL", "data/raw/docs_clean.jsonl")

MIN_LEN = int(os.environ.get("MIN_LEN", "25"))
MAX_SYMBOL_RATIO = float(os.environ.get("MAX_SYMBOL_RATIO", "0.35"))

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


def symbol_ratio(text):
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 1.0
    symbols = sum(1 for c in chars if not c.isalnum())
    return symbols / len(chars)


def iter_paragraphs(text):
    buf = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if buf:
                yield " ".join(buf)
                buf = []
            continue
        buf.append(line)
    if buf:
        yield " ".join(buf)


def has_markup(text):
    low = text.lower()
    for pat in _MARKUP_PATTERNS:
        if pat in low:
            return True
    return False


def clean_doc(text, counts):
    kept = []
    for para in iter_paragraphs(text):
        counts["paras_seen"] += 1
        if not para.strip():
            counts["dropped_empty"] += 1
            continue
        if has_markup(para):
            counts["dropped_markup"] += 1
            continue
        if para.count("|") >= 4 or para.count("=") >= 6:
            counts["dropped_pipe_eq"] += 1
            continue
        if len(para) < MIN_LEN:
            counts["dropped_short"] += 1
            continue
        if symbol_ratio(para) > MAX_SYMBOL_RATIO:
            counts["dropped_symbol_ratio"] += 1
            continue
        kept.append(para)
    counts["paras_kept"] += len(kept)
    return "\n\n".join(kept)


def main():
    out_dir = os.path.dirname(OUT_JSONL)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    counts = {
        "paras_seen": 0,
        "paras_kept": 0,
        "dropped_empty": 0,
        "dropped_markup": 0,
        "dropped_pipe_eq": 0,
        "dropped_short": 0,
        "dropped_symbol_ratio": 0,
    }
    total_docs = 0
    kept_docs = 0
    skipped_docs = 0

    with open(DOCS_JSONL, "r", encoding="utf-8") as fin, open(
        OUT_JSONL, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            total_docs += 1
            rec = json.loads(line)
            text = rec.get("text")
            if text is None:
                text = rec.get("content", "")
            cleaned = clean_doc(text, counts)
            if not cleaned:
                skipped_docs += 1
                continue
            if "text" in rec:
                rec["text"] = cleaned
            elif "content" in rec:
                rec["content"] = cleaned
            else:
                rec["text"] = cleaned
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept_docs += 1

    print(f"in={DOCS_JSONL}")
    print(f"out={OUT_JSONL}")
    print(
        "total_docs={total_docs} kept_docs={kept_docs} skipped_docs={skipped_docs}".format(
            total_docs=total_docs,
            kept_docs=kept_docs,
            skipped_docs=skipped_docs,
        )
    )
    print(
        "paras_seen={paras_seen} paras_kept={paras_kept} dropped_empty={dropped_empty} "
        "dropped_markup={dropped_markup} dropped_pipe_eq={dropped_pipe_eq} "
        "dropped_short={dropped_short} dropped_symbol_ratio={dropped_symbol_ratio}".format(
            **counts
        )
    )


if __name__ == "__main__":
    main()

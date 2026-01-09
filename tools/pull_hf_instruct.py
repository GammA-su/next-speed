import json
import os
import re
import random

from datasets import load_dataset, get_dataset_split_names

SEED = int(os.environ.get("SEED", "0"))
MAX_TRAIN = int(os.environ.get("MAX_TRAIN", "200000"))
MAX_VAL = int(os.environ.get("MAX_VAL", "5000"))
ULTRACHAT_FRACTION = float(os.environ.get("ULTRACHAT_FRACTION", "0.5"))
OPENORCA_FRACTION = float(os.environ.get("OPENORCA_FRACTION", "0.5"))
MAX_CHARS_PER_MSG = int(os.environ.get("MAX_CHARS_PER_MSG", "2000"))
SHUFFLE_BUFFER = int(os.environ.get("SHUFFLE_BUFFER", "10000"))

OUT_TRAIN = os.environ.get("OUT_TRAIN", "data/instruct/train.jsonl")
OUT_VAL = os.environ.get("OUT_VAL", "data/instruct/val.jsonl")

_WS = re.compile(r"\s+")
_MARKUP_PATTERNS = ["{|", "|}", "align=", "data-sort-value", "<ref", "</ref>"]


def _clean_text(text):
    text = (text or "").replace("\u00a0", " ")
    text = _WS.sub(" ", text).strip()
    if not text:
        return ""
    low = text.lower()
    for pat in _MARKUP_PATTERNS:
        if pat in low:
            return ""
    if len(text) > MAX_CHARS_PER_MSG:
        text = text[:MAX_CHARS_PER_MSG].rstrip()
    return text


def _norm_role(role):
    role = (role or "").strip().lower()
    if role in ("system",):
        return "system"
    if role in ("user", "human", "prompter"):
        return "user"
    if role in ("assistant", "gpt", "bot"):
        return "assistant"
    return role if role else "user"


def _normalize_messages(raw_messages):
    messages = []
    for msg in raw_messages:
        role = _norm_role(msg.get("role") or msg.get("from") or msg.get("speaker"))
        content = msg.get("content") or msg.get("value") or msg.get("text")
        content = _clean_text(content)
        if not content:
            return None
        messages.append({"role": role, "content": content})
    return messages if messages else None


def _normalize_ultrachat(ex):
    if "messages" in ex and isinstance(ex["messages"], list):
        return _normalize_messages(ex["messages"])
    if "conversation" in ex and isinstance(ex["conversation"], list):
        return _normalize_messages(ex["conversation"])
    return None


def _normalize_openorca(ex):
    if "messages" in ex and isinstance(ex["messages"], list):
        return _normalize_messages(ex["messages"])
    system_prompt = _clean_text(ex.get("system_prompt", ""))
    question = _clean_text(ex.get("question") or ex.get("prompt") or ex.get("instruction") or "")
    response = _clean_text(ex.get("response") or ex.get("output") or ex.get("answer") or "")
    if not question or not response:
        return None
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": response})
    return messages


def _resolve_split(name, candidates):
    available = []
    try:
        available = get_dataset_split_names(name)
    except Exception:
        available = []
    for cand in candidates:
        if cand in available:
            return cand
    return candidates[0]


def _iter_dataset(name, split, seed_offset):
    try:
        ds = load_dataset(name, split=split, streaming=True)
        ds = ds.shuffle(seed=SEED + seed_offset, buffer_size=SHUFFLE_BUFFER)
        return ds
    except Exception:
        ds = load_dataset(name, split=split)
        if hasattr(ds, "shuffle"):
            ds = ds.shuffle(seed=SEED + seed_offset)
        return ds


def _write_dataset(ds, normalize_fn, max_rows, fout, stats):
    written = 0
    for ex in ds:
        if max_rows > 0 and written >= max_rows:
            break
        stats["rows_in"] += 1
        msgs = normalize_fn(ex)
        if not msgs:
            stats["rows_dropped"] += 1
            continue
        fout.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
        written += 1
        stats["rows_out"] += 1
    return written


def _alloc_counts(total, frac_a, frac_b):
    frac_sum = frac_a + frac_b
    if frac_sum <= 0:
        frac_a = frac_b = 0.5
        frac_sum = 1.0
    frac_a /= frac_sum
    frac_b /= frac_sum
    a = int(round(total * frac_a))
    b = max(0, total - a)
    return a, b


def main():
    out_dir = os.path.dirname(OUT_TRAIN)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_dir = os.path.dirname(OUT_VAL)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    uc_train = _resolve_split("HuggingFaceH4/ultrachat_200k", ["train_sft", "train"])
    uc_val = _resolve_split("HuggingFaceH4/ultrachat_200k", ["test_sft", "validation", "val", "test"])
    oo_train = _resolve_split("Open-Orca/OpenOrca", ["train"])
    oo_val = _resolve_split("Open-Orca/OpenOrca", ["validation", "val", "test", "train"])

    uc_train_target, oo_train_target = _alloc_counts(MAX_TRAIN, ULTRACHAT_FRACTION, OPENORCA_FRACTION)
    uc_val_target, oo_val_target = _alloc_counts(MAX_VAL, ULTRACHAT_FRACTION, OPENORCA_FRACTION)

    stats = {
        "ultrachat_train": {"rows_in": 0, "rows_out": 0, "rows_dropped": 0},
        "ultrachat_val": {"rows_in": 0, "rows_out": 0, "rows_dropped": 0},
        "openorca_train": {"rows_in": 0, "rows_out": 0, "rows_dropped": 0},
        "openorca_val": {"rows_in": 0, "rows_out": 0, "rows_dropped": 0},
    }

    with open(OUT_TRAIN, "w", encoding="utf-8") as ftrain:
        ds = _iter_dataset("HuggingFaceH4/ultrachat_200k", uc_train, seed_offset=0)
        _write_dataset(ds, _normalize_ultrachat, uc_train_target, ftrain, stats["ultrachat_train"])

        ds = _iter_dataset("Open-Orca/OpenOrca", oo_train, seed_offset=1)
        _write_dataset(ds, _normalize_openorca, oo_train_target, ftrain, stats["openorca_train"])

    with open(OUT_VAL, "w", encoding="utf-8") as fval:
        ds = _iter_dataset("HuggingFaceH4/ultrachat_200k", uc_val, seed_offset=2)
        _write_dataset(ds, _normalize_ultrachat, uc_val_target, fval, stats["ultrachat_val"])

        ds = _iter_dataset("Open-Orca/OpenOrca", oo_val, seed_offset=3)
        _write_dataset(ds, _normalize_openorca, oo_val_target, fval, stats["openorca_val"])

    print(json.dumps(stats, indent=2))
    total_train = stats["ultrachat_train"]["rows_out"] + stats["openorca_train"]["rows_out"]
    total_val = stats["ultrachat_val"]["rows_out"] + stats["openorca_val"]["rows_out"]
    print(f"OUT_TRAIN={OUT_TRAIN} rows={total_train}")
    print(f"OUT_VAL={OUT_VAL} rows={total_val}")


if __name__ == "__main__":
    main()

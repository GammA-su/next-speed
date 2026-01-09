import hashlib
import json
import os
from typing import List

import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from utils import seed_all, save_json

TRAIN_PATH = os.environ.get("DECODER_TRAIN", os.environ.get("TRAIN_PATH", "data/decoder_train.jsonl"))
VAL_PATH = os.environ.get("DECODER_VAL", os.environ.get("VAL_PATH", "data/decoder_val.jsonl"))
OUT_DIR = os.environ.get("OUT_DIR", "out/decoder")
RESUME_FROM = os.environ.get("RESUME_FROM", "")
MODEL = os.environ.get("MODEL", "google/flan-t5-small")
MAX_IN = int(os.environ.get("MAX_IN", "512"))
MAX_OUT = int(os.environ.get("MAX_OUT", "96"))

CODE_MODE = os.environ.get("CODE_MODE", "special")  # special | text | none
K = int(os.environ.get("K", "8"))
V = int(os.environ.get("V", "1024"))
ADD_SPECIAL_TOKENS = os.environ.get("ADD_SPECIAL_TOKENS", "1") == "1"

NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "1"))
LR = float(os.environ.get("LR", "2e-5"))
TRAIN_BS = int(os.environ.get("TRAIN_BS", "8"))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "50"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "500"))
MAX_ROWS = int(os.environ.get("MAX_ROWS", "-1"))
NUM_PROC = int(os.environ.get("NUM_PROC", "0"))
CACHE_TOKENIZED = int(os.environ.get("CACHE_TOKENIZED", "1")) == "1"
TOKENIZED_CACHE_DIR = os.environ.get("TOKENIZED_CACHE_DIR", "out/cache/tokenized_decoder")
KEEP_IN_MEMORY = int(os.environ.get("KEEP_IN_MEMORY", "0")) == "1"
PRETOKENIZE_ONLY = int(os.environ.get("PRETOKENIZE_ONLY", "0")) == "1"
MAP_BATCH_SIZE = int(os.environ.get("MAP_BATCH_SIZE", "1000"))
SEED = int(os.environ.get("SEED", "0"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "1") == "1"
FP16 = int(os.environ.get("FP16", "0"))
BF16 = int(os.environ.get("BF16", "1"))
OPTIM = os.environ.get("OPTIM", "adamw_torch")
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", "1.0"))
LOGGING_NAN_INF_FILTER = int(os.environ.get("LOGGING_NAN_INF_FILTER", "0"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_TOKENIZER_CACHE = {}


def code_token(k: int, v: int, width: int) -> str:
    return f"<c{k}_{v:0{width}d}>"


def codes_to_special(codes: List[int], width: int) -> str:
    return " ".join([code_token(i, c, width) for i, c in enumerate(codes)])


def codes_to_text(codes: List[int]) -> str:
    return " ".join([f"c{i}={c}" for i, c in enumerate(codes)])


def parse_codes_str(code_str: str) -> List[int]:
    codes = []
    for part in code_str.strip().split():
        if "=" not in part:
            continue
        try:
            codes.append(int(part.split("=")[1]))
        except ValueError:
            continue
    return codes


def build_code_tokens(k: int, v: int) -> List[str]:
    width = len(str(v - 1))
    tokens = []
    for i in range(k):
        for j in range(v):
            tokens.append(code_token(i, j, width))
    return tokens


def format_prompt(ctx_text: str, codes: List[int]) -> str:
    if CODE_MODE == "none":
        return f"CTX: {ctx_text}\nWrite exactly one next sentence:"
    if CODE_MODE == "text":
        code_str = codes_to_text(codes)
        return f"CTX: {ctx_text}\nCODES: {code_str}\nWrite exactly one next sentence:"
    if CODE_MODE == "special":
        width = len(str(V - 1))
        code_str = codes_to_special(codes, width)
        return f"CTX: {ctx_text}\nCODES: {code_str}\nWrite exactly one next sentence:"
    raise ValueError(f"Unknown CODE_MODE: {CODE_MODE}")


def _read_rows(path: str, max_rows: int) -> List[dict]:
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ctx_text = r["ctx_text"]
            tgt_text = r["tgt_text"]
            codes = r.get("tgt_codes")
            if codes is None:
                codes = parse_codes_str(r.get("tgt_codes_str", ""))
            inp = format_prompt(ctx_text, codes)
            rows.append({"input": inp, "target": tgt_text})
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def _file_sig(path: str) -> dict:
    if path and os.path.exists(path):
        return {
            "path": os.path.abspath(path),
            "size": os.path.getsize(path),
            "mtime": os.path.getmtime(path),
        }
    return {
        "path": os.path.abspath(path) if path else "",
        "size": 0,
        "mtime": 0,
        "missing": True,
    }


def _cache_key(train_path: str, val_path: str, tokenizer_name: str, vocab_size: int) -> str:
    payload = {
        "train": _file_sig(train_path),
        "val": _file_sig(val_path),
        "tokenizer": tokenizer_name,
        "vocab_size": vocab_size,
        "code_mode": CODE_MODE,
        "K": K,
        "V": V,
        "add_special_tokens": ADD_SPECIAL_TOKENS,
        "max_in": MAX_IN,
        "max_out": MAX_OUT,
        "max_rows": MAX_ROWS,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _get_tokenizer(cfg: dict):
    key = (cfg["model"], cfg["code_mode"], cfg["k"], cfg["v"], cfg["add_special_tokens"])
    tok = _TOKENIZER_CACHE.get(key)
    if tok is not None:
        return tok
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if cfg["code_mode"] == "special" and cfg["add_special_tokens"]:
        code_tokens = build_code_tokens(cfg["k"], cfg["v"])
        tok.add_special_tokens({"additional_special_tokens": code_tokens})
    _TOKENIZER_CACHE[key] = tok
    return tok


def _tokenize_fn(batch, cfg: dict):
    tok = _get_tokenizer(cfg)
    x = tok(
        batch["input"],
        padding="max_length",
        truncation=True,
        max_length=cfg["max_in"],
    )
    y = tok(
        batch["target"],
        padding="max_length",
        truncation=True,
        max_length=cfg["max_out"],
    )
    x["labels"] = y["input_ids"]
    return x


def _set_torch_format(ds: Dataset) -> None:
    cols = [c for c in ["input_ids", "attention_mask", "labels"] if c in ds.column_names]
    if cols:
        ds.set_format(type="torch", columns=cols)


def _tokenize_dataset(
    ds: Dataset,
    cache_dir: str,
    cache_key: str,
    label: str,
    num_proc: int,
    map_batch_size: int,
    cfg: dict,
    keep_in_memory: bool,
) -> Dataset:
    rows = len(ds)
    cache_path = os.path.join(cache_dir, cache_key, label)
    print(
        f"tokenize {label}: rows={rows} num_proc={num_proc} batch_size={map_batch_size} "
        f"cache_key={cache_key} cache_dir={cache_path}"
    )

    if CACHE_TOKENIZED and os.path.isdir(cache_path):
        print(f"tokenize {label}: loading from cache")
        try:
            ds_cached = load_from_disk(cache_path)
            _set_torch_format(ds_cached)
            return ds_cached
        except Exception as exc:
            print(f"tokenize {label}: cache load failed ({exc}), rebuilding")

    try:
        print(f"tokenize {label}: building cache")
        ds_mapped = ds.map(
            _tokenize_fn,
            batched=True,
            batch_size=map_batch_size,
            num_proc=num_proc if num_proc > 0 else None,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
            keep_in_memory=keep_in_memory,
            fn_kwargs={"cfg": cfg},
        )
        _set_torch_format(ds_mapped)
        if CACHE_TOKENIZED:
            try:
                os.makedirs(cache_path, exist_ok=True)
                ds_mapped.save_to_disk(cache_path)
            except Exception as exc:
                print(f"tokenize {label}: cache save failed ({exc}), continuing")
        return ds_mapped
    except Exception as exc:
        print(f"tokenize {label}: cache build failed ({exc}), falling back")
        ds_fallback = ds.map(
            _tokenize_fn,
            batched=True,
            batch_size=map_batch_size,
            num_proc=None,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
            keep_in_memory=keep_in_memory,
            fn_kwargs={"cfg": cfg},
        )
        _set_torch_format(ds_fallback)
        return ds_fallback


def main():
    seed_all(SEED, deterministic=DETERMINISTIC)

    print(f"DECODER_TRAIN={TRAIN_PATH}")
    print(f"DECODER_VAL={VAL_PATH}")
    print(f"OUT_DIR={OUT_DIR}")
    if RESUME_FROM:
        print(f"RESUME_FROM={RESUME_FROM}")

    num_proc = NUM_PROC
    if os.name == "nt" and num_proc > 0:
        print("NUM_PROC>0 not supported on Windows; using NUM_PROC=0")
        num_proc = 0

    train_rows = _read_rows(TRAIN_PATH, MAX_ROWS)
    val_rows = _read_rows(VAL_PATH, -1)

    ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows) if val_rows else None
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

    use_fp16 = FP16 == 1
    use_bf16 = BF16 == 1 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype_mode = "fp32"
    if use_fp16:
        dtype_mode = "fp16"
    elif use_bf16:
        dtype_mode = "bf16"
    print(
        f"train_decoder: dtype={dtype_mode} optim={OPTIM} lr={LR} "
        f"batch={TRAIN_BS} max_grad_norm={MAX_GRAD_NORM} code_mode={CODE_MODE}"
    )

    tokens_added = 0
    if CODE_MODE == "special" and ADD_SPECIAL_TOKENS:
        code_tokens = build_code_tokens(K, V)
        tokens_added = tok.add_special_tokens({"additional_special_tokens": code_tokens})
        if tokens_added > 0:
            model.resize_token_embeddings(len(tok))

    cache_key = _cache_key(TRAIN_PATH, VAL_PATH, tok.name_or_path, len(tok))
    tok_cfg = {
        "model": MODEL,
        "code_mode": CODE_MODE,
        "k": K,
        "v": V,
        "add_special_tokens": ADD_SPECIAL_TOKENS,
        "max_in": MAX_IN,
        "max_out": MAX_OUT,
    }
    if CACHE_TOKENIZED:
        os.makedirs(TOKENIZED_CACHE_DIR, exist_ok=True)

    ds = _tokenize_dataset(
        ds,
        cache_dir=TOKENIZED_CACHE_DIR,
        cache_key=cache_key,
        label="train",
        num_proc=num_proc,
        map_batch_size=MAP_BATCH_SIZE,
        cfg=tok_cfg,
        keep_in_memory=KEEP_IN_MEMORY,
    )
    if val_ds is not None:
        _tokenize_dataset(
            val_ds,
            cache_dir=TOKENIZED_CACHE_DIR,
            cache_key=cache_key,
            label="val",
            num_proc=num_proc,
            map_batch_size=MAP_BATCH_SIZE,
            cfg=tok_cfg,
            keep_in_memory=KEEP_IN_MEMORY,
        )

    if PRETOKENIZE_ONLY:
        print("PRETOKENIZE_ONLY=1; exiting after cache build")
        return

    collator = DataCollatorForSeq2Seq(tok, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=TRAIN_BS,
        learning_rate=LR,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        report_to=[],
        fp16=use_fp16,
        bf16=use_bf16,
        optim=OPTIM,
        max_grad_norm=MAX_GRAD_NORM,
        logging_nan_inf_filter=LOGGING_NAN_INF_FILTER == 1,
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tok,
    )
    trainer.train(resume_from_checkpoint=RESUME_FROM or None)
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)

    meta = {
        "train_path": TRAIN_PATH,
        "out_dir": OUT_DIR,
        "model": MODEL,
        "code_mode": CODE_MODE,
        "K": K,
        "V": V,
        "max_in": MAX_IN,
        "max_out": MAX_OUT,
        "num_epochs": NUM_EPOCHS,
        "lr": LR,
        "train_batch_size": TRAIN_BS,
        "tokens_added": tokens_added,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
        "device": DEVICE,
    }
    save_json(os.path.join(OUT_DIR, "config.json"), meta)

    print("Saved:", OUT_DIR)
    print("Meta:", os.path.join(OUT_DIR, "config.json"))


if __name__ == "__main__":
    main()

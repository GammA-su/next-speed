import json
import os
import re
import time

import numpy as np
import torch
import faiss

from sentence_transformers import SentenceTransformer

INSTRUCT_IN = os.environ.get("INSTRUCT_IN", "data/instruct/train.jsonl")
INSTRUCT_VAL = os.environ.get("INSTRUCT_VAL", "data/instruct/val.jsonl")
OUT_TRAIN = os.environ.get("OUT_TRAIN", "data/decoder_train_instruct.jsonl")
OUT_VAL = os.environ.get("OUT_VAL", "data/decoder_val_instruct.jsonl")

RVQ_PATH = os.environ.get("RVQ_PATH", "out/rvq.npz")
EMB_MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

K = int(os.environ.get("K", "4"))
V = int(os.environ.get("V", "256"))
BS = int(os.environ.get("BS", "512"))
MIN_LEN = int(os.environ.get("MIN_LEN", "5"))
MAX_LEN = int(os.environ.get("MAX_LEN", "220"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "50"))
FAISS_NUM_THREADS = int(os.environ.get("FAISS_NUM_THREADS", "16"))

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = os.environ.get("DEVICE", DEFAULT_DEVICE)
if DEVICE != "cpu" and not torch.cuda.is_available():
    print(f"DEVICE={DEVICE} but cuda not available; falling back to cpu.")
    DEVICE = "cpu"
AMP = os.environ.get("AMP", "1") == "1" and DEVICE == "cuda"
USE_FAISS_GPU = os.environ.get("USE_FAISS_GPU", "1") == "1"
FAISS_GPU_DEVICE = int(os.environ.get("FAISS_GPU_DEVICE", "0"))

_WS = re.compile(r"\s+")
_CTRL_TOK = re.compile(r"(?:<c\d+_\d+>|\bc\d+_\d+\b)")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def clean_text(text):
    text = (text or "").replace("\u00a0", " ")
    return _WS.sub(" ", text).strip()


def split_sents(text):
    text = clean_text(text)
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


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
    return role[:1].upper() + role[1:] if role else "User"


def render_history(history):
    lines = []
    for role, content in history:
        if not content:
            continue
        lines.append(f"{role}: {content}")
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def rvq_encode_batch(x: np.ndarray, centroids: np.ndarray, use_gpu: bool) -> np.ndarray:
    n, d = x.shape
    codes = np.zeros((n, centroids.shape[0]), dtype=np.int32)
    residual = x.copy()
    faiss.normalize_L2(residual)

    index = faiss.IndexFlatIP(d)
    gpu_res = None
    if use_gpu and hasattr(faiss, "StandardGpuResources"):
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, FAISS_GPU_DEVICE, index)

    for k in range(centroids.shape[0]):
        C = centroids[k]
        index.reset()
        index.add(C)
        _, ids = index.search(residual, 1)
        ids = ids.reshape(-1)
        codes[:, k] = ids
        residual = residual - C[ids]
        faiss.normalize_L2(residual)

    return codes


def main():
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
    d = centroids.shape[2]
    faiss.normalize_L2(centroids.reshape(-1, d))

    if FAISS_NUM_THREADS > 0 and hasattr(faiss, "omp_set_num_threads"):
        faiss.omp_set_num_threads(FAISS_NUM_THREADS)
    use_gpu = USE_FAISS_GPU and hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0
    enc = SentenceTransformer(EMB_MODEL, device=DEVICE)
    amp_enabled = AMP and DEVICE == "cuda"
    print(
        "instruct_decoder: device={device} amp={amp} faiss_gpu={faiss_gpu} faiss_threads={threads} bs={bs}".format(
            device=DEVICE,
            amp=amp_enabled,
            faiss_gpu=use_gpu,
            threads=FAISS_NUM_THREADS,
            bs=BS,
        )
    )

    counts = {
        "bad_json": 0,
        "missing_messages": 0,
        "missing_content": 0,
        "no_assistant": 0,
        "control_stripped_empty": 0,
        "too_short": 0,
        "too_long": 0,
    }
    totals = {"train_in": 0, "val_in": 0, "train_out": 0, "val_out": 0}
    samples = []

    def flush(buf, fout, split_label):
        if not buf:
            return
        t0 = time.time()
        answers = [row["tgt_text"] for row in buf]
        ctxs = [row["ctx_text"] for row in buf]
        with torch.no_grad():
            if amp_enabled:
                amp_dtype = torch.float16
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    embs = enc.encode(
                        answers,
                        normalize_embeddings=True,
                        batch_size=BS,
                        convert_to_numpy=True,
                    ).astype(np.float32)
            else:
                embs = enc.encode(
                    answers,
                    normalize_embeddings=True,
                    batch_size=BS,
                    convert_to_numpy=True,
                ).astype(np.float32)

        codes = rvq_encode_batch(embs, centroids, use_gpu=use_gpu)
        for ctx_text, tgt_text, code_vec in zip(ctxs, answers, codes):
            tgt_codes_str = " ".join([f"c{i}={int(c)}" for i, c in enumerate(code_vec)])
            out = {
                "ctx_text": ctx_text,
                "tgt_text": tgt_text,
                "tgt_codes_str": tgt_codes_str,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            totals[f"{split_label}_out"] += 1
            if len(samples) < 3:
                samples.append(out)
        elapsed = time.time() - t0
        if elapsed > 0:
            rate = len(buf) / elapsed
            print(f"encoded_batch={len(buf)} split={split_label} rows/sec={rate:.2f}", flush=True)

    def process_file(path, fout, split_label):
        buf = []
        rows_seen = 0
        rows_out = 0
        last_log = time.time()
        if not os.path.isfile(path):
            print(f"Missing {split_label} input: {path}")
            return
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                totals[f"{split_label}_in"] += 1
                rows_seen += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    counts["bad_json"] += 1
                    continue
                messages = rec.get("messages")
                if not isinstance(messages, list):
                    counts["missing_messages"] += 1
                    continue

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
                            continue
                        if len(tgt) < MIN_LEN:
                            counts["too_short"] += 1
                            continue
                        if len(tgt) > MAX_LEN:
                            counts["too_long"] += 1
                            continue

                        hist_text = render_history(history)
                        if prefix:
                            hist_text += f"Assistant: {prefix}\n"
                        ctx_text = hist_text + "Assistant:"
                        buf.append({"ctx_text": ctx_text, "tgt_text": tgt})
                        if len(buf) >= BS:
                            flush(buf, fout, split_label)
                            buf = []
                            rows_out = totals[f"{split_label}_out"]
                            now = time.time()
                            if LOG_EVERY > 0 and rows_out % (LOG_EVERY * BS) == 0:
                                rate = (rows_out / max(1e-6, now - last_log))
                                print(
                                    f"progress split={split_label} rows_in={rows_seen} rows_out={rows_out} rows/sec={rate:.2f}",
                                    flush=True,
                                )
                                last_log = now

                        prefix = (prefix + " " + sent).strip()

                    if prefix:
                        history.append((role, prefix))

                if not saw_assistant:
                    counts["no_assistant"] += 1

        if buf:
            flush(buf, fout, split_label)

    with open(OUT_TRAIN, "w", encoding="utf-8") as ftrain, open(
        OUT_VAL, "w", encoding="utf-8"
    ) as fval:
        process_file(INSTRUCT_IN, ftrain, "train")
        process_file(INSTRUCT_VAL, fval, "val")

    skipped = totals["train_in"] + totals["val_in"] - totals["train_out"] - totals["val_out"]
    print(
        json.dumps(
            {
                "train_in": totals["train_in"],
                "val_in": totals["val_in"],
                "train_out": totals["train_out"],
                "val_out": totals["val_out"],
                "skipped": skipped,
                "skipped_by_reason": counts,
                "samples": samples,
            },
            indent=2,
        )
    )
    print(f"OUT_TRAIN={OUT_TRAIN}")
    print(f"OUT_VAL={OUT_VAL}")


if __name__ == "__main__":
    main()

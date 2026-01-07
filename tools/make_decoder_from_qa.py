import os, re, json, time
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

QA_IN = os.environ.get("QA_IN", "data/qa/train.jsonl")
QA_OUT = os.environ.get("QA_OUT", "data/decoder_train_qa.jsonl")
RVQ_PATH = os.environ.get("RVQ_PATH", "out/rvq.npz")
EMB_MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
K = int(os.environ.get("K", "4"))
V = int(os.environ.get("V", "256"))
MAX_ANS_CHARS = int(os.environ.get("MAX_ANS_CHARS", "220"))
BATCH = int(os.environ.get("BATCH", "1024"))
MAX_ROWS = int(os.environ.get("MAX_ROWS", "-1"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "50"))
USE_CUDA = os.environ.get("USE_CUDA", "1") == "1"
PROMPT_STYLE = os.environ.get("PROMPT_STYLE", "chat")  # chat|qa

_ws = re.compile(r"\s+")
def clean(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = _ws.sub(" ", s).strip()
    return s

def rvq_encode(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # x: [d] unit norm, centroids: [K,V,d]
    r = x.copy()
    codes = []
    for k in range(centroids.shape[0]):
        C = centroids[k]  # [V,d]
        # pick nearest by L2 (for unit vectors, equivalent to max dot)
        dots = C @ r
        idx = int(np.argmax(dots))
        codes.append(idx)
        r = r - C[idx]
    return np.array(codes, dtype=np.int32)

def rvq_encode_batch(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # x: [n,d] unit norm, centroids: [K,V,d]
    r = x.copy()
    codes = np.zeros((x.shape[0], centroids.shape[0]), dtype=np.int32)
    for k in range(centroids.shape[0]):
        C = centroids[k]  # [V,d]
        dots = r @ C.T  # [n,V]
        idx = np.argmax(dots, axis=1).astype(np.int32)
        codes[:, k] = idx
        r = r - C[idx]
    return codes

def fmt_ctx(q: str) -> str:
    if PROMPT_STYLE == "qa":
        return f"Q: {q}\nA:"
    return f"User: {q}\nAssistant:"

def main():
    rvq = np.load(RVQ_PATH)
    centroids = rvq["centroids"]  # [K,V,d]
    assert centroids.shape[0] == K, (centroids.shape, K)
    assert centroids.shape[1] == V, (centroids.shape, V)

    torch_cuda = torch.cuda.is_available()
    device = "cuda" if (USE_CUDA and torch_cuda) else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch_cuda else "none"
    print(
        f"device={device} torch_cuda={torch_cuda} torch_version={torch.__version__} "
        f"n_gpu={torch.cuda.device_count()} gpu_name={gpu_name}"
    )
    if USE_CUDA and not torch_cuda:
        print("USE_CUDA=1 but torch.cuda.is_available()=False; falling back to cpu.")

    enc = SentenceTransformer(EMB_MODEL, device=device)
    model_device = str(getattr(enc, "device", getattr(enc, "_target_device", "unknown")))
    if device == "cuda" and "cuda" not in model_device:
        print(f"Warning: expected cuda model device, got {model_device}")

    n_in = 0
    n_out = 0
    skipped_empty = 0
    skipped_short = 0
    n_batches = 0
    start_t = time.time()

    with open(QA_IN, "r", encoding="utf-8") as fin, open(QA_OUT, "w", encoding="utf-8") as fout:
        buf = []
        for line in tqdm(fin, desc="qa->decoder"):
            if MAX_ROWS > 0 and n_in >= MAX_ROWS:
                break
            n_in += 1
            r = json.loads(line)
            q = clean(r.get("question",""))
            a = clean(r.get("answer",""))
            if not q or not a:
                skipped_empty += 1
                continue
            a = a[:MAX_ANS_CHARS].strip()
            # 1 sentence target (your decoder is sentence-level)
            # If there are multiple sentences, keep the first.
            if "." in a:
                a = a.split(".", 1)[0].strip()
                if a:
                    a = a + "."
            if len(a) < 2:
                skipped_short += 1
                continue

            buf.append((q, a))
            if len(buf) >= BATCH:
                answers = [row[1] for row in buf]
                try:
                    embs = enc.encode(
                        answers,
                        normalize_embeddings=True,
                        batch_size=BATCH,
                        convert_to_numpy=True,
                    ).astype(np.float32)
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    if "out of memory" in msg and device == "cuda":
                        print("CUDA OOM during encode; lower BATCH to reduce memory.")
                        torch.cuda.empty_cache()
                    raise
                codes = rvq_encode_batch(embs, centroids)
                for (q_i, a_i), code_vec in zip(buf, codes):
                    tgt_codes_str = " ".join([f"c{i}={int(c)}" for i,c in enumerate(code_vec)])
                    out = {
                        "ctx_text": fmt_ctx(q_i),
                        "tgt_text": a_i,
                        "tgt_codes_str": tgt_codes_str,
                    }
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    n_out += 1
                buf.clear()
                n_batches += 1
                if LOG_EVERY > 0 and n_batches % LOG_EVERY == 0:
                    elapsed = max(1e-6, time.time() - start_t)
                    avg_it_s = n_out / elapsed
                    print(
                        f"qa->decoder batches={n_batches} rows={n_out} avg_it_s={avg_it_s:.2f} "
                        f"model_device={model_device}"
                    )

        if buf:
            answers = [row[1] for row in buf]
            try:
                embs = enc.encode(
                    answers,
                    normalize_embeddings=True,
                    batch_size=BATCH,
                    convert_to_numpy=True,
                ).astype(np.float32)
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg and device == "cuda":
                    print("CUDA OOM during encode; lower BATCH to reduce memory.")
                    torch.cuda.empty_cache()
                raise
            codes = rvq_encode_batch(embs, centroids)
            for (q_i, a_i), code_vec in zip(buf, codes):
                tgt_codes_str = " ".join([f"c{i}={int(c)}" for i,c in enumerate(code_vec)])
                out = {
                    "ctx_text": fmt_ctx(q_i),
                    "tgt_text": a_i,
                    "tgt_codes_str": tgt_codes_str,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_out += 1

    print(json.dumps({
        "qa_in": QA_IN,
        "qa_out": QA_OUT,
        "k": K,
        "v": V,
        "rows_in": n_in,
        "rows_out": n_out,
        "skipped_empty": skipped_empty,
        "skipped_short": skipped_short,
        "device": device,
        "batch": BATCH,
    }, indent=2))

if __name__ == "__main__":
    main()

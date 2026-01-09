import os
import time

import numpy as np
import faiss

from utils import seed_all, save_json

EMB_PATH = os.environ.get("EMB_PATH", "data/embeddings.npy")
RVQ_PATH = os.environ.get("RVQ_PATH", "out/rvq.npz")
OUT_CODES = os.environ.get("OUT_CODES", "data/codes.npy")
META = os.environ.get("CODES_META", "data/codes.meta.json")

K = int(os.environ.get("K", "8"))
V = int(os.environ.get("V", "1024"))
USE_FAISS_GPU = os.environ.get("USE_FAISS_GPU", "1") == "1"
FAISS_GPU_DEVICE = int(os.environ.get("FAISS_GPU_DEVICE", "0"))
SEED = int(os.environ.get("SEED", "0"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "1") == "1"


def encode_rvq(x, centroids, use_gpu=False, gpu_res=None, gpu_device=0):
    # x: (N,D) normalized
    n, d = x.shape
    codes = np.zeros((n, centroids.shape[0]), dtype=np.int32)
    residual = x.copy()
    index = faiss.IndexFlatIP(d)
    if use_gpu and gpu_res is not None:
        index = faiss.index_cpu_to_gpu(gpu_res, gpu_device, index)

    for k in range(centroids.shape[0]):
        C = centroids[k]
        index.reset()
        index.add(C)
        _, ids = index.search(residual, 1)
        ids = ids.reshape(-1)
        codes[:, k] = ids

        residual = residual - C[ids]
        rn = np.linalg.norm(residual, axis=1, keepdims=True) + 1e-8
        residual = residual / rn

    return codes


def main():
    seed_all(SEED, deterministic=DETERMINISTIC)
    os.makedirs(os.path.dirname(OUT_CODES), exist_ok=True)

    num_gpus = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
    use_gpu = USE_FAISS_GPU and num_gpus > 0 and hasattr(faiss, "StandardGpuResources")
    print(
        f"faiss_version={getattr(faiss, '__version__', 'unknown')} num_gpus={num_gpus} "
        f"use_faiss_gpu={use_gpu} faiss_gpu_device={FAISS_GPU_DEVICE if use_gpu else 'n/a'}"
    )

    print("Phase: load embeddings")
    x = np.load(EMB_PATH)
    if x.dtype != np.float32 or not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x, dtype="float32")

    print("Phase: load RVQ centroids")
    rvq = np.load(RVQ_PATH)
    centroids = np.ascontiguousarray(rvq["centroids"], dtype="float32")
    if centroids.ndim != 3:
        raise ValueError(f"Expected centroids to be 3D, got shape {centroids.shape}")
    if centroids.shape[0] != K:
        raise ValueError(f"centroids K mismatch: {centroids.shape[0]} != K={K}")
    if centroids.shape[1] != V:
        raise ValueError(f"centroids V mismatch: {centroids.shape[1]} != V={V}")

    d = centroids.shape[2]
    faiss.normalize_L2(centroids.reshape(-1, d))

    gpu_res = None
    if use_gpu:
        try:
            gpu_res = faiss.StandardGpuResources()
        except Exception:
            use_gpu = False
            print("Warning: failed to init FAISS GPU; falling back to CPU.")

    start = time.time()
    print("Phase: encode RVQ codes")
    codes = encode_rvq(x, centroids, use_gpu=use_gpu, gpu_res=gpu_res, gpu_device=FAISS_GPU_DEVICE)
    np.save(OUT_CODES, codes)
    elapsed = time.time() - start
    rate = codes.shape[0] / elapsed if elapsed > 0 else 0.0

    meta = {
        "emb_path": EMB_PATH,
        "rvq_path": RVQ_PATH,
        "out_codes": OUT_CODES,
        "n": int(codes.shape[0]),
        "d": int(codes.shape[1]),
        "K": K,
        "V": V,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
        "elapsed_sec": float(elapsed),
        "rows_per_sec": float(rate),
    }
    save_json(META, meta)

    print("Saved codes:", OUT_CODES, codes.shape)
    print(f"Encode time: {elapsed:.1f}s rows/sec={rate:.2f}")
    print("Meta:", META)


if __name__ == "__main__":
    main()

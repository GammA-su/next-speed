import os
import time

import numpy as np
import faiss

from utils import seed_all, save_json

EMB = os.environ.get("EMB_PATH", "data/embeddings.npy")
OUT = os.environ.get("RVQ_PATH", "out/rvq.npz")
META = os.environ.get("RVQ_META", "out/rvq_meta.json")

K = int(os.environ.get("K", "8"))  # number of codebooks
V = int(os.environ.get("V", "1024"))  # centroids per codebook
FAISS_NITER = int(os.environ.get("FAISS_NITER", "25"))
TRAIN_N = int(os.environ.get("TRAIN_N", "1000000"))
USE_FAISS_GPU = os.environ.get("USE_FAISS_GPU", "1") == "1"
FAISS_SEED = int(os.environ.get("FAISS_SEED", "0"))
FAISS_NUM_THREADS = int(os.environ.get("FAISS_NUM_THREADS", "0"))
SEED = int(os.environ.get("SEED", "0"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "1") == "1"


def main():
    seed_all(SEED, deterministic=DETERMINISTIC)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if FAISS_NUM_THREADS > 0:
        faiss.omp_set_num_threads(FAISS_NUM_THREADS)

    num_gpus = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
    print(
        f"faiss_version={getattr(faiss, '__version__', 'unknown')} num_gpus={num_gpus} "
        f"use_faiss_gpu={USE_FAISS_GPU}"
    )

    x = np.load(EMB, mmap_mode="r")
    n, d = x.shape
    train_n = n if TRAIN_N <= 0 or TRAIN_N >= n else TRAIN_N
    rng = np.random.default_rng(FAISS_SEED)
    if train_n < n:
        idx = rng.choice(n, size=train_n, replace=False)
        x_train = np.asarray(x[idx], dtype="float32")
    else:
        x_train = np.asarray(x, dtype="float32")

    use_gpu = USE_FAISS_GPU and num_gpus > 0 and hasattr(faiss, "StandardGpuResources")
    gpu_res = None
    if use_gpu:
        try:
            gpu_res = faiss.StandardGpuResources()
        except Exception:
            use_gpu = False
    if use_gpu:
        print(f"Using FAISS GPU (gpus={num_gpus})")
    else:
        print("Using FAISS CPU")

    residual = x_train.copy()
    centroids = np.zeros((K, V, d), dtype=np.float32)
    total_start = time.time()

    for k in range(K):
        stage_start = time.time()
        try:
            km = faiss.Kmeans(
                d,
                V,
                niter=FAISS_NITER,
                seed=FAISS_SEED + k,
                verbose=True,
                gpu=use_gpu,
            )
        except TypeError:
            km = faiss.Kmeans(
                d,
                V,
                niter=FAISS_NITER,
                seed=FAISS_SEED + k,
                verbose=True,
            )
        km.train(residual)
        C = km.centroids.reshape(V, d).astype(np.float32)

        # assign nearest centroid
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(C)  # keep centroids normalized too
        if use_gpu and gpu_res is not None:
            index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
        index.add(C)
        _, ids = index.search(residual, 1)
        ids = ids.reshape(-1)

        # subtract chosen centroid (residual quantization)
        residual = residual - C[ids]
        # renormalize residual to keep scale stable (optional but helps)
        rn = np.linalg.norm(residual, axis=1, keepdims=True) + 1e-8
        residual = residual / rn

        centroids[k] = C
        stage_elapsed = time.time() - stage_start
        print(f"Codebook {k}: trained in {stage_elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    np.savez_compressed(OUT, centroids=centroids)
    meta = {
        "emb_path": EMB,
        "out_path": OUT,
        "n": int(n),
        "train_n": int(train_n),
        "d": int(d),
        "K": K,
        "V": V,
        "niter": FAISS_NITER,
        "seed": SEED,
        "faiss_seed": FAISS_SEED,
        "use_faiss_gpu": USE_FAISS_GPU,
        "faiss_num_threads": FAISS_NUM_THREADS,
        "deterministic": DETERMINISTIC,
        "normalize_residual": True,
        "elapsed_sec": float(total_elapsed),
    }
    save_json(META, meta)

    print("Saved RVQ:", OUT, "centroids:", centroids.shape)
    print(f"Total time: {total_elapsed:.1f}s")
    print("Meta:", META)


if __name__ == "__main__":
    main()

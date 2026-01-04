import os
import numpy as np
import faiss

from utils import seed_all, save_json

EMB = os.environ.get("EMB_PATH", "data/embeddings.npy")
OUT = os.environ.get("RVQ_PATH", "out/rvq.npz")
META = os.environ.get("RVQ_META", "out/rvq_meta.json")

K = int(os.environ.get("K", "8"))  # number of codebooks
V = int(os.environ.get("V", "1024"))  # centroids per codebook
NITER = int(os.environ.get("NITER", "25"))  # kmeans iters
SEED = int(os.environ.get("SEED", "0"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "1") == "1"


def main():
    seed_all(SEED, deterministic=DETERMINISTIC)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    x = np.load(EMB).astype("float32")  # (N, D) normalized already
    n, d = x.shape

    residual = x.copy()
    centroids = np.zeros((K, V, d), dtype=np.float32)

    for k in range(K):
        km = faiss.Kmeans(d, V, niter=NITER, seed=SEED + k, verbose=True, gpu=False)
        km.train(residual)
        C = km.centroids.reshape(V, d).astype(np.float32)

        # assign nearest centroid
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(C)  # keep centroids normalized too
        index.add(C)
        _, ids = index.search(residual, 1)
        ids = ids.reshape(-1)

        # subtract chosen centroid (residual quantization)
        residual = residual - C[ids]
        # renormalize residual to keep scale stable (optional but helps)
        rn = np.linalg.norm(residual, axis=1, keepdims=True) + 1e-8
        residual = residual / rn

        centroids[k] = C
        print(f"Codebook {k}: trained")

    np.savez_compressed(OUT, centroids=centroids)
    meta = {
        "emb_path": EMB,
        "out_path": OUT,
        "n": int(n),
        "d": int(d),
        "K": K,
        "V": V,
        "niter": NITER,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
        "normalize_residual": True,
    }
    save_json(META, meta)

    print("Saved RVQ:", OUT, "centroids:", centroids.shape)
    print("Meta:", META)


if __name__ == "__main__":
    main()

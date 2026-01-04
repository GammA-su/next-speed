import os
import numpy as np
import faiss

from utils import seed_all, save_json

EMB_PATH = os.environ.get("EMB_PATH", "data/embeddings.npy")
RVQ_PATH = os.environ.get("RVQ_PATH", "out/rvq.npz")
OUT_PATH = os.environ.get("OUT_PATH", "out/eval/rvq_recon_error.json")
MAX_N = int(os.environ.get("MAX_N", "-1"))
SEED = int(os.environ.get("SEED", "0"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "1") == "1"


def main():
    seed_all(SEED, deterministic=DETERMINISTIC)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    x = np.load(EMB_PATH).astype("float32")
    if MAX_N > 0:
        x = x[:MAX_N]
    centroids = np.load(RVQ_PATH)["centroids"].astype("float32")

    n, d = x.shape
    k_total = centroids.shape[0]

    residual = x.copy()
    recon = np.zeros_like(x)
    stats = []

    for k in range(k_total):
        C = centroids[k].astype("float32")
        faiss.normalize_L2(C)
        index = faiss.IndexFlatIP(d)
        index.add(C)
        _, ids = index.search(residual, 1)
        ids = ids.reshape(-1)

        recon += C[ids]
        recon_norm = recon / (np.linalg.norm(recon, axis=1, keepdims=True) + 1e-8)

        mse = float(np.mean(np.sum((x - recon_norm) ** 2, axis=1)))
        cos_sim = float(np.mean(np.sum(x * recon_norm, axis=1)))
        stats.append({"stage": k, "mse": mse, "cos_sim": cos_sim, "cos_dist": 1.0 - cos_sim})

        residual = residual - C[ids]
        rn = np.linalg.norm(residual, axis=1, keepdims=True) + 1e-8
        residual = residual / rn

    out = {
        "emb_path": EMB_PATH,
        "rvq_path": RVQ_PATH,
        "n": int(n),
        "d": int(d),
        "stages": stats,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
    }
    save_json(OUT_PATH, out)
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()

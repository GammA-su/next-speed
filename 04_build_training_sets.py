import os
import json
import random
import numpy as np
import faiss
from tqdm import tqdm

from utils import seed_all, save_json

SENT_PATH = os.environ.get("SENT_PATH", "data/sentences.jsonl")
EMB_PATH = os.environ.get("EMB_PATH", "data/embeddings.npy")
RVQ_PATH = os.environ.get("RVQ_PATH", "out/rvq.npz")

OUT_CODES = os.environ.get("OUT_CODES", "data/codes.npy")
OUT_LM = os.environ.get("OUT_LM", "data/sent_lm_train.jsonl")
OUT_LM_VAL = os.environ.get("OUT_LM_VAL", "data/sent_lm_val.jsonl")
OUT_DEC = os.environ.get("OUT_DEC", "data/decoder_train.jsonl")
OUT_DEC_VAL = os.environ.get("OUT_DEC_VAL", "data/decoder_val.jsonl")
META = os.environ.get("TRAIN_META", "data/training_meta.json")

K = int(os.environ.get("K", "8"))
V = int(os.environ.get("V", "1024"))
CONTEXT_SENTS = int(os.environ.get("CONTEXT_SENTS", "6"))
VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.05"))
SEED = int(os.environ.get("SEED", "0"))
DETERMINISTIC = os.environ.get("DETERMINISTIC", "1") == "1"


def encode_rvq(x, centroids):
    # x: (N,D) normalized
    n, d = x.shape
    codes = np.zeros((n, centroids.shape[0]), dtype=np.int32)
    residual = x.copy()

    for k in range(centroids.shape[0]):
        C = centroids[k].astype("float32")
        faiss.normalize_L2(C)
        index = faiss.IndexFlatIP(d)
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
    os.makedirs(os.path.dirname(OUT_LM), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_LM_VAL), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_DEC), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_DEC_VAL), exist_ok=True)

    x = np.load(EMB_PATH).astype("float32")
    centroids = np.load(RVQ_PATH)["centroids"].astype("float32")

    codes = encode_rvq(x, centroids)
    np.save(OUT_CODES, codes)
    print("Saved codes:", OUT_CODES, codes.shape)

    # load sentences and group by doc
    docs = {}
    with open(SENT_PATH, "r", encoding="utf-8") as f:
        idx = 0
        for line in f:
            r = json.loads(line)
            docs.setdefault(r["doc_id"], []).append((idx, r["text"], r["sent_id"]))
            idx += 1

    doc_ids = list(docs.keys())
    rng = random.Random(SEED)
    rng.shuffle(doc_ids)
    n_val = int(len(doc_ids) * VAL_FRAC)
    val_set = set(doc_ids[:n_val]) if n_val > 0 else set()

    counts = {
        "lm_train": 0,
        "lm_val": 0,
        "dec_train": 0,
        "dec_val": 0,
    }

    with open(OUT_LM, "w", encoding="utf-8") as flm_train, \
        open(OUT_LM_VAL, "w", encoding="utf-8") as flm_val, \
        open(OUT_DEC, "w", encoding="utf-8") as fdec_train, \
        open(OUT_DEC_VAL, "w", encoding="utf-8") as fdec_val:
        for doc_id in tqdm(doc_ids, desc="build"):
            items = docs[doc_id]
            if len(items) <= CONTEXT_SENTS:
                continue

            is_val = doc_id in val_set
            flm = flm_val if is_val else flm_train
            fdec = fdec_val if is_val else fdec_train

            # per sentence position
            for i in range(CONTEXT_SENTS, len(items)):
                ctx_items = items[i - CONTEXT_SENTS : i]
                tgt_item = items[i]

                ctx_idxs = [t[0] for t in ctx_items]
                tgt_idx = tgt_item[0]

                ctx_codes = codes[ctx_idxs].reshape(-1).tolist()  # flatten CONTEXT_SENTS*K
                tgt_codes = codes[tgt_idx].tolist()  # K codes

                ctx_text = " ".join([t[1] for t in ctx_items])
                tgt_text = tgt_item[1]

                flm.write(
                    json.dumps({"ctx_codes": ctx_codes, "tgt_codes": tgt_codes}) + "\n"
                )
                if is_val:
                    counts["lm_val"] += 1
                else:
                    counts["lm_train"] += 1

                # textual code rendering (simple, works with any tokenizer)
                code_str = " ".join([f"c{j}={tgt_codes[j]}" for j in range(len(tgt_codes))])
                fdec.write(
                    json.dumps(
                        {
                            "ctx_text": ctx_text,
                            "ctx_codes": ctx_codes,
                            "tgt_codes": tgt_codes,
                            "tgt_codes_str": code_str,
                            "tgt_text": tgt_text,
                            "doc_id": doc_id,
                            "sent_id": tgt_item[2],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if is_val:
                    counts["dec_val"] += 1
                else:
                    counts["dec_train"] += 1

    meta = {
        "sent_path": SENT_PATH,
        "emb_path": EMB_PATH,
        "rvq_path": RVQ_PATH,
        "out_codes": OUT_CODES,
        "out_lm_train": OUT_LM,
        "out_lm_val": OUT_LM_VAL,
        "out_dec_train": OUT_DEC,
        "out_dec_val": OUT_DEC_VAL,
        "K": K,
        "V": V,
        "context_sents": CONTEXT_SENTS,
        "val_frac": VAL_FRAC,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
        "counts": counts,
    }
    save_json(META, meta)

    print("Wrote:", OUT_LM)
    print("Wrote:", OUT_LM_VAL)
    print("Wrote:", OUT_DEC)
    print("Wrote:", OUT_DEC_VAL)
    print("Meta:", META)


if __name__ == "__main__":
    main()

import os
from transformers import AutoTokenizer

DECODER_PATH = os.getenv("DECODER_PATH", "out/decoder/checkpoint-50000")
K = int(os.getenv("K", "4"))
V = int(os.getenv("V", "256"))

tok = AutoTokenizer.from_pretrained(DECODER_PATH, use_fast=True)

bad = 0
multi = 0
missing = 0
for s in range(K):
    for i in range(V):
        t = f"<c{s}_{i:04d}>"
        ids = tok(t, add_special_tokens=False).input_ids
        if len(ids) == 0:
            missing += 1
        elif len(ids) != 1:
            multi += 1

print("DECODER_PATH:", DECODER_PATH)
print("K,V:", K, V)
print("missing:", missing, "multi_token:", multi, "total:", K*V)

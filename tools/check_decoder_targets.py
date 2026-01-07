import os, json, re

P = os.getenv("P", "data/decoder_train.jsonl")
pat = re.compile(r"(\|\||<c\d+_\d{4}>|c\d+_\d+>)")

n = 0
hit = 0
with open(P, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        t = r.get("tgt_text","")
        n += 1
        if pat.search(t):
            hit += 1
            if hit <= 5:
                print("example tgt_text:", t[:200].replace("\n","\\n"))
print("rows:", n, "targets_with_control_tokens:", hit, "pct:", hit/max(1,n))

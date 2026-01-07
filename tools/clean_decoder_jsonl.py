import json, re, os

IN_TRAIN = os.getenv("IN_TRAIN", "data/decoder_train.jsonl")
IN_VAL   = os.getenv("IN_VAL",   "data/decoder_val.jsonl")
OUT_TRAIN= os.getenv("OUT_TRAIN","data/decoder_train.clean.jsonl")
OUT_VAL  = os.getenv("OUT_VAL",  "data/decoder_val.clean.jsonl")

# Heuristics for Wikipedia tables / markup leakage
BAD = re.compile(
    r"(\|\||^\s*\||align=right|data-sort-value=|rowspan=|colspan=|style=|class=|scope=|bgcolor=)",
    re.IGNORECASE
)

def clean(in_path, out_path):
    kept = 0
    dropped = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            tgt = r.get("tgt_text","")
            if BAD.search(tgt):
                dropped += 1
                continue
            # also drop super-short targets (often junk)
            if len(tgt.strip()) < 5:
                dropped += 1
                continue
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1
    print(f"{in_path} -> {out_path} kept={kept} dropped={dropped} pct_dropped={dropped/max(1,(kept+dropped)):.4f}")

clean(IN_TRAIN, OUT_TRAIN)
clean(IN_VAL, OUT_VAL)

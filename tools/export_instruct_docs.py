import os, json, uuid
from datasets import load_dataset

OUT = os.getenv("OUT", "data/raw/docs_instruct.jsonl")
MAX_DOCS = int(os.getenv("MAX_DOCS", "200000"))

# Comma-separated: dataset_name:split
# Good defaults:
# - HuggingFaceH4/ultrachat_200k:train_sft
# - allenai/tulu-v2-sft-long-mixture:train
# - databricks/databricks-dolly-15k:train
DATASETS = os.getenv(
    "DATASETS",
    "HuggingFaceH4/ultrachat_200k:train_sft,allenai/tulu-v2-sft-long-mixture:train,databricks/databricks-dolly-15k:train",
)

def norm_role(r: str) -> str:
    r = (r or "").lower()
    if r in ("user", "human", "prompter"):
        return "User"
    if r in ("assistant", "gpt", "bot"):
        return "Assistant"
    return r[:1].upper() + r[1:] if r else "Unknown"

def format_messages(msgs):
    lines = []
    for m in msgs:
        role = norm_role(m.get("role") or m.get("from") or m.get("speaker"))
        content = m.get("content") or m.get("value") or m.get("text")
        if not content:
            continue
        content = str(content).strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

def format_dolly(ex):
    inst = (ex.get("instruction") or "").strip()
    ctx  = (ex.get("context") or "").strip()
    resp = (ex.get("response") or "").strip()
    parts = []
    if inst:
        parts.append(f"Instruction: {inst}")
    if ctx:
        parts.append(f"Context: {ctx}")
    if resp:
        parts.append(f"Assistant: {resp}")
    return "\n".join(parts)

def iter_dataset(name, split):
    # Try streaming first (fast start, less disk), fallback if unsupported
    try:
        ds = load_dataset(name, split=split, streaming=True)
        for ex in ds:
            yield ex
    except Exception:
        ds = load_dataset(name, split=split)
        for ex in ds:
            yield ex

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    wrote = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for spec in [s.strip() for s in DATASETS.split(",") if s.strip()]:
            if ":" not in spec:
                raise SystemExit(f"Bad DATASETS entry: {spec} (expected name:split)")
            name, split = spec.split(":", 1)
            for ex in iter_dataset(name, split):
                if wrote >= MAX_DOCS:
                    break

                text = ""
                if "messages" in ex and isinstance(ex["messages"], list):
                    text = format_messages(ex["messages"])
                elif "conversations" in ex and isinstance(ex["conversations"], list):
                    # some sharegpt-style sets use conversations[{from,value}]
                    text = format_messages(ex["conversations"])
                elif "instruction" in ex and "response" in ex:
                    text = format_dolly(ex)
                else:
                    # last resort: try common keys
                    msg = ex.get("text") or ex.get("content") or ""
                    text = str(msg).strip()

                if not text or len(text) < 50:
                    continue

                rec = {
                    "doc_id": str(uuid.uuid4()),
                    "source": f"{name}:{split}",
                    "text": text,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                wrote += 1

    print(f"Wrote {wrote} docs -> {OUT}")

if __name__ == "__main__":
    main()

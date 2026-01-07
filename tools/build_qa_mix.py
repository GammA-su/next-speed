import os, re, json, random
from typing import Iterable, Dict, Any, List, Optional
from datasets import load_dataset

SEED = int(os.environ.get("SEED", "0"))
OUT_DIR = os.environ.get("OUT_DIR", "data/qa")
TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", "0.98"))
MAX_ROWS = int(os.environ.get("MAX_ROWS", "0"))  # 0 = all

random.seed(SEED)

_ws = re.compile(r"\s+")
_bad = re.compile(r"(\|\|+)|(\{\{.*?\}\})|(<ref.*?>.*?</ref>)|(<.*?>)")
_many_pipes = re.compile(r"\|{2,}")
def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\u00a0", " ")
    s = _bad.sub(" ", s)
    s = _many_pipes.sub(" ", s)
    s = _ws.sub(" ", s).strip()
    return s

def keep_answer(a: str) -> bool:
    if not a:
        return False
    if len(a) < 2:
        return False
    if len(a) > 220:
        return False
    # drop “table-y” garbage
    if a.count("|") >= 2:
        return False
    return True

def add_item(buf: List[Dict[str, Any]], source: str, q: str, a: str):
    q = clean_text(q)
    a = clean_text(a)
    if not q or not a:
        return
    if not keep_answer(a):
        return
    buf.append({"source": source, "question": q, "answer": a})

def load_squad_v2(buf):
    ds = load_dataset("squad_v2")
    for split in ["train", "validation"]:
        for r in ds[split]:
            if r.get("is_impossible"):
                continue
            ans = r.get("answers", {}).get("text", [])
            if not ans:
                continue
            add_item(buf, "squad_v2", r["question"], ans[0])

def load_web_questions(buf):
    ds = load_dataset("web_questions")
    for split in ds.keys():
        for r in ds[split]:
            ans = r.get("answers", [])
            if not ans:
                continue
            add_item(buf, "web_questions", r["question"], ans[0])

def load_openbookqa(buf):
    ds = load_dataset("openbookqa", "main")
    for split in ds.keys():
        for r in ds[split]:
            q = r["question_stem"]
            choices = r["choices"]
            # Build a compact multiple-choice prompt; answer is the correct option text
            letters = choices["label"]
            texts = choices["text"]
            opts = " ".join([f"({l}) {t}" for l, t in zip(letters, texts)])
            q2 = f"{q} {opts}"
            key = r["answerKey"]
            ans = None
            for l, t in zip(letters, texts):
                if l == key:
                    ans = t
                    break
            if ans:
                add_item(buf, "openbookqa", q2, ans)

def load_triviaqa(buf):
    # trivia_qa configs differ; try most useful ones first
    tried = []
    for cfg in ["rc.nocontext", "rc", "unfiltered.nocontext", "unfiltered"]:
        try:
            ds = load_dataset("trivia_qa", cfg)
            for split in ds.keys():
                for r in ds[split]:
                    q = r.get("question", "")
                    a = None
                    ans = r.get("answer", {})
                    if isinstance(ans, dict):
                        a = ans.get("value") or (ans.get("normalized_value") if isinstance(ans.get("normalized_value"), str) else None)
                    if not a:
                        continue
                    add_item(buf, f"trivia_qa:{cfg}", q, a)
            return
        except Exception as e:
            tried.append((cfg, str(e)))
    raise RuntimeError("Failed to load trivia_qa configs. Tried: " + "; ".join([c for c,_ in tried]))

def load_truthfulqa(buf):
    ds = load_dataset("truthful_qa", "generation")
    for split in ds.keys():
        for r in ds[split]:
            q = r.get("question", "")
            a = r.get("best_answer", "")
            add_item(buf, "truthful_qa:generation", q, a)

def dedup(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for x in items:
        key = x["question"].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    buf: List[Dict[str, Any]] = []

    load_squad_v2(buf)
    load_web_questions(buf)
    load_openbookqa(buf)
    load_triviaqa(buf)
    load_truthfulqa(buf)

    buf = dedup(buf)

    if MAX_ROWS > 0:
        buf = buf[:MAX_ROWS]

    random.shuffle(buf)
    n_train = int(len(buf) * TRAIN_FRAC)
    train = buf[:n_train]
    val = buf[n_train:]

    def dump(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(os.path.join(OUT_DIR, "train.jsonl"), train)
    dump(os.path.join(OUT_DIR, "val.jsonl"), val)

    print(json.dumps({
        "out_dir": OUT_DIR,
        "seed": SEED,
        "total": len(buf),
        "train": len(train),
        "val": len(val),
        "train_frac": TRAIN_FRAC
    }, indent=2))

if __name__ == "__main__":
    main()

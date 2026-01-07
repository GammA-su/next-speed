import json
import os
import random
import re

BASE_TRAIN = os.environ.get("BASE_TRAIN", "data/decoder_train.jsonl")
QA_TRAIN = os.environ.get("QA_TRAIN", "data/decoder_train_qa.jsonl")
OUT_TRAIN = os.environ.get("OUT_TRAIN", "data/decoder_train_mix.jsonl")

BASE_VAL = os.environ.get("BASE_VAL", "data/decoder_val.jsonl")
QA_VAL = os.environ.get("QA_VAL", "data/decoder_val_qa.jsonl")
OUT_VAL = os.environ.get("OUT_VAL", "data/decoder_val_mix.jsonl")

MIX_QA_RATIO = float(os.environ.get("MIX_QA_RATIO", "0.30"))
SEED = int(os.environ.get("SEED", "0"))
MAX_ROWS = int(os.environ.get("MAX_ROWS", "-1"))

CONTROL_RE = re.compile(r"(<c\d+_\d+>|\bc\d+=\d+)")


def _count_lines(path, max_rows=-1):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
            if max_rows > 0 and n >= max_rows:
                break
    return n


def _reservoir_sample_indices(path, k, seed):
    rng = random.Random(seed)
    selected = []
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            if k > 0:
                if total < k:
                    selected.append(total)
                else:
                    j = rng.randint(0, total)
                    if j < k:
                        selected[j] = total
            total += 1
    selected.sort()
    return selected, total


def _iter_lines(path, max_rows=-1):
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_rows > 0 and idx >= max_rows:
                break
            yield line


def _iter_selected_lines(path, selected_indices):
    idx = 0
    want = 0
    want_total = len(selected_indices)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if want >= want_total:
                break
            if idx == selected_indices[want]:
                yield line
                want += 1
            idx += 1


def _write_checked(line, source, fout, stats):
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        stats["bad_json"] += 1
        return False

    tgt = rec.get("tgt_text", "")
    if not tgt or not str(tgt).strip():
        stats["skipped_empty"] += 1
        return False

    if CONTROL_RE.search(str(tgt)):
        stats["control_in_tgt"] += 1

    if not line.endswith("\n"):
        line = line + "\n"
    fout.write(line)
    stats["rows_out"] += 1
    if source == "base":
        stats["rows_out_base"] += 1
    else:
        stats["rows_out_qa"] += 1
    return True


def _mix_split(base_path, qa_path, out_path, ratio, seed, max_rows=-1):
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Missing base file: {base_path}")
    if not os.path.exists(qa_path):
        print(f"Missing QA file: {qa_path} (will copy base only)")
        qa_path = None

    if ratio < 0.0 or ratio >= 1.0:
        raise ValueError(f"MIX_QA_RATIO must be in [0,1), got {ratio}")

    base_count = _count_lines(base_path, max_rows=max_rows)
    target_qa = int(round(base_count * ratio / max(1e-9, 1.0 - ratio)))

    qa_indices = []
    qa_total = 0
    if qa_path:
        qa_indices, qa_total = _reservoir_sample_indices(qa_path, target_qa, seed)

    qa_used_target = len(qa_indices)
    if qa_path and qa_used_target < target_qa:
        print(
            f"QA smaller than target ratio: qa_total={qa_total} target_qa={target_qa} using_all_qa={qa_used_target}"
        )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    stats = {
        "rows_in_base": base_count,
        "rows_in_qa": qa_used_target,
        "rows_out": 0,
        "rows_out_base": 0,
        "rows_out_qa": 0,
        "skipped_empty": 0,
        "control_in_tgt": 0,
        "bad_json": 0,
    }

    rng = random.Random(seed)
    base_iter = _iter_lines(base_path, max_rows=max_rows)
    qa_iter = _iter_selected_lines(qa_path, qa_indices) if qa_path else iter(())

    base_remaining = base_count
    qa_remaining = qa_used_target

    with open(out_path, "w", encoding="utf-8") as fout:
        while base_remaining > 0 or qa_remaining > 0:
            if base_remaining == 0:
                line = next(qa_iter)
                qa_remaining -= 1
                _write_checked(line, "qa", fout, stats)
                continue
            if qa_remaining == 0:
                line = next(base_iter)
                base_remaining -= 1
                _write_checked(line, "base", fout, stats)
                continue

            p_qa = qa_remaining / (base_remaining + qa_remaining)
            if rng.random() < p_qa:
                line = next(qa_iter)
                qa_remaining -= 1
                _write_checked(line, "qa", fout, stats)
            else:
                line = next(base_iter)
                base_remaining -= 1
                _write_checked(line, "base", fout, stats)

    ctrl_pct = 100.0 * stats["control_in_tgt"] / max(1, stats["rows_out"])
    print(
        json.dumps(
            {
                "base_path": base_path,
                "qa_path": qa_path or "",
                "out_path": out_path,
                "rows_in_base": stats["rows_in_base"],
                "rows_in_qa": stats["rows_in_qa"],
                "rows_out": stats["rows_out"],
                "rows_out_base": stats["rows_out_base"],
                "rows_out_qa": stats["rows_out_qa"],
                "skipped_empty": stats["skipped_empty"],
                "control_in_tgt": stats["control_in_tgt"],
                "control_in_tgt_pct": round(ctrl_pct, 4),
                "bad_json": stats["bad_json"],
                "seed": seed,
                "mix_qa_ratio": ratio,
            },
            indent=2,
        )
    )
    if not os.path.exists(out_path):
        raise RuntimeError(f"Failed to write output: {out_path}")
    print(f"Wrote: {out_path} size_bytes={os.path.getsize(out_path)}")


def main():
    _mix_split(BASE_TRAIN, QA_TRAIN, OUT_TRAIN, MIX_QA_RATIO, SEED, max_rows=MAX_ROWS)

    if os.path.exists(BASE_VAL):
        _mix_split(BASE_VAL, QA_VAL, OUT_VAL, MIX_QA_RATIO, SEED, max_rows=-1)
    else:
        print(f"VAL base missing; skipping val mix ({BASE_VAL})")


if __name__ == "__main__":
    main()

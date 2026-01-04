# Sentence-Step Autoregressive Generation with RVQ Codes

This repo builds a hierarchical text generator where the autoregressive unit is a **sentence code tuple** (RVQ), and a token-level decoder renders the surface text.

## Quickstart (uv)

```bash
uv sync

# 1) Extract sentences (default HF dataset path)
uv run python 01_make_sentences.py

# 2) Embed sentences
uv run python 02_embed_sentences.py

# 3) Train RVQ codebooks
uv run python 03_fit_rvq.py

# 4) Build training sets (+ split)
uv run python 04_build_training_sets.py

# 5) Train Sentence-LM planner
uv run python 05_train_sentence_lm.py

# 6) Train decoder
uv run python 06_train_decoder.py

# 7) Generate
uv run python 07_generate.py
```

## LLM-style inference

Install the server extras (already in `pyproject.toml`):
```bash
uv sync
# or, if you prefer explicit adds:
uv add fastapi uvicorn
```

Interactive chat (reuses the same planner+decoder logic as `07_generate.py`):
```bash
uv run python 08_chat.py
```

Python API:
```bash
uv run python -c "from infer import SentenceGenerator; g=SentenceGenerator(); print(g.generate('Paris is the capital of France.', 2))"
```

FastAPI server:
```bash
uv run python 09_server.py
```

Example request:
```bash
curl -X POST http://localhost:8000/generate \\
  -H 'Content-Type: application/json' \\
  -d '{\"prompt\":\"Paris is the capital of France.\",\"max_sentences\":3,\"params\":{\"rerank\":true}}'
```

## Wikipedia ingestion (recommended)

If your environment does not already include `datasets` and `pysbd`, install them with:
```bash
uv add datasets pysbd
```

Export raw Wikipedia docs (streaming by default):
```bash
STREAMING=1 MAX_DOCS=10000 MIN_CHARS=800 HF_CONFIG=20231101.en \
  uv run python tools/export_wikimedia_wikipedia.py
```

Split into sentences using the local docs file:
```bash
DOCS_JSONL=data/raw/docs.jsonl uv run python 01_make_sentences.py
```

Then run the existing pipeline steps 02..07 as in Quickstart.

Sanity checks:
```bash
wc -l data/raw/docs.jsonl
wc -l data/sentences.jsonl
```

Scaling note: increase `MAX_DOCS` and watch the size of `data/embeddings.npy` and downstream files.

## Pipeline outputs

- `data/raw/docs.jsonl` and `data/raw/docs.meta.json`
- `data/sentences.jsonl` and `data/sentences.meta.json`
- `data/embeddings.npy` and `data/emb_meta.json`
- `out/rvq.npz` and `out/rvq_meta.json`
- `data/codes.npy`
- `data/sent_lm_train.jsonl`, `data/sent_lm_val.jsonl`
- `data/decoder_train.jsonl`, `data/decoder_val.jsonl`
- `out/sentence_lm/sentence_lm.pt` and `out/sentence_lm/config.json`
- `out/decoder/` and `out/decoder/config.json`
- `out/generate/*.json` (generation logs)

## Reproducibility

All scripts honor `SEED` and `DETERMINISTIC` for deterministic runs and write a metadata/config JSON alongside outputs.

- `SEED` (default: `0`)
- `DETERMINISTIC` (default: `1`) sets deterministic torch/cuDNN settings

## Environment variables (common knobs)

General:
- `SEED`, `DETERMINISTIC`

Data:
- `HF_DATASET`, `HF_CONFIG`, `HF_SPLIT` (defaults: `wikitext`, `wikitext-2-raw-v1`, `train`)
- `DOCS_JSONL` (use local docs JSONL when set)
- `SENT_PATH`, `SENT_META`
- `MIN_LEN`, `MAX_LEN`, `LANG`

Wikipedia export (`tools/export_wikimedia_wikipedia.py`):
- `HF_CONFIG` (default: `20231101.en`)
- `HF_SPLIT` (default: `train`)
- `STREAMING`, `SHUFFLE_BUFFER`
- `MIN_CHARS`, `MAX_DOCS`
- `OUT_PATH`, `OUT_META`

Embeddings:
- `EMB_MODEL`, `BS`, `MAXLEN`, `OUT_EMB`, `OUT_META`

RVQ:
- `K`, `V`, `NITER`, `RVQ_PATH`, `RVQ_META`

Training sets:
- `CONTEXT_SENTS`, `VAL_FRAC`
- `OUT_LM`, `OUT_LM_VAL`, `OUT_DEC`, `OUT_DEC_VAL`, `TRAIN_META`

Sentence-LM:
- `D_MODEL`, `N_LAYERS`, `N_HEADS`, `DROP`
- `BS`, `LR`, `STEPS`, `LOG_EVERY`, `OUT_DIR`

Decoder:
- `MODEL`, `MAX_IN`, `MAX_OUT`
- `CODE_MODE` (`special`, `text`, `none`)
- `K`, `V`, `ADD_SPECIAL_TOKENS`
- `NUM_EPOCHS`, `LR`, `TRAIN_BS`, `MAX_ROWS`

Generation:
- `GEN_STEPS`, `N_CAND`, `RERANK`
- `CODE_MODE` (`special`, `text`, `none`)
- `DO_SAMPLE`, `TOP_P`, `TEMP`, `MAX_NEW_TOKENS`
- `RUN_ID`, `OUT_PATH`

## Decoder conditioning

Set `CODE_MODE`:
- `special` (default): uses compact special tokens like `<c0_0001> <c1_0512> ...`.
- `text`: uses the fallback string format `c0=.. c1=..`.
- `none`: ablation that removes code conditioning entirely.

Notes:
- `special` adds `K * V` tokens to the tokenizer and resizes the decoder embeddings.
- If you change `K` or `V`, re-run `06_train_decoder.py` so token IDs and embeddings align.

## Evaluation scripts

RVQ quantization error:
```bash
uv run python rvq_recon_error.py
```

Sentence-LM accuracy (per-head, tuple exact match):
```bash
uv run python sent_lm_eval.py
```

Code agreement between predicted codes and generated sentence codes:
```bash
uv run python code_agreement_eval.py
```

## Ablations and sanity checks

Decoder without codes:
```bash
CODE_MODE=none uv run python 06_train_decoder.py
```

Decoder tokenization cache (faster reruns):
```bash
NUM_PROC=16 CACHE_TOKENIZED=1 MAP_BATCH_SIZE=2000 uv run python 06_train_decoder.py
```

Pretokenize only (build cache and exit):
```bash
PRETOKENIZE_ONLY=1 NUM_PROC=16 CACHE_TOKENIZED=1 MAP_BATCH_SIZE=2000 uv run python 06_train_decoder.py
```

Generation without reranking:
```bash
RERANK=0 uv run python 07_generate.py
```

Smaller codebooks:
```bash
K=4 V=256 uv run python 03_fit_rvq.py
K=4 V=256 uv run python 04_build_training_sets.py
K=4 V=256 uv run python 05_train_sentence_lm.py
K=4 V=256 uv run python 06_train_decoder.py
```

## Notes

- The Sentence-LM predicts the next sentence code tuple (K categorical heads).
- The decoder renders a single sentence conditioned on context text + predicted codes.
- Candidate reranking re-encodes generated sentences and prefers code match.
# next-speed

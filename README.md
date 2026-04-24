# Automatic Error Correction for Non-Native Russian Speakers with Corpus Expansion

Grammatical Error Correction (GEC) for Russian as a foreign language. This diploma project trains and compares two models on the RLC-GEC corpus: a seq2seq T5 baseline and a Gemma-2B model fine-tuned with SimPO (Simple Preference Optimization). The corpus is expanded with rule-based synthetic errors generated from a Russian Wikipedia dump.

## Repository contents

| File | Description |
|---|---|
| `preprocessing.ipynb` | Normalises `sentences_full.csv`: unicode NFC, XLM-RoBERTa punctuation restoration, Cyrillic filter, identical-pair removal, preserves prev/next context columns |
| `data_prep_new.ipynb` | Reconstructs (text, corrected) pairs from raw annotator DB tables (`annotator_*.csv`); joins neighbours for prev/next sentence context |
| `data_analysis.ipynb` | Exploratory analysis of the preprocessed corpus: length distributions, POS tags, n-gram counts, outlier detection |
| `synthetic_data.ipynb` | Generates ~130k synthetic (correct, corrupted) pairs from a Russian Wikipedia dump; error types and confusion lists are calibrated from `annotator_annotation.csv` (123k annotations); includes a `real_sub` injector that replays actual learner substitution patterns |
| `t5-baseline-fixed-2.ipynb` | Fine-tunes [cointegrated/rut5-base-multitask](https://huggingface.co/cointegrated/rut5-base-multitask) on the GEC task with PyTorch Lightning; always includes prev/next context in input; evaluates with ROUGE and BLEU |
| `simpo-gemma-2.ipynb` | Fine-tunes [Vikhrmodels/Vikhr-Gemma-2B-instruct](https://huggingface.co/Vikhrmodels/Vikhr-Gemma-2B-instruct) with a custom SimPO loss, 4-bit quantization, and frozen bottom layers; always includes prev/next context in prompt |
| `annotator_sentence.csv` | Raw DB dump: 198,907 learner sentences |
| `annotator_annotation.csv` | Raw DB dump: 123,605 annotations with span corrections |
| `annotator_document.csv` | Raw DB dump: 15,255 documents with metadata (L1, year, corpus) |
| `sentences_preprocessed.csv` | Preprocessed output of `preprocessing.ipynb` |

## Data

The project uses the **RLC-GEC corpus** (Russian Learner Corpus for Grammatical Error Correction), 198k sentences from 15k documents, 50+ L1 backgrounds, sourced from three raw annotator DB dumps:

| File | Rows | Key columns |
|---|---|---|
| `annotator_sentence.csv` | 198,907 | `id`, `text`, `document_id`, `sentence_index` |
| `annotator_annotation.csv` | 123,605 | `sentence_id`, JSON `data` field (`corrs`, `quote`, `tags`, `ranges`) |
| `annotator_document.csv` | 15,255 | `id`, task type, corpus, year, L1 |

`data_prep_new.ipynb` reconstructs ~51k (text, corrected) pairs by applying `quote→corrs` span replacements, then joins each sentence with its document neighbours to produce `prev_sent` and `next_sent` context columns. Output: `sentences_full.csv`.

`preprocessing.ipynb` then normalises `sentences_full.csv` (NFC, XLM-RoBERTa punctuation restoration, Cyrillic filter, dedup) and outputs `sentences_preprocessed.csv` with all four columns preserved.

## Pipeline

```
annotator_*.csv ─► data_prep_new.ipynb ──► sentences_full.csv ─────────────────┐
                     │                         (+ prev/next ctx)               │
                     │                                                          │
                     └──► synthetic_data.ipynb ──► synthetic_errors.csv        │
                     │       (annotation freq + real_sub + confusion)           │
Wikipedia dump ──────┘                                   │                      │
                                                         │                      ▼
                                            preprocessing.ipynb ──► sentences_preprocessed.csv
                                            (NFC + XLM-RoBERTa punct + filter)         │
                                                                                        ▼
                                               t5-baseline-fixed-2.ipynb: load_data(sentences_preprocessed.csv,
                                               simpo-gemma-2.ipynb:                    synthetic_errors.csv)
```

Both training notebooks always include prev/next sentence context in the model input (context is truncated to 50 characters per side).

## Kaggle setup

All notebooks are configured for Kaggle (GPU T4 x2 or P100 recommended for training). Paths are hardcoded to `/kaggle/input/` — no edits needed as long as you attach the right datasets.

### Step 1 — Create Kaggle datasets

Upload the following as separate Kaggle datasets, using the dataset slug shown:

| Dataset slug | Files to upload | Used by |
|---|---|---|
| `rlc-raw` | `annotator_sentence.csv`, `annotator_annotation.csv`, `annotator_document.csv` | `data_prep_new.ipynb`, `synthetic_data.ipynb` |
| `sentences` | `sentences_preprocessed.csv`, `sentences_full.csv`, `rlc_test.csv` | `preprocessing.ipynb`, training notebooks |
| `new-gen` | `synthetic_errors.csv` (output of `synthetic_data.ipynb`) | training notebooks |
| `ruwiki` | Russian Wikipedia `.bz2` dump | `synthetic_data.ipynb` |

Attach each dataset to the relevant notebook via **Notebook settings → Add data**.

### Step 2 — Set credentials

In `t5-baseline-fixed-2.ipynb` and `simpo-gemma-2.ipynb`, fill in:

```python
WANDB_KEY = 'your-key-here'      # https://wandb.ai/authorize
WANDB_PROJECT = 'your-project'
```

### Step 3 — Run order

```
1. data_prep_new.ipynb        →  produces sentences_full.csv (51k pairs + context columns)
                                  (needs rlc-raw dataset)
2. synthetic_data.ipynb       →  produces synthetic_errors.csv
                                  (needs rlc-raw for annotation calibration + ruwiki dump)
3. preprocessing.ipynb        →  produces sentences_preprocessed.csv
                                  (NFC + XLM-RoBERTa punct + filter; needs sentences_full.csv)
4. data_analysis.ipynb        →  exploratory analysis (no output files required downstream)
5. t5-baseline-fixed-2.ipynb  →  trains T5 baseline, saves model + predictions
6. simpo-gemma-2.ipynb        →  trains SimPO Gemma-2B, saves model + predictions
```

In both training notebooks `load_data(SENTENCES_CSV, SYNTHETIC_CSV)` stacks `sentences_preprocessed.csv`
and `synthetic_errors.csv` into one combined training set. Prev/next sentence context (truncated to 50 chars)
is always included in model inputs.

### Dependencies

The install cells at the top of each notebook handle all Python packages via `pip`. No additional system packages are required.

## Models

### T5 Baseline (`t5-baseline-fixed-2.ipynb`)

- **Model:** `cointegrated/rut5-base-multitask` (Russian T5-base)
- **Task prefix:** `"Исправь грамматические ошибки в предложении: "` with `[до: …]` / `[после: …]` context brackets (always on; omitted only when the neighbour sentence is absent)
- **Training:** sequence-to-sequence cross-entropy, cosine LR schedule with warmup
- **Decoding:** beam search (4 beams), `max_length=192`
- **Metrics:** ROUGE-1/2/L, BLEU-4

### SimPO Gemma-2B (`simpo-gemma-2.ipynb`)

- **Model:** `Vikhrmodels/Vikhr-Gemma-2B-instruct`
- **Quantization:** 4-bit NF4 (BitsAndBytes), compute dtype bfloat16
- **Prompt:** always includes `Предыдущее: …` / `Следующее: …` context lines (omitted only when the neighbour sentence is absent)
- **Optimization:** SimPO loss — preference learning without a reference model. The chosen response is the gold correction; the rejected response is the uncorrected input sentence.
- **SimPO loss:**

```
L = -log σ(β · (log p_θ(chosen|prompt) - log p_θ(rejected|prompt)))
```

  Log-probabilities are length-normalized over response tokens only (prompt tokens are masked).
- **Memory efficiency:** gradient checkpointing, bottom 8 layers frozen, `accumulate_grad_batches=8`
- **Precision:** `bf16-true`

## Synthetic data generation (`synthetic_data.ipynb`)

Error frequencies are computed from the annotated RLC-GEC corpus and used as sampling weights (softmax-normalized). The following error types are implemented:

| Tag | Description |
|---|---|
| `ortho` | Orthographic substitution (ы↔и, тся↔ться, etc.) |
| `gov` | Case government error |
| `prep` | Preposition substitution or deletion |
| `ref` | Pronoun reference error |
| `conj` | Conjunction substitution |
| `agrcase` | Case agreement error |
| `agrnum` | Number agreement error |
| `agrgender` | Gender agreement error |
| `agrpers` | Person agreement error (я пишет instead of я пишу) |
| `num` | Number flip (singular↔plural) |
| `space` | Missing space (mainly with "не") |
| `tense` | Verb tense error |
| `wo` | Word order swap |
| `infl` | Inflection confusion (-ите/-ете, -ют/-ят) |
| `keyboard_ortho` | Keyboard proximity error on the ЙЦУКЕН layout |
| `refl` | Reflexive suffix added or removed (-ся/-сь) |
| `asp` | Verb aspect swap (perfective↔imperfective) from curated pairs |
| `brev` | Short vs. full adjective confusion |
| `graph` | Cyrillic character replaced by a visually identical Latin one |
| `hyphen` | Hyphen deleted from a compound word (куда-то→кудато) |
| `lex` | MLM-based contextual substitution via RuBERT (`DeepPavlov/rubert-base-cased`) |
| `real_sub` | Corpus-derived learner substitution patterns replayed from `annotator_annotation.csv` (e.g. также→тоже) |

Source corpus: Russian Wikipedia dump (~390k sentences after filtering).
Error type frequencies and confusion lists (prepositions, conjunctions, `real_sub` map) are derived directly from `annotator_annotation.csv`.

## Annotation tools

Two web tools were built for corpus annotation, for internal use only.

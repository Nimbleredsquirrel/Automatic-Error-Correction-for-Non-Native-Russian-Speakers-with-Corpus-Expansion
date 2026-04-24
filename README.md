# Automatic Error Correction for Non-Native Russian Speakers with Corpus Expansion

Grammatical Error Correction (GEC) for Russian as a foreign language. This diploma project trains and compares two models on the RLC-GEC corpus: a seq2seq T5 baseline and a Gemma-2B model fine-tuned with SimPO (Simple Preference Optimization). The corpus is expanded with rule-based synthetic errors generated from Russian Wikipedia.

## Repository contents

| File | Description |
|---|---|
| `data-prep-new.ipynb` | Reconstructs (text, corrected) pairs from raw annotator DB tables (`annotator_*.csv`); joins neighbours for prev/next sentence context; outputs `sentences_full.csv` (51,776 pairs) |
| `preprocessing.ipynb` | Normalises `sentences_full.csv`: unicode NFC, XLM-RoBERTa punctuation restoration (`1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase`, ONNX), Cyrillic filter, identical-pair removal; outputs `sentences_preprocessed.csv` (50,329 pairs) |
| `data_analysis.ipynb` | Exploratory analysis of the raw RLC corpus (`annotator_*.csv`): length distributions, edit distance breakdown, POS tags, n-gram counts, error type frequencies, per-document error rates, outlier detection; outputs `outliers.csv`, `corpus_summary.csv`, `empty.csv` |
| `synthetic-data.ipynb` | Generates synthetic (correct, corrupted) pairs from Russian Wikipedia (`wikimedia/wikipedia`, HuggingFace); error types and confusion lists are calibrated from `annotator_annotation.csv`; includes a `real_sub` injector that replays actual learner substitution patterns; outputs `synthetic_errors.csv` |
| `t5-baseline-fixed-2.ipynb` | Fine-tunes [cointegrated/rut5-base-multitask](https://huggingface.co/cointegrated/rut5-base-multitask) on the GEC task; inputs `sentences_preprocessed.csv` + full `synthetic_errors.csv`; evaluates with ROUGE and BLEU |
| `simpo-gemma-2.ipynb` | Fine-tunes [Vikhrmodels/Vikhr-Gemma-2B-instruct](https://huggingface.co/Vikhrmodels/Vikhr-Gemma-2B-instruct) with a custom SimPO loss, 4-bit NF4 quantization, and frozen bottom layers; inputs full `sentences_preprocessed.csv` + full `synthetic_errors.csv` |

## Data

The project uses the **RLC-GEC corpus** (Russian Learner Corpus for Grammatical Error Correction), 198k sentences from 15k documents, 50+ L1 backgrounds, sourced from three raw annotator DB dumps:

| File | Rows | Key columns |
|---|---|---|
| `annotator_sentence.csv` | 198,907 | `id`, `text`, `document_id`, `sentence_index` |
| `annotator_annotation.csv` | 123,605 | `sentence_id`, JSON `data` field (`corrs`, `quote`, `tags`, `ranges`) |
| `annotator_document.csv` | 15,255 | `id`, task type, corpus, year, L1 |

`data-prep-new.ipynb` reconstructs 51,776 (text, corrected) pairs by applying `quote→corrs` span replacements from 58,907 annotations, then joins each sentence with its document neighbours to produce `prev_sent` and `next_sent` context columns.

`preprocessing.ipynb` normalises `sentences_full.csv` (NFC, XLM-RoBERTa punctuation restoration, Cyrillic filter, dedup) and outputs `sentences_preprocessed.csv` (50,329 pairs) with all four columns preserved.

## Pipeline

```
annotator_*.csv ──► data-prep-new.ipynb ──► sentences_full.csv
                                               (51,776 pairs + prev/next ctx)
                                                    │
                         ┌──────────────────────────┤
                         │                          │
                         ▼                          ▼
              synthetic-data.ipynb         preprocessing.ipynb
              (Wikipedia HuggingFace)      (NFC + punct + filter)
                         │                          │
                         ▼                          ▼
              synthetic_errors.csv     sentences_preprocessed.csv
                         │
                         └──────────────────────────┐
                                                     ▼
                              t5-baseline-fixed-2.ipynb   (sentences_full + 10% synthetic)
                              simpo-gemma-2.ipynb          (10% sentences_full + 5% synthetic)
```

Both training notebooks always include prev/next sentence context in the model input (truncated to 50 characters per side).

## Kaggle setup

All notebooks are configured for Kaggle. Paths are hardcoded to `/kaggle/input/` — no edits needed as long as you attach the right datasets.

### Step 1 — Create Kaggle datasets

| Dataset slug | Files to upload | Used by |
|---|---|---|
| `rlc-raw` | `annotator_sentence.csv`, `annotator_annotation.csv`, `annotator_document.csv` | `data-prep-new.ipynb`, `synthetic-data.ipynb` |
| `sentences` | `sentences_preprocessed.csv`, `sentences_full.csv`, `rlc_test.csv` | `preprocessing.ipynb`, training notebooks |
| `new-gen` | `synthetic_errors.csv` (output of `synthetic-data.ipynb`) | training notebooks |

Wikipedia is loaded directly from HuggingFace (`wikimedia/wikipedia`, `20231101.ru`) — no separate dataset needed.

### Step 2 — Set credentials

In `t5-baseline-fixed-2.ipynb` and `simpo-gemma-2.ipynb`, fill in:

```python
WANDB_KEY = 'your-key-here'      # https://wandb.ai/authorize
WANDB_PROJECT = 'your-project'
```

### Step 3 — Run order

```
1. data-prep-new.ipynb        →  sentences_full.csv (51,776 pairs + context)
                                  needs: rlc-raw dataset
2. synthetic-data.ipynb       →  synthetic_errors.csv
                                  needs: rlc-raw (annotation calibration)
                                         internet access for wikimedia/wikipedia
3. preprocessing.ipynb        →  sentences_preprocessed.csv (50,329 pairs)
                                  needs: sentences_full.csv
4. data_analysis.ipynb        →  exploratory only, no downstream outputs required
5. t5-baseline-fixed-2.ipynb  →  trains T5 baseline, saves model + predictions
6. simpo-gemma-2.ipynb        →  trains SimPO Gemma-2B, saves model + predictions
```

## Models

### T5 Baseline (`t5-baseline-fixed-2.ipynb`)

- **Model:** `cointegrated/rut5-base-multitask` (Russian T5-base)
- **Task prefix:** `"Исправь грамматические ошибки в предложении: "` with `[до: …]` / `[после: …]` context brackets
- **Training:** 1 epoch, AdamW (lr=3e-5, weight_decay=0.01), cosine schedule with 10% warmup, gradient clipping 0.5
- **Data:** `sentences_preprocessed.csv` + full `synthetic_errors.csv`, max length 192 tokens, batch size 16
- **Decoding:** beam search (4 beams), `max_length=192`
- **Metrics:** ROUGE-1/2/L, BLEU-4

### SimPO Gemma-2B (`simpo-gemma-2.ipynb`)

- **Model:** `Vikhrmodels/Vikhr-Gemma-2B-instruct`
- **Quantization:** 4-bit NF4 (BitsAndBytes), compute dtype bfloat16
- **Prompt:** always includes `Предыдущее: …` / `Следующее: …` context lines
- **Optimization:** SimPO loss — preference learning without a reference model. Chosen response = gold correction; rejected response = uncorrected input.
- **SimPO loss:**

```
L = -log σ(β · (log p_θ(chosen|prompt) - log p_θ(rejected|prompt)))
```

  β=2.0; log-probabilities are length-normalized over response tokens only.
- **Training:** 2 epochs, AdamW (lr=5e-5, weight_decay=0.01), cosine schedule with 10% warmup, gradient clipping 0.5
- **Data:** full `sentences_preprocessed.csv` + full `synthetic_errors.csv`, max length 192 tokens, batch size 1 × grad accumulation 8
- **Memory:** gradient checkpointing, bottom 8 layers frozen, bf16
- **Decoding:** `max_new_tokens=32`

## Synthetic data generation (`synthetic-data.ipynb`)

Source: Russian Wikipedia via HuggingFace (`wikimedia/wikipedia`, `20231101.ru`), 2.8M sentences available; 150k sampled for generation (3:1 ratio to real data). Error type frequencies and confusion lists (prepositions, conjunctions, `real_sub` map) are derived from `annotator_annotation.csv` and softmax-normalized for sampling.

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
| `real_sub` | Corpus-derived learner substitution patterns replayed from `annotator_annotation.csv` |

## Annotation tools

Two web tools were built for corpus annotation, for internal use only.

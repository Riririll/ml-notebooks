# Clickbait spoiling: spoiler type classification

Notebooks:
- [`clickbait_spoiling_baseline.ipynb`](clickbait_spoiling_baseline.ipynb): fine-tuned transformer baselines
- [`span_tagging_classification.ipynb`](span_tagging_classification.ipynb): a two-stage method that tags spoiler spans first and then classifies the spoiler type

## Context

This is part of a two-person course project for *2209678 Natural Language Processing Systems* at Chulalongkorn University, on SemEval 2023 Task 5 (Webis Clickbait Spoiling). The project report also describes a third, larger model (SPACEBERT) that is a separate component and is not included here.

A clickbait post promises information it does not give. The linked article contains the answer, and a *spoiler* is the text that delivers it. Subtask 1 asks which kind of spoiler a post needs: a short `phrase`, a longer `passage`, or `multi` (several separate parts). The metric is **balanced accuracy**, the mean of per-class recall, so a model that always predicts one class scores 1/3.

## Data

The task provides `train.jsonl`, `val.jsonl` (called `validation.jsonl` in the second notebook) and `test.jsonl`. `multi` is the rarest class.

| Split | Posts | phrase | passage | multi |
|---|---|---|---|---|
| Train | 3,200 | 42.7% | 39.8% | 17.5% |
| Validation | 800 | 41.9% | 40.2% | 17.9% |
| Test | 1,000 | 423 | 403 | 174 |

Each record has the post text, the article title, the article paragraphs, the spoiler text, the spoiler character positions and a spoiler tag. The data is not included here. Download it from the task page and place the files where the notebook expects them.

## Notebook 1: baselines

**Approach.** One input string per example: post text, then ` - `, then the article title, then the article paragraphs. The label is the first spoiler tag. A pretrained encoder with a classification head is fine-tuned through a small wrapper around the Hugging Face `Trainer`: 5 epochs, batch size 8, max length 256, library-default learning rate, unweighted loss. The epoch with the best validation balanced accuracy is kept, and the model is scored on the 1,000 test posts. A `simpletransformers` version is included at the end as scratch work.

**Results.** Test balanced accuracy, one training run per cell. Run 2 repeats three of the models from run 1.

| Model | Run 1 | Run 2 |
|---|---|---|
| `roberta-base` | 0.333 | **0.695** |
| `bert-base-uncased` | **0.671** | |
| `distilroberta-base` | **0.665** | |
| `distilbert-base-uncased` | **0.660** | |
| `microsoft/deberta-v3-base` | 0.352 | |
| `bert-large-uncased` | 0.333 | 0.333 |
| `roberta-large` | 0.333 | 0.333 |

The best run (`roberta-base`, run 2) reaches 0.695 balanced accuracy and 0.71 plain accuracy. Its per-class F1 is 0.73 for `phrase`, 0.73 for `passage` and 0.64 for `multi`. On the validation set the same run reached 0.709.

**What the results show, and what they do not.**
- Five of the ten runs scored exactly 0.333, where the model predicted the same class (`phrase`) for every input. `deberta-v3-base` came close to the same failure. The same model and settings gave 0.333 in run 1 and 0.695 in run 2 for `roberta-base`, so the outcome depends strongly on the random seed.
- The large models failed in both runs. A likely explanation is that the default learning rate is too aggressive for them at batch size 8, but this was not tested.
- In the best run, validation loss is lowest at epoch 3 and rises afterwards while training loss keeps falling, so overfitting starts early.
- These are single runs without seed control, so differences of a few points between the base-size models are within what seed noise could explain.

## Notebook 2: span tagging, then classification

**Approach.** Two stages.

1. *Span tagging.* `SpanBERT/spanbert-base-cased` is fine-tuned as a token classifier with BIO tags over the article paragraphs joined into one string (max length 256). The dataset gives spoiler positions as paragraph index plus character offsets, so the notebook converts them to absolute character positions in the joined text and then maps them to tokens with the tokenizer's offset mapping. 3,199 of the 3,200 training posts have usable positions.
2. *Classification.* The predicted spans are appended to the input (`post - title - paragraphs [SPANS] span1 span2 ...`) and a `bert-base-uncased` classifier (dropout 0.3, weight decay 0.01) predicts the spoiler type. Training stops early on validation loss with patience 2 and restores the best weights.

Settings behind the stored results: learning rate 5e-6 with linear decay and no warmup, gradient clipping at 1.0, seed 42. Stage 1 ran 3 epochs at batch size 16. The stored Stage 2 run used 5 epochs with batch size 8, and reloaded the tagger saved by the first run.

**Results.** Scored on the validation set (800 posts). The notebook writes test predictions to `submission.csv` but does not score them.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| phrase | 0.60 | 0.83 | 0.70 |
| passage | 0.69 | 0.58 | 0.63 |
| multi | 0.92 | 0.42 | 0.58 |

Balanced accuracy 0.6111, accuracy 0.6575. For comparison, the `roberta-base` baseline reached 0.709 on the same validation set, so this method did not beat the plain baseline. The notebook also stores a second Stage 2 run with the default settings in the source (3 epochs, batch size 16), which scored 0.361 and never predicted `multi`.

**Observations and open questions.**
- The model rarely predicts `multi`, but when it does it is almost always right (precision 0.92, recall 0.42). `multi` is 17.5% of the training data. Code that oversamples `multi` is present in `run_classification_only` but is commented out, so it was not part of these results.
- Stage 1 token-level validation accuracy is 0.9694 in all three epochs. Spoiler tokens are a small share of all tokens (8 of 256 positions in the sample the notebook prints, with padding counted as `O`), so a tagger that predicts `O` everywhere would also score a high accuracy. The constant value hints that the tagger may not have learned to find spans, but no span-level metric is reported, so this is not established. The accuracy code also zips all validation predictions with only the last batch's attention mask, so it covers just the last 32 validation posts.
- The spans are appended after the full article text and the input is truncated to 256 tokens. For long articles the spans would be cut off. How often this happens was not measured.
- Stage 2 is trained on spans predicted by a tagger on its own training data, while validation and test spans come from unseen posts, so the training spans are probably cleaner than the ones seen at test time.
- Validation data both picks the stopping epoch (in the current code) and is the data the result is reported on, so the number is slightly optimistic.

## Comparing the two

The baselines are scored on the test set and the span method on the validation set, so the two tables are not directly comparable. Only the validation figure for `roberta-base` (0.709 against 0.611) is a like-for-like comparison. The baselines also use library defaults, while the span method uses its own learning rate schedule and early stopping.

## What I would do next

Fix the random seeds and repeat runs, check what the tagger actually predicts at the span level, put the spans before the article text (or shorten the article) so they survive truncation, train Stage 2 on out-of-fold spans, turn on `multi` oversampling or a class-weighted loss, and score everything on the test set. None of this is done in the notebooks.

## Reproduce

```
pip install -r requirements.txt
```

Both notebooks were written for Google Colab with a GPU, so edit the data paths near the top. For the second notebook, put `train.jsonl`, `validation.jsonl` and `test.jsonl` in the working directory, run `run_full_pipeline`, and set `TRAIN_BATCH_SIZE = 8`, `VALID_BATCH_SIZE = 16` and `EPOCHS = 5` to match the stored result. `run_classification_only` reuses the saved tagger (`span_tagger_model.pt`) and retrains only Stage 2. The last cells of each notebook are scratch code or tests.

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

**Notes from the stored outputs.**
- Five of the ten runs scored exactly 0.333: the model predicted `phrase` for every test post. `deberta-v3-base` scored 0.352.
- `roberta-base` scored 0.333 in run 1 and 0.695 in run 2 with the same code and settings. Seeds are not fixed in the notebook.
- In the best run, validation loss is lowest at epoch 3 (0.728) and is 1.084 at epoch 5, while training loss keeps falling. Validation balanced accuracy still rises to its best value at epoch 5 (0.709).
- Every number above comes from a single run.

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

**Notes from the code and stored outputs.**
- `multi` precision is 0.92 and recall is 0.42. The code that oversamples `multi` in `run_classification_only` is commented out, so it was not used for these results.
- Stage 1 token-level validation accuracy is 0.9694 in all three epochs. No span-level metric is computed. The accuracy code zips all validation predictions with only the last batch's attention mask, so it covers the last 32 validation posts only.
- Spans are appended after the full article text and the input is truncated to 256 tokens.
- Stage 2 training uses spans predicted by the tagger on its own training data.
- The validation set is used for early stopping and is also the set the result is reported on.

## Comparing the two

The baselines are scored on the test set and the span method on the validation set, so the two tables are not directly comparable. Only the validation figure for `roberta-base` (0.709 against 0.611) is a like-for-like comparison. The baselines also use library defaults, while the span method uses its own learning rate schedule and early stopping.

## Reproduce

```
pip install -r requirements.txt
```

Both notebooks were written for Google Colab with a GPU, so edit the data paths near the top. For the second notebook, put `train.jsonl`, `validation.jsonl` and `test.jsonl` in the working directory, run `run_full_pipeline`, and set `TRAIN_BATCH_SIZE = 8`, `VALID_BATCH_SIZE = 16` and `EPOCHS = 5` to match the stored result. `run_classification_only` reuses the saved tagger (`span_tagger_model.pt`) and retrains only Stage 2. The last cells of each notebook are scratch code or tests.

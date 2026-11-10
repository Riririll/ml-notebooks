# mBERT fine-tuning for IMDb sentiment classification

Notebook: [`mbert_imdb_sentiment.ipynb`](mbert_imdb_sentiment.ipynb)

## What this does

Fine-tunes `bert-base-multilingual-cased` (mBERT) with a sequence classification head for binary sentiment classification of English movie reviews, and scores it with precision, recall and F1 for the positive class.

## Data

The `stanfordnlp/imdb` dataset from Hugging Face (labels: 0 negative, 1 positive). The notebook builds its own split from the 25,000-review train split:

- Test: 1,000 reviews, held out with `train_test_split(test_size=1000, seed=42)`.
- Train: the first 5,000 of the remaining 24,000 reviews.

The original IMDb test split and the unsupervised split are not used. There is no separate validation set.

## Method

Reviews are tokenized with the mBERT tokenizer, padded or truncated to 128 tokens. Training uses the Hugging Face `Trainer`: 3 epochs, batch size 16 (939 steps), 500 warmup steps, weight decay 0.01, and the default optimizer settings of the library. The test set is evaluated after every epoch, but nothing is selected on it: the model from the last epoch is the one that is scored.

## Result

Scores on the 1,000 held-out reviews, positive class, from one training run:

| Precision | Recall | F1 |
|---|---|---|
| 0.8264 | 0.8197 | 0.8230 |

Training took about 8.6 minutes (517 s) on a Colab T4 GPU. This is a single run. The notebook does not fix the seeds beyond the data split and does not compare against another model.

## Notes from the code

- Reviews longer than 128 tokens are truncated.
- The notebook also writes the test reviews with gold and predicted labels to `prediction_results.csv`. The file is not tracked in this repository.

## Reproduce

```
pip install -r requirements.txt
```

The stored run used `transformers` 5.22.1 and `datasets` 4.8.5 on Google Colab with Python 3.13. Older `transformers` versions name some `TrainingArguments` options differently (for example `evaluation_strategy` instead of `eval_strategy`).

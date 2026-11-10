# Clickbait spoiling: spoiler type classification

Notebook: [`clickbait_spoiling_baseline.ipynb`](clickbait_spoiling_baseline.ipynb)

## Overview

A clickbait post promises information it does not give ("You won't believe what this CEO did"). The linked article contains the answer, and a *spoiler* is the short text that delivers it. SemEval 2023 Task 5 (Webis Clickbait Spoiling) asks for two things: classify what kind of spoiler a post needs (subtask 1), then generate it (subtask 2). This notebook covers **subtask 1**.

There are three spoiler types: a short `phrase`, a longer `passage`, or `multi` (several separate parts). The task is scored with **balanced accuracy**, the mean of per-class recall, so a model that always predicts one class scores 1/3.

## Data

The task provides `train.jsonl` (3,200 posts), `val.jsonl` and `test.jsonl`. The test set has 1,000 posts, with 423 `phrase`, 403 `passage` and 174 `multi` examples. Each record has the post text, the article title, the article paragraphs and a spoiler tag.

The notebook builds one input string per example: post text, then ` - `, then the article title, then the article paragraphs joined with spaces. The label is the first spoiler tag. The data is not included here. Download it from the task page and put the three files in `data/`.

## Approach

1. Convert the JSONL files to dataframes and then to CSV.
2. Fine-tune a pretrained encoder with a classification head, using a small wrapper class around the Hugging Face `Trainer`: 5 epochs, batch size 8, max sequence length 256, evaluation after every epoch, and the epoch with the best validation balanced accuracy is kept.
3. Report balanced accuracy and a per-class report on the test set.
4. Repeat for several encoders. A `simpletransformers` version of the same setup is included at the end as an alternative baseline.

## Results

Test balanced accuracy, one training run per cell. Run 2 repeats three of the models from run 1.

| Model | Run 1 | Run 2 |
|---|---|---|
| `roberta-base` | 0.333 | **0.695** |
| `bert-base-uncased` | 0.671 | |
| `distilroberta-base` | 0.665 | |
| `distilbert-base-uncased` | 0.660 | |
| `microsoft/deberta-v3-base` | 0.352 | |
| `bert-large-uncased` | 0.333 | 0.333 |
| `roberta-large` | 0.333 | 0.333 |

The best run (`roberta-base`, run 2) reaches 0.695 balanced accuracy and 0.71 plain accuracy. Its per-class F1 is 0.73 for `phrase`, 0.73 for `passage` and 0.64 for `multi`, so the rare `multi` class is the hardest. Its validation balanced accuracy was 0.709 at the last epoch.

## What the results show, and what they do not

- **Five of the ten runs scored exactly 0.333.** In these runs the model predicted the same class for every input (`phrase`). `deberta-v3-base` came close to the same failure, with 0.352. The same model and settings gave 0.333 in run 1 and 0.695 in run 2 for `roberta-base`, so the outcome depends strongly on the random seed.
- **The large models failed in both runs.** My likely explanation is that the default learning rate is too aggressive for them at batch size 8, but I did not test this.
- **Overfitting starts early.** In the best run, validation loss is lowest at epoch 3 and rises afterwards while training loss keeps falling, although validation balanced accuracy still improves slightly.
- **The numbers are not a model ranking.** Each model was trained once or twice, with default hyperparameters and no seed control, so differences of a few points between `bert-base`, `distilroberta` and `distilbert` are within what I would expect from seed noise.

## What I would do next

Fix the random seeds and run each model several times, lower the learning rate with warmup for the large models, stop on validation loss, and try a class-weighted loss for `multi`. None of this is done in the notebook.

## Reproduce

```
pip install -r requirements.txt
```

Open the notebook, point the path cells at your data folder (the original paths are Google Drive paths) and run it top to bottom. A GPU is needed for reasonable training time. The last cells (`simpletransformers` section) are scratch experiments without saved results.

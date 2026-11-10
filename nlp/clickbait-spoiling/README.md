# Clickbait spoiling type classification (SemEval 2023 Task 5)

**Problem.** Given a clickbait post and the linked article, predict what kind of spoiler the post needs: a short `phrase`, a longer `passage`, or `multi` (several separate parts). This is Task 1 of SemEval 2023 Task 5 and is scored with balanced accuracy.

**Method.** The post text, the article title and the article paragraphs are joined into one string and fed to a pretrained transformer with a classification head. The notebook compares several encoders through a Hugging Face `Trainer` wrapper (5 epochs, batch size 8, max length 256, model selected by validation balanced accuracy). A `simpletransformers` version is included as an alternative baseline.

**Results.** Test balanced accuracy on 1,000 examples: `roberta-base` reached 0.695 in the second run, `bert-base-uncased` 0.671, `distilroberta-base` 0.665 and `distilbert-base-uncased` 0.660. Several runs collapsed to predicting a single class (balanced accuracy 0.333), including `roberta-large` and `bert-large-uncased` in both runs and `roberta-base` in the first run. No hyperparameter tuning or seed control was done, so these runs show training instability and should not be read as a model ranking.

**Data.** Not included. Download the Webis Clickbait Challenge 2023 files (`train.jsonl`, `val.jsonl`, `test.jsonl`) from the task page and put them in `data/`. The first cells expect them under a Google Drive folder and use Colab paths, so change the path in the second cell.

**Run.** `pip install -r requirements.txt`, then run the notebook top to bottom (a GPU is recommended). Some cells near the end are scratch experiments.

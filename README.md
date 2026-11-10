# ml-notebooks

Machine learning notebooks from course projects, grouped by topic. Each project folder has its own README covering the problem, the approach, the results (with the caveats) and how to run it. Datasets and model files are not tracked.

## Projects

| Project | Area | Approach | Key result |
|---|---|---|---|
| [Clickbait spoiling](nlp/clickbait-spoiling) | NLP | Two-person course project on SemEval 2023 Task 5, subtask 1: fine-tuned transformer baselines, and a two-stage method (SpanBERT span tagging, then BERT classification) | Best baseline test balanced accuracy 0.695 (`roberta-base`). The span method reached 0.611 on the validation set, below the baseline's 0.709 on the same set. The README covers the failed runs and open questions |
| [mBERT sentiment classification](nlp/sentiment-mbert) | NLP | Fine-tuned `bert-base-multilingual-cased` on 5,000 IMDb reviews, evaluated on 1,000 held-out reviews | Precision 0.826, recall 0.820, F1 0.823 (positive class, one run) |
| [Spine segmentation](computer-vision/spine-segmentation) | Computer vision | Classical image processing (CLAHE, edge maps, brightness profiles) to split a frontal spine X-ray into individual vertebrae | Qualitative: 22 vertebra crops on a sample X-ray, no quantitative metric (group project, 3 members) |
| [Synthetic Thai text data](computer-vision/thai-text-synthetic-data) | Computer vision | Generator for labelled images of Thai words on textured backgrounds, with bounding boxes and augmentation (brightness, contrast, perspective, noise, blur), for a billboard text reading term project | 10,000 images (2,500 base x 4) in the stored run. A model trained on it was very accurate on the generated fonts and much lower on other fonts in the real task (qualitative, the model is not in this repo) |

## Repository layout

```
ml-notebooks/
  nlp/
    clickbait-spoiling/        notebooks, README, requirements.txt
    sentiment-mbert/           notebook, README, requirements.txt
  computer-vision/
    spine-segmentation/        notebook, README, requirements.txt
    thai-text-synthetic-data/  notebook, README, requirements.txt
```

## Running the notebooks

The notebooks were written in Google Colab, so some cells use Colab paths such as `/content/`. To run one locally or in a fresh Colab session, install the project's `requirements.txt`, place the data where the project README says, and edit the path cells near the top. Stored outputs are kept in the notebooks so the results can be read without rerunning anything.

## Related work

My thesis project on recommendation systems is in a separate repository: [Balancing-Novelty-and-Accuracy-in-Recommendation](https://github.com/Riririll/Balancing-Novelty-and-Accuracy-in-Recommendation).

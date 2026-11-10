# ml-notebooks

Machine learning notebooks from course projects, grouped by topic. Each project folder has its own README covering the problem, the approach, the results (with the caveats) and how to run it. Datasets and model files are not tracked.

## Projects

| Project | Area | Approach | Key result |
|---|---|---|---|
| [Clickbait spoiling](nlp/clickbait-spoiling) | NLP | Two-person course project on SemEval 2023 Task 5, subtask 1: fine-tuned transformer baselines, and a two-stage method (SpanBERT span tagging, then BERT classification) | Best baseline test balanced accuracy 0.695 (`roberta-base`). The span method reached 0.611 on the validation set, below the baseline's 0.709 on the same set. The README lists the runs that collapsed to 0.333 |
| [Spine segmentation](computer-vision/spine-segmentation) | Computer vision | Classical image processing (CLAHE, edge maps, brightness profiles) to split a frontal spine X-ray into individual vertebrae | Qualitative: 22 vertebra crops on a sample X-ray, no quantitative metric (group project, 3 members) |

## Repository layout

```
ml-notebooks/
  nlp/
    clickbait-spoiling/        notebook, README, requirements.txt
  computer-vision/
    spine-segmentation/        notebook, README, requirements.txt
```

## Running the notebooks

The notebooks were written in Google Colab, so some cells use Colab paths such as `/content/` and a mounted Google Drive. To run one locally or in a fresh Colab session, install the project's `requirements.txt`, place the data where the project README says, and edit the path cells near the top. Stored outputs are kept in the notebooks so the results can be read without rerunning anything.

## Related work

My thesis project on recommendation systems is in a separate repository: [Balancing-Novelty-and-Accuracy-in-Recommendation](https://github.com/Riririll/Balancing-Novelty-and-Accuracy-in-Recommendation).

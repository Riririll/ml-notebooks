# ml-notebooks

Notebooks from coursework in natural language processing and computer vision. Each folder is one project with its notebook, a README that explains the problem, what was done, the results and the limitations, and a `requirements.txt`. Datasets, generated images and model weights are not included; each README says where the data comes from.

## Projects

### Clickbait spoiler classification (NLP)
[`nlp/clickbait-spoiling`](nlp/clickbait-spoiling) · two-person course project, SemEval 2023 Task 5 subtask 1

A clickbait post promises information that is only in the linked article. The task is to predict what kind of spoiler the post needs: a short phrase, a longer passage, or several separate parts. This folder has two notebooks. The first fine-tunes seven pretrained transformer encoders with the same training code and compares them. The second is a two-stage method: SpanBERT first tags the spoiler tokens in the article, then a BERT classifier predicts the type from the tagged text.

Result: the best baseline (`roberta-base`) reaches 0.695 balanced accuracy on the test set. The two-stage method reaches 0.611 on the validation set, below the baseline's 0.709 on the same set. The README also documents the baseline runs that collapsed to always predicting one class (0.333).

### Sentiment classification with mBERT (NLP)
[`nlp/sentiment-mbert`](nlp/sentiment-mbert)

Fine-tunes `bert-base-multilingual-cased` on 5,000 IMDb movie reviews to classify them as positive or negative, and evaluates it on 1,000 held-out reviews.

Result: precision 0.826, recall 0.820 and F1 0.823 for the positive class, from one run.

### Vertebra segmentation from a spine X-ray (computer vision)
[`computer-vision/spine-segmentation`](computer-vision/spine-segmentation) · group term project, 3 members

Splits a frontal full-spine X-ray into one image per vertebra, labelled from the sacrum to C1, using only classical image processing (gamma correction, CLAHE, edge enhancement, brightness profiles). There is no trained model.

Result: 22 vertebra crops on a sample X-ray. The evaluation is qualitative, with no ground truth and no metric.

### Synthetic Thai text data for billboard text reading (computer vision)
[`computer-vision/thai-text-synthetic-data`](computer-vision/thai-text-synthetic-data) · term project

A generator for labelled training images: Thai words from Wikipedia drawn on many kinds of background surfaces, with bounding boxes, plus augmented copies (brightness, contrast, perspective, noise, blur). The stored run made 10,000 images from 2,500 base images.

Result: a model trained on this data was very accurate on the fonts used for generation and much lower on other fonts in the real task. This is a qualitative summary; the model is not in this repository.

## Repository layout

```
ml-notebooks/
  nlp/
    clickbait-spoiling/        2 notebooks, README, requirements.txt
    sentiment-mbert/           notebook, README, requirements.txt
  computer-vision/
    spine-segmentation/        notebook, README, requirements.txt
    thai-text-synthetic-data/  notebook, README, requirements.txt
```

## Running the notebooks

The notebooks were written in Google Colab, so some cells use Colab paths such as `/content/`. To run one locally or in a fresh Colab session, install the project's `requirements.txt`, place the data where the project README says, and edit the path cells near the top. Stored outputs are kept in the notebooks so the results can be read without rerunning anything.

## Related work

My thesis project on recommendation systems is in a separate repository: [Balancing-Novelty-and-Accuracy-in-Recommendation](https://github.com/Riririll/Balancing-Novelty-and-Accuracy-in-Recommendation).

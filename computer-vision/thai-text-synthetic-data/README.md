# Synthetic Thai text images for billboard text reading

Notebook: [`thai_text_dataset_generator.ipynb`](thai_text_dataset_generator.ipynb)

## Purpose

This was written for a computer vision term project on reading Thai text from billboards. The real task data contained only a few examples, so the notebook generates a labelled training set: Thai words rendered on textured backgrounds, with a bounding box and the text for every image. The fonts were chosen from the test examples provided by the instructor.

The detector and recognizer that were trained on this data are not part of this repository. Only the data generator is.

## What the generator does

1. **Text.** A random Thai word is taken from Thai Wikipedia (`pythainlp/thai-wiki-dataset-v3`, first 5% of the train split, 9,827 articles in the stored run). The article is segmented with PyThaiNLP (`newmm`), and a word is kept only if it has 5 to 20 characters and contains Thai characters only (no digits or symbols).
2. **Rendering.** A random background is resized to 800 x 600 and the word is drawn in the centre in a random font (size 120 to 150, reduced until it fits), in a random bright colour (each RGB channel between 200 and 255).
3. **Annotation.** One JSON file per image with the image size, the text, the language, the font and background names, and a bounding box `[x1, y1, x2, y2]` that includes some padding around the text.
4. **Augmentation.** Each image gets 3 augmented copies. Every copy has a random brightness factor (0.7 to 1.0) and contrast factor (0.7 to 1.3). With probability 0.7 a perspective transform moves each corner of the text box by up to 10% of its width and height, and the bounding box is transformed with it. With probability 0.3 each, Gaussian noise (sigma 15) and a 3x3 or 5x5 Gaussian blur are added.
5. **Split.** The annotation files are shuffled and written to `train.txt`, `val.txt` and `test.txt` lists (70% / 15% / 15%).

The stored run used 2,500 base images, so 2,500 x (1 original + 3 augmented) = 10,000 images with 10,000 annotations, split into 7,000 / 1,500 / 1,500. It used 3 fonts (Prompt, Sarabun, Kanit) and 20 background images. The notebook also contains a multi-line generator (`create_multi_text_image`) and fixed phrase lists (`_load_thai_texts`); neither was used in the stored run (`num_multi_images = 0`).

The last cell of the stored outputs shows 100 random samples with their boxes drawn. The image is stored at reduced resolution.

## Result

On the fonts used for generation, a model trained on this data was very accurate. On other fonts that appear in the real task data, accuracy was much lower. The generated data therefore works for the fonts it covers, and coverage of more fonts is what it lacks. This is a qualitative summary from the project. The notebook does not contain the model or its metrics.

## Data and licensing

- Fonts are downloaded from the `ofl` directory of the `google/fonts` repository, and are not stored here.
- Text comes from the Hugging Face dataset `pythainlp/thai-wiki-dataset-v3`. Check the dataset card for its licence before redistributing generated images.
- The 20 background images are not included, and no generated images are stored in this repository. Put your own images in `backgrounds/`.

## Known issues in the code

- The `perspective`, `noise` and `blur` entries in an augmented image's metadata are drawn again with `random.random()`, so they do not record what was applied to that image. The brightness and contrast entries are the real factors.
- Single-text images store the box as `[x1, y1, x2, y2]`. The multi-line generator stores `[x, y, width, height]`.
- `split_dataset` shuffles annotation files, not base images, so an original image and its augmented copies can end up in different splits.

## Reproduce

```
pip install -r requirements.txt
```

The notebook was run in Google Colab. Run the cells in order. It creates `fonts/`, `backgrounds/` and `thai_text_dataset/` in the working directory, downloads the fonts and the Wikipedia data, and writes images, annotations and the split lists. The last two cells zip the dataset and download it through Colab, so skip them elsewhere.

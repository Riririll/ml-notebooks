# Vertebra segmentation from a frontal spine X-ray

Notebook: [`spine_segmentation.ipynb`](spine_segmentation.ipynb)

Group term project (3 members).

## Overview

Given a frontal full-spine X-ray, split the spine into one image per vertebra and label each crop from the sacrum up to C1. The method uses classical image processing only, with no trained model, so every step can be inspected and explained.

## Pipeline

**Part 1: enhancement.** Gamma correction (gamma 4.5) is applied first, then CLAHE (clip limit 3, 8 by 8 tiles) boosts local contrast. Sobel and Canny edge maps are combined and blended back into the image (weights 0.7 and 0.3). The notebook also renders the original and enhanced image with a JET color map to make bone structure easier to see.

**Part 2: isolating the spine column.** The image is cropped to its central region (width divided by 2.5) and cut into 25 horizontal strips. In each strip the brightest region is kept through a mask, and everything else is set to black. The strips are then placed back on a black background, which leaves only the spine.

**Part 3: splitting into vertebrae.** Columns outside a central band are zeroed and the pixel values of each row are summed. The row profile is smoothed with a moving average (window of height divided by 50), and its local minima are taken as the gaps between vertebrae. Each smoothed minimum is snapped to the lowest raw row sum within height divided by 100 rows. The image is then cropped between consecutive cut points with a small margin, and segments shorter than height divided by 40 are skipped. Cut points are processed from the bottom up, so the crops are labelled `Sacrum, L5 ... L1, T12 ... T1, C7 ... C1`.

## Results

The stored outputs show each stage: the enhancement panels, the isolated spine, the cut points on the row profile, and the individual crops (22 in the stored run). The evaluation is qualitative only.

## Known limitations of the implementation

- Every constant (gamma, strip count, window sizes, thresholds) is a fixed value set in the code. There is no ground truth and no quantitative metric.
- The vertebra labels are assigned by counting cut points from the bottom. A missed or extra cut point shifts every label above it.

## Data

The input X-ray files are not included as separate files. The notebook expects two images (`4.jpeg` and `Spine_Segments.jpeg`) in `/content/`, and the stored cell outputs show the results on them.

## Reproduce

```
pip install -r requirements.txt
```

Place your own frontal spine X-ray at the image paths used in the notebook (or edit them) and run the cells in order. The last cells write the crops to a Google Drive folder, so change that output path as well.

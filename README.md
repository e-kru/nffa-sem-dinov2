# Microscopy Image Classification with DINOv2

This project explores how a pretrained vision model, DINOv2, can be used to classify scanning electron microscopy images from the NFFA-EUROPE dataset. The goal is to assess how well general visual representations transfer to scientific microscopy images.

## Data preparation

We removed metadata strips and excluded images containing measurement annotations, reducing the dataset from 21,169 to 18,286 images. A duplicate audit using SHA-256 revealed identical images across the original training and evaluation subsets. To reduce this source of data leakage, we created two group-based train/validation/test splits:

- **Split A:** keeps detected exact duplicates in the same subset.
- **Split B:** additionally groups visually similar images using perceptual hashes, providing a more conservative evaluation of the baseline.

Entire groups are assigned to one subset so that their images cannot appear in both training and evaluation.

## Initial results

All three baselines use a frozen DINOv2 encoder with a logistic-regression classifier.

| Dataset and split | Evaluation | Accuracy | Macro-F1 |
| --- | --- | --- | --- |
| Original data, before cleaning and duplicate-aware splitting | Validation | 92.76% | 0.8980 |
| Cleaned data, Split A | Test | 92.45% | 0.9087 |
| Cleaned data, Split B | Test | 90.96% | 0.8785 |

The cleaned baselines retain over 90% test accuracy under both split strategies. Since the evaluation sets differ, these results do not isolate the effect of data cleaning.

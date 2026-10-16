# Semantic Segmentation on Pascal VOC 2007: U-Net, SAM 2, and DeepLabV3+

**Course:** SHBT 261 Artificial Intelligence in Medicine (Mini-Project 2)
**Author:** Cindy Sun
**Institution:** Harvard University, Spring 2026
**Instructors:** Mengyu Wang and Tobias Elze

---

## Overview

Semantic segmentation assigns a class label to every pixel in an image. It is a core task in medical imaging, where it supports structure delineation, lesion detection, and surgical planning. This project compares four segmentation approaches on a small subset of Pascal VOC 2007 (209 training images, 213 test images, 21 classes):

| Model | Type | Trainable parameters |
|---|---|---|
| U-Net | Custom CNN trained from scratch | All |
| SAM 2 (zero-shot) | Foundation model with oracle-informed label assignment | None |
| SAM 2 (fine-tuned) | Frozen SAM 2 encoder plus a lightweight segmentation head | 370K (1.1% of total) |
| DeepLabV3+ | ResNet-50 backbone, COCO-pretrained, fine-tuned end to end | All |

A total of **10 model configurations** were trained and evaluated: the four models above plus six ablations (data augmentation, loss function, test-time augmentation, and prompt density). Results are reported with mIoU, mean Dice, pixel accuracy, HD95, and per-class IoU and accuracy, with extra focus on the clinically relevant **person** class.

---

## Key Results

Test set: the 213-image official validation split, held out for testing throughout.

| Model | Pixel Acc | mIoU | mDice | HD95 | Person IoU |
|---|---|---|---|---|---|
| U-Net (from scratch) | 0.7410 | 0.0386 | 0.0467 | 87.13 | 0.0693 |
| SAM 2 zero-shot (oracle labels) | 0.9137 | 0.6651 | 0.7704 | 47.16 | 0.7555 |
| SAM 2 fine-tuned | 0.8135 | 0.1425 | 0.1987 | 100.35 | 0.6699 |
| DeepLabV3+ | 0.7285 | 0.0433 | 0.0542 | 60.50 | 0.1663 |

### Main findings

1. **Background collapse.** U-Net and DeepLabV3+ reach about 74% pixel accuracy but near-zero mIoU. Both predict background for almost every pixel and score IoU = 0 on 19 of 21 classes. Class imbalance, a very small training set (209 images), and unweighted cross-entropy are the likely causes. Pixel accuracy was misleading here, and mIoU exposed the failure.
2. **SAM 2 zero-shot is an upper bound, not a fair comparison.** It assigns each SAM segment the most common ground-truth class inside it, so it uses ground-truth labels at inference time. Its high mIoU shows that SAM 2's masks are spatially accurate, not that it solves the semantic task unaided.
3. **Parameter-efficient adaptation works for some classes.** A frozen SAM 2 encoder with a 370K-parameter head reaches 0.670 IoU on person. Most other classes still score near zero with only 209 training images.
4. **Dice loss helps the most.** Switching from cross-entropy to Dice loss gave the largest single gain across all ablations (SAM 2 fine-tuned mIoU: 0.143 to 0.213).

---

## Ablation Studies

| Model | Ablation | Setting | mIoU | mDice | Pixel Acc |
|---|---|---|---|---|---|
| U-Net | Augmentation | None | 0.0386 | 0.0467 | 0.7410 |
| U-Net | Augmentation | Flip + color jitter | 0.0309 | 0.0424 | 0.4979 |
| U-Net | Loss | Cross-entropy | 0.0386 | 0.0467 | 0.7410 |
| U-Net | Loss | Dice | 0.0455 | 0.0628 | 0.6117 |
| SAM 2 zero-shot | Test-time augmentation | None | 0.6651 | 0.7704 | 0.9137 |
| SAM 2 zero-shot | Test-time augmentation | Horizontal flip | 0.6928 | 0.7908 | 0.9198 |
| SAM 2 zero-shot | Points per side | 16 (default) | 0.6651 | 0.7704 | 0.9137 |
| SAM 2 zero-shot | Points per side | 8 | 0.6124 | 0.7308 | 0.9020 |
| SAM 2 fine-tuned | Augmentation | None | 0.1425 | 0.1987 | 0.8135 |
| SAM 2 fine-tuned | Augmentation | Flip + color jitter | 0.1617 | 0.2234 | 0.8160 |
| SAM 2 fine-tuned | Loss | Cross-entropy | 0.1425 | 0.1987 | 0.8135 |
| SAM 2 fine-tuned | Loss | Dice | 0.2130 | 0.3055 | 0.8057 |

Notes:
- Augmentation hurt U-Net (unstable training on 209 images) but modestly helped SAM 2 fine-tuned, where the frozen encoder keeps training stable.
- Dice loss raised mIoU for both U-Net and SAM 2 fine-tuned, at a small cost in pixel accuracy.
- Reducing SAM 2 points per side from 16 to 8 lowered accuracy but sped up inference (about 3.1 to 8.3 iterations per second).

---

## Dataset

- **Source:** [Pascal VOC 2007 on Kaggle](https://www.kaggle.com/datasets/zaraks/pascal-voc-2007)
- **Classes:** 21 (20 object categories plus background)
- **Split used:** 209 training images and 213 validation images (the validation split serves as the test set)
- **Preprocessing:** images and masks resized to 256 x 256, images normalized with ImageNet statistics. Boundary pixels (label 255) are excluded from all losses and metrics through `ignore_index=255`.

This subset is much smaller than the standard VOC segmentation split (about 1,464 training and 1,449 validation images), which amplifies class imbalance.

---

## Models and Training

| Setting | Value |
|---|---|
| Image size | 256 x 256 |
| Batch size | 4 |
| Epochs | 10 |
| Optimizer | Adam (learning rate 1e-3) |
| Scheduler | StepLR (step size 5, gamma 0.5) |
| Primary loss | Cross-entropy with `ignore_index=255` |
| Hardware | Google Colab, CUDA GPU (PyTorch 2.10.0+cu128) |

- **U-Net:** four encoder DoubleConv blocks (64, 128, 256, 512 channels) with 2 x 2 max pooling, a 1024-channel bottleneck, four transposed-convolution decoder stages with skip connections, and a 1 x 1 convolution producing 21-class logits.
- **SAM 2 zero-shot:** `sam2_hiera_small` with `SAM2AutomaticMaskGenerator` (16 points per side). Each segment receives the majority ground-truth class within its region.
- **SAM 2 fine-tuned:** the Hiera small image encoder (34.3M parameters) stays frozen. A head of two Conv-BN-ReLU layers (down to 64 channels), bilinear upsampling to 256 x 256, and a 1 x 1 output convolution is trained on the encoder output.
- **DeepLabV3+:** torchvision `deeplabv3_resnet50` with COCO-pretrained weights. The classifier and auxiliary classifier are replaced with 1 x 1 convolutions for 21 classes, and the whole network is fine-tuned.

---

## Evaluation Metrics

- **mIoU:** macro-averaged Intersection-over-Union over all 21 classes
- **mDice:** macro-averaged Dice coefficient over all 21 classes
- **Pixel accuracy:** fraction of correctly labeled pixels (boundary pixels excluded)
- **HD95:** 95th-percentile Hausdorff distance, computed with KD-tree nearest-neighbor distances in both directions and combined across images per class
- **Per-class IoU and accuracy:** class-wise breakdown

The notebook also contains `compute_hd95_fast`, a subsampled approximation kept for reference. The reported results use the full `compute_hd95` function.

---

## Repository Structure

```
SHBT261_miniproject2/
├── final_code.ipynb          # full pipeline: data, 4 models, ablations, metrics, plots
├── final_report.pdf          # written report
├── figures/                  # saved plots (see list below)
└── README.md
```

When run, the notebook writes the following to the working folder in Google Drive (`base_path`):

- Model checkpoints (`*.pth`) for each trained model and ablation
- Cached predictions and masks (`*.pt`)
- Metrics (`*_metrics.json`) and a combined `results_summary.csv`
- Comparison plots: `comparison_all_metrics.png`, `comparison_perclass_iou.png`, `comparison_perclass_acc.png`

Reruns load saved checkpoints and predictions when they exist, so training and SAM 2 inference are skipped on later runs.

---

## How to Run

The code was run on **Google Colab** with Google Drive mounted.

1. **Get the data.** Download the dataset from [Kaggle](https://www.kaggle.com/datasets/zaraks/pascal-voc-2007) and place it in a `data` folder containing:
   ```
   data/
   ├── VOCtrainval_06-Nov-2007/
   ├── VOCtest_06-Nov-2007/
   ├── PASCAL_VOC/
   └── pascal-voc-2007.zip
   ```
2. **Open the notebook.** Upload `final_code.ipynb` to Colab and mount Drive (Section 1).
3. **Set your paths.** In Section 2, change `notebook_path` to the location of your own project folder in Drive. The `data` folder must sit inside the folder that contains that notebook.
4. **Install SAM 2.** Section 1 installs it from source:
   ```bash
   pip install git+https://github.com/facebookresearch/sam2.git
   ```
5. **Run all cells in order.** The SAM 2 checkpoint (`sam2_hiera_small.pt`) downloads automatically on first use if it is not already in the project folder.

**Main libraries:** PyTorch, torchvision, SAM 2, NumPy, SciPy, pandas, Matplotlib, Pillow, tqdm.

---

## Limitations

- The dataset is a small subset (209 training and 213 test images), which amplifies class imbalance.
- No class-weighted loss, focal loss, or oversampling was applied to the supervised models.
- The SAM 2 zero-shot result relies on ground-truth labels for class assignment, so it is an upper-bound baseline.
- Training used 10 epochs per run; longer training might improve the fine-tuned models.
- A confusion matrix was not generated.
- Ablations were run for three of the four models (not DeepLabV3+).

---

## Possible Next Steps

- Class-weighted cross-entropy, focal loss, or oversampling of rare classes
- A fair zero-shot setup that assigns classes without ground-truth access
- Longer training and a larger training split
- Applying the same pipeline to a medical segmentation dataset

---

## Acknowledgments

Thanks to the SHBT 261 instructors, Mengyu Wang and Tobias Elze, for the project guidance. This project uses [SAM 2](https://github.com/facebookresearch/sam2) from Meta AI, torchvision's DeepLabV3 implementation, and the Pascal VOC 2007 dataset.

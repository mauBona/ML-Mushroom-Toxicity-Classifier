# Mushroom Toxicity Classifier (Keras + EfficientNetB0)

A **binary image classifier** that predicts whether a mushroom is **toxic (1)** or **edible (0)** from photos, built with **Keras** and **EfficientNetB0** via **transfer learning**.

It includes:

- **Kaggle auto‑download** of a mushroom **images** dataset (MO‑106; ~27k images across 94 species).
- A **species → toxicity** mapping step to create a binary folder‑of‑folders dataset: `edible/` and `toxic/`.
- A `tf.data` pipeline with **`AUTOTUNE`** for performance, plus normalization and light augmentation.
- **EfficientNetB0** feature extraction → **fine‑tuning**, **class weights**, and **threshold tuning** to prioritize safety (recall on toxic).
- (Optional) a tiny **scratch CNN baseline** for comparison.

> **References**
> - Keras EfficientNet fine‑tuning example (official): https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
> - EfficientNetB0 API docs (TensorFlow): https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0
> - Kaggle MO‑106 dataset: https://www.kaggle.com/datasets/iftekhar08/mo-106

---

## Repository structure

```
.
├─ mushrooms_effnet_binary.ipynb    # main notebook (binary toxic vs edible)
├─ README.md                        # this file
└─ data/                            # created at runtime (downloads, processed binary set)
   ├─ mo106_raw/                    # raw images after unzip (by species)
   └─ mushrooms_binary/             # edible/ and toxic/ after mapping
```

---

## Quick start

### 1) Environment

- Python 3.9+
- TensorFlow 2.12+ (GPU-enabled if available), scikit‑learn, Kaggle CLI

Install (example):

```bash
pip install -U tensorflow scikit-learn kaggle
```

> GPU acceleration requires a proper NVIDIA/CUDA/cuDNN setup for TensorFlow on Linux/Windows, or **tensorflow-metal** on Apple silicon (macOS). See official TensorFlow GPU/Metal docs.

### 2) Kaggle credentials

You need a **Kaggle API token** to auto‑download the dataset.

1. Go to Kaggle → **Profile** → **Account** → **Create New API Token** (downloads `kaggle.json`).
2. Place it at `~/.kaggle/kaggle.json` and set permissions:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3) Run the notebook

Open `mushrooms_effnet_binary.ipynb` and run cells in order. The notebook will:

- (Optionally) **download and unzip** the MO‑106 dataset into `data/mo106_raw/`.
- **Map species → toxicity** to create `data/mushrooms_binary/{edible,toxic}`.
- Build the **`tf.data` pipeline** with `AUTOTUNE`.
- Train **EfficientNetB0** in two phases (freeze → fine‑tune) with **class weights**.
- Sweep the **decision threshold** to focus on **toxic recall**.

---

## How it works

### Dataset & mapping

MO‑106 is organized by **species**. For a **binary classifier**, we map species to **edible** or **toxic** and copy images into `edible/` and `toxic/`.

- The notebook includes a **fallback heuristic** (searches “edible”, “toxic”, “poison” tokens in folder names) to build the binary set.
- **Recommended**: If your copy of MO‑106 provides **metadata/CSV** with edibility per species, **replace** the heuristic with that mapping for accuracy and reproducibility.

### Training recipe

- **Phase 1 (feature extraction)**: Freeze the pre‑trained EfficientNetB0 backbone (ImageNet weights), train a new classification head.
- **Phase 2 (fine‑tune top blocks)**: Unfreeze only the last ~30 layers; lower learning rate (e.g., `1e-4`).
- Use **class weights** if the dataset is imbalanced.
- **Threshold tuning**: Sweep decision thresholds and choose one that **maximizes recall on toxic** (to minimize false “edible” on true toxic).

---

## Example commands (optional)

If you prefer to download MO‑106 manually before running the notebook:

```bash
mkdir -p data
kaggle datasets download -d iftekhar08/mo-106 -p data -o
unzip data/*.zip -d data/mo106_raw
```

Then open the notebook and **skip** the download cell.

---

## Results & evaluation

- The notebook prints **validation accuracy** during training.
- The **threshold sweep** prints Precision/Recall/F1 at the selected threshold and a **confusion matrix**.
- For safety‑critical goals, prefer **higher toxic recall** even at the cost of more conservative predictions.

---

## Extending the project

- **Species‑level** classifier: switch to a softmax head with `NUM_CLASSES` and train longer.
- **Deployment**: export to **SavedModel** or **TFLite** (float16/int8) for mobile/edge.
- **Calibration**: add Platt scaling or temperature scaling for better probability estimates.
- **MLOps**: log predictions, collect hard cases, retrain periodically.

---

## License

Add a license (e.g., **MIT**) appropriate for your organization. If you use MO‑106 or other datasets, ensure you comply with their respective **terms of use**.

---

## Safety disclaimer

This repository is a **technical demonstration**. **Never** consume wild mushrooms based solely on AI predictions. Use expert mycological identification and follow local guidance. The authors and contributors assume **no liability** for misuse or decisions made based on this code or models.

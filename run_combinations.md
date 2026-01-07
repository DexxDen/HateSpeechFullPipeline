# Pipeline Run Combinations

This document lists various command combinations for the `kd_prune_quant` pipeline, tailored for different goals.

## 1. Balanced (Recommended Start)
**Goal:** A good trade-off between accuracy, model size, and training speed.

```bash
python main.py \
    --pipeline kd_prune_quant \
    --dataset_path data/HateSpeech.csv \
    --author_name "Researcher" \
    --teacher_path "csebuetnlp/banglabert" \
    --student_path "distilbert-base-multilingual-cased" \
    --teacher_epochs 5 \
    --epochs 10 \
    --fine_tune_epochs 3 \
    --batch 32 \
    --prune_method magnitude \
    --prune_sparsity 0.4 \
    --quant_method fp16
```

### 💡 Why this combination?
*   **Student (`distilbert-base-multilingual-cased`)**: A standard, reliable multilingual model that supports Bengali. It has 6 layers (vs 12 in BERT), offering immediate 2x speedup.
*   **Pruning (`magnitude`, 0.4)**: Magnitude pruning is robust and simple. 40% sparsity is a "safe zone" where accuracy loss is usually negligible for Transformer models.
*   **Quantization (`fp16`)**: FP16 is the native format for modern GPUs (T4, A100). It reduces memory usage by 50% with zero accuracy loss and requires no calibration.

---

## 2. Maximum Compression (GPU)
**Goal:** The smallest possible model size that still runs on GPU.

```bash
python main.py \
    --pipeline kd_prune_quant \
    --dataset_path data/HateSpeech.csv \
    --author_name "Researcher" \
    --teacher_path "csebuetnlp/banglabert" \
    --student_path "distilbert-base-multilingual-cased" \
    --teacher_epochs 5 \
    --epochs 10 \
    --fine_tune_epochs 5 \
    --batch 32 \
    --prune_method wanda \
    --prune_sparsity 0.5 \
    --quant_method int4
```

### 💡 Why this combination?
*   **Pruning (`wanda`, 0.5)**: "Wanda" (Pruning by Weights and activations) is a state-of-the-art method (2023) that considers input activations. It typically outperforms magnitude pruning at higher sparsities (50%+).
*   **Quantization (`int4`)**: 4-bit quantization (via `bitsandbytes`) offers extreme compression (8x smaller than FP32).
*   **Fine-tuning (5 epochs)**: Higher sparsity requires more recovery time, so we increase fine-tuning epochs to 5.

---

## 3. CPU Deployment (Dynamic Quantization)
**Goal:** Deploying the model on a standard server or laptop without a GPU.

```bash
python main.py \
    --pipeline kd_prune_quant \
    --dataset_path data/HateSpeech.csv \
    --author_name "Researcher" \
    --teacher_path "csebuetnlp/banglabert" \
    --student_path "distilbert-base-multilingual-cased" \
    --teacher_epochs 5 \
    --epochs 10 \
    --fine_tune_epochs 3 \
    --batch 32 \
    --prune_method magnitude \
    --prune_sparsity 0.4 \
    --quant_method dynamic
```

### 💡 Why this combination?
*   **Quantization (`dynamic`)**: Dynamic quantization stores weights in INT8 (4x smaller) but quantizes activations dynamically at runtime. This is the **standard recommendation for CPU inference** in PyTorch, as it balances speed and accuracy better than static quantization for Transformers.

---

## 4. Ultra-Compact Bengali Model (SahajBERT)
**Goal:** The absolute most efficient model specifically for the Bengali language.

```bash
python main.py \
    --pipeline kd_prune_quant \
    --dataset_path data/HateSpeech.csv \
    --author_name "Researcher" \
    --teacher_path "csebuetnlp/banglabert" \
    --student_path "neuropark/sahajBERT" \
    --teacher_epochs 5 \
    --epochs 15 \
    --fine_tune_epochs 5 \
    --batch 32 \
    --prune_method magnitude \
    --prune_sparsity 0.4 \
    --quant_method dynamic
```

### 💡 Why this combination?
*   **Student (`neuropark/sahajBERT`)**: This model is pre-trained specifically on Bengali data. It has only 4 layers (vs 6 in DistilBERT) and ~18M parameters (vs ~135M).
*   **Epochs (15)**: Since `sahajBERT` is smaller, it might need more training iterations to fully absorb knowledge from the larger teacher.
*   **Result**: Combined with pruning and quantization, this can produce a model under **5 MB**.

---

## 5. High Accuracy Focus
**Goal:** Maximizing F1 score while still getting some compression benefits.

```bash
python main.py \
    --pipeline kd_prune_quant \
    --dataset_path data/HateSpeech.csv \
    --author_name "Researcher" \
    --teacher_path "csebuetnlp/banglabert" \
    --student_path "distilbert-base-multilingual-cased" \
    --teacher_epochs 10 \
    --epochs 15 \
    --fine_tune_epochs 5 \
    --batch 16 \
    --prune_method magnitude \
    --prune_sparsity 0.3 \
    --quant_method fp16
```

### 💡 Why this combination?
*   **Sparsity (0.3)**: Lower sparsity (30%) is very conservative, ensuring almost zero drop in accuracy.
*   **Teacher Epochs (10)**: A better-trained teacher provides better soft labels for distillation.
*   **Batch Size (16)**: Smaller batch sizes can sometimes help the model generalize better (though training is slower).

---

## 6. Fast Experimentation (Debug)
**Goal:** Verifying the pipeline works without waiting hours.

```bash
python main.py \
    --pipeline kd_prune_quant \
    --dataset_path data/HateSpeech.csv \
    --author_name "Researcher" \
    --teacher_path "csebuetnlp/banglabert" \
    --student_path "distilbert-base-multilingual-cased" \
    --teacher_epochs 1 \
    --epochs 1 \
    --fine_tune_epochs 1 \
    --batch 32 \
    --data_fraction 0.1 \
    --prune_method magnitude \
    --prune_sparsity 0.1 \
    --quant_method fp16
```

### 💡 Why this combination?
*   **Data Fraction (0.1)**: Uses only 10% of the dataset.
*   **Epochs (1)**: Runs only 1 epoch per stage.
*   **Use Case**: Perfect for checking if your code crashes or if the pipeline flows correctly before launching a full overnight run.

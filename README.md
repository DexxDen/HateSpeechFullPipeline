# Binary Hate Speech Model Compression Framework

A modular framework for compressing NLP models using Knowledge Distillation (KD), Pruning, and Quantization, specifically adapted for **Binary Hate Speech Detection**.

## 🚀 Overview

This framework allows you to systematically compress large Transformer models (like BERT) into smaller, faster versions while maintaining high performance on binary classification tasks.

### Key Features
- **Knowledge Distillation**: Transfer knowledge from a large Teacher (e.g., BERT-base) to a smaller Student (e.g., DistilBERT or a custom small BERT).
- **Pruning**: Remove redundant weights using Magnitude, Wanda, or Gradual pruning.
- **Quantization**: Reduce precision to FP16 or INT8/INT4 for significant speedup and size reduction.
- **Binary Optimized**: Uses Binary Cross-Entropy (BCE) loss and binary-specific metrics (F1 Binary, ROC-AUC).
- **Kaggle Ready**: Includes templates and configurations optimized for Kaggle environments.

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

*Note: For INT4 quantization, ensure `bitsandbytes` is installed.*

---

## 📊 Dataset Structure

The framework expects a CSV file with at least two columns:
1. **Text Column**: Named `comment`, `text`, or `content`.
2. **Label Column**: Named `HateSpeech` (0 for Non-Hate, 1 for Hate).

**Example (`data/HateSpeech.csv`):**
```csv
comment,HateSpeech
"This is a nice comment",0
"I hate this",1
```

---

## 🧪 Running Experiments

The framework uses the `--pipeline` flag to control which compression stages are executed. Below are the commands for every possible variant.

### 1. Single-Stage Pipelines
| Variant | Command |
| :--- | :--- |
| **Baseline** | `python main.py --pipeline baseline --dataset_path data/HateSpeech.csv --author_name "test"` |
| **KD Only** | `python main.py --pipeline kd_only --dataset_path data/HateSpeech.csv --author_name "test"` |
| **Pruning Only** | `python main.py --pipeline prune_only --dataset_path data/HateSpeech.csv --author_name "test"` |
| **Quant Only** | `python main.py --pipeline quant_only --dataset_path data/HateSpeech.csv --author_name "test"` |

### 2. Multi-Stage Pipelines
| Variant | Command |
| :--- | :--- |
| **KD + Pruning** | `python main.py --pipeline kd_prune --dataset_path data/HateSpeech.csv --author_name "test"` |
| **KD + Quant** | `python main.py --pipeline kd_quant --dataset_path data/HateSpeech.csv --author_name "test"` |
| **Prune + Quant** | `python main.py --pipeline prune_quant --dataset_path data/HateSpeech.csv --author_name "test"` |
| **Full (KD+P+Q)** | `python main.py --pipeline kd_prune_quant --dataset_path data/HateSpeech.csv --author_name "test"` |

### 3. Comparison & Ablation
| Variant | Command |
| :--- | :--- |
| **Ablation Study** | `python main.py --run_ablation --dataset_path data/HateSpeech.csv --author_name "test"` |

---

## 💡 Pro-Tips for Experiments

- **Skip Teacher Training**: If you already have a fine-tuned teacher, use `--teacher_checkpoint "your-username/your-model"` to save time.
- **Adjust Sparsity**: Use `--prune_sparsity 0.7` to target 70% sparsity during pruning.
- **INT4 Quantization**: Use `--quant_method int4` for maximum compression (requires `bitsandbytes`).
- **Quick Test**: Add `--teacher_epochs 1 --epochs 1 --fine_tune_epochs 1` to quickly verify the pipeline logic.

---

## ⚙️ Key Configuration Options

| Category | Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| **General** | `--teacher_path` | `bert-base-uncased` | Base model for teacher. |
| | `--student_path` | `distilbert-base-uncased` | Base model for student. |
| **KD** | `--kd_alpha` | `0.7` | Weight for soft labels (0 to 1). |
| | `--kd_temperature`| `4.0` | Softness of teacher predictions. |
| **Pruning** | `--prune_method` | `magnitude` | `magnitude`, `wanda`, `gradual`. |
| | `--prune_sparsity`| `0.5` | Target sparsity (e.g., 0.5 = 50%). |
| **Quant** | `--quant_method` | `dynamic` | `dynamic`, `static`, `fp16`, `int4`. |

---

## 📈 Output & Results

All results are saved in the `./compressed_models` directory:
- `results_final.csv`: Comprehensive metrics for all stages.
- `ablation_summary.csv`: Comparison table for ablation studies.
- `model_hf/`: Final compressed model in HuggingFace-compatible format.
- `plots/`: Visualizations of performance vs. size.

---

## 📖 Usage Example (Full Pipeline)

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

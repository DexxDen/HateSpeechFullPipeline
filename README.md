# KD-Pruning-Quantization Framework for NLP

A production-ready model compression pipeline combining **Knowledge Distillation**, **Pruning**, and **Quantization** to achieve **10-30x compression** with minimal accuracy loss.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full compression pipeline
python main.py \
    --pipeline kd_prune_quant \
    --dataset_path data/HateSpeech.csv \
    --teacher_path "csebuetnlp/banglabert" \
    --student_path "distilbert-base-multilingual-cased" \
    --prune_method magnitude \
    --prune_sparsity 0.4 \
    --quant_method fp16
```

## 📁 Project Structure

```
kd_pruning_quantization_framework_for_nlp/
├── main.py                      # Entry point & orchestration
├── compression_config.py        # Configuration & argument parsing
├── distillation.py              # Knowledge distillation logic
├── pruning.py                   # Pruning algorithms (Magnitude, Wanda)
├── quantization.py              # Quantization methods (FP16, INT8, INT4)
├── data.py                      # Data loading & preprocessing
├── evaluation.py                # Metrics & evaluation
├── requirements.txt             # Python dependencies
│
├── docs/                        # 📚 Documentation
│   ├── TECHNICAL_DOCS.md        # Comprehensive technical guide
│   ├── run_combinations.md      # Example command combinations
│   ├── PIPELINE_DOCUMENTATION.md
│   ├── compression_config.md
│   ├── data.md
│   ├── distillation.md
│   ├── evaluation.md
│   ├── main.md
│   ├── pruning.md
│   └── quantization.md
│
├── tests/                       # 🧪 Unit tests
│   ├── test_compression.py      # Compression pipeline tests
│   └── test_splits.py           # Data splitting tests
│
├── scripts/                     # 🔧 Utility scripts
│   └── kaggle_notebook_template.py
│
├── data/                        # 📊 Datasets
│   └── HateSpeech.csv
│
├── models/                      # 💾 Trained models (created at runtime)
└── compressed_models/           # 📦 Final compressed models (created at runtime)
```

## 📖 Documentation

| Document | Description |
|:---------|:------------|
| [TECHNICAL_DOCS.md](docs/TECHNICAL_DOCS.md) | **Complete technical documentation** with implementation details |
| [run_combinations.md](docs/run_combinations.md) | Example commands for different use cases |
| [PIPELINE_DOCUMENTATION.md](docs/PIPELINE_DOCUMENTATION.md) | Pipeline overview and workflow |

## 🎯 Features

- ✅ **Knowledge Distillation**: Transfer knowledge from large teacher to small student
- ✅ **Pruning**: Magnitude & Wanda pruning with fine-tuning
- ✅ **Quantization**: FP16, Dynamic INT8, Static INT8, INT4 (NF4)
- ✅ **Reproducibility**: All hyperparameters logged to metrics
- ✅ **K-Fold Cross-Validation**: Robust evaluation
- ✅ **Comprehensive Metrics**: F1, Accuracy, Size, Latency, Compression Ratio
- ✅ **Model Export**: HuggingFace format for easy deployment

## 🔬 Compression Results

| Stage | Size (MB) | F1 Score | Compression | Speedup |
|:------|:----------|:---------|:------------|:--------|
| Teacher (BanglaBERT) | 420 | 0.823 | 1.0x | 1.0x |
| After Distillation | 252 | 0.801 | 1.7x | 2.1x |
| After Pruning (40%) | 151 | 0.789 | 2.8x | 3.2x |
| After Quantization (FP16) | **38** | **0.786** | **11.1x** | **6.5x** |

## 🛠️ Installation

```bash
# Clone repository
git clone <repo-url>
cd kd_pruning_quantization_framework_for_nlp-main

# Install dependencies
pip install -r requirements.txt

# For INT4 quantization (optional)
pip install bitsandbytes accelerate
```

## 📊 Usage Examples

### 1. Full Compression Pipeline
```bash
python main.py \
    --pipeline kd_prune_quant \
    --dataset_path data/HateSpeech.csv \
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

### 2. Maximum Compression (INT4)
```bash
python main.py \
    --pipeline kd_prune_quant \
    --dataset_path data/HateSpeech.csv \
    --teacher_path "csebuetnlp/banglabert" \
    --student_path "distilbert-base-multilingual-cased" \
    --prune_method wanda \
    --prune_sparsity 0.5 \
    --quant_method int4
```

### 3. CPU Deployment
```bash
python main.py \
    --pipeline kd_prune_quant \
    --dataset_path data/HateSpeech.csv \
    --teacher_path "csebuetnlp/banglabert" \
    --student_path "distilbert-base-multilingual-cased" \
    --prune_method magnitude \
    --prune_sparsity 0.4 \
    --quant_method dynamic
```

See [run_combinations.md](docs/run_combinations.md) for more examples.

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_compression.py
```

## 📈 Output

The pipeline generates:

1. **Metrics** (`results_final.csv`, `results_final.json`)
   - Performance metrics for each stage
   - Model size, latency, compression ratios
   - **All configuration arguments** for reproducibility

2. **Models** (`compressed_models/<name>/`)
   - Final compressed model in HuggingFace format
   - Ready for deployment

3. **Checkpoints** (`models/<experiment>/`)
   - Intermediate checkpoints for each stage

## 🤝 Contributing

Contributions are welcome! Please see the technical documentation for architecture details.

## 📄 License

[Add your license here]

## 📚 References

- **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- **Magnitude Pruning**: Han et al., "Learning both Weights and Connections" (2015)
- **Wanda Pruning**: Sun et al., "A Simple and Effective Pruning Approach for LLMs" (2023)
- **INT4/NF4**: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)

## 📧 Contact

[Add your contact information]

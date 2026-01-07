# Documentation Index

This directory contains all project documentation.

## 📚 Main Documentation

### [TECHNICAL_DOCS.md](TECHNICAL_DOCS.md)
**Comprehensive technical documentation** covering:
- System architecture and pipeline flow
- Detailed implementation of each phase (Teacher, KD, Pruning, Quantization)
- Mathematical foundations
- Code architecture and module descriptions
- Configuration system
- Performance optimization techniques
- Troubleshooting guide

**Audience**: Developers, researchers, contributors

---

### [run_combinations.md](run_combinations.md)
**Example command combinations** for different scenarios:
- Balanced (recommended start)
- Maximum compression (GPU)
- CPU deployment
- Ultra-compact Bengali model
- High accuracy focus
- Fast experimentation

**Audience**: Users, practitioners

---

### [PIPELINE_DOCUMENTATION.md](PIPELINE_DOCUMENTATION.md)
Pipeline overview and workflow documentation.

---

## 📖 Module Documentation

Detailed documentation for each Python module:

- [compression_config.md](compression_config.md) - Configuration and argument parsing
- [data.md](data.md) - Data loading and preprocessing
- [distillation.md](distillation.md) - Knowledge distillation implementation
- [evaluation.md](evaluation.md) - Metrics and evaluation
- [main.md](main.md) - Main pipeline orchestration
- [pruning.md](pruning.md) - Pruning algorithms
- [quantization.md](quantization.md) - Quantization methods

---

## 🚀 Quick Navigation

| I want to... | Read this |
|:-------------|:----------|
| Understand the entire system | [TECHNICAL_DOCS.md](TECHNICAL_DOCS.md) |
| Run the pipeline with different settings | [run_combinations.md](run_combinations.md) |
| Understand a specific module | Module-specific `.md` files |
| Get started quickly | [../README.md](../README.md) |

---

## 📝 Documentation Standards

All documentation follows these principles:
- **Clarity**: Clear, concise explanations
- **Completeness**: Comprehensive coverage of features
- **Examples**: Code snippets and usage examples
- **Structure**: Logical organization with table of contents
- **Maintenance**: Keep in sync with code changes

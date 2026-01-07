# Scripts

This directory contains utility scripts and templates.

## 📜 Available Scripts

### [kaggle_notebook_template.py](kaggle_notebook_template.py)
Template for running the compression pipeline on Kaggle.

**Features**:
- Pre-configured for Kaggle environment
- GPU optimization settings
- Dataset mounting and path configuration
- Example pipeline execution

**Usage**:
1. Copy this file to a new Kaggle notebook
2. Modify dataset paths and model names
3. Run the notebook

---

## 🔧 Adding New Scripts

When adding utility scripts:
1. Add a descriptive name (e.g., `export_to_onnx.py`)
2. Include a docstring explaining purpose and usage
3. Update this README with script description
4. Make scripts executable if needed: `chmod +x script.py`

---

## 📝 Script Guidelines

- **Modularity**: Scripts should be standalone and reusable
- **Documentation**: Include clear usage instructions
- **Error Handling**: Handle common errors gracefully
- **Logging**: Use print statements or logging for progress updates

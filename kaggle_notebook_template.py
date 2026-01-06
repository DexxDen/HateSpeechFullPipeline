"""
================================================================================
KAGGLE NOTEBOOK TEMPLATE FOR COMPRESSION EXPERIMENTS
================================================================================

Copy this entire file to a Kaggle notebook and run cell by cell.

ESTIMATED TIME: ~5-6 hours for full ablation study
KAGGLE LIMIT: 12 hours GPU → You have plenty of time!

PREREQUISITES:
1. Upload your fine-tuned teacher model to HuggingFace (optional, speeds up baseline)
2. Upload your dataset to Kaggle (e.g., as a private dataset)
"""

# =============================================================================
# CELL 1: SETUP AND INSTALL DEPENDENCIES
# =============================================================================

# Install required packages
!pip install transformers datasets scikit-learn pandas tqdm --quiet
!pip install iterative-stratification --quiet  # For stratification
!pip install bitsandbytes --quiet  # For INT4 quantization
!pip install accelerate --quiet # For efficient training

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("✅ Dependencies installed!")

# =============================================================================
# CELL 2: CREATE COMPRESSION FRAMEWORK FILES
# =============================================================================

# Create directory structure
!mkdir -p /kaggle/working/compression_framework
!mkdir -p /kaggle/working/cache
!mkdir -p /kaggle/working/results

# Option 1: Upload files manually (RECOMMENDED)
# Upload the following files to /kaggle/working/compression_framework/:
# - main.py
# - data.py
# - evaluation.py
# - distillation.py
# - pruning.py
# - quantization.py
# - compression_config.py

# Option 2: Clone from GitHub (if you pushed your code)
# !git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git /kaggle/working/compression_framework

print("✅ Framework directory ready. Please ensure files are uploaded!")

# =============================================================================
# CELL 3: VERIFY GPU
# =============================================================================

import torch

if torch.cuda.is_available():
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ No GPU available! Training will be slow.")

# =============================================================================
# CELL 4: CONFIGURATION
# =============================================================================

# ==================== MODIFY THESE ====================
DATASET_PATH = "/kaggle/input/YOUR-DATASET/data.csv"  # Change this!
TEACHER_CHECKPOINT = "csebuetnlp/banglabert"  # Or your fine-tuned model path
AUTHOR_NAME = "Your Name"  # Change this!
# ======================================================

# Verify paths
import os
if not os.path.exists(DATASET_PATH):
    print(f"❌ Dataset not found: {DATASET_PATH}")
    print("   Please update DATASET_PATH to your dataset location")
else:
    print(f"✅ Dataset found: {DATASET_PATH}")

# =============================================================================
# CELL 5: RUN SINGLE EXPERIMENT (KD ONLY - QUICK TEST)
# =============================================================================

# Quick test to make sure everything works
!cd /kaggle/working/compression_framework && python main.py \
    --dataset_path {DATASET_PATH} \
    --author_name "{AUTHOR_NAME}" \
    --pipeline kd_only \
    --teacher_path {TEACHER_CHECKPOINT} \
    --epochs 1 \
    --teacher_epochs 1 \
    --data_fraction 0.05 \
    --output_dir /kaggle/working/results/test_run \
    --cache_dir /kaggle/working/cache

print("\n✅ Test run complete! Check results in /kaggle/working/results/test_run/")

# =============================================================================
# CELL 6: RUN FULL ABLATION STUDY
# =============================================================================

# This runs all pipeline configurations (baseline, kd_only, prune_only, etc.)
# Estimated time: ~5-6 hours

!cd /kaggle/working/compression_framework && python main.py \
    --dataset_path {DATASET_PATH} \
    --author_name "{AUTHOR_NAME}" \
    --run_ablation \
    --teacher_path {TEACHER_CHECKPOINT} \
    --output_dir /kaggle/working/results/ablation \
    --cache_dir /kaggle/working/cache

# =============================================================================
# CELL 7: ANALYZE RESULTS
# =============================================================================

import pandas as pd

# Load ablation results
results_path = "/kaggle/working/results/ablation/ablation_summary.csv"
if os.path.exists(results_path):
    df = pd.read_csv(results_path)
    print("\n📊 ABLATION STUDY RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    
    # Find best configuration
    # Note: 'F1 Macro' might be a string with 4 decimals, convert to float for sorting
    df['F1_Float'] = pd.to_numeric(df['F1 Macro'], errors='coerce')
    best_idx = df['F1_Float'].idxmax()
    best = df.loc[best_idx]
    print(f"\n🏆 Best Configuration: {best['Pipeline']}")
    print(f"   F1 Macro: {best['F1 Macro']}")
    print(f"   Compression: {best['Compression']}")
else:
    print("❌ Results file not found. Did the ablation study complete?")

# =============================================================================
# CELL 8: VISUALIZE RESULTS
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

if os.path.exists(results_path):
    df = pd.read_csv(results_path)
    df['F1_Float'] = pd.to_numeric(df['F1 Macro'], errors='coerce')
    df['Compression_Float'] = df['Compression'].str.replace('×', '').astype(float)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: F1 vs Compression
    ax1 = axes[0]
    ax1.scatter(df['Compression_Float'], df['F1_Float'], s=100, c='blue', alpha=0.7)
    for i, row in df.iterrows():
        ax1.annotate(row['Pipeline'], (row['Compression_Float'], row['F1_Float']),
                    fontsize=8, ha='center', va='bottom')
    ax1.set_xlabel('Compression Ratio (×)')
    ax1.set_ylabel('F1 Macro')
    ax1.set_title('Accuracy vs Compression Trade-off')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bar chart of F1 scores
    ax2 = axes[1]
    colors = ['green' if x == df['F1_Float'].max() else 'steelblue' 
              for x in df['F1_Float']]
    bars = ax2.barh(df['Pipeline'], df['F1_Float'], color=colors)
    ax2.set_xlabel('F1 Macro')
    ax2.set_title('F1 Scores by Configuration')
    ax2.set_xlim(0.0, 1.0)
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/results/ablation_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Plot saved to /kaggle/working/results/ablation_plot.png")

# =============================================================================
# CELL 9: RUN ADDITIONAL EXPERIMENTS (OPTIONAL)
# =============================================================================

# KD Method Comparison
for kd_method in ['logit', 'hidden', 'attention', 'multi_level']:
    print(f"\n{'='*60}")
    print(f"Running KD Method: {kd_method}")
    print('='*60)
    
    !cd /kaggle/working/compression_framework && python main.py \
        --dataset_path {DATASET_PATH} \
        --author_name "{AUTHOR_NAME}" \
        --pipeline kd_only \
        --teacher_path {TEACHER_CHECKPOINT} \
        --kd_method {kd_method} \
        --output_dir /kaggle/working/results/kd_{kd_method} \
        --cache_dir /kaggle/working/cache

# =============================================================================
# CELL 10: EXPORT RESULTS
# =============================================================================

# Zip all results for download
!cd /kaggle/working && zip -r results_all.zip results/

print("\n✅ All results zipped!")
print("   Download: /kaggle/working/results_all.zip")

# List output files
print("\n📁 Output Files:")
!find /kaggle/working/results -name "*.csv" -o -name "*.json" | head -20

# =============================================================================
# CELL 11: GENERATE PAPER TABLES (LATEX)
# =============================================================================

if os.path.exists(results_path):
    df = pd.read_csv(results_path)
    
    # Generate LaTeX table
    latex_table = df.to_latex(index=False, float_format="%.4f")
    
    print("\n📝 LATEX TABLE FOR PAPER:")
    print("="*60)
    print(latex_table)
    
    # Save to file
    with open('/kaggle/working/results/table_for_paper.tex', 'w') as f:
        f.write(latex_table)
    print("\n✅ LaTeX table saved to /kaggle/working/results/table_for_paper.tex")

print("\n" + "="*60)
print("🎉 ALL EXPERIMENTS COMPLETE!")
print("="*60)

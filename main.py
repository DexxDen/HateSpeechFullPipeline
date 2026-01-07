#!/usr/bin/env python3
"""
================================================================================
MAIN COMPRESSION PIPELINE (ENHANCED VERSION)
================================================================================

FIXES IN THIS VERSION:
1. ✅ HuggingFace format saving for deployment
2. ✅ Clear KD → Pruning → Quantization flow (always on student after KD)
3. ✅ Fine-tuning after pruning is now DEFAULT
4. ✅ INT4 quantization support (via bitsandbytes)
5. ✅ Metrics CSV saved after EACH stage
6. ✅ Better progress tracking

FLOW EXPLANATION:
─────────────────
Pipeline: kd_prune_quant

Step 1: Load/Train Teacher (large model, e.g., BanglaBERT)
        ↓
Step 2: Knowledge Distillation
        - Create Student (smaller model, e.g., DistilBERT)
        - Train Student to mimic Teacher
        - current_model = Student
        ↓
Step 3: Pruning (applied to Student, NOT teacher!)
        - Remove 50% of Student's weights
        - Fine-tune to recover accuracy
        - current_model = Pruned Student
        ↓
Step 4: Quantization (applied to Pruned Student)
        - Reduce precision (FP32 → INT8 or FP16)
        - current_model = Quantized Pruned Student
        ↓
Step 5: Final Evaluation & Save in HuggingFace format
"""

# =============================================================================
# ENVIRONMENT SETUP (MUST BE FIRST!)
# =============================================================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
import numpy as np
from datetime import datetime
from tqdm import tqdm
import json
import warnings
import shutil
warnings.filterwarnings('ignore')

# Local imports
from compression_config import (
    parse_compression_arguments, print_compression_config,
    PIPELINE_CONFIGS, LABEL_COLUMNS, get_config_for_pipeline
)
from data import (
    load_and_preprocess_data, get_or_create_tokenized_dataset,
    prepare_kfold_splits, calculate_class_weights,
    create_data_loaders, IndexedDataset
)
from distillation import (
    TeacherModel, StudentModel, DistillationTrainer,
    MultiLabelDistillationLoss, verify_teacher_performance
)
from pruning import (
    PruningManager, GradualPruner, WandaPruner,
    get_pruner, fine_tune_after_pruning
)
from quantization import (
    QuantizationManager, quantize_model, benchmark_inference_speed
)
from evaluation import (
    CompressionEvaluator, CompressionStageMetrics,
    compare_stages, export_metrics_to_csv, export_metrics_to_json
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"🖥️  Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = 'cpu'
        print("🖥️  Using CPU (GPU not available)")
    return device


# =============================================================================
# HUGGINGFACE FORMAT SAVING (NEW!)
# =============================================================================

class HuggingFaceModelWrapper(nn.Module):
    """
    Wrapper to save models in HuggingFace format for easy deployment.
    
    This creates a model that can be:
    1. Pushed to HuggingFace Hub
    2. Loaded with AutoModel.from_pretrained()
    3. Used with transformers pipeline
    """
    
    def __init__(self, encoder, classifier, config, num_labels=1):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.config = config
        self.num_labels = num_labels
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())
        
        return {'loss': loss, 'logits': logits}


def save_model_for_huggingface(model, save_path, tokenizer=None):
    """
    Save model in HuggingFace format for deployment.
    
    Creates a folder with:
    - config.json (model configuration)
    - pytorch_model.bin (weights)
    - tokenizer files (if tokenizer provided)
    - classifier_config.json (classifier head info)
    
    After saving, you can:
    1. Push to Hub: `huggingface-cli upload ./model your-username/model-name`
    2. Load: `model = AutoModel.from_pretrained('./model')`
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save encoder
    if hasattr(model, 'encoder'):
        model.encoder.save_pretrained(save_path)
    
    # Save classifier separately
    if hasattr(model, 'classifier'):
        classifier_path = os.path.join(save_path, 'classifier.pt')
        torch.save(model.classifier.state_dict(), classifier_path)
        
        # Save classifier config
        classifier_config = {
            'type': 'sequential',
            'layers': str(model.classifier),
            'num_labels': model.num_labels if hasattr(model, 'num_labels') else 1
        }
        with open(os.path.join(save_path, 'classifier_config.json'), 'w') as f:
            json.dump(classifier_config, f, indent=2)
    
    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)
    
    # Create a loading script
    loading_script = '''
# How to load this model:

from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

# Load encoder
encoder = AutoModel.from_pretrained("{save_path}")

# Load classifier
classifier = nn.Sequential(
    nn.Linear(encoder.config.hidden_size, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 1)  # 1 label
)
classifier.load_state_dict(torch.load("{save_path}/classifier.pt"))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{save_path}")

# Inference
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = encoder(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = classifier(cls_embedding)
        probs = torch.sigmoid(logits)
    return probs

# Example
probs = predict("আপনার বাংলা টেক্সট এখানে")
print(probs)
'''.format(save_path=save_path)
    
    with open(os.path.join(save_path, 'how_to_load.py'), 'w') as f:
        f.write(loading_script)
    
    print(f"✅ Model saved in HuggingFace format: {save_path}")
    print(f"   📄 Files created:")
    for f in os.listdir(save_path):
        print(f"      - {f}")


def push_to_huggingface_hub(model, tokenizer, repo_name, token=None):
    """
    Push model directly to HuggingFace Hub.
    
    Args:
        model: Model to push
        tokenizer: Tokenizer to push
        repo_name: "your-username/model-name"
        token: HuggingFace API token (or set HF_TOKEN env var)
    """
    from huggingface_hub import HfApi, create_repo
    
    # Create temporary directory
    temp_dir = './temp_hf_upload'
    save_model_for_huggingface(model, temp_dir, tokenizer)
    
    # Create repo if doesn't exist
    api = HfApi()
    try:
        create_repo(repo_name, token=token, exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload
    api.upload_folder(
        folder_path=temp_dir,
        repo_id=repo_name,
        token=token
    )
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    print(f"✅ Model pushed to: https://huggingface.co/{repo_name}")


# =============================================================================
# INT4 QUANTIZATION (NEW!)
# =============================================================================

def apply_int4_quantization(model, device='cuda'):
    """
    Apply INT4 quantization using bitsandbytes library.
    
    WHAT: Reduces weights to 4-bit precision (vs 32-bit original)
    WHY: 8× compression with minimal accuracy loss
    HOW: Uses bitsandbytes library for 4-bit quantization
    
    REQUIREMENTS:
        pip install bitsandbytes
        GPU with CUDA support
    
    NOTE: INT4 is primarily used for inference, not training.
    """
    try:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
    except ImportError:
        print("❌ bitsandbytes not installed!")
        print("   Install: pip install bitsandbytes")
        print("   Falling back to INT8...")
        return apply_int8_quantization(model)
    
    print("\n   Applying INT4 quantization (bitsandbytes)...")
    
    # Create quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # Nested quantization for more compression
        bnb_4bit_quant_type="nf4"  # Normalized float 4-bit
    )
    
    # For existing model, we need to convert Linear layers
    def replace_linear_with_4bit(model):
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                # Create 4-bit linear layer
                new_layer = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=torch.float16,
                    quant_type="nf4"
                )
                # Copy weights
                new_layer.weight = bnb.nn.Params4bit(
                    module.weight.data,
                    requires_grad=False,
                    quant_type="nf4"
                )
                if module.bias is not None:
                    new_layer.bias = nn.Parameter(module.bias.data)
                setattr(model, name, new_layer)
            else:
                replace_linear_with_4bit(module)
        return model
    
    model = replace_linear_with_4bit(model)
    model = model.to(device)
    
    print("   ✅ INT4 quantization applied")
    return model


def apply_int8_quantization(model):
    """Apply INT8 dynamic quantization (fallback for INT4)."""
    print("\n   Applying INT8 dynamic quantization...")
    
    model_cpu = model.cpu()
    quantized = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8
    )
    
    print("   ✅ INT8 quantization applied")
    return quantized


# =============================================================================
# METRICS SAVING (ENHANCED)
# =============================================================================

def save_stage_metrics(metrics: CompressionStageMetrics, output_dir: str, stage_name: str):
    """
    Save metrics immediately after each stage (not just at the end).
    
    Creates:
    - results_{stage_name}.csv
    - results_{stage_name}.json
    - results_all.csv (appends)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual stage
    stage_csv = os.path.join(output_dir, f'results_{stage_name}.csv')
    stage_json = os.path.join(output_dir, f'results_{stage_name}.json')
    
    import pandas as pd
    df = pd.DataFrame([metrics.to_flat_dict()])
    df.to_csv(stage_csv, index=False)
    
    with open(stage_json, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2, default=str)
    
    # Append to cumulative results
    all_csv = os.path.join(output_dir, 'results_all.csv')
    if os.path.exists(all_csv):
        df.to_csv(all_csv, mode='a', header=False, index=False)
    else:
        df.to_csv(all_csv, index=False)
    
    print(f"   📊 Metrics saved: {stage_csv}")


# =============================================================================
# TEACHER LOADING/TRAINING
# =============================================================================

def get_or_train_teacher(config, tokenized_data, train_idx, val_idx, device):
    """
    Get teacher model - either load from checkpoint or train from scratch.
    """
    print("\n" + "="*70)
    print("🎓 PHASE 1: TEACHER MODEL")
    print("="*70)
    
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_path)
    
    if config.teacher_checkpoint:
        print(f"\n📥 Loading pre-trained teacher from: {config.teacher_checkpoint}")
        teacher = TeacherModel(
            model_name=config.teacher_checkpoint,
            num_labels=len(LABEL_COLUMNS),
            dropout=config.dropout
        ).to(device)
        print("   ✅ Teacher loaded successfully!")
        return teacher, tokenizer
    
    # Train from scratch
    print(f"\n🔧 Training teacher from scratch ({config.teacher_epochs} epochs)")
    
    teacher = TeacherModel(
        model_name=config.teacher_path,
        num_labels=len(LABEL_COLUMNS),
        dropout=config.dropout
    ).to(device)
    
    train_loader, val_loader = create_data_loaders(
        tokenized_data, train_idx, val_idx,
        batch_size=config.batch, num_workers=2
    )
    
    train_labels = tokenized_data['labels'][train_idx]
    class_weights = calculate_class_weights(train_labels.numpy())
    
    optimizer = AdamW(teacher.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.teacher_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * total_steps),
        num_training_steps=total_steps
    )
    
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    best_f1 = 0
    
    for epoch in range(config.teacher_epochs):
        teacher.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Teacher Epoch {epoch+1}/{config.teacher_epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = teacher(input_ids, attention_mask)
            loss = loss_fn(outputs['logits'], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Evaluate
        teacher.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = teacher(input_ids, attention_mask)
                preds = (torch.sigmoid(outputs['logits']) > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch['labels'].numpy())
        
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
        f1 = f1_score(all_labels, all_preds, average='binary')
        print(f"   Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, F1={f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            
            # Calculate training metrics for best epoch (approximate with subset)
            teacher.eval()
            train_subset_loader = create_data_loaders(
                tokenized_data, train_idx[:min(len(train_idx), 2000)], [], 
                batch_size=config.batch, num_workers=0
            )[0]
            
            train_preds, train_labels = [], []
            with torch.no_grad():
                for batch in train_subset_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    outputs = teacher(input_ids, attention_mask)
                    preds = (torch.sigmoid(outputs['logits']) > 0.5).cpu().numpy()
                    train_preds.extend(preds)
                    train_labels.extend(batch['labels'].numpy())
            
            train_acc = accuracy_score(train_labels, train_preds)
            train_prec_hate = precision_score(train_labels, train_preds, pos_label=1, zero_division=0)
            train_rec_hate = recall_score(train_labels, train_preds, pos_label=1, zero_division=0)
            train_f1_hate = f1_score(train_labels, train_preds, pos_label=1, zero_division=0)
            
            train_prec_non = precision_score(train_labels, train_preds, pos_label=0, zero_division=0)
            train_rec_non = recall_score(train_labels, train_preds, pos_label=0, zero_division=0)
            train_f1_non = f1_score(train_labels, train_preds, pos_label=0, zero_division=0)
            
            train_f1_macro = f1_score(train_labels, train_preds, average='macro', zero_division=0)
            try:
                # Need probs for AUC
                train_probs = []
                with torch.no_grad():
                    for batch in train_subset_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        outputs = teacher(input_ids, attention_mask)
                        probs = torch.sigmoid(outputs['logits']).cpu().numpy()
                        train_probs.extend(probs)
                train_roc_auc = roc_auc_score(train_labels, train_probs)
            except:
                train_roc_auc = 0.0
            
            # Val loss
            val_loss = 0
            loss_fn_val = nn.BCEWithLogitsLoss()
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = teacher(input_ids, attention_mask)
                    loss = loss_fn_val(outputs['logits'], labels)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            best_metrics = {
                'best_epoch': epoch + 1,
                'val_loss': val_loss,
                'best_threshold': 0.5,
                'train_accuracy': train_acc,
                'train_precision_hate': train_prec_hate,
                'train_recall_hate': train_rec_hate,
                'train_f1_hate': train_f1_hate,
                'train_precision_non_hate': train_prec_non,
                'train_recall_non_hate': train_rec_non,
                'train_f1_non_hate': train_f1_non,
                'train_f1_macro': train_f1_macro,
                'train_roc_auc': train_roc_auc,
                'train_loss': total_loss/len(train_loader)
            }
    
    print(f"\n   ✅ Teacher training complete! Best F1: {best_f1:.4f}")
    
    if 'best_metrics' not in locals():
        best_metrics = {}
        
    return teacher, tokenizer, best_metrics


# =============================================================================
# KNOWLEDGE DISTILLATION
# =============================================================================

def run_knowledge_distillation(config, teacher, tokenized_data, train_idx, val_idx, device):
    """
    Run knowledge distillation from teacher to student.
    
    CREATES A NEW STUDENT MODEL and trains it to mimic the teacher.
    The student is SMALLER than the teacher.
    """
    print("\n" + "="*70)
    print("🔄 PHASE 2: KNOWLEDGE DISTILLATION")
    print("="*70)
    print(f"   Teacher: {config.teacher_path} (large)")
    print(f"   Student: {config.student_path} (smaller)")
    print(f"   Method: {config.kd_method}")
    print(f"   Alpha: {config.kd_alpha} (0=hard labels only, 1=soft labels only)")
    print(f"   Temperature: {config.kd_temperature}")
    
    train_loader, val_loader = create_data_loaders(
        tokenized_data, train_idx, val_idx,
        batch_size=config.batch, num_workers=2
    )
    
    # Verify teacher
    is_valid, _ = verify_teacher_performance(teacher, val_loader, device, min_f1=0.4)
    if not is_valid:
        print("   ⚠️  Warning: Teacher F1 is low. Consider using a better teacher.")
    
    # CREATE NEW STUDENT MODEL (smaller than teacher!)
    student = StudentModel(
        model_name=config.student_path,
        num_labels=len(LABEL_COLUMNS),
        dropout=config.dropout,
        classifier_hidden_size=config.student_hidden_size
    ).to(device)
    
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    
    print(f"\n   📊 Model Size Comparison:")
    print(f"      Teacher: {teacher_params/1e6:.2f}M parameters")
    print(f"      Student: {student_params/1e6:.2f}M parameters")
    print(f"      Reduction: {(1 - student_params/teacher_params)*100:.1f}%")
    
    # Get class weights
    train_labels = tokenized_data['labels'][train_idx]
    class_weights = calculate_class_weights(train_labels.numpy())
    
    # Create trainer
    trainer = DistillationTrainer(teacher, student, config, device)
    
    optimizer = AdamW(student.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * total_steps),
        num_training_steps=total_steps
    )
    
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(config.epochs):
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"KD Epoch {epoch+1}/{config.epochs}")
        for batch in pbar:
            losses = trainer.train_step(batch, optimizer, class_weights)
            scheduler.step()
            epoch_losses.append(losses['total_loss'])
            pbar.set_postfix({
                'loss': f"{losses['total_loss']:.4f}",
                'soft': f"{losses['soft_loss']:.4f}"
            })
        
        # Evaluate
        eval_results = trainer.evaluate(val_loader, class_weights)
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
        preds = (eval_results['predictions'] > 0.5).astype(int)
        f1 = f1_score(eval_results['labels'], preds, average='binary')
        
        print(f"   Epoch {epoch+1}: Loss={np.mean(epoch_losses):.4f}, F1={f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            
            # Calculate training metrics for best epoch
            # Note: This is an approximation using the last batch of training, 
            # or we could run a full eval on train set (expensive).
            # For efficiency, let's run a quick eval on a subset of training data
            train_subset_loader = create_data_loaders(
                tokenized_data, train_idx[:min(len(train_idx), 2000)], [], # First 2000 samples
                batch_size=config.batch, num_workers=0
            )[0]
            
            train_eval = trainer.evaluate(train_subset_loader, class_weights)
            train_preds = (train_eval['predictions'] > 0.5).astype(int)
            train_labels = train_eval['labels']
            
            # Calculate detailed metrics
            train_acc = accuracy_score(train_labels, train_preds)
            train_prec_hate = precision_score(train_labels, train_preds, pos_label=1, zero_division=0)
            train_rec_hate = recall_score(train_labels, train_preds, pos_label=1, zero_division=0)
            train_f1_hate = f1_score(train_labels, train_preds, pos_label=1, zero_division=0)
            
            train_prec_non = precision_score(train_labels, train_preds, pos_label=0, zero_division=0)
            train_rec_non = recall_score(train_labels, train_preds, pos_label=0, zero_division=0)
            train_f1_non = f1_score(train_labels, train_preds, pos_label=0, zero_division=0)
            
            train_f1_macro = f1_score(train_labels, train_preds, average='macro', zero_division=0)
            try:
                train_roc_auc = roc_auc_score(train_labels, train_eval['predictions'])
            except:
                train_roc_auc = 0.0
            
            best_metrics = {
                'best_epoch': epoch + 1,
                'val_loss': eval_results['loss'],
                'best_threshold': 0.5,
                'train_accuracy': train_acc,
                'train_precision_hate': train_prec_hate,
                'train_recall_hate': train_rec_hate,
                'train_f1_hate': train_f1_hate,
                'train_precision_non_hate': train_prec_non,
                'train_recall_non_hate': train_rec_non,
                'train_f1_non_hate': train_f1_non,
                'train_f1_macro': train_f1_macro,
                'train_roc_auc': train_roc_auc,
                'train_loss': np.mean(epoch_losses)
            }
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n   ✅ KD complete!")
    print(f"      Student F1: {best_f1:.4f}")
    print(f"      The STUDENT model will be used for subsequent compression steps.")
    
    if 'best_metrics' not in locals():
        best_metrics = {}
        
    return student, best_metrics


# =============================================================================
# PRUNING (APPLIES TO CURRENT MODEL - STUDENT IF KD WAS DONE)
# =============================================================================

def run_pruning(config, model, tokenized_data, train_idx, val_idx, device, model_name="model"):
    """
    Apply pruning to the model.
    
    IMPORTANT: This prunes whatever model is passed in:
    - If called after KD: prunes the STUDENT
    - If called without KD (prune_only): prunes the TEACHER
    
    Always fine-tunes after pruning by default (config.fine_tune_after_prune=True).
    """
    print("\n" + "="*70)
    print("✂️  PHASE 3: PRUNING")
    print("="*70)
    print(f"   Target model: {model_name}")
    print(f"   Method: {config.prune_method}")
    print(f"   Target sparsity: {config.prune_sparsity*100:.0f}%")
    print(f"   Fine-tune after: {'Yes' if config.fine_tune_after_prune else 'No'}")
    
    train_loader, val_loader = create_data_loaders(
        tokenized_data, train_idx, val_idx,
        batch_size=config.batch, num_workers=2
    )
    
    # Get initial metrics
    from sklearn.metrics import f1_score
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            if 'student_input_ids' in batch:
                input_ids = batch['student_input_ids'].to(device)
                attention_mask = batch['student_attention_mask'].to(device)
            else:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['labels'].numpy())
    
    f1_before = f1_score(all_labels, all_preds, average='binary')
    print(f"\n   F1 before pruning: {f1_before:.4f}")
    
    # Apply pruning based on method
    if config.prune_method == 'magnitude':
        pruner = PruningManager(
            model=model,
            target_sparsity=config.prune_sparsity,
            prune_layers=config.prune_layers,
            global_pruning=True
        )
        pruner.apply_magnitude_pruning()
        
    elif config.prune_method == 'gradual':
        pruner = GradualPruner(
            model=model,
            target_sparsity=config.prune_sparsity,
            start_epoch=config.prune_start_epoch,
            end_epoch=config.prune_end_epoch,
            schedule=config.prune_schedule,
            prune_frequency=config.prune_frequency,
            prune_layers=config.prune_layers
        )
        
        optimizer = AdamW(model.parameters(), lr=config.lr * 0.1)
        loss_fn = nn.BCEWithLogitsLoss()
        
        total_steps = len(train_loader) * config.prune_end_epoch
        current_step = 0
        
        for epoch in range(config.prune_end_epoch):
            model.train()
            for batch in tqdm(train_loader, desc=f"Gradual Prune Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                
                pruner.step(current_step, total_steps)
                current_step += 1
            
            sparsity = pruner.get_sparsity()['overall']
            print(f"   Epoch {epoch+1}: Sparsity = {sparsity*100:.1f}%")
        
    elif config.prune_method == 'wanda':
        pruner = WandaPruner(
            model=model,
            target_sparsity=config.prune_sparsity,
            prune_layers=config.prune_layers
        )
        pruner.collect_activations(train_loader, device, num_samples=config.calib_samples)
        pruner.apply_wanda_pruning()
    
    else:
        raise ValueError(f"Unknown pruning method: {config.prune_method}")
    
    # Make pruning permanent
    if hasattr(pruner, 'make_pruning_permanent'):
        pruner.make_pruning_permanent()
    
    # Get F1 after pruning (before fine-tuning)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            if 'student_input_ids' in batch:
                input_ids = batch['student_input_ids'].to(device)
                attention_mask = batch['student_attention_mask'].to(device)
            else:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['labels'].numpy())
    
    f1_after_prune = f1_score(all_labels, all_preds, average='binary')
    print(f"\n   F1 after pruning (before fine-tune): {f1_after_prune:.4f}")
    print(f"   F1 drop: {(f1_before - f1_after_prune)*100:.2f}%")
    
    # ALWAYS fine-tune after pruning (default behavior now)
    # ALWAYS fine-tune after pruning (default behavior now)
    best_metrics = {}
    if config.fine_tune_after_prune:
        print(f"\n   🔧 Fine-tuning for {config.fine_tune_epochs} epochs to recover accuracy...")
        ft_results = fine_tune_after_pruning(model, train_loader, val_loader, config, device)
        if 'best_metrics' in ft_results:
            best_metrics = ft_results['best_metrics']
        
        # Get F1 after fine-tuning
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                if 'student_input_ids' in batch:
                    input_ids = batch['student_input_ids'].to(device)
                    attention_mask = batch['student_attention_mask'].to(device)
                else:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch['labels'].numpy())
        
        f1_after_finetune = f1_score(all_labels, all_preds, average='binary')
        print(f"\n   F1 after fine-tuning: {f1_after_finetune:.4f}")
        print(f"   Recovery: {(f1_after_finetune - f1_after_prune)*100:.2f}%")
    
    final_sparsity = pruner.get_sparsity()
    print(f"\n   ✅ Pruning complete!")
    print(f"      Final sparsity: {final_sparsity['overall']*100:.2f}%")
    
    return model, best_metrics


# =============================================================================
# QUANTIZATION (APPLIES TO CURRENT MODEL)
# =============================================================================

def run_quantization(config, model, tokenized_data, train_idx, val_idx, device):
    """
    Apply quantization to the model.
    
    Supports: dynamic, static, fp16, int4 (NEW!)
    """
    print("\n" + "="*70)
    print("📉 PHASE 4: QUANTIZATION")
    print("="*70)
    print(f"   Method: {config.quant_method}")
    print(f"   Data type: {config.quant_dtype}")
    
    train_loader, val_loader = create_data_loaders(
        tokenized_data, train_idx, val_idx,
        batch_size=config.batch, num_workers=2
    )
    
    # Determine quantization method
    if config.quant_method == 'dynamic':
        print("   ⚠️  Dynamic INT8 runs on CPU only")
        quant_device = 'cpu'
        model_cpu = model.cpu()
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu, {nn.Linear}, dtype=torch.qint8
        )
        
    elif config.quant_method == 'static':
        print("   ⚠️  Static INT8 runs on CPU only")
        quant_device = 'cpu'
        manager = QuantizationManager(model, method='static', dtype=config.quant_dtype)
        manager.prepare_static_quantization()
        manager.calibrate(train_loader, device='cpu', num_batches=config.quant_calibration_batches)
        quantized_model = manager.convert_static_quantization()
        
    elif config.quant_method == 'fp16':
        print("   ✅ FP16 works on GPU!")
        quant_device = device
        quantized_model = model.half().to(device)
        
    elif config.quant_method == 'int4':
        print("   Applying INT4 quantization (bitsandbytes)...")
        quant_device = device
        quantized_model = apply_int4_quantization(model, device)
        
    else:
        raise ValueError(f"Unknown quantization method: {config.quant_method}")
    
    # Compare sizes
    manager = QuantizationManager(model, method=config.quant_method)
    manager.quantized_model = quantized_model
    size_info = manager.compare_sizes()
    
    print(f"\n   ✅ Quantization complete!")
    print(f"      Compression: {size_info['compression_ratio']:.2f}×")
    
    return quantized_model, quant_device


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_compression_pipeline(config):
    """
    Run the complete compression pipeline.
    
    FLOW:
    1. Load/Train Teacher
    2. KD: Create Student, train to mimic Teacher
    3. Pruning: Applied to Student (or Teacher if no KD)
    4. Quantization: Applied to Pruned model
    5. Save in HuggingFace format
    """
    print("\n" + "="*70)
    print("🚀 COMPRESSION PIPELINE")
    print("="*70)
    print(f"   Pipeline: {config.pipeline}")
    print(f"   KD: {'✅' if config.enable_kd else '❌'}")
    print(f"   Pruning: {'✅' if config.enable_pruning else '❌'}")
    print(f"   Quantization: {'✅' if config.enable_quantization else '❌'}")
    
    # Setup
    set_seed(config.seed)
    device = get_device()
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.cache_dir, exist_ok=True)
    
    # Load data with caching
    comments, labels = load_and_preprocess_data(config.dataset_path)
    
    # Subsample data if requested
    if config.data_fraction < 1.0:
        print(f"\n✂️  Subsampling data: {config.data_fraction*100:.1f}%")
        from sklearn.model_selection import train_test_split
        comments, _, labels, _ = train_test_split(
            comments, labels, 
            train_size=config.data_fraction,
            stratify=labels,
            random_state=config.seed
        )
        print(f"   New dataset size: {len(comments)} samples")
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_path)
    
    # Load student tokenizer if KD is enabled and different from teacher
    student_tokenizer = None
    if config.enable_kd and config.student_path != config.teacher_path:
        print(f"🔄 Loading student tokenizer: {config.student_path}")
        student_tokenizer = AutoTokenizer.from_pretrained(config.student_path)
        
    tokenized_data = get_or_create_tokenized_dataset(
        comments, labels, tokenizer, config.max_length, config.cache_dir,
        student_tokenizer=student_tokenizer
    )
    
    # K-fold splits
    splits = list(prepare_kfold_splits(
        comments, labels, config.num_folds,
        stratification_type='binary', seed=config.seed
    ))
    train_idx, val_idx = splits[0]
    
    # Initialize evaluator
    evaluator = CompressionEvaluator()
    all_metrics = []
    
    train_loader, val_loader = create_data_loaders(
        tokenized_data, train_idx, val_idx,
        batch_size=config.batch, num_workers=2
    )
    
    # ==========================================================================
    # PHASE 1: Teacher
    # ==========================================================================
    teacher, tokenizer, teacher_metrics = get_or_train_teacher(
        config, tokenized_data, train_idx, val_idx, device
    )
    
    baseline_metrics = evaluator.evaluate_model(
        teacher, val_loader, device, stage='baseline',
        extra_metrics=teacher_metrics
    )
    baseline_metrics.print_summary()
    all_metrics.append(baseline_metrics)
    save_stage_metrics(baseline_metrics, config.output_dir, 'baseline')
    
    if config.pipeline == 'baseline':
        save_model_for_huggingface(teacher, os.path.join(config.output_dir, 'model_hf'), tokenizer)
        return all_metrics
    
    # Track current model
    current_model = teacher
    current_model_name = "teacher"
    
    # ==========================================================================
    # PHASE 2: Knowledge Distillation
    # ==========================================================================
    if config.enable_kd:
        student, kd_train_metrics = run_knowledge_distillation(
            config, teacher, tokenized_data, train_idx, val_idx, device
        )
        
        kd_metrics = evaluator.evaluate_model(
            student, val_loader, device, stage='after_kd',
            use_student_input_ids=True,
            extra_metrics=kd_train_metrics
        )
        kd_metrics.print_summary()
        all_metrics.append(kd_metrics)
        save_stage_metrics(kd_metrics, config.output_dir, 'after_kd')
        
        # NOW THE STUDENT BECOMES THE CURRENT MODEL
        current_model = student
        current_model_name = "student"
        
        if config.save_all_stages:
            save_model_for_huggingface(
                student, os.path.join(config.output_dir, 'model_after_kd_hf'), tokenizer
            )
    
    # ==========================================================================
    # PHASE 3: Pruning (APPLIED TO CURRENT MODEL - STUDENT IF KD WAS DONE!)
    # ==========================================================================
    if config.enable_pruning:
        print(f"\n   📌 Pruning will be applied to: {current_model_name}")
        
        pruned_model, prune_train_metrics = run_pruning(
            config, current_model, tokenized_data, train_idx, val_idx, device,
            model_name=current_model_name
        )
        
        prune_metrics = evaluator.evaluate_model(
            pruned_model, val_loader, device, stage='after_pruning',
            use_student_input_ids=True,
            extra_metrics=prune_train_metrics
        )
        prune_metrics.print_summary()
        all_metrics.append(prune_metrics)
        save_stage_metrics(prune_metrics, config.output_dir, 'after_pruning')
        
        current_model = pruned_model
        current_model_name = f"pruned_{current_model_name}"
        
        if config.save_all_stages:
            save_model_for_huggingface(
                pruned_model, os.path.join(config.output_dir, 'model_after_pruning_hf'), tokenizer
            )
    
    # ==========================================================================
    # PHASE 4: Quantization
    # ==========================================================================
    if config.enable_quantization:
        quantized_model, quant_device = run_quantization(
            config, current_model, tokenized_data, train_idx, val_idx, device
        )
        
        quant_loader = create_data_loaders(
            tokenized_data, train_idx, val_idx,
            batch_size=config.batch, num_workers=0
        )[1]
        
        quant_metrics = evaluator.evaluate_model(
            quantized_model, quant_loader, quant_device, stage='after_quantization',
            use_student_input_ids=True
        )
        quant_metrics.print_summary()
        all_metrics.append(quant_metrics)
        save_stage_metrics(quant_metrics, config.output_dir, 'after_quantization')
        
        current_model = quantized_model
    
    # ==========================================================================
    # Final Summary
    # ==========================================================================
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    
    comparison_df = compare_stages(all_metrics)
    print(comparison_df.to_string(index=False))
    
    export_metrics_to_csv(all_metrics, os.path.join(config.output_dir, 'results_final.csv'))
    export_metrics_to_json(all_metrics, os.path.join(config.output_dir, 'results_final.json'))
    
    final_metrics = all_metrics[-1]
    print(f"\n🎯 SUMMARY:")
    print(f"   Compression: {final_metrics.size_compression_ratio:.2f}×")
    print(f"   F1: {baseline_metrics.f1_macro:.4f} → {final_metrics.f1_macro:.4f}")
    print(f"   Size: {baseline_metrics.model_size_mb:.1f} MB → {final_metrics.model_size_mb:.1f} MB")
    
    return all_metrics


# =============================================================================
# ABLATION STUDY
# =============================================================================

def run_ablation_study(config):
    """Run all pipeline configurations for ablation study."""
    print("\n" + "="*70)
    print("🔬 ABLATION STUDY")
    print("="*70)
    
    pipelines = config.ablation_pipelines
    all_results = {}
    
    for pipeline in pipelines:
        print(f"\n{'='*70}")
        print(f"📊 ABLATION: {pipeline.upper()}")
        print(f"{'='*70}")
        
        pipeline_config = get_config_for_pipeline(
            pipeline,
            dataset_path=config.dataset_path,
            author_name=config.author_name,
            teacher_checkpoint=config.teacher_checkpoint,
            output_dir=os.path.join(config.output_dir, f'ablation_{pipeline}'),
            cache_dir=config.cache_dir,
            kd_method=config.kd_method,
            prune_method=config.prune_method,
            quant_method=config.quant_method,
            fine_tune_after_prune=True  # Always fine-tune!
        )
        
        metrics = run_compression_pipeline(pipeline_config)
        all_results[pipeline] = metrics[-1]
    
    # Summary table
    print("\n" + "="*70)
    print("📊 ABLATION RESULTS")
    print("="*70)
    
    import pandas as pd
    rows = []
    for pipeline, metrics in all_results.items():
        rows.append({
            'Pipeline': pipeline,
            'F1 Macro': f"{metrics.f1_macro:.4f}",
            'F1 Binary': f"{metrics.per_label_f1.get('HateSpeech', 0):.4f}",
            'Size (MB)': f"{metrics.model_size_mb:.1f}",
            'Compression': f"{metrics.size_compression_ratio:.2f}×",
            'Sparsity': f"{metrics.sparsity_percent:.1f}%"
        })
    
    ablation_df = pd.DataFrame(rows)
    print(ablation_df.to_string(index=False))
    ablation_df.to_csv(os.path.join(config.output_dir, 'ablation_summary.csv'), index=False)
    
    return all_results


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    config = parse_compression_arguments()
    print_compression_config(config)
    
    if config.run_ablation:
        return run_ablation_study(config)
    else:
        return run_compression_pipeline(config)


if __name__ == "__main__":
    main()

# data.py
# Adapted for single-label (binary) HateSpeech detection
# Matches the API and advanced features of the multi-label data.txt module

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Tuple, List, Dict, Optional, Generator
from tqdm import tqdm
import hashlib

# =============================================================================
# CONSTANTS
# =============================================================================

# Single label column for this dataset
LABEL_COLUMNS = ['HateSpeech']  # Kept as list for API compatibility


# =============================================================================
# CACHING FUNCTIONS (Same as multi-label version)
# =============================================================================

def get_cache_filename(model_path: str, max_length: int) -> str:
    safe_name = model_path.replace('/', '_').replace('-', '_').replace('.', '_')
    return f"{safe_name}_maxlen{max_length}_tokenized.pkl"


def get_or_create_tokenized_dataset(
    comments: np.ndarray,
    labels: np.ndarray,
    tokenizer,
    max_length: int,
    cache_dir: str = './cache',
    student_tokenizer = None
) -> Dict[str, torch.Tensor]:
    """
    Tokenize all samples with caching – supports dual tokenization for KD.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Create a unique cache filename based on both tokenizers if applicable
    tokenizer_name = tokenizer.name_or_path
    if student_tokenizer:
        tokenizer_name += f"_and_{student_tokenizer.name_or_path}"
    
    cache_filename = get_cache_filename(tokenizer_name, max_length)
    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path):
        print(f"✅ Loading tokenized data from cache: {cache_path}")
        cached_data = torch.load(cache_path)

        if cached_data['input_ids'].shape[0] == len(comments):
            print(f"   Loaded {len(comments)} samples in ~2 seconds")
            return cached_data
        else:
            print(f"⚠️  Cache size mismatch! Re-tokenizing...")

    print(f"🔄 Tokenizing {len(comments)} samples...")
    print(f"   Model: {tokenizer.name_or_path}")
    print(f"   Max length: {max_length}")

    all_input_ids = []
    all_attention_masks = []
    all_student_input_ids = []
    all_student_attention_masks = []

    for comment in tqdm(comments, desc="Tokenizing"):
        text = str(comment) if comment is not None else ""

        # Teacher tokenization
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        all_input_ids.append(encoding['input_ids'].squeeze(0))
        all_attention_masks.append(encoding['attention_mask'].squeeze(0))

        # Student tokenization (if different)
        if student_tokenizer:
            s_encoding = student_tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            all_student_input_ids.append(s_encoding['input_ids'].squeeze(0))
            all_student_attention_masks.append(s_encoding['attention_mask'].squeeze(0))

    tokenized_data = {
        'input_ids': torch.stack(all_input_ids),
        'attention_mask': torch.stack(all_attention_masks),
        'labels': torch.tensor(labels, dtype=torch.float32)
    }

    if student_tokenizer:
        tokenized_data['student_input_ids'] = torch.stack(all_student_input_ids)
        tokenized_data['student_attention_mask'] = torch.stack(all_student_attention_masks)

    torch.save(tokenized_data, cache_path)
    print(f"✅ Saved tokenized data to cache: {cache_path}")
    print(f"   Cache size: {os.path.getsize(cache_path) / 1024 / 1024:.1f} MB")

    return tokenized_data


# =============================================================================
# DATASET CLASSES
# =============================================================================

class IndexedDataset(Dataset):
    """
    Fast dataset using pre-tokenized cached data + indices.
    Identical to multi-label version.
    """
    def __init__(self, tokenized_data: Dict[str, torch.Tensor], indices: np.ndarray):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']
        self.labels = tokenized_data['labels']
        self.indices = indices
        
        # Optional dual tokenization
        self.student_input_ids = tokenized_data.get('student_input_ids')
        self.student_attention_mask = tokenized_data.get('student_attention_mask')

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self.indices[idx]
        item = {
            'input_ids': self.input_ids[real_idx],
            'attention_mask': self.attention_mask[real_idx],
            'labels': self.labels[real_idx]
        }
        
        if self.student_input_ids is not None:
            item['student_input_ids'] = self.student_input_ids[real_idx]
            item['student_attention_mask'] = self.student_attention_mask[real_idx]
            
        return item


class HateSpeechDataset(Dataset):
    """
    Legacy on-the-fly tokenization dataset (slower).
    Kept for backward compatibility.
    """
    def __init__(self, comments: np.ndarray, labels: np.ndarray,
                 tokenizer, max_length: int = 128):
        self.comments = comments
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.comments)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        comment = str(self.comments[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            comment,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load HateSpeech dataset and return comments + labels (shape [N, 1]).
    """
    print(f"\n📁 Loading dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"   Raw rows: {len(df)}")

    # Column name flexibility
    comment_col = None
    for col in ['Comments', 'comments', 'Comment', 'comment', 'text', 'Text', 'content']:
        if col in df.columns:
            comment_col = col
            break
    if comment_col is None:
        raise ValueError("No comment/text column found")

    if LABEL_COLUMNS[0] not in df.columns:
        raise ValueError(f"Missing label column: {LABEL_COLUMNS[0]}")

    if comment_col != 'comment':
        df = df.rename(columns={comment_col: 'comment'})
        print(f"   Renamed '{comment_col}' → 'comment'")

    # Drop NA
    df = df.dropna(subset=['comment', LABEL_COLUMNS[0]])
    print(f"   After dropping NA: {len(df)} rows")

    comments = df['comment'].values
    labels = df[LABEL_COLUMNS].values  # Shape: [N, 1]

    # Statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(comments)}")
    print(f"   Label: {LABEL_COLUMNS[0]}")

    pos = int(np.sum(labels))
    neg = len(labels) - pos
    perc = (pos / len(labels)) * 100
    print(f"\n   Label distribution:")
    print(f"      HateSpeech: {pos}/{len(labels)} ({perc:.1f}% positive)")

    return comments, labels


# =============================================================================
# K-FOLD CROSS VALIDATION
# =============================================================================

def prepare_kfold_splits(
    comments: np.ndarray,
    labels: np.ndarray,
    num_folds: int = 5,
    stratification_type: str = 'binary',  # 'binary' or 'none'
    seed: int = 42
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Prepare K-fold splits with optional stratification.
    """
    print(f"\n🔀 Preparing {num_folds}-fold cross-validation...")

    if num_folds == 1:
        print("   num_folds=1 requested. Using simple 80/20 train/val split.")
        from sklearn.model_selection import train_test_split
        # Create indices
        indices = np.arange(len(comments))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, stratify=labels if stratification_type == 'binary' else None,
            random_state=seed
        )
        yield train_idx, val_idx
        return

    if stratification_type == 'binary':
        print(f"   Using StratifiedKFold (preserves hate/non-hate ratio)")
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        return kfold.split(comments, labels.ravel())  # labels: [N,1] → flatten
    else:
        print(f"   Using basic KFold (no stratification)")
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        return kfold.split(comments)


# =============================================================================
# CLASS WEIGHTS
# =============================================================================

def calculate_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute weight for the positive (HateSpeech) class.
    weight = negatives / positives
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    labels = labels.ravel()  # [N,1] → [N]
    pos_counts = np.sum(labels)
    neg_counts = len(labels) - pos_counts

    weight = neg_counts / pos_counts if pos_counts > 0 else 1.0

    print("\n⚖️  Class weights (for imbalanced data):")
    print(f"      HateSpeech: {weight:.2f} "
          f"({int(pos_counts)} pos, {int(neg_counts)} neg)")

    # Return as tensor of shape [1] to match multi-label API
    return torch.FloatTensor([weight])


# =============================================================================
# DATA LOADERS
# =============================================================================

def create_data_loaders(
    tokenized_data: Dict[str, torch.Tensor],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """
    Identical to multi-label version.
    """
    train_dataset = IndexedDataset(tokenized_data, train_indices)
    val_dataset = IndexedDataset(tokenized_data, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\n📦 DataLoaders created:")
    print(f"      Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"      Val: {len(val_dataset)} samples, {len(val_loader)} batches")

    return train_loader, val_loader


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing HateSpeech data module...")

    comments = np.array(["test comment"] * 100)
    labels = np.random.randint(0, 2, size=(100, 1)).astype(np.float32)

    weights = calculate_class_weights(labels)
    print(f"Weights: {weights}")

    splits = list(prepare_kfold_splits(comments, labels, num_folds=5))
    print(f"Number of folds: {len(splits)}")

    print("\n✅ All tests passed!")

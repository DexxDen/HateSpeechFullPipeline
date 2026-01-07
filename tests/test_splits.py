
import numpy as np
from data import prepare_kfold_splits

def test_splits():
    print("Testing prepare_kfold_splits...")
    comments = np.array(["test"] * 100)
    labels = np.random.randint(0, 2, size=(100, 1))
    
    # Test binary stratification
    splits = list(prepare_kfold_splits(comments, labels, num_folds=5, stratification_type='binary'))
    print(f"Binary splits count: {len(splits)}")
    assert len(splits) == 5, f"Expected 5 splits, got {len(splits)}"
    
    # Test no stratification
    splits_simple = list(prepare_kfold_splits(comments, labels, num_folds=5, stratification_type='none'))
    print(f"Simple splits count: {len(splits_simple)}")
    assert len(splits_simple) == 5, f"Expected 5 splits, got {len(splits_simple)}"
    
    print("✅ Verification passed!")

if __name__ == "__main__":
    test_splits()

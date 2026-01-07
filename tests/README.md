# Tests

This directory contains all unit tests for the compression pipeline.

## 🧪 Test Files

### [test_compression.py](test_compression.py)
Comprehensive tests for the compression pipeline:
- **Sparsity Preservation**: Verifies pruning maintains target sparsity after fine-tuning
- **Metrics Configuration**: Checks that config arguments are logged to metrics
- **INT4 Quantization**: Tests INT4 quantization flow (mocked)
- **Static Quantization**: Tests static INT8 quantization
- **FP16 Quantization**: Tests FP16 half-precision conversion

### [test_splits.py](test_splits.py)
Tests for K-fold data splitting functionality.

---

## 🚀 Running Tests

### Run All Tests
```bash
# From project root
python -m pytest tests/

# Or run directly
python tests/test_compression.py
```

### Run Specific Test
```bash
python tests/test_compression.py TestCompression.test_sparsity_preservation
```

### Run with Verbose Output
```bash
python -m pytest tests/ -v
```

---

## ✅ Test Coverage

Current test coverage:
- ✅ Pruning sparsity preservation
- ✅ Metric logging with config arguments
- ✅ Quantization methods (FP16, INT8, INT4)
- ✅ Data splitting

---

## 🔧 Adding New Tests

To add a new test:

1. Create a test method in `test_compression.py`:
```python
def test_my_feature(self):
    # Arrange
    model = MockModel()
    
    # Act
    result = my_function(model)
    
    # Assert
    self.assertEqual(result, expected_value)
```

2. Run the test to verify:
```bash
python tests/test_compression.py
```

---

## 📝 Test Requirements

Tests use:
- `unittest` - Python's built-in testing framework
- `unittest.mock` - For mocking dependencies
- `torch` - For model operations

No additional dependencies required beyond `requirements.txt`.

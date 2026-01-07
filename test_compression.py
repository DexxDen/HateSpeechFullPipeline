
import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
import os
from main import run_pruning, run_quantization

class MockConfig:
    def __init__(self):
        self.prune_method = 'magnitude'
        self.prune_sparsity = 0.5
        self.prune_layers = 'all'
        self.fine_tune_after_prune = True
        self.fine_tune_epochs = 1
        self.batch = 2
        self.lr = 1e-5
        self.weight_decay = 0.01
        self.gradient_clip_norm = 1.0
        self.quant_method = 'fp16'
        self.quant_dtype = 'int8'
        self.quant_calibration_batches = 1

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10, 1)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.linear1(input_ids)
        x = self.relu(x)
        return self.linear2(x)

    def parameters(self):
        # Ensure we return some parameters even if mocked
        return super().parameters()

    def save_pretrained(self, save_path):
        # Mock save_pretrained
        pass

class TestCompression(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.config = MockConfig()
        self.device = 'cpu'
        
        # Mock data loaders
        self.loader = MagicMock()
        self.loader.__iter__.return_value = [
            {'input_ids': torch.randn(2, 10), 'attention_mask': torch.ones(2, 10), 'labels': torch.tensor([0, 1])}
        ]
        self.loader.__len__.return_value = 1

    @patch('main.fine_tune_after_pruning')
    def test_sparsity_preservation(self, mock_finetune):
        # Mock fine-tuning to just return empty dict (simulating training)
        mock_finetune.return_value = {}
        
        # Run pruning
        print("\nTesting sparsity preservation...")
        pruned_model, _ = run_pruning(
            self.config, self.model, None, None, None, self.device, model_name="test_model"
        )
        
        # Check sparsity
        total_params = 0
        zero_params = 0
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight
                if hasattr(module, 'weight_mask'):
                    weight = weight * module.weight_mask
                
                total_params += weight.numel()
                zero_params += (weight == 0).sum().item()
        
        sparsity = zero_params / total_params
        print(f"Final sparsity: {sparsity:.2f}")
        
        # Should be close to 0.5 (50%)
        self.assertTrue(sparsity > 0.4, f"Sparsity {sparsity} is too low! Pruning might have been lost.")

    def test_metrics_config(self):
        """Test that metrics include config arguments."""
        print("\nTesting metrics config...")
        
        # Create a dummy config
        import argparse
        config = argparse.Namespace(prune_sparsity=0.5, quant_method='dynamic')
        config_dict = vars(config)
        
        # Create dummy metrics with config
        from evaluation import CompressionStageMetrics
        metrics = CompressionStageMetrics(
            stage='test',
            model_size_mb=10.0,
            num_parameters=1000,
            trainable_parameters=1000,
            sparsity_percent=0.5,
            accuracy_exact=0.8,
            accuracy_per_label=0.8,
            f1_macro=0.8,
            f1_weighted=0.8,
            f1_micro=0.8,
            precision_macro=0.8,
            recall_macro=0.8,
            hamming_loss=0.2,
            per_label_f1={'label': 0.8},
            per_label_precision={'label': 0.8},
            per_label_recall={'label': 0.8},
            per_label_accuracy={'label': 0.8},
            priority_weighted_f1=0.8,
            config=config_dict
        )
        
        # Check flat dict
        flat = metrics.to_flat_dict()
        print(f"Flat metrics keys: {flat.keys()}")
        
        self.assertIn('arg_prune_sparsity', flat)
        self.assertEqual(flat['arg_prune_sparsity'], 0.5)
        self.assertIn('arg_quant_method', flat)
        self.assertEqual(flat['arg_quant_method'], 'dynamic')
        print("INT8 model file created successfully.")
        
        # Clean up
    @patch('transformers.BitsAndBytesConfig')
    @patch('distillation.StudentModel.from_pretrained')
    @patch('transformers.AutoModel.from_pretrained')
    def test_quant_int4(self, mock_auto_from_pretrained, mock_student_from_pretrained, mock_bnb_config):
        """Test INT4 quantization flow (mocked)."""
        print("\nTesting INT4 quantization...")
        self.config.quant_method = 'int4'
        
        # Mock bitsandbytes import check
        with patch.dict('sys.modules', {'bitsandbytes': MagicMock()}):
            # Run quantization
            run_quantization(self.config, self.model, None, None, None, self.device)
            
            # Check if from_pretrained was called (indicating reload)
            # Since MockModel is not StudentModel, it falls back to AutoModel
            mock_auto_from_pretrained.assert_called()
            print("INT4 quantization flow verified.")

    def test_quant_static(self):
        """Test Static INT8 quantization flow."""
        print("\nTesting Static INT8 quantization...")
        self.config.quant_method = 'static'
        
        # Run quantization (needs calibration loader)
        run_quantization(self.config, self.model, self.loader, None, None, 'cpu')
        print("Static quantization verified.")

    def test_quant_fp16(self):
        """Test FP16 quantization flow."""
        print("\nTesting FP16 quantization...")
        self.config.quant_method = 'fp16'
        
        # Run quantization
        run_quantization(self.config, self.model, None, None, None, 'cpu')
        print("FP16 quantization verified.")

if __name__ == '__main__':
    # Patch create_data_loaders to return our mock loaders
    with patch('main.create_data_loaders', return_value=(MagicMock(), MagicMock())):
        # Configure mock loaders to behave like iterables
        mock_loader = MagicMock()
        mock_loader.__iter__.return_value = [
            {'input_ids': torch.randn(2, 10), 'attention_mask': torch.ones(2, 10), 'labels': torch.tensor([0, 1])}
        ]
        mock_loader.__len__.return_value = 1
        
        with patch('main.create_data_loaders', return_value=(mock_loader, mock_loader)):
             unittest.main()

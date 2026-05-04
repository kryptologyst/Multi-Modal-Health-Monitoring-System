"""Tests for multi-modal health monitoring system.

This module contains unit tests for the health monitoring system components.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models import HealthMonitoringModel, MultiModalFusionLayer, AudioEncoder, UncertaintyCalibrationLoss
from src.data import HealthMonitoringDataset, DataProcessor
from src.eval import MedicalEvaluationMetrics, UncertaintyAnalyzer
from src.viz import HealthMonitoringVisualizer


class TestHealthMonitoringModel:
    """Test cases for HealthMonitoringModel."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = HealthMonitoringModel(num_classes=6)
        assert model.num_classes == 6
        assert model.text_encoder is not None
        assert model.image_encoder is not None
        assert model.audio_encoder is not None
        assert model.fusion_layer is not None
        assert model.classifier is not None
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = HealthMonitoringModel(num_classes=6)
        
        # Create dummy inputs
        batch_size = 2
        text_input_ids = torch.randint(0, 1000, (batch_size, 512))
        text_attention_mask = torch.ones(batch_size, 512)
        audio_features = torch.randn(batch_size, 128, 313)
        image_features = torch.randn(batch_size, 3, 224, 224)
        
        # Forward pass
        outputs = model(text_input_ids, text_attention_mask, audio_features, image_features)
        
        # Check output shapes
        assert outputs["logits"].shape == (batch_size, 6)
        assert outputs["probabilities"].shape == (batch_size, 6)
        assert outputs["uncertainty"].shape == (batch_size, 1)
        assert outputs["calibration"].shape == (batch_size, 1)
        assert outputs["fused_features"].shape == (batch_size, 512)
        
        # Check probability normalization
        assert torch.allclose(outputs["probabilities"].sum(dim=-1), torch.ones(batch_size))
        
        # Check uncertainty positivity
        assert torch.all(outputs["uncertainty"] >= 0)
        
        # Check calibration range
        assert torch.all(outputs["calibration"] >= 0)
        assert torch.all(outputs["calibration"] <= 1)


class TestMultiModalFusionLayer:
    """Test cases for MultiModalFusionLayer."""
    
    def test_fusion_layer_initialization(self):
        """Test fusion layer initialization."""
        fusion_layer = MultiModalFusionLayer(
            text_dim=768,
            audio_dim=512,
            image_dim=512,
            hidden_dim=512
        )
        assert fusion_layer.text_dim == 768
        assert fusion_layer.audio_dim == 512
        assert fusion_layer.image_dim == 512
        assert fusion_layer.hidden_dim == 512
    
    def test_fusion_layer_forward(self):
        """Test fusion layer forward pass."""
        fusion_layer = MultiModalFusionLayer()
        
        batch_size = 4
        text_features = torch.randn(batch_size, 768)
        audio_features = torch.randn(batch_size, 512)
        image_features = torch.randn(batch_size, 512)
        
        fused_features, uncertainty = fusion_layer(text_features, audio_features, image_features)
        
        assert fused_features.shape == (batch_size, 512)
        assert uncertainty.shape == (batch_size, 1)
        assert torch.all(uncertainty >= 0)


class TestAudioEncoder:
    """Test cases for AudioEncoder."""
    
    def test_audio_encoder_initialization(self):
        """Test audio encoder initialization."""
        encoder = AudioEncoder(input_dim=128, output_dim=512)
        assert encoder.conv_layers is not None
        assert encoder.global_pool is not None
        assert encoder.projection is not None
    
    def test_audio_encoder_forward(self):
        """Test audio encoder forward pass."""
        encoder = AudioEncoder(input_dim=128, output_dim=512)
        
        batch_size = 3
        seq_len = 1000
        audio_features = torch.randn(batch_size, seq_len, 128)
        
        output = encoder(audio_features)
        
        assert output.shape == (batch_size, 512)


class TestUncertaintyCalibrationLoss:
    """Test cases for UncertaintyCalibrationLoss."""
    
    def test_loss_initialization(self):
        """Test loss function initialization."""
        loss_fn = UncertaintyCalibrationLoss(alpha=0.1, beta=0.1)
        assert loss_fn.alpha == 0.1
        assert loss_fn.beta == 0.1
        assert loss_fn.ce_loss is not None
    
    def test_loss_forward(self):
        """Test loss function forward pass."""
        loss_fn = UncertaintyCalibrationLoss()
        
        batch_size = 4
        num_classes = 6
        logits = torch.randn(batch_size, num_classes)
        uncertainty = torch.rand(batch_size, 1)
        calibration = torch.rand(batch_size, 1)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        loss_dict = loss_fn(logits, uncertainty, calibration, targets)
        
        assert "total_loss" in loss_dict
        assert "ce_loss" in loss_dict
        assert "uncertainty_loss" in loss_dict
        assert "calibration_loss" in loss_dict
        
        assert loss_dict["total_loss"] >= 0
        assert loss_dict["ce_loss"] >= 0
        assert loss_dict["uncertainty_loss"] >= 0
        assert loss_dict["calibration_loss"] >= 0


class TestHealthMonitoringDataset:
    """Test cases for HealthMonitoringDataset."""
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        # Create temporary data directory
        data_dir = Path("temp_data")
        data_dir.mkdir(exist_ok=True)
        
        try:
            dataset = HealthMonitoringDataset(data_dir, split="train")
            assert len(dataset) > 0
            assert dataset.labels is not None
            assert dataset.label_to_idx is not None
        finally:
            # Clean up
            import shutil
            shutil.rmtree(data_dir)
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval."""
        data_dir = Path("temp_data")
        data_dir.mkdir(exist_ok=True)
        
        try:
            dataset = HealthMonitoringDataset(data_dir, split="train")
            sample = dataset[0]
            
            assert "text_input_ids" in sample
            assert "text_attention_mask" in sample
            assert "audio_features" in sample
            assert "image_features" in sample
            assert "label" in sample
            assert "sample_id" in sample
            assert "metadata" in sample
            
            assert isinstance(sample["label"], torch.Tensor)
            assert sample["text_input_ids"].shape == (512,)
            assert sample["text_attention_mask"].shape == (512,)
            assert sample["audio_features"].shape == (128, 313)
            assert sample["image_features"].shape == (3, 224, 224)
        finally:
            import shutil
            shutil.rmtree(data_dir)


class TestMedicalEvaluationMetrics:
    """Test cases for MedicalEvaluationMetrics."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = MedicalEvaluationMetrics()
        assert metrics.class_names is not None
        assert len(metrics.class_names) == 6
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        metrics = MedicalEvaluationMetrics()
        
        batch_size = 10
        num_classes = 6
        
        predictions = torch.randint(0, num_classes, (batch_size,))
        targets = torch.randint(0, num_classes, (batch_size,))
        probabilities = torch.softmax(torch.randn(batch_size, num_classes), dim=-1)
        uncertainty = torch.rand(batch_size, 1)
        calibration = torch.rand(batch_size, 1)
        
        results = metrics.compute_metrics(
            predictions, targets, probabilities, uncertainty, calibration
        )
        
        assert "accuracy" in results
        assert "f1_macro" in results
        assert "f1_weighted" in results
        assert "auc_macro" in results
        assert "uncertainty_correlation" in results
        assert "ece" in results
        
        assert 0 <= results["accuracy"] <= 1
        assert 0 <= results["f1_macro"] <= 1
        assert 0 <= results["f1_weighted"] <= 1


class TestUncertaintyAnalyzer:
    """Test cases for UncertaintyAnalyzer."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = UncertaintyAnalyzer()
        assert analyzer is not None
    
    def test_uncertainty_analysis(self):
        """Test uncertainty analysis."""
        analyzer = UncertaintyAnalyzer()
        
        batch_size = 20
        predictions = torch.randint(0, 6, (batch_size,))
        targets = torch.randint(0, 6, (batch_size,))
        uncertainty = torch.rand(batch_size, 1)
        
        analysis = analyzer.analyze_uncertainty_patterns(
            predictions, targets, uncertainty
        )
        
        assert "uncertainty_stats" in analysis
        assert "uncertainty_by_correctness" in analysis
        assert "uncertainty_by_class" in analysis
        assert "high_uncertainty_cases" in analysis
        
        assert "mean" in analysis["uncertainty_stats"]
        assert "std" in analysis["uncertainty_stats"]
        assert "min" in analysis["uncertainty_stats"]
        assert "max" in analysis["uncertainty_stats"]


class TestHealthMonitoringVisualizer:
    """Test cases for HealthMonitoringVisualizer."""
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        visualizer = HealthMonitoringVisualizer()
        assert visualizer.class_names is not None
        assert visualizer.output_dir is not None
    
    def test_plot_confusion_matrix(self):
        """Test confusion matrix plotting."""
        visualizer = HealthMonitoringVisualizer()
        
        batch_size = 20
        predictions = torch.randint(0, 6, (batch_size,))
        targets = torch.randint(0, 6, (batch_size,))
        
        fig = visualizer.plot_confusion_matrix(predictions, targets)
        assert fig is not None


# Integration tests
class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_inference(self):
        """Test end-to-end inference pipeline."""
        model = HealthMonitoringModel(num_classes=6)
        model.eval()
        
        # Create dummy inputs
        batch_size = 2
        text_input_ids = torch.randint(0, 1000, (batch_size, 512))
        text_attention_mask = torch.ones(batch_size, 512)
        audio_features = torch.randn(batch_size, 128, 313)
        image_features = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            outputs = model(text_input_ids, text_attention_mask, audio_features, image_features)
            
            # Check that all outputs are valid
            assert outputs["logits"].shape == (batch_size, 6)
            assert outputs["probabilities"].shape == (batch_size, 6)
            assert outputs["uncertainty"].shape == (batch_size, 1)
            assert outputs["calibration"].shape == (batch_size, 1)
            
            # Check probability normalization
            assert torch.allclose(outputs["probabilities"].sum(dim=-1), torch.ones(batch_size))
            
            # Check uncertainty positivity
            assert torch.all(outputs["uncertainty"] >= 0)
    
    def test_data_processing_pipeline(self):
        """Test data processing pipeline."""
        data_dir = Path("temp_data")
        data_dir.mkdir(exist_ok=True)
        
        try:
            processor = DataProcessor()
            train_loader, val_loader, test_loader = processor.create_data_loaders(
                data_dir=data_dir,
                batch_size=2,
                num_workers=0  # Avoid multiprocessing issues in tests
            )
            
            # Test that loaders are created
            assert train_loader is not None
            assert val_loader is not None
            assert test_loader is not None
            
            # Test that we can get a batch
            batch = next(iter(train_loader))
            assert "text_input_ids" in batch
            assert "text_attention_mask" in batch
            assert "audio_features" in batch
            assert "image_features" in batch
            assert "labels" in batch
            
        finally:
            import shutil
            shutil.rmtree(data_dir)


if __name__ == "__main__":
    pytest.main([__file__])

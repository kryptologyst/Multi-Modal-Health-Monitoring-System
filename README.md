# Multi-Modal Health Monitoring System

A comprehensive multi-modal AI system for health monitoring that combines text, audio, and image analysis for health assessment. This system is designed for research and educational purposes with appropriate medical disclaimers.

## ⚠️ IMPORTANT MEDICAL DISCLAIMER ⚠️

**This system is for research and educational purposes only. It is NOT intended for:**
- Clinical diagnosis
- Medical treatment decisions  
- Healthcare recommendations
- Professional medical advice

**Always consult qualified healthcare professionals for medical concerns.**

## Overview

This multi-modal health monitoring system integrates three types of data:

- **Text Analysis**: Health reports, symptoms, and medical history using NLP models
- **Audio Analysis**: Voice recordings for stress, fatigue, and emotional state detection
- **Image Analysis**: Medical images (X-rays, skin photos) using computer vision

The system provides:
- Multi-modal fusion with attention mechanisms
- Uncertainty quantification for medical reliability
- Calibration metrics for prediction confidence
- Comprehensive evaluation with medical-grade metrics

## Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/kryptologyst/multimodal-health-monitoring.git
cd multimodal-health-monitoring
```

2. **Install dependencies:**
```bash
pip install -e .
```

3. **Run the demo:**
```bash
streamlit run demo/app.py
```

### Basic Usage

```python
from src.models import HealthMonitoringModel
from src.data import DataProcessor

# Load model
model = HealthMonitoringModel()

# Process data
processor = DataProcessor()
train_loader, val_loader, test_loader = processor.create_data_loaders("data/")

# Train model
from scripts.train import HealthMonitoringTrainer
trainer = HealthMonitoringTrainer(model, train_loader, val_loader, test_loader, config)
trainer.train()
```

## Project Structure

```
multimodal-health-monitoring/
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   ├── data/                     # Data processing
│   ├── eval/                     # Evaluation metrics
│   ├── viz/                      # Visualization tools
│   └── utils/                    # Utility functions
├── configs/                      # Configuration files
│   ├── model/                    # Model configurations
│   ├── train/                    # Training configurations
│   └── eval/                     # Evaluation configurations
├── data/                         # Data directory
│   ├── images/                   # Image data
│   ├── audio/                    # Audio data
│   ├── text/                     # Text data
│   └── annotations.json          # Data annotations
├── scripts/                      # Training and evaluation scripts
├── demo/                         # Streamlit demo application
├── tests/                        # Unit tests
├── assets/                       # Generated assets and visualizations
├── checkpoints/                  # Model checkpoints
└── notebooks/                    # Jupyter notebooks
```

## Model Architecture

### Multi-Modal Fusion

The system uses an attention-based fusion mechanism:

1. **Text Encoder**: DistilBERT for health report analysis
2. **Audio Encoder**: CNN-based encoder for voice analysis  
3. **Image Encoder**: CLIP vision model for medical image analysis
4. **Fusion Layer**: Cross-attention mechanism for modality integration
5. **Classification Head**: Multi-layer perceptron with uncertainty quantification

### Uncertainty Quantification

- **Epistemic Uncertainty**: Model uncertainty estimation
- **Calibration**: Prediction confidence calibration
- **Medical Metrics**: AUROC, AUPRC, calibration curves

## Evaluation Metrics

### Classification Performance
- Accuracy, F1-Score (macro/weighted)
- Per-class precision, recall, F1
- Confusion matrix analysis

### Medical-Specific Metrics
- Sensitivity at specific FPR thresholds
- Average precision per condition
- ROC curves and AUC scores

### Uncertainty Metrics
- Uncertainty correlation with errors
- Calibration error (ECE, MCE)
- High uncertainty case analysis

## Configuration

### Training Configuration

```yaml
# configs/train/default.yaml
training:
  num_epochs: 10
  batch_size: 8
  learning_rate: 1e-4
  weight_decay: 1e-5
  patience: 5

loss:
  uncertainty_weight: 0.1
  calibration_weight: 0.1
```

### Model Configuration

```yaml
# configs/model/health_monitoring.yaml
model:
  num_classes: 6
  text_model_name: "distilbert-base-uncased"
  image_model_name: "openai/clip-vit-base-patch32"
  fusion_hidden_dim: 512
```

## Training

### Command Line Training

```bash
python scripts/train.py \
    --config configs/train/default.yaml \
    --data_dir data/ \
    --output_dir outputs/ \
    --seed 42
```

### Programmatic Training

```python
from scripts.train import HealthMonitoringTrainer

trainer = HealthMonitoringTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader, 
    test_loader=test_loader,
    config=config
)

trainer.train()
trainer.evaluate()
```

## Demo Application

The Streamlit demo provides an interactive interface for:

- **Text Input**: Health reports and symptom descriptions
- **Audio Upload**: Voice recordings for analysis
- **Image Upload**: Medical images for visual analysis
- **Results Display**: Predictions with uncertainty and calibration
- **Visualization**: Attention maps and confidence scores

### Running the Demo

```bash
streamlit run demo/app.py
```

Access the demo at `http://localhost:8501`

## Data Format

### Dataset Structure

```json
{
  "id": "sample_001",
  "text": "Patient reports feeling fatigued and experiencing shortness of breath.",
  "audio_file": "sample_001.wav",
  "image_file": "sample_001.jpg", 
  "label": "fatigue",
  "metadata": {
    "age": 45,
    "gender": "female",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Supported Formats

- **Text**: Plain text health reports
- **Audio**: WAV, MP3, M4A files (16kHz recommended)
- **Images**: JPG, PNG, JPEG files (224x224 recommended)

## Testing

### Run Tests

```bash
pytest tests/ -v
```

### Test Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

## API Reference

### Core Classes

#### `HealthMonitoringModel`
Main multi-modal health monitoring model with uncertainty quantification.

#### `MultiModalFusionLayer`
Attention-based fusion layer for combining text, audio, and image features.

#### `HealthMonitoringDataset`
Dataset class for loading and preprocessing multi-modal health data.

#### `MedicalEvaluationMetrics`
Comprehensive evaluation metrics for medical applications.

#### `HealthMonitoringVisualizer`
Visualization tools for health monitoring results.

### Key Methods

```python
# Model inference
outputs = model(text_input_ids, text_attention_mask, audio_features, image_features)

# Evaluation
metrics = evaluator.compute_metrics(predictions, targets, probabilities, uncertainty, calibration)

# Visualization
visualizer.create_evaluation_dashboard(predictions, targets, probabilities, uncertainty, calibration)
```

## Privacy and Security

### Data Protection
- No data is stored permanently
- Temporary files are cleaned up automatically
- All processing is done locally

### Medical Compliance
- Research-only usage
- No clinical decision support
- Appropriate disclaimers throughout

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run linting
black src/ tests/
ruff check src/ tests/
mypy src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Transformers library for NLP models
- CLIP for vision-language understanding
- PyTorch for deep learning framework
- Streamlit for demo interface

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the medical disclaimers

## Final Medical Disclaimer

**This system is for research and educational purposes only. It is NOT intended for clinical diagnosis, treatment, or medical decision-making. Always consult qualified healthcare professionals for medical advice and diagnosis.**

---

**Repository**: [github.com/kryptologyst/Multi-Modal-Health-Monitoring-System](https://github.com/kryptologyst/Multi-Modal-Health-Monitoring-System)
# Multi-Modal-Health-Monitoring-System

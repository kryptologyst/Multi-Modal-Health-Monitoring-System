"""Inference script for multi-modal health monitoring.

This script provides command-line inference capabilities for the
multi-modal health monitoring system.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from PIL import Image
import librosa

from src.models import HealthMonitoringModel
from src import DEVICE, set_seed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Health condition labels
HEALTH_CONDITIONS = [
    "diabetes",
    "heart_disease",
    "fatigue",
    "stress",
    "healthy",
    "overweight"
]


def load_model(checkpoint_path: Optional[str] = None) -> HealthMonitoringModel:
    """Load the health monitoring model.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        
    Returns:
        Loaded health monitoring model.
    """
    model = HealthMonitoringModel(
        num_classes=6,
        text_model_name="distilbert-base-uncased",
        image_model_name="openai/clip-vit-base-patch32",
        fusion_hidden_dim=512,
        dropout=0.1,
    )
    
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model from {checkpoint_path}")
    else:
        logger.info("Using randomly initialized model")
    
    model.eval()
    return model.to(DEVICE)


def preprocess_text(text: str) -> Dict[str, torch.Tensor]:
    """Preprocess text input.
    
    Args:
        text: Input text.
        
    Returns:
        Preprocessed text features.
    """
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    encoding = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
    }


def preprocess_audio(audio_path: str) -> torch.Tensor:
    """Preprocess audio input.
    
    Args:
        audio_path: Path to audio file.
        
    Returns:
        Preprocessed audio features.
    """
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Pad or truncate to 10 seconds
        target_length = 10 * sr
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Convert to mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=128,
            hop_length=512,
        )
        
        # Convert to log scale and normalize
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
        
        return torch.tensor(log_mel_spec, dtype=torch.float32)
    
    except Exception as e:
        logger.error(f"Error processing audio {audio_path}: {e}")
        # Return synthetic audio features
        return torch.randn(128, 313)


def preprocess_image(image_path: str) -> torch.Tensor:
    """Preprocess image input.
    
    Args:
        image_path: Path to image file.
        
    Returns:
        Preprocessed image features.
    """
    try:
        # Load image
        image = Image.open(image_path)
        image = image.convert("RGB")
        
        # Resize to 224x224
        image = image.resize((224, 224))
        
        # Convert to numpy array
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Convert to tensor and normalize for CLIP
        image_tensor = torch.tensor(image_array).permute(2, 0, 1)  # CHW format
        
        # Normalize for CLIP (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        image_tensor = (image_tensor - mean.view(3, 1, 1)) / std.view(3, 1, 1)
        
        return image_tensor
    
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        # Return synthetic image features
        return torch.randn(3, 224, 224)


def predict_health_condition(
    model: HealthMonitoringModel,
    text: str,
    audio_path: Optional[str] = None,
    image_path: Optional[str] = None,
) -> Dict[str, any]:
    """Predict health condition from multi-modal inputs.
    
    Args:
        model: Health monitoring model.
        text: Health report text.
        audio_path: Path to audio file.
        image_path: Path to image file.
        
    Returns:
        Dictionary containing predictions and uncertainty.
    """
    model.eval()
    
    # Preprocess inputs
    text_features = preprocess_text(text)
    
    if audio_path:
        audio_features = preprocess_audio(audio_path)
    else:
        audio_features = torch.randn(128, 313)  # Synthetic audio
    
    if image_path:
        image_features = preprocess_image(image_path)
    else:
        image_features = torch.randn(3, 224, 224)  # Synthetic image
    
    with torch.no_grad():
        # Add batch dimension
        text_input_ids = text_features["input_ids"].unsqueeze(0).to(DEVICE)
        text_attention_mask = text_features["attention_mask"].unsqueeze(0).to(DEVICE)
        audio_features = audio_features.unsqueeze(0).to(DEVICE)
        image_features = image_features.unsqueeze(0).to(DEVICE)
        
        # Forward pass
        outputs = model(
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            audio_features=audio_features,
            image_features=image_features,
        )
        
        # Get predictions
        logits = outputs["logits"]
        probabilities = outputs["probabilities"]
        uncertainty = outputs["uncertainty"]
        calibration = outputs["calibration"]
        
        # Get predicted class
        predicted_class = logits.argmax(dim=-1).item()
        predicted_probability = probabilities[0, predicted_class].item()
        
        return {
            "predicted_class": predicted_class,
            "predicted_condition": HEALTH_CONDITIONS[predicted_class],
            "predicted_probability": predicted_probability,
            "all_probabilities": probabilities[0].cpu().numpy().tolist(),
            "uncertainty": uncertainty[0].item(),
            "calibration": calibration[0].item(),
        }


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Multi-Modal Health Monitoring Inference")
    parser.add_argument("--text", type=str, required=True, 
                       help="Health report text")
    parser.add_argument("--audio", type=str, 
                       help="Path to audio file")
    parser.add_argument("--image", type=str, 
                       help="Path to image file")
    parser.add_argument("--checkpoint", type=str, 
                       help="Path to model checkpoint")
    parser.add_argument("--output", type=str, 
                       help="Path to save results")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Display medical disclaimer
    print("=" * 80)
    print("MULTI-MODAL HEALTH MONITORING SYSTEM")
    print("=" * 80)
    print("IMPORTANT MEDICAL DISCLAIMER:")
    print("This system is for research and educational purposes only.")
    print("It is NOT intended for clinical diagnosis or medical decision-making.")
    print("Always consult qualified healthcare professionals for medical advice.")
    print("=" * 80)
    
    # Set seed
    set_seed(args.seed)
    
    # Load model
    logger.info("Loading model...")
    model = load_model(args.checkpoint)
    
    # Run inference
    logger.info("Running inference...")
    results = predict_health_condition(
        model=model,
        text=args.text,
        audio_path=args.audio,
        image_path=args.image,
    )
    
    # Display results
    print("\nPREDICTION RESULTS:")
    print("-" * 40)
    print(f"Predicted Condition: {results['predicted_condition'].replace('_', ' ').title()}")
    print(f"Confidence: {results['predicted_probability']:.1%}")
    print(f"Uncertainty Score: {results['uncertainty']:.3f}")
    print(f"Calibration Score: {results['calibration']:.3f}")
    
    print("\nALL CONDITION PROBABILITIES:")
    print("-" * 40)
    for i, condition in enumerate(HEALTH_CONDITIONS):
        prob = results['all_probabilities'][i]
        print(f"{condition.replace('_', ' ').title():15}: {prob:.1%}")
    
    # Uncertainty analysis
    print("\nUNCERTAINTY ANALYSIS:")
    print("-" * 40)
    if results['uncertainty'] < 0.3:
        print("✅ Low uncertainty - High confidence prediction")
    elif results['uncertainty'] < 0.6:
        print("⚠️  Moderate uncertainty - Consider additional data")
    else:
        print("❌ High uncertainty - Prediction may be unreliable")
    
    # Calibration analysis
    print("\nCALIBRATION ANALYSIS:")
    print("-" * 40)
    if results['calibration'] > 0.8:
        print("✅ Well-calibrated prediction")
    elif results['calibration'] > 0.6:
        print("⚠️  Moderately calibrated")
    else:
        print("❌ Poorly calibrated - Low reliability")
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    print("\n" + "=" * 80)
    print("REMEMBER: This is for research/educational purposes only!")
    print("NOT for clinical diagnosis or medical decision-making.")
    print("=" * 80)


if __name__ == "__main__":
    main()

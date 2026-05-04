"""Streamlit demo application for multi-modal health monitoring.

This demo provides an interactive interface for uploading and analyzing
health data using the multi-modal health monitoring system.

IMPORTANT MEDICAL DISCLAIMER:
This demo is for research and educational purposes only. It is NOT intended
for clinical diagnosis, treatment, or medical decision-making.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import torch
from PIL import Image
import librosa
import cv2

# Import our modules
from src.models import HealthMonitoringModel
from src.data import DataProcessor
from src.eval import MedicalEvaluationMetrics
from src.viz import HealthMonitoringVisualizer
from src import DEVICE, set_seed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Multi-Modal Health Monitoring",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Medical disclaimer
MEDICAL_DISCLAIMER = """
⚠️ **IMPORTANT MEDICAL DISCLAIMER** ⚠️

This system is for **research and educational purposes only**. It is **NOT intended** for:
- Clinical diagnosis
- Medical treatment decisions
- Healthcare recommendations
- Professional medical advice

**Always consult qualified healthcare professionals for medical concerns.**
"""

# Health condition labels
HEALTH_CONDITIONS = [
    "diabetes",
    "heart_disease", 
    "fatigue",
    "stress",
    "healthy",
    "overweight"
]

# Initialize session state
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model" not in st.session_state:
    st.session_state.model = None


@st.cache_resource
def load_model() -> HealthMonitoringModel:
    """Load the health monitoring model.
    
    Returns:
        Loaded health monitoring model.
    """
    try:
        model = HealthMonitoringModel(
            num_classes=6,
            text_model_name="distilbert-base-uncased",
            image_model_name="openai/clip-vit-base-patch32",
            fusion_hidden_dim=512,
            dropout=0.1,
        )
        
        # Try to load a checkpoint if available
        checkpoint_path = Path("checkpoints/best_model.pt")
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded model from checkpoint")
        else:
            logger.info("Using randomly initialized model")
        
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        return None


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


def preprocess_audio(audio_file) -> torch.Tensor:
    """Preprocess audio input.
    
    Args:
        audio_file: Uploaded audio file.
        
    Returns:
        Preprocessed audio features.
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        # Load audio
        audio, sr = librosa.load(tmp_path, sr=16000)
        
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
        
        # Clean up temporary file
        Path(tmp_path).unlink()
        
        return torch.tensor(log_mel_spec, dtype=torch.float32)
    
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        st.error(f"Error processing audio: {e}")
        # Return synthetic audio features
        return torch.randn(128, 313)  # Synthetic mel-spectrogram


def preprocess_image(image_file) -> torch.Tensor:
    """Preprocess image input.
    
    Args:
        image_file: Uploaded image file.
        
    Returns:
        Preprocessed image features.
    """
    try:
        # Load image
        image = Image.open(image_file)
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
        logger.error(f"Error processing image: {e}")
        st.error(f"Error processing image: {e}")
        # Return synthetic image features
        return torch.randn(3, 224, 224)


def predict_health_condition(
    model: HealthMonitoringModel,
    text_features: Dict[str, torch.Tensor],
    audio_features: torch.Tensor,
    image_features: torch.Tensor,
) -> Dict[str, any]:
    """Predict health condition from multi-modal inputs.
    
    Args:
        model: Health monitoring model.
        text_features: Preprocessed text features.
        audio_features: Preprocessed audio features.
        image_features: Preprocessed image features.
        
    Returns:
        Dictionary containing predictions and uncertainty.
    """
    model.eval()
    
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
            "all_probabilities": probabilities[0].cpu().numpy(),
            "uncertainty": uncertainty[0].item(),
            "calibration": calibration[0].item(),
        }


def main():
    """Main Streamlit application."""
    
    # Display disclaimer
    st.markdown(MEDICAL_DISCLAIMER)
    
    # Title
    st.title("🏥 Multi-Modal Health Monitoring System")
    st.markdown("**Research and Educational Demo**")
    
    # Sidebar
    with st.sidebar:
        st.header("System Information")
        st.info(f"Device: {DEVICE}")
        st.info("Model: Multi-Modal Health Monitoring")
        st.info("Purpose: Research & Education")
        
        st.header("Data Upload")
        st.markdown("Upload your health data for analysis:")
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("Loading model..."):
            model = load_model()
            if model is not None:
                st.session_state.model = model
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
            else:
                st.error("Failed to load model!")
                st.stop()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Health Report Text")
        health_text = st.text_area(
            "Enter health report or symptoms:",
            value="Patient reports feeling fatigued and experiencing shortness of breath.",
            height=100,
            help="Describe symptoms, health concerns, or medical history"
        )
        
        st.header("🎤 Audio Recording")
        audio_file = st.file_uploader(
            "Upload audio file (voice recording):",
            type=["wav", "mp3", "m4a"],
            help="Upload a voice recording for analysis"
        )
        
        if audio_file:
            st.audio(audio_file, format="audio/wav")
    
    with col2:
        st.header("📷 Medical Image")
        image_file = st.file_uploader(
            "Upload medical image:",
            type=["jpg", "jpeg", "png"],
            help="Upload a medical image (X-ray, skin photo, etc.)"
        )
        
        if image_file:
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Analysis button
    if st.button("🔍 Analyze Health Data", type="primary"):
        if not health_text.strip():
            st.error("Please enter health report text!")
            return
        
        with st.spinner("Analyzing health data..."):
            try:
                # Preprocess inputs
                text_features = preprocess_text(health_text)
                audio_features = preprocess_audio(audio_file) if audio_file else torch.randn(128, 313)
                image_features = preprocess_image(image_file) if image_file else torch.randn(3, 224, 224)
                
                # Get predictions
                results = predict_health_condition(
                    st.session_state.model,
                    text_features,
                    audio_features,
                    image_features,
                )
                
                # Display results
                st.header("📊 Analysis Results")
                
                # Main prediction
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Predicted Condition",
                        results["predicted_condition"].replace("_", " ").title(),
                        delta=f"{results['predicted_probability']:.1%} confidence"
                    )
                
                with col2:
                    st.metric(
                        "Uncertainty Score",
                        f"{results['uncertainty']:.3f}",
                        help="Lower values indicate higher confidence"
                    )
                
                with col3:
                    st.metric(
                        "Calibration Score",
                        f"{results['calibration']:.3f}",
                        help="Model's confidence calibration"
                    )
                
                # Detailed probabilities
                st.subheader("📈 Condition Probabilities")
                
                prob_data = []
                for i, condition in enumerate(HEALTH_CONDITIONS):
                    prob_data.append({
                        "Condition": condition.replace("_", " ").title(),
                        "Probability": f"{results['all_probabilities'][i]:.1%}",
                        "Score": results['all_probabilities'][i]
                    })
                
                # Sort by probability
                prob_data.sort(key=lambda x: x['Score'], reverse=True)
                
                # Display as table
                for i, item in enumerate(prob_data):
                    if i == 0:
                        st.success(f"🥇 {item['Condition']}: {item['Probability']}")
                    elif i == 1:
                        st.info(f"🥈 {item['Condition']}: {item['Probability']}")
                    elif i == 2:
                        st.warning(f"🥉 {item['Condition']}: {item['Probability']}")
                    else:
                        st.write(f"   {item['Condition']}: {item['Probability']}")
                
                # Uncertainty analysis
                st.subheader("🎯 Uncertainty Analysis")
                
                if results['uncertainty'] < 0.3:
                    st.success("✅ Low uncertainty - High confidence prediction")
                elif results['uncertainty'] < 0.6:
                    st.warning("⚠️ Moderate uncertainty - Consider additional data")
                else:
                    st.error("❌ High uncertainty - Prediction may be unreliable")
                
                # Calibration analysis
                st.subheader("📏 Calibration Analysis")
                
                if results['calibration'] > 0.8:
                    st.success("✅ Well-calibrated prediction")
                elif results['calibration'] > 0.6:
                    st.warning("⚠️ Moderately calibrated")
                else:
                    st.error("❌ Poorly calibrated - Low reliability")
                
                # Recommendations
                st.subheader("💡 Recommendations")
                
                recommendations = [
                    "This analysis is for research/educational purposes only",
                    "Consult qualified healthcare professionals for medical advice",
                    "Do not use this system for clinical decision-making",
                    "Consider multiple data sources for comprehensive assessment"
                ]
                
                for rec in recommendations:
                    st.write(f"• {rec}")
                
                # Technical details
                with st.expander("🔧 Technical Details"):
                    st.json({
                        "predicted_class": results["predicted_class"],
                        "predicted_probability": results["predicted_probability"],
                        "uncertainty": results["uncertainty"],
                        "calibration": results["calibration"],
                        "all_probabilities": results["all_probabilities"].tolist(),
                    })
                
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                logger.error(f"Analysis error: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Multi-Modal Health Monitoring System** | 
    Research & Educational Demo | 
    [GitHub Repository](https://github.com/kryptologyst/multimodal-health-monitoring)
    """)


if __name__ == "__main__":
    main()

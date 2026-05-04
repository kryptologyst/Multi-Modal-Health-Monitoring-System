"""Data processing and loading utilities for multi-modal health monitoring.

This module provides data loaders, preprocessing pipelines, and augmentation
strategies for text, audio, and image data in health monitoring applications.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class HealthMonitoringDataset(Dataset):
    """Dataset class for multi-modal health monitoring data.
    
    This dataset handles text, audio, and image data with proper preprocessing
    and augmentation for health monitoring applications.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        max_text_length: int = 512,
        audio_sample_rate: int = 16000,
        audio_max_length: float = 10.0,
        image_size: Tuple[int, int] = (224, 224),
        augment: bool = True,
    ) -> None:
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing the dataset.
            split: Dataset split ('train', 'val', 'test').
            max_text_length: Maximum text sequence length.
            audio_sample_rate: Audio sample rate in Hz.
            audio_max_length: Maximum audio length in seconds.
            image_size: Target image size (height, width).
            augment: Whether to apply data augmentation.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_text_length = max_text_length
        self.audio_sample_rate = audio_sample_rate
        self.audio_max_length = audio_max_length
        self.image_size = image_size
        self.augment = augment and split == "train"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Load data
        self.data = self._load_data()
        
        # Health condition labels
        self.labels = [
            "diabetes",
            "heart_disease", 
            "fatigue",
            "stress",
            "healthy",
            "overweight"
        ]
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load dataset from JSON file or create synthetic data.
        
        Returns:
            List of data samples.
        """
        data_file = self.data_dir / f"{self.split}.json"
        
        if data_file.exists():
            with open(data_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Data file {data_file} not found. Creating synthetic data.")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> List[Dict[str, Any]]:
        """Create synthetic data for demonstration purposes.
        
        Returns:
            List of synthetic data samples.
        """
        synthetic_data = []
        
        # Create synthetic samples
        for i in range(50):  # Small dataset for demo
            sample = {
                "id": f"synthetic_{i}",
                "text": self._generate_synthetic_text(i),
                "audio_file": f"synthetic_audio_{i}.wav",
                "image_file": f"synthetic_image_{i}.jpg",
                "label": self.labels[i % len(self.labels)],
                "metadata": {
                    "age": np.random.randint(20, 80),
                    "gender": np.random.choice(["male", "female"]),
                    "timestamp": f"2024-01-{i%28+1:02d}T{np.random.randint(0,24):02d}:00:00Z"
                }
            }
            synthetic_data.append(sample)
        
        return synthetic_data
    
    def _generate_synthetic_text(self, idx: int) -> str:
        """Generate synthetic health report text.
        
        Args:
            idx: Sample index.
            
        Returns:
            Synthetic health report text.
        """
        templates = [
            "Patient reports feeling {condition} and experiencing {symptom}.",
            "Health assessment shows signs of {condition} with {symptom}.",
            "Patient describes {symptom} and mentions {condition} concerns.",
            "Medical evaluation indicates {condition} based on {symptom}.",
            "Patient presents with {symptom} suggesting possible {condition}.",
        ]
        
        conditions = ["fatigue", "stress", "diabetes", "heart disease", "healthy"]
        symptoms = ["tiredness", "shortness of breath", "chest pain", "dizziness", "normal vitals"]
        
        template = templates[idx % len(templates)]
        condition = conditions[idx % len(conditions)]
        symptom = symptoms[idx % len(symptoms)]
        
        return template.format(condition=condition, symptom=symptom)
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a data sample.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary containing processed data.
        """
        sample = self.data[idx]
        
        # Process text
        text_features = self._process_text(sample["text"])
        
        # Process audio
        audio_features = self._process_audio(sample["audio_file"])
        
        # Process image
        image_features = self._process_image(sample["image_file"])
        
        # Get label
        label = self.label_to_idx[sample["label"]]
        
        return {
            "text_input_ids": text_features["input_ids"],
            "text_attention_mask": text_features["attention_mask"],
            "audio_features": audio_features,
            "image_features": image_features,
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": sample["id"],
            "metadata": sample.get("metadata", {}),
        }
    
    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Process text data.
        
        Args:
            text: Input text.
            
        Returns:
            Tokenized text features.
        """
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
    
    def _process_audio(self, audio_file: str) -> torch.Tensor:
        """Process audio data.
        
        Args:
            audio_file: Path to audio file.
            
        Returns:
            Processed audio features.
        """
        # Create synthetic audio features if file doesn't exist
        audio_path = self.data_dir / "audio" / audio_file
        
        if not audio_path.exists():
            # Generate synthetic audio features
            duration = np.random.uniform(1.0, self.audio_max_length)
            n_samples = int(duration * self.audio_sample_rate)
            
            # Generate synthetic audio (simple sine wave with noise)
            t = np.linspace(0, duration, n_samples)
            frequency = np.random.uniform(100, 1000)  # Random frequency
            audio = np.sin(2 * np.pi * frequency * t)
            audio += 0.1 * np.random.randn(n_samples)  # Add noise
            
            # Pad or truncate to fixed length
            target_length = int(self.audio_max_length * self.audio_sample_rate)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]
        else:
            # Load real audio file
            try:
                audio, sr = librosa.load(audio_path, sr=self.audio_sample_rate)
                
                # Pad or truncate to fixed length
                target_length = int(self.audio_max_length * self.audio_sample_rate)
                if len(audio) < target_length:
                    audio = np.pad(audio, (0, target_length - len(audio)))
                else:
                    audio = audio[:target_length]
            except Exception as e:
                logger.warning(f"Error loading audio {audio_path}: {e}")
                # Fallback to synthetic audio
                audio = np.random.randn(int(self.audio_max_length * self.audio_sample_rate))
        
        # Convert to mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.audio_sample_rate,
            n_mels=128,
            hop_length=512,
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
        
        return torch.tensor(log_mel_spec, dtype=torch.float32)
    
    def _process_image(self, image_file: str) -> torch.Tensor:
        """Process image data.
        
        Args:
            image_file: Path to image file.
            
        Returns:
            Processed image features.
        """
        # Create synthetic image if file doesn't exist
        image_path = self.data_dir / "images" / image_file
        
        if not image_path.exists():
            # Generate synthetic image
            image = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
            
            # Add some structure to make it more realistic
            center_x, center_y = self.image_size[1] // 2, self.image_size[0] // 2
            cv2.circle(image, (center_x, center_y), 50, (255, 255, 255), -1)
            cv2.circle(image, (center_x, center_y), 30, (0, 0, 0), -1)
        else:
            # Load real image
            try:
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.warning(f"Error loading image {image_path}: {e}")
                # Fallback to synthetic image
                image = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
        
        # Resize image
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        
        # Apply augmentation if training
        if self.augment:
            image = self._augment_image(image)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and normalize for CLIP
        image_tensor = torch.tensor(image).permute(2, 0, 1)  # CHW format
        
        # Normalize for CLIP (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        image_tensor = (image_tensor - mean.view(3, 1, 1)) / std.view(3, 1, 1)
        
        return image_tensor
    
    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply image augmentation.
        
        Args:
            image: Input image.
            
        Returns:
            Augmented image.
        """
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
        
        # Random contrast adjustment
        contrast_factor = np.random.uniform(0.8, 1.2)
        image = np.clip(128 + contrast_factor * (image - 128), 0, 255).astype(np.uint8)
        
        return image


class DataProcessor:
    """Data processor for multi-modal health monitoring.
    
    Provides utilities for data preprocessing, augmentation, and batch creation.
    """
    
    def __init__(
        self,
        text_tokenizer_name: str = "distilbert-base-uncased",
        image_size: Tuple[int, int] = (224, 224),
        audio_sample_rate: int = 16000,
    ) -> None:
        """Initialize the data processor.
        
        Args:
            text_tokenizer_name: Name of the text tokenizer.
            image_size: Target image size.
            audio_sample_rate: Audio sample rate.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name)
        self.image_size = image_size
        self.audio_sample_rate = audio_sample_rate
        
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader.
        
        Args:
            batch: List of samples.
            
        Returns:
            Batched data.
        """
        # Stack tensors
        text_input_ids = torch.stack([item["text_input_ids"] for item in batch])
        text_attention_mask = torch.stack([item["text_attention_mask"] for item in batch])
        audio_features = torch.stack([item["audio_features"] for item in batch])
        image_features = torch.stack([item["image_features"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        
        return {
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "audio_features": audio_features,
            "image_features": image_features,
            "labels": labels,
            "sample_ids": [item["sample_id"] for item in batch],
            "metadata": [item["metadata"] for item in batch],
        }
    
    def create_data_loaders(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 8,
        num_workers: int = 4,
        shuffle_train: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for train, validation, and test splits.
        
        Args:
            data_dir: Directory containing the dataset.
            batch_size: Batch size.
            num_workers: Number of worker processes.
            shuffle_train: Whether to shuffle training data.
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        # Create datasets
        train_dataset = HealthMonitoringDataset(data_dir, split="train")
        val_dataset = HealthMonitoringDataset(data_dir, split="val")
        test_dataset = HealthMonitoringDataset(data_dir, split="test")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
        
        return train_loader, val_loader, test_loader

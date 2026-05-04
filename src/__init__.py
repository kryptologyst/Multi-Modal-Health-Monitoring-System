"""Multi-modal Health Monitoring System.

This module provides a comprehensive multi-modal health monitoring system that combines
text, audio, and image analysis for health assessment. The system is designed for
research and educational purposes with appropriate medical disclaimers.

IMPORTANT MEDICAL DISCLAIMER:
This system is for research and educational purposes only. It is NOT intended for
clinical diagnosis, treatment, or medical decision-making. Always consult qualified
healthcare professionals for medical advice and diagnosis.
"""

from __future__ import annotations

import logging
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    pipeline,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set device with fallback
def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Initialize device and seed
DEVICE = get_device()
set_seed(42)

logger.info(f"Using device: {DEVICE}")
logger.info("IMPORTANT: This system is for research/educational purposes only. NOT for clinical use.")

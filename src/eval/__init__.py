"""Evaluation metrics for multi-modal health monitoring.

This module provides comprehensive evaluation metrics specifically designed for
medical applications, including uncertainty quantification and calibration metrics.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class MedicalEvaluationMetrics:
    """Comprehensive evaluation metrics for medical applications.
    
    This class provides various metrics including classification performance,
    uncertainty quantification, and calibration metrics for health monitoring.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None) -> None:
        """Initialize the evaluation metrics.
        
        Args:
            class_names: List of class names for reporting.
        """
        self.class_names = class_names or [
            "diabetes",
            "heart_disease",
            "fatigue", 
            "stress",
            "healthy",
            "overweight"
        ]
        
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: torch.Tensor,
        uncertainty: torch.Tensor,
        calibration: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics.
        
        Args:
            predictions: Model predictions [batch_size]
            targets: Ground truth labels [batch_size]
            probabilities: Prediction probabilities [batch_size, num_classes]
            uncertainty: Uncertainty estimates [batch_size]
            calibration: Calibration scores [batch_size]
            
        Returns:
            Dictionary containing all computed metrics.
        """
        # Convert to numpy for sklearn compatibility
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        probs_np = probabilities.cpu().numpy()
        uncertainty_np = uncertainty.cpu().numpy()
        calibration_np = calibration.cpu().numpy()
        
        metrics = {}
        
        # Basic classification metrics
        metrics.update(self._compute_classification_metrics(preds_np, targets_np, probs_np))
        
        # Uncertainty metrics
        metrics.update(self._compute_uncertainty_metrics(preds_np, targets_np, uncertainty_np))
        
        # Calibration metrics
        metrics.update(self._compute_calibration_metrics(preds_np, targets_np, calibration_np))
        
        # Medical-specific metrics
        metrics.update(self._compute_medical_metrics(preds_np, targets_np, probs_np))
        
        return metrics
    
    def _compute_classification_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: np.ndarray,
    ) -> Dict[str, float]:
        """Compute basic classification metrics.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth labels.
            probabilities: Prediction probabilities.
            
        Returns:
            Dictionary of classification metrics.
        """
        metrics = {}
        
        # Accuracy
        metrics["accuracy"] = accuracy_score(targets, predictions)
        
        # F1 scores
        metrics["f1_macro"] = f1_score(targets, predictions, average="macro")
        metrics["f1_weighted"] = f1_score(targets, predictions, average="weighted")
        
        # Per-class F1 scores
        f1_per_class = f1_score(targets, predictions, average=None)
        for i, class_name in enumerate(self.class_names):
            metrics[f"f1_{class_name}"] = f1_per_class[i] if i < len(f1_per_class) else 0.0
        
        # AUC for multi-class (one-vs-rest)
        try:
            if len(np.unique(targets)) > 2:
                metrics["auc_macro"] = roc_auc_score(
                    targets, probabilities, multi_class="ovr", average="macro"
                )
                metrics["auc_weighted"] = roc_auc_score(
                    targets, probabilities, multi_class="ovr", average="weighted"
                )
            else:
                metrics["auc"] = roc_auc_score(targets, probabilities[:, 1])
        except ValueError:
            logger.warning("Could not compute AUC metrics")
            metrics["auc_macro"] = 0.0
            metrics["auc_weighted"] = 0.0
        
        return metrics
    
    def _compute_uncertainty_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainty: np.ndarray,
    ) -> Dict[str, float]:
        """Compute uncertainty quantification metrics.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth labels.
            uncertainty: Uncertainty estimates.
            
        Returns:
            Dictionary of uncertainty metrics.
        """
        metrics = {}
        
        # Uncertainty correlation with errors
        errors = (predictions != targets).astype(float)
        uncertainty_correlation = np.corrcoef(errors, uncertainty)[0, 1]
        metrics["uncertainty_correlation"] = uncertainty_correlation if not np.isnan(uncertainty_correlation) else 0.0
        
        # Average uncertainty for correct vs incorrect predictions
        correct_mask = predictions == targets
        metrics["uncertainty_correct"] = np.mean(uncertainty[correct_mask]) if np.any(correct_mask) else 0.0
        metrics["uncertainty_incorrect"] = np.mean(uncertainty[~correct_mask]) if np.any(~correct_mask) else 0.0
        
        # Uncertainty ratio (incorrect / correct)
        if metrics["uncertainty_correct"] > 0:
            metrics["uncertainty_ratio"] = metrics["uncertainty_incorrect"] / metrics["uncertainty_correct"]
        else:
            metrics["uncertainty_ratio"] = 0.0
        
        return metrics
    
    def _compute_calibration_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        calibration: np.ndarray,
    ) -> Dict[str, float]:
        """Compute calibration metrics.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth labels.
            calibration: Calibration scores.
            
        Returns:
            Dictionary of calibration metrics.
        """
        metrics = {}
        
        # Expected Calibration Error (ECE)
        accuracy = (predictions == targets).astype(float)
        ece = np.mean(np.abs(calibration - accuracy))
        metrics["ece"] = ece
        
        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(calibration - accuracy))
        metrics["mce"] = mce
        
        # Calibration correlation
        calibration_correlation = np.corrcoef(accuracy, calibration)[0, 1]
        metrics["calibration_correlation"] = calibration_correlation if not np.isnan(calibration_correlation) else 0.0
        
        return metrics
    
    def _compute_medical_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: np.ndarray,
    ) -> Dict[str, float]:
        """Compute medical-specific metrics.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth labels.
            probabilities: Prediction probabilities.
            
        Returns:
            Dictionary of medical metrics.
        """
        metrics = {}
        
        # Sensitivity and Specificity for each class
        for i, class_name in enumerate(self.class_names):
            # Binary classification for this class
            binary_targets = (targets == i).astype(int)
            binary_probs = probabilities[:, i]
            
            if len(np.unique(binary_targets)) > 1:
                try:
                    # ROC curve
                    fpr, tpr, _ = roc_curve(binary_targets, binary_probs)
                    
                    # Sensitivity (True Positive Rate) at specific FPR thresholds
                    for fpr_threshold in [0.01, 0.05, 0.1]:
                        idx = np.where(fpr <= fpr_threshold)[0]
                        if len(idx) > 0:
                            sensitivity = tpr[idx[-1]]
                            metrics[f"sensitivity_{class_name}_fpr{fpr_threshold}"] = sensitivity
                        else:
                            metrics[f"sensitivity_{class_name}_fpr{fpr_threshold}"] = 0.0
                    
                    # Precision-Recall curve
                    precision, recall, _ = precision_recall_curve(binary_targets, binary_probs)
                    
                    # Average Precision
                    avg_precision = np.trapz(precision, recall)
                    metrics[f"avg_precision_{class_name}"] = avg_precision
                    
                except ValueError:
                    logger.warning(f"Could not compute medical metrics for class {class_name}")
                    metrics[f"sensitivity_{class_name}_fpr0.01"] = 0.0
                    metrics[f"sensitivity_{class_name}_fpr0.05"] = 0.0
                    metrics[f"sensitivity_{class_name}_fpr0.1"] = 0.0
                    metrics[f"avg_precision_{class_name}"] = 0.0
        
        return metrics
    
    def generate_report(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: torch.Tensor,
        uncertainty: torch.Tensor,
        calibration: torch.Tensor,
    ) -> str:
        """Generate a comprehensive evaluation report.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth labels.
            probabilities: Prediction probabilities.
            uncertainty: Uncertainty estimates.
            calibration: Calibration scores.
            
        Returns:
            Formatted evaluation report.
        """
        metrics = self.compute_metrics(predictions, targets, probabilities, uncertainty, calibration)
        
        report = "=" * 80 + "\n"
        report += "MULTI-MODAL HEALTH MONITORING EVALUATION REPORT\n"
        report += "=" * 80 + "\n\n"
        
        report += "IMPORTANT MEDICAL DISCLAIMER:\n"
        report += "This evaluation is for research and educational purposes only.\n"
        report += "NOT intended for clinical diagnosis or medical decision-making.\n\n"
        
        # Classification Performance
        report += "CLASSIFICATION PERFORMANCE:\n"
        report += "-" * 40 + "\n"
        report += f"Accuracy: {metrics['accuracy']:.4f}\n"
        report += f"F1-Score (Macro): {metrics['f1_macro']:.4f}\n"
        report += f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}\n"
        report += f"AUC (Macro): {metrics['auc_macro']:.4f}\n"
        report += f"AUC (Weighted): {metrics['auc_weighted']:.4f}\n\n"
        
        # Per-class Performance
        report += "PER-CLASS PERFORMANCE:\n"
        report += "-" * 40 + "\n"
        for class_name in self.class_names:
            f1_key = f"f1_{class_name}"
            if f1_key in metrics:
                report += f"{class_name:15}: F1={metrics[f1_key]:.4f}\n"
        report += "\n"
        
        # Uncertainty Metrics
        report += "UNCERTAINTY QUANTIFICATION:\n"
        report += "-" * 40 + "\n"
        report += f"Uncertainty Correlation: {metrics['uncertainty_correlation']:.4f}\n"
        report += f"Uncertainty (Correct): {metrics['uncertainty_correct']:.4f}\n"
        report += f"Uncertainty (Incorrect): {metrics['uncertainty_incorrect']:.4f}\n"
        report += f"Uncertainty Ratio: {metrics['uncertainty_ratio']:.4f}\n\n"
        
        # Calibration Metrics
        report += "CALIBRATION METRICS:\n"
        report += "-" * 40 + "\n"
        report += f"Expected Calibration Error: {metrics['ece']:.4f}\n"
        report += f"Maximum Calibration Error: {metrics['mce']:.4f}\n"
        report += f"Calibration Correlation: {metrics['calibration_correlation']:.4f}\n\n"
        
        # Medical Metrics
        report += "MEDICAL-SPECIFIC METRICS:\n"
        report += "-" * 40 + "\n"
        for class_name in self.class_names:
            avg_precision_key = f"avg_precision_{class_name}"
            if avg_precision_key in metrics:
                report += f"{class_name:15}: Avg Precision={metrics[avg_precision_key]:.4f}\n"
        report += "\n"
        
        report += "=" * 80 + "\n"
        
        return report


class UncertaintyAnalyzer:
    """Analyzer for uncertainty quantification in medical predictions.
    
    Provides tools for analyzing and visualizing uncertainty patterns
    in health monitoring predictions.
    """
    
    def __init__(self) -> None:
        """Initialize the uncertainty analyzer."""
        pass
    
    def analyze_uncertainty_patterns(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty: torch.Tensor,
        metadata: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Analyze uncertainty patterns in predictions.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth labels.
            uncertainty: Uncertainty estimates.
            metadata: Optional metadata for analysis.
            
        Returns:
            Dictionary containing uncertainty analysis results.
        """
        analysis = {}
        
        # Convert to numpy
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        uncertainty_np = uncertainty.cpu().numpy()
        
        # Overall uncertainty statistics
        analysis["uncertainty_stats"] = {
            "mean": np.mean(uncertainty_np),
            "std": np.std(uncertainty_np),
            "min": np.min(uncertainty_np),
            "max": np.max(uncertainty_np),
            "median": np.median(uncertainty_np),
        }
        
        # Uncertainty by prediction correctness
        correct_mask = preds_np == targets_np
        analysis["uncertainty_by_correctness"] = {
            "correct_mean": np.mean(uncertainty_np[correct_mask]) if np.any(correct_mask) else 0.0,
            "incorrect_mean": np.mean(uncertainty_np[~correct_mask]) if np.any(~correct_mask) else 0.0,
            "correct_std": np.std(uncertainty_np[correct_mask]) if np.any(correct_mask) else 0.0,
            "incorrect_std": np.std(uncertainty_np[~correct_mask]) if np.any(~correct_mask) else 0.0,
        }
        
        # Uncertainty by predicted class
        unique_preds = np.unique(preds_np)
        analysis["uncertainty_by_class"] = {}
        for pred_class in unique_preds:
            class_mask = preds_np == pred_class
            analysis["uncertainty_by_class"][f"class_{pred_class}"] = {
                "mean": np.mean(uncertainty_np[class_mask]),
                "std": np.std(uncertainty_np[class_mask]),
                "count": np.sum(class_mask),
            }
        
        # High uncertainty cases
        high_uncertainty_threshold = np.percentile(uncertainty_np, 90)
        high_uncertainty_mask = uncertainty_np > high_uncertainty_threshold
        analysis["high_uncertainty_cases"] = {
            "count": np.sum(high_uncertainty_mask),
            "percentage": np.mean(high_uncertainty_mask) * 100,
            "accuracy": accuracy_score(targets_np[high_uncertainty_mask], preds_np[high_uncertainty_mask]) if np.any(high_uncertainty_mask) else 0.0,
        }
        
        return analysis

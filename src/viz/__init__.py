"""Visualization utilities for multi-modal health monitoring.

This module provides visualization tools for health monitoring results,
including attention maps, uncertainty plots, and medical data visualization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Set style for medical visualizations
plt.style.use('default')
sns.set_palette("husl")


class HealthMonitoringVisualizer:
    """Visualizer for multi-modal health monitoring results.
    
    Provides comprehensive visualization tools for health monitoring
    predictions, uncertainty, and attention patterns.
    """
    
    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the visualizer.
        
        Args:
            class_names: List of class names for visualization.
            output_dir: Directory to save visualizations.
        """
        self.class_names = class_names or [
            "diabetes",
            "heart_disease",
            "fatigue",
            "stress", 
            "healthy",
            "overweight"
        ]
        self.output_dir = Path(output_dir) if output_dir else Path("assets/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_confusion_matrix(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        title: str = "Confusion Matrix",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Plot confusion matrix for health monitoring predictions.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth labels.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Convert to numpy
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Compute confusion matrix
        cm = np.array([[0 for _ in range(len(self.class_names))] for _ in range(len(self.class_names))])
        for i in range(len(preds_np)):
            cm[targets_np[i]][preds_np[i]] += 1
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Health Condition', fontsize=12)
        ax.set_ylabel('True Health Condition', fontsize=12)
        
        # Add disclaimer
        fig.text(0.5, 0.02, 
                'DISCLAIMER: For research/educational purposes only. NOT for clinical use.',
                ha='center', fontsize=8, style='italic', color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_uncertainty_distribution(
        self,
        uncertainty: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        title: str = "Uncertainty Distribution",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Plot uncertainty distribution by prediction correctness.
        
        Args:
            uncertainty: Uncertainty estimates.
            predictions: Model predictions.
            targets: Ground truth labels.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Convert to numpy
        uncertainty_np = uncertainty.cpu().numpy()
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Overall uncertainty distribution
        ax1.hist(uncertainty_np, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Uncertainty Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Overall Uncertainty Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Uncertainty by correctness
        correct_mask = preds_np == targets_np
        correct_uncertainty = uncertainty_np[correct_mask]
        incorrect_uncertainty = uncertainty_np[~correct_mask]
        
        ax2.hist(correct_uncertainty, bins=20, alpha=0.7, label='Correct Predictions', color='green')
        ax2.hist(incorrect_uncertainty, bins=20, alpha=0.7, label='Incorrect Predictions', color='red')
        ax2.set_xlabel('Uncertainty Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Uncertainty by Prediction Correctness')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        
        # Add disclaimer
        fig.text(0.5, 0.02, 
                'DISCLAIMER: For research/educational purposes only. NOT for clinical use.',
                ha='center', fontsize=8, style='italic', color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_calibration_curve(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        calibration: torch.Tensor,
        title: str = "Calibration Curve",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Plot calibration curve for health monitoring predictions.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth labels.
            calibration: Calibration scores.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Convert to numpy
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        calibration_np = calibration.cpu().numpy()
        
        # Compute accuracy for different calibration bins
        bin_edges = np.linspace(0, 1, 11)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        accuracies = []
        counts = []
        
        for i in range(len(bin_edges) - 1):
            mask = (calibration_np >= bin_edges[i]) & (calibration_np < bin_edges[i + 1])
            if np.any(mask):
                bin_accuracy = np.mean(preds_np[mask] == targets_np[mask])
                bin_count = np.sum(mask)
                accuracies.append(bin_accuracy)
                counts.append(bin_count)
            else:
                accuracies.append(0)
                counts.append(0)
        
        # Plot calibration curve
        ax.plot(bin_centers, accuracies, 'o-', linewidth=2, markersize=8, label='Model Calibration')
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
        
        # Add count information
        for i, (center, acc, count) in enumerate(zip(bin_centers, accuracies, counts)):
            if count > 0:
                ax.annotate(f'n={count}', (center, acc), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Calibration Score')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add disclaimer
        fig.text(0.5, 0.02, 
                'DISCLAIMER: For research/educational purposes only. NOT for clinical use.',
                ha='center', fontsize=8, style='italic', color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_class_performance(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: torch.Tensor,
        title: str = "Per-Class Performance",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Plot per-class performance metrics.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth labels.
            probabilities: Prediction probabilities.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Convert to numpy
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        probs_np = probabilities.cpu().numpy()
        
        # Per-class accuracy
        class_accuracies = []
        for i in range(len(self.class_names)):
            class_mask = targets_np == i
            if np.any(class_mask):
                class_acc = np.mean(preds_np[class_mask] == targets_np[class_mask])
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0)
        
        ax1.bar(self.class_names, class_accuracies, color='skyblue', alpha=0.7)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Per-Class Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # Per-class F1 scores
        from sklearn.metrics import f1_score
        f1_scores = f1_score(targets_np, preds_np, average=None)
        
        ax2.bar(self.class_names, f1_scores, color='lightgreen', alpha=0.7)
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Per-Class F1 Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Confidence distribution by class
        class_confidences = []
        for i in range(len(self.class_names)):
            class_mask = preds_np == i
            if np.any(class_mask):
                class_conf = np.mean(probs_np[class_mask, i])
                class_confidences.append(class_conf)
            else:
                class_confidences.append(0)
        
        ax3.bar(self.class_names, class_confidences, color='orange', alpha=0.7)
        ax3.set_ylabel('Average Confidence')
        ax3.set_title('Per-Class Confidence')
        ax3.tick_params(axis='x', rotation=45)
        
        # Sample counts per class
        class_counts = [np.sum(targets_np == i) for i in range(len(self.class_names))]
        
        ax4.bar(self.class_names, class_counts, color='pink', alpha=0.7)
        ax4.set_ylabel('Sample Count')
        ax4.set_title('Per-Class Sample Count')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        
        # Add disclaimer
        fig.text(0.5, 0.02, 
                'DISCLAIMER: For research/educational purposes only. NOT for clinical use.',
                ha='center', fontsize=8, style='italic', color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        modality_names: List[str] = None,
        title: str = "Multi-Modal Attention Heatmap",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Figure:
        """Plot attention heatmap for multi-modal fusion.
        
        Args:
            attention_weights: Attention weights [batch_size, num_modalities, num_modalities]
            modality_names: Names of modalities.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Convert to numpy and average across batch
        attention_np = attention_weights.cpu().numpy()
        avg_attention = np.mean(attention_np, axis=0)
        
        if modality_names is None:
            modality_names = ['Text', 'Audio', 'Image']
        
        # Create heatmap
        sns.heatmap(
            avg_attention,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            xticklabels=modality_names,
            yticklabels=modality_names,
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Query Modality')
        ax.set_ylabel('Key Modality')
        
        # Add disclaimer
        fig.text(0.5, 0.02, 
                'DISCLAIMER: For research/educational purposes only. NOT for clinical use.',
                ha='center', fontsize=8, style='italic', color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_evaluation_dashboard(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: torch.Tensor,
        uncertainty: torch.Tensor,
        calibration: torch.Tensor,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Create a comprehensive evaluation dashboard.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth labels.
            probabilities: Prediction probabilities.
            uncertainty: Uncertainty estimates.
            calibration: Calibration scores.
            save_dir: Directory to save visualizations.
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.output_dir
        
        logger.info("Creating evaluation dashboard...")
        
        # Create all visualizations
        self.plot_confusion_matrix(
            predictions, targets,
            save_path=save_dir / "confusion_matrix.png"
        )
        
        self.plot_uncertainty_distribution(
            uncertainty, predictions, targets,
            save_path=save_dir / "uncertainty_distribution.png"
        )
        
        self.plot_calibration_curve(
            predictions, targets, calibration,
            save_path=save_dir / "calibration_curve.png"
        )
        
        self.plot_class_performance(
            predictions, targets, probabilities,
            save_path=save_dir / "class_performance.png"
        )
        
        logger.info(f"Evaluation dashboard saved to {save_dir}")
        
        # Create summary figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 
               'Multi-Modal Health Monitoring\nEvaluation Dashboard\n\n'
               'All visualizations have been generated and saved.\n'
               'Please refer to individual plot files for detailed analysis.\n\n'
               'IMPORTANT: This system is for research/educational purposes only.\n'
               'NOT intended for clinical diagnosis or medical decision-making.',
               ha='center', va='center', fontsize=16,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.title('Evaluation Dashboard Summary', fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / "dashboard_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Evaluation dashboard completed!")

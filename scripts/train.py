"""Main training script for multi-modal health monitoring.

This script provides training, evaluation, and inference capabilities for the
multi-modal health monitoring system.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.data import DataProcessor, HealthMonitoringDataset
from src.eval import MedicalEvaluationMetrics, UncertaintyAnalyzer
from src.models import HealthMonitoringModel, UncertaintyCalibrationLoss
from src.viz import HealthMonitoringVisualizer
from src import DEVICE, set_seed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthMonitoringTrainer:
    """Trainer for multi-modal health monitoring model.
    
    Provides comprehensive training, validation, and evaluation capabilities
    with medical-grade uncertainty quantification.
    """
    
    def __init__(
        self,
        model: HealthMonitoringModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
    ) -> None:
        """Initialize the trainer.
        
        Args:
            model: Health monitoring model.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            test_loader: Test data loader.
            config: Training configuration.
        """
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        # Loss function
        self.criterion = UncertaintyCalibrationLoss(
            alpha=config.get("uncertainty_weight", 0.1),
            beta=config.get("calibration_weight", 0.1),
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 1e-5),
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * config.get("num_epochs", 10)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps,
        )
        
        # Evaluation metrics
        self.evaluator = MedicalEvaluationMetrics()
        self.uncertainty_analyzer = UncertaintyAnalyzer()
        self.visualizer = HealthMonitoringVisualizer()
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
        }
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number.
            
        Returns:
            Dictionary containing training metrics.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            # Move to device
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                text_input_ids=batch["text_input_ids"],
                text_attention_mask=batch["text_attention_mask"],
                audio_features=batch["audio_features"],
                image_features=batch["image_features"],
            )
            
            # Compute loss
            loss_dict = self.criterion(
                logits=outputs["logits"],
                uncertainty=outputs["uncertainty"],
                calibration=outputs["calibration"],
                targets=batch["labels"],
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict["total_loss"].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss_dict["total_loss"].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_dict['total_loss'].item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })
        
        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            epoch: Current epoch number.
            
        Returns:
            Dictionary containing validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_uncertainty = []
        all_calibration = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                # Move to device
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    text_input_ids=batch["text_input_ids"],
                    text_attention_mask=batch["text_attention_mask"],
                    audio_features=batch["audio_features"],
                    image_features=batch["image_features"],
                )
                
                # Compute loss
                loss_dict = self.criterion(
                    logits=outputs["logits"],
                    uncertainty=outputs["uncertainty"],
                    calibration=outputs["calibration"],
                    targets=batch["labels"],
                )
                
                # Collect predictions
                predictions = outputs["logits"].argmax(dim=-1)
                all_predictions.append(predictions)
                all_targets.append(batch["labels"])
                all_probabilities.append(outputs["probabilities"])
                all_uncertainty.append(outputs["uncertainty"])
                all_calibration.append(outputs["calibration"])
                
                total_loss += loss_dict["total_loss"].item()
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        all_probabilities = torch.cat(all_probabilities)
        all_uncertainty = torch.cat(all_uncertainty)
        all_calibration = torch.cat(all_calibration)
        
        # Compute metrics
        metrics = self.evaluator.compute_metrics(
            all_predictions, all_targets, all_probabilities, 
            all_uncertainty, all_calibration
        )
        
        metrics["val_loss"] = total_loss / len(self.val_loader)
        
        return metrics
    
    def train(self) -> None:
        """Train the model."""
        logger.info("Starting training...")
        logger.info("IMPORTANT: This system is for research/educational purposes only. NOT for clinical use.")
        
        best_val_f1 = 0.0
        patience = self.config.get("patience", 5)
        patience_counter = 0
        
        for epoch in range(self.config.get("num_epochs", 10)):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate(epoch)
            
            # Update history
            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1_macro"])
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}:")
            logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"  Val F1: {val_metrics['f1_macro']:.4f}")
            logger.info(f"  Val AUC: {val_metrics['auc_macro']:.4f}")
            
            # Early stopping
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("Training completed!")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on test set.
        
        Returns:
            Dictionary containing test metrics.
        """
        logger.info("Evaluating on test set...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_uncertainty = []
        all_calibration = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            for batch in pbar:
                # Move to device
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    text_input_ids=batch["text_input_ids"],
                    text_attention_mask=batch["text_attention_mask"],
                    audio_features=batch["audio_features"],
                    image_features=batch["image_features"],
                )
                
                # Collect predictions
                predictions = outputs["logits"].argmax(dim=-1)
                all_predictions.append(predictions)
                all_targets.append(batch["labels"])
                all_probabilities.append(outputs["probabilities"])
                all_uncertainty.append(outputs["uncertainty"])
                all_calibration.append(outputs["calibration"])
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        all_probabilities = torch.cat(all_probabilities)
        all_uncertainty = torch.cat(all_uncertainty)
        all_calibration = torch.cat(all_calibration)
        
        # Compute comprehensive metrics
        metrics = self.evaluator.compute_metrics(
            all_predictions, all_targets, all_probabilities,
            all_uncertainty, all_calibration
        )
        
        # Generate evaluation report
        report = self.evaluator.generate_report(
            all_predictions, all_targets, all_probabilities,
            all_uncertainty, all_calibration
        )
        
        logger.info(report)
        
        # Create visualizations
        self.visualizer.create_evaluation_dashboard(
            all_predictions, all_targets, all_probabilities,
            all_uncertainty, all_calibration,
            save_dir="assets/evaluation"
        )
        
        # Analyze uncertainty patterns
        uncertainty_analysis = self.uncertainty_analyzer.analyze_uncertainty_patterns(
            all_predictions, all_targets, all_uncertainty
        )
        
        logger.info("Uncertainty Analysis:")
        logger.info(f"  Mean Uncertainty: {uncertainty_analysis['uncertainty_stats']['mean']:.4f}")
        logger.info(f"  Uncertainty Correlation: {uncertainty_analysis['uncertainty_correlation']:.4f}")
        logger.info(f"  High Uncertainty Cases: {uncertainty_analysis['high_uncertainty_cases']['percentage']:.1f}%")
        
        return metrics
    
    def save_checkpoint(
        self, 
        epoch: int, 
        metrics: Dict[str, float], 
        is_best: bool = False
    ) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch.
            metrics: Validation metrics.
            is_best: Whether this is the best model.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model at epoch {epoch+1}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Multi-Modal Health Monitoring Model")
    parser.add_argument("--config", type=str, default="configs/train/default.yaml", 
                       help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="data", 
                       help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="outputs", 
                       help="Path to output directory")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        logger.warning(f"Config file {config_path} not found. Using default config.")
        config = {
            "num_epochs": 10,
            "batch_size": 8,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "uncertainty_weight": 0.1,
            "calibration_weight": 0.1,
            "patience": 5,
        }
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data processor
    data_processor = DataProcessor()
    
    # Create data loaders
    train_loader, val_loader, test_loader = data_processor.create_data_loaders(
        data_dir=args.data_dir,
        batch_size=config.get("batch_size", 8),
        num_workers=2,  # Reduced for compatibility
    )
    
    # Create model
    model = HealthMonitoringModel(
        num_classes=6,
        text_model_name="distilbert-base-uncased",
        image_model_name="openai/clip-vit-base-patch32",
        fusion_hidden_dim=512,
        dropout=0.1,
    )
    
    # Create trainer
    trainer = HealthMonitoringTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
    )
    
    # Train model
    trainer.train()
    
    # Evaluate model
    test_metrics = trainer.evaluate()
    
    # Save final results
    results = {
        "test_metrics": test_metrics,
        "config": config,
        "history": trainer.history,
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()

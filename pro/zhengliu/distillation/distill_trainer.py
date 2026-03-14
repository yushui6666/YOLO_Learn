"""
Distillation Trainer for YOLO Knowledge Distillation

Handles the training loop for knowledge distillation:
- Teacher model inference (frozen)
- Student model training
- Loss computation and backpropagation
- Checkpoint saving and logging
"""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from pathlib import Path

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

from utils import (
    AverageMeter, ProgressMeter, save_checkpoint,
    get_device, create_output_directories, print_header, print_subheader
)
from distill_loss import DistillationLoss, create_distillation_criterion


class DistillationTrainer:
    """
    Trainer for YOLO knowledge distillation.
    
    Manages:
    - Teacher and student model loading
    - Training loop with distillation
    - Logging and checkpointing
    """
    
    def __init__(
        self,
        config: Dict,
        device: Optional[torch.device] = None
    ):
        """
        Initialize distillation trainer.
        
        Args:
            config: Configuration dictionary
            device: Torch device (optional, will be inferred from config)
        """
        self.config = config
        self.device = device or get_device(config.get('device', '0'))
        
        # Training state
        self.epoch = 0
        self.best_metric = 0.0
        self.train_losses = []
        self.val_metrics = []
        
        # Models
        self.teacher_model = None
        self.student_model = None
        
        # Loss and optimizer
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        
        # Output directories
        self.output_dirs = None
    
    def load_models(
        self,
        teacher_path: str,
        student_path: Optional[str] = None,
        student_arch: str = "yolov8x.pt"
    ):
        """
        Load teacher and student models.
        
        Args:
            teacher_path: Path to teacher model weights
            student_path: Path to student model weights (optional)
            student_arch: Student architecture to use if no weights provided
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics package required. Install with: pip install ultralytics")
        
        print_subheader("Loading Models")
        
        # Load teacher model (frozen)
        print(f"Loading teacher model from: {teacher_path}")
        self.teacher_model = YOLO(teacher_path)
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        
        # Freeze teacher parameters
        for param in self.teacher_model.model.parameters():
            param.requires_grad = False
        
        print(f"Teacher model loaded: {type(self.teacher_model)}")
        
        # Load student model
        if student_path and os.path.exists(student_path):
            print(f"Loading student model from: {student_path}")
            self.student_model = YOLO(student_path)
        else:
            print(f"Initializing student model with architecture: {student_arch}")
            self.student_model = YOLO(student_arch)
        
        self.student_model.to(self.device)
        self.student_model.train()
        
        print(f"Student model loaded: {type(self.student_model)}")
        
        # Get number of classes from teacher model
        self.num_classes = self._get_num_classes()
        print(f"Number of classes: {self.num_classes}")
    
    def _get_num_classes(self) -> int:
        """Get number of classes from teacher model."""
        # Try to get from model's names attribute
        if hasattr(self.teacher_model, 'names') and self.teacher_model.names:
            return len(self.teacher_model.names)
        return 80  # Default COCO classes
    
    def setup_training(self):
        """Setup optimizer, scheduler, and loss function."""
        print_subheader("Setting Up Training")
        
        # Get student model parameters
        params = [p for p in self.student_model.model.parameters() if p.requires_grad]
        
        train_cfg = self.config.get('train', {})
        
        # Optimizer
        self.optimizer = AdamW(
            params,
            lr=train_cfg.get('lr0', 0.01),
            weight_decay=train_cfg.get('weight_decay', 0.0005),
            betas=(train_cfg.get('momentum', 0.937), 0.999)
        )
        print(f"Optimizer: AdamW with lr={train_cfg.get('lr0', 0.01)}")
        
        # Learning rate scheduler (cosine annealing)
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg.get('epochs', 50),
            eta_min=train_cfg.get('lr0', 0.01) * train_cfg.get('lrf', 0.1)
        )
        print(f"Scheduler: CosineAnnealingLR with T_max={train_cfg.get('epochs', 50)}")
        
        # Distillation loss
        distill_cfg = self.config.get('distill', {})
        self.criterion = create_distillation_criterion(
            temperature=distill_cfg.get('temperature', 4.0),
            alpha=distill_cfg.get('alpha', 0.7),
            beta=distill_cfg.get('beta', 0.3)
        )
        print(f"Distillation: T={distill_cfg.get('temperature', 4.0)}, alpha={distill_cfg.get('alpha', 0.7)}, beta={distill_cfg.get('beta', 0.3)}")
    
    def prepare_outputs(
        self,
        model_outputs
    ) -> Dict:
        """
        Convert YOLO model outputs to standardized format.
        
        Args:
            model_outputs: Raw outputs from YOLO model forward pass
        
        Returns:
            Dictionary with cls_pred, box_pred, obj_pred
        """
        # YOLOv8 outputs format depends on the model configuration
        # Typically returns a list/tuple of tensors for different scales
        
        if isinstance(model_outputs, (list, tuple)):
            # Multi-scale outputs
            # Concatenate predictions from all scales
            cls_preds = []
            box_preds = []
            obj_preds = []
            
            for scale_out in model_outputs:
                if isinstance(scale_out, dict):
                    cls_preds.append(scale_out.get('cls', scale_out.get('cls_pred', None)))
                    box_preds.append(scale_out.get('box', scale_out.get('box_pred', None)))
                    obj_preds.append(scale_out.get('obj', scale_out.get('obj_pred', None)))
                elif isinstance(scale_out, torch.Tensor):
                    # Assume tensor format: (B, anchors, H, W)
                    # This is simplified - actual parsing depends on YOLO version
                    pass
            
            # For now, return a simplified output structure
            # This may need adjustment based on actual YOLO version
            return {
                'cls_pred': model_outputs[0] if len(model_outputs) > 0 else None,
                'box_pred': model_outputs[1] if len(model_outputs) > 1 else None,
                'obj_pred': model_outputs[2] if len(model_outputs) > 2 else None
            }
        elif isinstance(model_outputs, torch.Tensor):
            # Single tensor output
            return {
                'cls_pred': model_outputs,
                'box_pred': None,
                'obj_pred': None
            }
        elif isinstance(model_outputs, dict):
            return model_outputs
        else:
            return {
                'cls_pred': None,
                'box_pred': None,
                'obj_pred': None
            }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Dictionary of average losses
        """
        self.student_model.train()
        self.teacher_model.eval()
        
        # Metrics
        meters = {
            'loss': AverageMeter('Loss'),
            'distill_loss': AverageMeter('Distill'),
            'gt_loss': AverageMeter('GT'),
            'cls_loss': AverageMeter('Cls'),
            'box_loss': AverageMeter('Box'),
        }
        
        progress = ProgressMeter(
            len(train_loader),
            list(meters.values()),
            prefix=f"Epoch [{epoch}]"
        )
        
        log_freq = self.config.get('log', {}).get('print_freq', 10)
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['images'].to(self.device)
            target_boxes = batch['boxes'].to(self.device)
            target_labels = batch['labels'].to(self.device)
            
            # Teacher inference (no grad)
            with torch.no_grad():
                teacher_outputs_raw = self.teacher_model.model(images)
                teacher_outputs = self.prepare_outputs(teacher_outputs_raw)
            
            # Student inference
            student_outputs_raw = self.student_model.model(images)
            student_outputs = self.prepare_outputs(student_outputs_raw)
            
            # Prepare ground truth targets
            gt_targets = {
                'box_gt': target_boxes,
                'cls_gt': target_labels,
                'obj_gt': (target_labels != -1).float()  # Object mask
            }
            
            # Compute loss
            loss, loss_dict = self.criterion(
                student_outputs,
                teacher_outputs,
                gt_targets
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update meters
            meters['loss'].update(loss.item(), images.size(0))
            meters['distill_loss'].update(loss_dict.get('distill_loss', 0))
            meters['gt_loss'].update(loss_dict.get('gt_loss', 0))
            meters['cls_loss'].update(loss_dict.get('cls_loss_distill', 0))
            meters['box_loss'].update(loss_dict.get('box_loss_distill', 0))
            
            # Progress display
            if (batch_idx + 1) % log_freq == 0:
                progress.display(batch_idx + 1)
        
        # Return average losses
        return {name: meter.avg for name, meter in meters.items()}
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict:
        """
        Validate the student model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Dictionary of validation metrics
        """
        self.student_model.eval()
        self.teacher_model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in val_loader:
            images = batch['images'].to(self.device)
            target_boxes = batch['boxes'].to(self.device)
            target_labels = batch['labels'].to(self.device)
            
            # Teacher inference
            teacher_outputs_raw = self.teacher_model.model(images)
            teacher_outputs = self.prepare_outputs(teacher_outputs_raw)
            
            # Student inference
            student_outputs_raw = self.student_model.model(images)
            student_outputs = self.prepare_outputs(student_outputs_raw)
            
            # Compute loss
            gt_targets = {
                'box_gt': target_boxes,
                'cls_gt': target_labels,
                'obj_gt': (target_labels != -1).float()
            }
            
            loss, loss_dict = self.criterion(
                student_outputs,
                teacher_outputs,
                gt_targets
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return {
            'val_loss': avg_loss,
            'val_metric': avg_loss  # Using loss as metric for now
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        print_header("Starting Distillation Training")
        
        train_cfg = self.config.get('train', {})
        num_epochs = train_cfg.get('epochs', 50)
        checkpoint_cfg = self.config.get('checkpoint', {})
        
        # Create output directories
        output_dir = self.config.get('output_dir', 'outputs/distillation')
        self.output_dirs = create_output_directories(output_dir)
        
        print(f"Output directory: {self.output_dirs['checkpoint']}")
        print(f"Training for {num_epochs} epochs")
        print(f"Batch size: {train_cfg.get('batch_size', 8)}")
        print(f"Image size: {train_cfg.get('imgsz', 640)}")
        print()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_losses)
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.val_metrics.append(val_metrics)
                
                # Check for best model
                is_best = val_metrics.get('val_metric', float('inf')) < self.best_metric or self.best_metric == 0
                if is_best:
                    self.best_metric = val_metrics.get('val_metric', self.best_metric)
            else:
                val_metrics = {}
                is_best = False
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_losses['loss']:.4f}")
            if val_metrics:
                print(f"  Val Loss: {val_metrics.get('val_loss', 'N/A'):.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Update scheduler
            self.scheduler.step()
            
            # Save checkpoint
            save_freq = checkpoint_cfg.get('save_freq', 5)
            if (epoch + 1) % save_freq == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        print_header("Training Complete")
        print(f"Best validation metric: {self.best_metric:.4f}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        state = {
            'epoch': epoch,
            'student_state_dict': self.student_model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
            'config': self.config
        }
        
        filename = f'checkpoint_epoch_{epoch + 1}.pth'
        save_checkpoint(state, is_best, self.output_dirs['checkpoint'], filename)
        
        # Also save the student model in YOLO format
        student_path = os.path.join(self.output_dirs['checkpoint'], f'student_epoch_{epoch + 1}.pt')
        self.student_model.save(student_path)
        
        print(f"  Checkpoint saved: {filename}")
        if is_best:
            print(f"  New best model saved!")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.student_model.model.load_state_dict(checkpoint['student_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_metric = checkpoint['best_metric']
        self.train_losses = checkpoint['train_losses']
        self.val_metrics = checkpoint['val_metrics']
        
        print(f"Loaded checkpoint from epoch {self.epoch + 1}")


def create_trainer(config: Dict) -> DistillationTrainer:
    """
    Factory function to create a distillation trainer.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        DistillationTrainer instance
    """
    device = get_device(config.get('device', '0'))
    return DistillationTrainer(config, device)
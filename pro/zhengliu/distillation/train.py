#!/usr/bin/env python3
"""
Main Training Script for YOLO Knowledge Distillation

Usage:
    python train.py --config config.yaml
    python train.py --teacher teacher_model/yolo26x.pt --student student_model/yolov8_resnet101.pt

Example:
    python train.py --config config.yaml --epochs 50 --batch-size 8
"""
import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import torch

# Import local modules
from utils import (
    load_yaml_config, get_device, load_coco_classes,
    print_header, print_subheader
)
from dataset import create_dataloader
from distill_trainer import DistillationTrainer, create_trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='YOLO Knowledge Distillation Training'
    )
    
    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    # Model paths (override config)
    parser.add_argument(
        '--teacher', '-t',
        type=str,
        default=None,
        help='Path to teacher model weights (overrides config)'
    )
    parser.add_argument(
        '--student', '-s',
        type=str,
        default=None,
        help='Path to student model weights (overrides config)'
    )
    parser.add_argument(
        '--student-arch',
        type=str,
        default=None,
        help='Student architecture (e.g., yolov8n.pt, yolov8s.pt) (overrides config)'
    )
    
    # Dataset paths (override config)
    parser.add_argument(
        '--train-images',
        type=str,
        default=None,
        help='Path to training images directory'
    )
    parser.add_argument(
        '--train-labels',
        type=str,
        default=None,
        help='Path to training labels file'
    )
    parser.add_argument(
        '--val-images',
        type=str,
        default=None,
        help='Path to validation images directory'
    )
    parser.add_argument(
        '--val-labels',
        type=str,
        default=None,
        help='Path to validation labels file'
    )
    
    # Training hyperparameters (override config)
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=None,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=None,
        help='Batch size'
    )
    parser.add_argument(
        '--imgsz', '-i',
        type=int,
        default=None,
        help='Image size (default: 640)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Initial learning rate'
    )
    
    # Distillation hyperparameters (override config)
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='Distillation temperature (default: 4.0)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=None,
        help='Distillation loss weight (default: 0.7)'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=None,
        help='Ground truth loss weight (default: 0.3)'
    )
    
    # Device and output
    parser.add_argument(
        '--device', '-d',
        type=str,
        default=None,
        help='Device to use (cuda/cpu/device_id)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for checkpoints and logs'
    )
    
    # Resume training
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    # Debug mode
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (print more information)'
    )
    
    return parser.parse_args()


def update_config(config: dict, args: argparse.Namespace) -> dict:
    """Update configuration with command line arguments."""
    
    # Model paths
    if args.teacher is not None:
        config['teacher_model'] = args.teacher
    if args.student is not None:
        config['student_model'] = args.student
    if args.student_arch is not None:
        config['student_arch'] = args.student_arch
    
    # Dataset paths
    if args.train_images is not None:
        config['dataset']['train_images'] = args.train_images
    if args.train_labels is not None:
        config['dataset']['train_labels'] = args.train_labels
    if args.val_images is not None:
        config['dataset']['val_images'] = args.val_images
    if args.val_labels is not None:
        config['dataset']['val_labels'] = args.val_labels
    
    # Training hyperparameters
    if args.epochs is not None:
        config['train']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['train']['batch_size'] = args.batch_size
    if args.imgsz is not None:
        config['train']['imgsz'] = args.imgsz
    if args.lr is not None:
        config['train']['lr0'] = args.lr
    
    # Distillation hyperparameters
    if args.temperature is not None:
        config['distill']['temperature'] = args.temperature
    if args.alpha is not None:
        config['distill']['alpha'] = args.alpha
    if args.beta is not None:
        config['distill']['beta'] = args.beta
    
    # Device and output
    if args.device is not None:
        config['device'] = args.device
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    
    return config


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    print_header("YOLO Knowledge Distillation")
    
    if os.path.exists(args.config):
        print(f"Loading configuration from: {args.config}")
        config = load_yaml_config(args.config)
    else:
        print(f"Config file not found: {args.config}")
        print("Using default configuration")
        config = {
            'teacher_model': 'teacher_model/yolo26x.pt',
            'student_model': None,
            'student_arch': 'yolov8x.pt',
            'output_dir': 'outputs/distillation',
            'dataset': {
                'train_images': 'dataset/coco/train2017/image',
                'train_labels': 'dataset/coco/train2017/annotations/instances_train2017.json',
                'val_images': 'dataset/coco/val2017/image',
                'val_labels': 'dataset/coco/val2017/annotation/instances_val2017.json',
            },
            'train': {
                'epochs': 50,
                'batch_size': 8,
                'imgsz': 640,
                'lr0': 0.01,
                'lrf': 0.1,
                'momentum': 0.937,
                'weight_decay': 0.0005,
            },
            'distill': {
                'temperature': 4.0,
                'alpha': 0.7,
                'beta': 0.3,
            },
            'device': '0',
        }
    
    # Update config with command line arguments
    config = update_config(config, args)
    
    # Print configuration
    print_subheader("Configuration")
    print(f"Teacher model: {config['teacher_model']}")
    print(f"Student model: {config.get('student_model', 'None')}")
    print(f"Student architecture: {config.get('student_arch', 'yolov8x.pt')}")
    print(f"Device: {config['device']}")
    print(f"Output directory: {config['output_dir']}")
    
    # Get device
    device = get_device(config['device'])
    print(f"\nUsing device: {device}")
    
    # Create trainer
    print_subheader("Initializing Trainer")
    trainer = create_trainer(config)
    
    # Load models
    trainer.load_models(
        teacher_path=config['teacher_model'],
        student_path=config.get('student_model'),
        student_arch=config.get('student_arch', 'yolov8x.pt')
    )
    
    # Setup training
    trainer.setup_training()
    
    # Create data loaders
    print_subheader("Creating Data Loaders")
    
    dataset_cfg = config['dataset']
    train_cfg = config['train']
    
    # Check if dataset paths exist
    train_images = dataset_cfg['train_images']
    train_labels = dataset_cfg['train_labels']
    
    if not os.path.exists(train_images):
        print(f"Warning: Training images directory not found: {train_images}")
    if not os.path.exists(train_labels):
        print(f"Warning: Training labels file not found: {train_labels}")
    
    # Create training data loader
    train_loader = create_dataloader(
        image_dir=train_images,
        annotation_file=train_labels,
        batch_size=train_cfg['batch_size'],
        img_size=train_cfg['imgsz'],
        augment=True,
        num_workers=4,
        pin_memory=True
    )
    print(f"Training dataset: {len(train_loader.dataset)} images")
    
    # Create validation data loader (optional)
    val_images = dataset_cfg.get('val_images')
    val_labels = dataset_cfg.get('val_labels')
    
    val_loader = None
    if val_images and val_labels and os.path.exists(val_images) and os.path.exists(val_labels):
        val_loader = create_dataloader(
            image_dir=val_images,
            annotation_file=val_labels,
            batch_size=train_cfg['batch_size'],
            img_size=train_cfg['imgsz'],
            augment=False,
            num_workers=4,
            pin_memory=True
        )
        print(f"Validation dataset: {len(val_loader.dataset)} images")
    else:
        print("Validation dataset not found, training without validation")
    
    # Resume from checkpoint (optional)
    if args.resume:
        print_subheader("Resuming Training")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print_subheader("Starting Training")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    print_header("Training Finished")
    print(f"Student model saved to: {trainer.output_dirs['checkpoint']}")


if __name__ == '__main__':
    main()
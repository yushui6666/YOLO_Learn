"""
YOLOv8 Optuna 超参数调优脚本
支持两阶段训练：先搜索最佳超参数，再进行完整训练
"""

import os
import sys
import yaml
import json
import time
import random
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Optuna 导入
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, NopPruner
from optuna.exceptions import TrialPruned

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from models.yolov8 import create_model
from data.dataset import create_dataloader
from utils.loss import YOLOv8Loss
from utils.metrics import MetricsCalculator


def set_seed(seed: int = 42):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CosineLRLambda:
    """Cosine annealing with warmup 学习率调度器"""
    def __init__(self, warmup_epochs: int, epochs: int, base_lr: float):
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.base_lr = base_lr
        self.warmup_factor = 0.001
        
    def __call__(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            alpha = epoch / self.warmup_epochs
            return self.warmup_factor + alpha * (1.0 - self.warmup_factor)
        else:
            progress = (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))


class OptunaTrainer:
    """支持 Optuna 调优的训练器"""
    
    def __init__(
        self, 
        config: Dict, 
        trial: Optional[optuna.Trial] = None,
        output_dir: Optional[Path] = None
    ):
        self.config = config
        self.trial = trial
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() and not config['training'].get('use_cpu', False) else 'cpu')
        
        # 创建输出目录
        if output_dir is None:
            self.output_dir = Path(config['training'].get('save_dir', 'runs/train'))
        else:
            self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard 写入器（可选）
        self.writer = None
        
        # 训练参数
        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']
        self.num_workers = config['training']['num_workers']
        self.image_size = config['training']['image_size']
        
        # 评估参数
        self.eval_period = config['training'].get('eval_period', 5)
        
        # 早停参数
        self.early_stopping_patience = config['training'].get('early_stopping_patience', None)
        self.early_stopping_min_delta = config['training'].get('early_stopping_min_delta', 0.001)
        
        # 初始化
        self.best_map = 0.0
        self.start_epoch = 0
        self.no_improve_count = 0
        
        # 创建模型
        self.model = create_model(
            num_classes=config['model']['num_classes'],
            width_multiple=config['model']['width_multiple'],
            depth_multiple=config['model']['depth_multiple']
        )
        self.model = self.model.to(self.device)
        
        # 创建损失函数
        loss_cfg = config['loss']
        self.loss_fn = YOLOv8Loss(
            num_classes=config['model']['num_classes'],
            box_gain=loss_cfg['box_gain'],
            cls_gain=loss_cfg['cls_gain'],
            dfl_gain=loss_cfg['dfl_gain'],
            obj_gain=loss_cfg.get('obj_gain', 1.0),
            reg_max=loss_cfg.get('reg_max', 16),
            max_pos_per_gt=loss_cfg.get('max_pos_per_gt', 10),
            use_focal_loss=loss_cfg.get('use_focal_loss', False),
        )
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay'],
            betas=tuple(config['optimizer']['betas'])
        )
        
        # 创建学习率调度器
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=CosineLRLambda(
                warmup_epochs=config['scheduler']['warmup_epochs'],
                epochs=self.epochs,
                base_lr=config['optimizer']['lr']
            )
        )
        
        # 创建数据加载器
        pin_memory = config['training'].get('pin_memory', True)
        self.train_loader = create_dataloader(
            img_dir=config['dataset']['train'],
            ann_file=config['dataset']['annotations_train'],
            batch_size=self.batch_size,
            img_size=self.image_size,
            is_training=True,
            num_workers=self.num_workers,
            augmentation_config=config.get('augmentation', None),
            pin_memory=pin_memory
        )
        
        self.val_loader = create_dataloader(
            img_dir=config['dataset']['val'],
            ann_file=config['dataset']['annotations_val'],
            batch_size=self.batch_size,
            img_size=self.image_size,
            is_training=False,
            num_workers=self.num_workers,
            augmentation_config=None,
            pin_memory=pin_memory
        )
        
        # 创建评估器
        self.metrics_calculator = MetricsCalculator(
            num_classes=config['model']['num_classes'],
            iou_thresholds=config['evaluation']['iou_thresholds']
        )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        total_box_loss = 0.0
        total_cls_loss = 0.0
        total_dfl_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.epochs}', leave=False)
        for batch_idx, (images, targets, image_ids) in enumerate(pbar):
            images = images.to(self.device)
            targets = [torch.from_numpy(t).to(self.device) for t in targets]
            
            model_out = self.model(images, return_features=True)
            
            if isinstance(model_out, (tuple, list)):
                if len(model_out) == 2:
                    outputs, features = model_out
                elif len(model_out) == 3:
                    pred_cls, pred_dist, features = model_out
                    outputs = (pred_cls, pred_dist)
                else:
                    raise ValueError(f"Unexpected number of outputs from model: {len(model_out)}")
            else:
                raise TypeError("Model should return a tuple or list when return_features=True")
            
            loss_dict = self.loss_fn(
                outputs, targets,
                img_h=self.image_size, img_w=self.image_size,
                features=features
            )
            loss = loss_dict['total_loss']
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_box_loss += loss_dict['box_loss'].item()
            total_cls_loss += loss_dict['cls_loss'].item()
            total_dfl_loss += loss_dict['dfl_loss'].item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        num_batches = len(self.train_loader)
        return {
            'total_loss': total_loss / num_batches,
            'box_loss': total_box_loss / num_batches,
            'cls_loss': total_cls_loss / num_batches,
            'dfl_loss': total_dfl_loss / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        self.metrics_calculator.reset()
        
        inference_times = []
        
        for images, targets, image_ids in tqdm(self.val_loader, desc='Validation', leave=False):
            images = images.to(self.device)
            
            start_time = time.time()
            outputs = self.model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            predictions = self.model.decode_predictions(
                outputs, 
                img_h=self.image_size, 
                img_w=self.image_size, 
                conf_thres=self.config['inference']['conf_threshold']
            )
            
            predictions = [pred.cpu().numpy() for pred in predictions]
            
            pred_boxes = [pred[:, :4] for pred in predictions]
            pred_labels = [pred[:, 5].astype(np.int64) for pred in predictions]
            pred_scores = [pred[:, 4] for pred in predictions]
            
            gt_labels = [t[:, 0].astype(np.int64) for t in targets]
            gt_boxes = []
            for t in targets:
                boxes = np.zeros_like(t[:, 1:5])
                boxes[:, 0] = (t[:, 1] - t[:, 3] / 2) * self.image_size
                boxes[:, 1] = (t[:, 2] - t[:, 4] / 2) * self.image_size
                boxes[:, 2] = (t[:, 1] + t[:, 3] / 2) * self.image_size
                boxes[:, 3] = (t[:, 2] + t[:, 4] / 2) * self.image_size
                gt_boxes.append(boxes.astype(np.float32))
            
            self.metrics_calculator.update(
                pred_boxes, pred_labels, pred_scores,
                gt_boxes, gt_labels, image_ids
            )
        
        metrics = self.metrics_calculator.compute_metrics(
            conf_threshold=self.config['evaluation']['conf_threshold']
        )
        
        metrics['map50'] = metrics.get('mAP50', 0.0)
        metrics['map50_95'] = metrics.get('mAP50-95', 0.0)
        
        avg_inference_time = np.mean(inference_times)
        metrics['fps'] = self.batch_size / avg_inference_time
        
        return metrics
    
    def train(self) -> float:
        """完整训练流程，返回最佳 mAP@0.5:0.95"""
        for epoch in range(self.start_epoch, self.epochs):
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 定期验证
            if (epoch + 1) % self.eval_period == 0 or epoch == self.epochs - 1:
                val_metrics = self.validate()
                current_map = val_metrics['map50_95']
                
                # 打印进度
                print(f"Epoch {epoch + 1}/{self.epochs} - "
                      f"Loss: {train_metrics['total_loss']:.4f} - "
                      f"mAP@0.5: {val_metrics['map50']:.4f} - "
                      f"mAP@0.5:0.95: {current_map:.4f}")
                
                # 报告给 Optuna（用于剪枝）
                if self.trial is not None:
                    self.trial.report(current_map, epoch)
                    
                    # 检查是否应该剪枝
                    if self.trial.should_prune():
                        print(f"Trial pruned at epoch {epoch + 1}")
                        raise TrialPruned()
                
                # 更新最佳 mAP
                if current_map > self.best_map + self.early_stopping_min_delta:
                    self.best_map = current_map
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1
                
                # 早停检查
                if self.early_stopping_patience is not None:
                    if self.no_improve_count >= self.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch + 1} (no improvement for {self.no_improve_count} evaluations)")
                        break
        
        return self.best_map


def suggest_hyperparameters(trial: optuna.Trial, search_space: Dict) -> Dict:
    """根据搜索空间建议超参数"""
    params = {}
    
    # 训练参数
    if 'training' in search_space:
        for key, spec in search_space['training'].items():
            if spec['type'] == 'categorical':
                params[f'training_{key}'] = trial.suggest_categorical(f'training_{key}', spec['choices'])
            elif spec['type'] == 'uniform':
                params[f'training_{key}'] = trial.suggest_float(f'training_{key}', spec['low'], spec['high'])
            elif spec['type'] == 'loguniform':
                params[f'training_{key}'] = trial.suggest_float(f'training_{key}', spec['low'], spec['high'], log=True)
    
    # 优化器参数
    if 'optimizer' in search_space:
        for key, spec in search_space['optimizer'].items():
            if spec['type'] == 'loguniform':
                params[f'optimizer_{key}'] = trial.suggest_float(f'optimizer_{key}', spec['low'], spec['high'], log=True)
            elif spec['type'] == 'uniform':
                params[f'optimizer_{key}'] = trial.suggest_float(f'optimizer_{key}', spec['low'], spec['high'])
    
    # 损失函数权重
    if 'loss' in search_space:
        for key, spec in search_space['loss'].items():
            if spec['type'] == 'loguniform':
                params[f'loss_{key}'] = trial.suggest_float(f'loss_{key}', spec['low'], spec['high'], log=True)
            elif spec['type'] == 'uniform':
                params[f'loss_{key}'] = trial.suggest_float(f'loss_{key}', spec['low'], spec['high'])
    
    # 数据增强参数
    if 'augmentation' in search_space:
        for key, spec in search_space['augmentation'].items():
            if spec['type'] == 'uniform':
                params[f'aug_{key}'] = trial.suggest_float(f'aug_{key}', spec['low'], spec['high'])
            elif spec['type'] == 'categorical':
                params[f'aug_{key}'] = trial.suggest_categorical(f'aug_{key}', spec['choices'])
    
    # 模型参数
    if 'model' in search_space:
        for key, spec in search_space['model'].items():
            if spec['type'] == 'categorical':
                params[f'model_{key}'] = trial.suggest_categorical(f'model_{key}', spec['choices'])
            elif spec['type'] == 'uniform':
                params[f'model_{key}'] = trial.suggest_float(f'model_{key}', spec['low'], spec['high'])
    
    return params


def build_config(params: Dict, optuna_config: Dict) -> Dict:
    """根据超参数构建完整配置"""
    config = {}
    
    # 固定参数
    fixed = optuna_config['fixed_params']
    
    # 训练配置
    config['training'] = fixed['training'].copy()
    config['training']['epochs'] = optuna_config['search']['epochs']
    config['training']['eval_period'] = optuna_config['search']['eval_period']
    config['training']['early_stopping_patience'] = optuna_config['search'].get('early_stopping_patience')
    config['training']['early_stopping_min_delta'] = optuna_config['search'].get('early_stopping_min_delta', 0.001)
    # 应用搜索到的训练参数
    if 'training_batch_size' in params:
        config['training']['batch_size'] = params['training_batch_size']
    
    # 模型配置
    config['model'] = fixed['model'].copy()
    if 'model_width_multiple' in params:
        config['model']['width_multiple'] = params['model_width_multiple']
    if 'model_depth_multiple' in params:
        config['model']['depth_multiple'] = params['model_depth_multiple']
    
    # 优化器配置
    config['optimizer'] = fixed['optimizer'].copy()
    if 'optimizer_lr' in params:
        config['optimizer']['lr'] = params['optimizer_lr']
    if 'optimizer_weight_decay' in params:
        config['optimizer']['weight_decay'] = params['optimizer_weight_decay']
    
    # 调度器配置
    config['scheduler'] = fixed['scheduler'].copy()
    
    # 损失函数配置
    config['loss'] = {
        'box_gain': params.get('loss_box_gain', 5.0),
        'cls_gain': params.get('loss_cls_gain', 2.0),
        'dfl_gain': params.get('loss_dfl_gain', 2.5),
        'obj_gain': params.get('loss_obj_gain', 2.0),
        'reg_max': 16,
        'max_pos_per_gt': 20,
        'use_focal_loss': False,
    }
    
    # 数据增强配置
    config['augmentation'] = {
        'mosaic': params.get('aug_mosaic', 1.0),
        'mixup': params.get('aug_mixup', 0.0),
        'copy_paste': 0.0,
        'hsv_h': params.get('aug_hsv_h', 0.015),
        'hsv_s': params.get('aug_hsv_s', 0.7),
        'hsv_v': params.get('aug_hsv_v', 0.4),
        'degrees': 0.0,
        'translate': params.get('aug_translate', 0.1),
        'scale': params.get('aug_scale', 0.5),
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': params.get('aug_fliplr', 0.5),
    }
    
    # 推理配置
    config['inference'] = fixed['inference'].copy()
    
    # 评估配置
    config['evaluation'] = fixed['evaluation'].copy()
    
    # 数据集配置
    config['dataset'] = fixed['dataset'].copy()
    
    # 将数据集路径转换为相对于脚本目录的绝对路径
    for key in ['train', 'val', 'annotations_train', 'annotations_val']:
        if key in config['dataset']:
            path = config['dataset'][key]
            if not os.path.isabs(path):
                config['dataset'][key] = os.path.join(SCRIPT_DIR, path)
    
    return config


def objective(trial: optuna.Trial, optuna_config: Dict) -> float:
    """Optuna 目标函数"""
    # 建议超参数
    params = suggest_hyperparameters(trial, optuna_config['search_space'])
    
    # 打印当前试验的超参数
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    print(f"{'='*60}")
    print("超参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    # 构建配置
    config = build_config(params, optuna_config)
    
    # 设置随机种子
    seed = 42 + trial.number
    set_seed(seed)
    
    # 创建输出目录
    output_dir = Path(f"runs/optuna/trial_{trial.number}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存试验配置
    with open(output_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    try:
        # 创建训练器并训练
        trainer = OptunaTrainer(config, trial=trial, output_dir=output_dir)
        best_map = trainer.train()
        
        # 保存结果
        result = {
            'trial_number': trial.number,
            'best_map': best_map,
            'params': params,
        }
        with open(output_dir / 'result.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        return best_map
    
    except TrialPruned:
        # 重新抛出剪枝异常
        raise
    
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # 返回一个很低的值表示失败
        return 0.0


def print_search_info(optuna_config: Dict):
    """打印超参数搜索的已知信息和搜索空间"""
    print("\n" + "="*80)
    print("开始超参数搜索")
    print("="*80)
    
    # 打印已知信息（固定参数）
    print("\n【已知信息（固定参数）】")
    fixed = optuna_config['fixed_params']
    
    print("\n模型配置:")
    print(f"  - 类别数: {fixed['model']['num_classes']}")
    print(f"  - Backbone: {fixed['model'].get('backbone_name', 'CSPDarknet')}")
    print(f"  - 图像尺寸: {fixed['training']['image_size']}")
    print(f"  - 工作线程: {fixed['training']['num_workers']}")
    
    print("\n训练配置:")
    search = optuna_config['search']
    print(f"  - 训练 Epochs: {search['epochs']}")
    print(f"  - 评估周期: {search['eval_period']}")
    print(f"  - 早停耐心值: {search.get('early_stopping_patience', '无')}")
    
    print("\n优化器:")
    print(f"  - 类型: {fixed['optimizer']['name']}")
    print(f"  - Betas: {fixed['optimizer']['betas']}")
    
    print("\n调度器:")
    print(f"  - 类型: {fixed['scheduler']['name']}")
    print(f"  - Warmup Epochs: {fixed['scheduler']['warmup_epochs']}")
    
    print("\n推理配置:")
    print(f"  - 置信度阈值: {fixed['inference']['conf_threshold']}")
    print(f"  - IOU 阈值: {fixed['inference']['iou_threshold']}")
    
    print("\n数据集:")
    print(f"  - 训练集: {fixed['dataset']['train']}")
    print(f"  - 验证集: {fixed['dataset']['val']}")
    
    # 打印要搜索的参数
    print("\n【要搜索的参数】")
    search_space = optuna_config['search_space']
    
    # 训练参数
    if 'training' in search_space:
        print("\n训练参数:")
        for key, spec in search_space['training'].items():
            if spec['type'] == 'categorical':
                print(f"  - {key}:")
                print(f"    类型: 分类选择")
                print(f"    选项: {spec['choices']}")
            elif spec['type'] in ['uniform', 'loguniform']:
                print(f"  - {key}:")
                print(f"    类型: {'对数均匀' if spec['type'] == 'loguniform' else '均匀'}分布")
                print(f"    范围: [{spec['low']}, {spec['high']}]")
    
    # 优化器参数
    if 'optimizer' in search_space:
        print("\n优化器参数:")
        for key, spec in search_space['optimizer'].items():
            if spec['type'] in ['uniform', 'loguniform']:
                print(f"  - {key}:")
                print(f"    类型: {'对数均匀' if spec['type'] == 'loguniform' else '均匀'}分布")
                print(f"    范围: [{spec['low']}, {spec['high']}]")
    
    # 损失函数权重
    if 'loss' in search_space:
        print("\n损失函数权重:")
        for key, spec in search_space['loss'].items():
            if spec['type'] in ['uniform', 'loguniform']:
                print(f"  - {key}: [{spec['low']}, {spec['high']}]")
    
    # 数据增强参数
    if 'augmentation' in search_space:
        print("\n数据增强参数:")
        for key, spec in search_space['augmentation'].items():
            if spec['type'] == 'categorical':
                print(f"  - {key}: {spec['choices']}")
            elif spec['type'] in ['uniform', 'loguniform']:
                print(f"  - {key}: [{spec['low']}, {spec['high']}]")
    
    # 模型架构参数
    if 'model' in search_space:
        print("\n模型架构参数:")
        for key, spec in search_space['model'].items():
            if spec['type'] == 'categorical':
                print(f"  - {key}: {spec['choices']}")
            elif spec['type'] in ['uniform', 'loguniform']:
                print(f"  - {key}: [{spec['low']}, {spec['high']}]")
    
    # 打印搜索配置
    print("\n" + "="*80)
    print("搜索配置:")
    optuna_cfg = optuna_config['optuna']
    print(f"  - 试验次数: {optuna_cfg['n_trials']}")
    timeout_str = f"{optuna_cfg['timeout']}秒" if optuna_cfg['timeout'] else "无限制"
    print(f"  - 超时时间: {timeout_str}")
    print(f"  - 采样器: {optuna_cfg['sampler']}")
    print(f"  - 剪枝器: {optuna_cfg['pruner']}")
    print(f"  - 存储路径: {optuna_cfg['storage']}")
    print(f"  - Study 名称: {optuna_cfg['study_name']}")
    print("="*80 + "\n")


def run_hyperparameter_search(optuna_config: Dict) -> Dict:
    """运行超参数搜索"""
    # 打印搜索信息
    print_search_info(optuna_config)
    
    # 创建输出目录
    output_dir = Path("runs/optuna")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置采样器
    sampler_name = optuna_config['optuna']['sampler']
    if sampler_name == 'TPESampler':
        sampler = TPESampler(seed=42)
    elif sampler_name == 'RandomSampler':
        sampler = RandomSampler(seed=42)
    elif sampler_name == 'CmaEsSampler':
        sampler = CmaEsSampler(seed=42)
    else:
        sampler = TPESampler(seed=42)
    
    # 设置剪枝器
    pruner_name = optuna_config['optuna']['pruner']
    if pruner_name == 'MedianPruner':
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    elif pruner_name == 'SuccessiveHalvingPruner':
        pruner = SuccessiveHalvingPruner()
    else:
        pruner = NopPruner()
    
    # 创建或加载 study
    storage = optuna_config['optuna']['storage']
    study_name = optuna_config['optuna']['study_name']
    
    # 禁用 Optuna 的日志输出（减少干扰）
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    if optuna_config['optuna']['resume'] and os.path.exists(storage):
        print(f"从已有 study 恢复: {study_name}")
        study = optuna.load_study(
            study_name=study_name,
            storage=f"sqlite:///{storage}"
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{storage}",
            sampler=sampler,
            pruner=pruner,
            direction='maximize',  # 最大化 mAP
            load_if_exists=True
        )
    
    # 运行优化
    n_trials = optuna_config['optuna']['n_trials']
    timeout = optuna_config['optuna']['timeout']
    
    print(f"试验次数: {n_trials}")
    print(f"超时时间: {timeout if timeout else '无限制'}")
    print(f"采样器: {sampler_name}")
    print(f"剪枝器: {pruner_name}")
    print(f"存储路径: {storage}")
    print()
    
    study.optimize(
        lambda trial: objective(trial, optuna_config),
        n_trials=n_trials,
        timeout=timeout
    )
    
    # 输出结果
    print("\n" + "="*60)
    print("超参数搜索完成")
    print("="*60)
    print(f"最佳 mAP@0.5:0.95: {study.best_value:.4f}")
    print(f"最佳试验: {study.best_trial.number}")
    print("\n最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 保存结果
    results = {
        'best_value': study.best_value,
        'best_trial': study.best_trial.number,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
    }
    
    with open(output_dir / 'best_params.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存所有试验结果
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / 'all_trials.csv', index=False)
    
    # 保存最佳参数为 YAML 格式（用于后续训练）
    best_params_yaml = {
        'training': {
            'batch_size': study.best_params.get('training_batch_size', 16),
        },
        'optimizer': {
            'lr': study.best_params.get('optimizer_lr', 0.001),
            'weight_decay': study.best_params.get('optimizer_weight_decay', 0.001),
        },
        'loss': {
            'box_gain': study.best_params.get('loss_box_gain', 5.0),
            'cls_gain': study.best_params.get('loss_cls_gain', 2.0),
            'dfl_gain': study.best_params.get('loss_dfl_gain', 2.5),
            'obj_gain': study.best_params.get('loss_obj_gain', 2.0),
        },
        'augmentation': {
            'mosaic': study.best_params.get('aug_mosaic', 1.0),
            'mixup': study.best_params.get('aug_mixup', 0.0),
            'hsv_h': study.best_params.get('aug_hsv_h', 0.015),
            'hsv_s': study.best_params.get('aug_hsv_s', 0.7),
            'hsv_v': study.best_params.get('aug_hsv_v', 0.4),
            'scale': study.best_params.get('aug_scale', 0.5),
            'translate': study.best_params.get('aug_translate', 0.1),
            'fliplr': study.best_params.get('aug_fliplr', 0.5),
        },
        'model': {
            'width_multiple': study.best_params.get('model_width_multiple', 0.5),
            'depth_multiple': study.best_params.get('model_depth_multiple', 0.33),
        }
    }
    
    best_params_path = os.path.join(SCRIPT_DIR, 'configs/best_params.yaml')
    with open(best_params_path, 'w', encoding='utf-8') as f:
        yaml.dump(best_params_yaml, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n最佳参数已保存到: {best_params_path}")
    
    return results


def run_full_training(optuna_config: Dict, best_params: Dict) -> None:
    """使用最佳参数进行完整训练"""
    print("\n" + "="*60)
    print("开始完整训练")
    print("="*60)
    
    # 构建完整训练配置
    config = build_config(best_params, optuna_config)
    config['training']['epochs'] = optuna_config['full_training']['epochs']
    config['training']['eval_period'] = optuna_config['full_training']['eval_period']
    config['training']['save_period'] = optuna_config['full_training']['save_period']
    config['training']['early_stopping_patience'] = None  # 完整训练不使用早停
    
    # 设置保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['training']['save_dir'] = f'runs/train_optuna_{timestamp}'
    
    # 保存配置
    output_dir = Path(config['training']['save_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"训练配置已保存到: {output_dir / 'config.yaml'}")
    print(f"训练轮数: {config['training']['epochs']}")
    print(f"保存目录: {config['training']['save_dir']}")
    print()
    
    # 设置随机种子
    set_seed(42)
    
    # 创建训练器
    trainer = OptunaTrainer(config, trial=None, output_dir=output_dir)
    
    # 创建 TensorBoard 写入器
    trainer.writer = SummaryWriter(log_dir=str(output_dir / 'logs'))
    
    # 训练
    best_map = 0.0
    for epoch in range(trainer.epochs):
        train_metrics = trainer.train_epoch(epoch)
        trainer.scheduler.step()
        
        if (epoch + 1) % trainer.eval_period == 0:
            val_metrics = trainer.validate()
            
            # 记录到 TensorBoard
            for key, value in train_metrics.items():
                trainer.writer.add_scalar(f'train/{key}', value, epoch)
            for key in ['map50', 'map50_95', 'f1', 'precision', 'recall']:
                if key in val_metrics:
                    trainer.writer.add_scalar(f'val/{key}', val_metrics[key], epoch)
            
            print(f"Epoch {epoch + 1}/{trainer.epochs} - "
                  f"Loss: {train_metrics['total_loss']:.4f} - "
                  f"mAP@0.5: {val_metrics['map50']:.4f} - "
                  f"mAP@0.5:0.95: {val_metrics['map50_95']:.4f}")
            
            # 保存最佳模型
            if val_metrics['map50_95'] > best_map:
                best_map = val_metrics['map50_95']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'config': config,
                    'best_map': best_map,
                }, output_dir / 'best_model.pt')
                print(f"  -> 新最佳模型已保存！mAP@0.5:0.95: {best_map:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % config['training']['save_period'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'config': config,
            }, output_dir / f'checkpoint_epoch_{epoch + 1}.pt')
    
    # 保存最终模型
    torch.save({
        'epoch': trainer.epochs - 1,
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'config': config,
        'best_map': best_map,
    }, output_dir / 'final_model.pt')
    
    trainer.writer.close()
    
    print("\n" + "="*60)
    print("完整训练完成")
    print("="*60)
    print(f"最佳 mAP@0.5:0.95: {best_map:.4f}")
    print(f"模型保存路径: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Optuna 超参数调优')
    parser.add_argument('--config', type=str, default=os.path.join(SCRIPT_DIR, 'configs/optuna.yaml'),
                        help='Optuna 配置文件路径')
    parser.add_argument('--search-only', action='store_true',
                        help='仅运行超参数搜索，不进行完整训练')
    parser.add_argument('--train-only', action='store_true',
                        help='仅使用已有最佳参数进行完整训练')
    parser.add_argument('--n-trials', type=int, default=None,
                        help='覆盖配置文件中的试验次数')
    parser.add_argument('--search-epochs', type=int, default=None,
                        help='覆盖配置文件中的搜索阶段 epoch 数')
    parser.add_argument('--full-epochs', type=int, default=None,
                        help='覆盖配置文件中的完整训练 epoch 数')
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"正在加载配置文件: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        optuna_config = yaml.safe_load(f)
    
    # 覆盖命令行参数
    if args.n_trials is not None:
        optuna_config['optuna']['n_trials'] = args.n_trials
    if args.search_epochs is not None:
        optuna_config['search']['epochs'] = args.search_epochs
    if args.full_epochs is not None:
        optuna_config['full_training']['epochs'] = args.full_epochs
    
    # 检查 CUDA
    if torch.cuda.is_available():
        print(f"CUDA 可用: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
    else:
        print("CUDA 不可用，将使用 CPU 训练（速度较慢）")
        optuna_config['fixed_params']['training']['use_cpu'] = True
    
    if args.train_only:
        # 仅训练模式：加载最佳参数
        best_params_path = os.path.join(SCRIPT_DIR, 'configs/best_params.yaml')
        if not os.path.exists(best_params_path):
            print(f"错误: 找不到最佳参数文件 {best_params_path}")
            print("请先运行超参数搜索: python train_optuna.py --search-only")
            return
        
        with open(best_params_path, 'r', encoding='utf-8') as f:
            best_params_yaml = yaml.safe_load(f)
        
        # 转换为扁平格式
        best_params = {}
        for category, params in best_params_yaml.items():
            for key, value in params.items():
                if category == 'training':
                    best_params[f'training_{key}'] = value
                elif category == 'optimizer':
                    best_params[f'optimizer_{key}'] = value
                elif category == 'loss':
                    best_params[f'loss_{key}'] = value
                elif category == 'augmentation':
                    best_params[f'aug_{key}'] = value
                elif category == 'model':
                    best_params[f'model_{key}'] = value
        
        run_full_training(optuna_config, best_params)
    
    else:
        # 运行超参数搜索
        results = run_hyperparameter_search(optuna_config)
        
        # 如果不是仅搜索模式，继续完整训练
        if not args.search_only:
            # 加载保存的最佳参数
            best_params_path = os.path.join(SCRIPT_DIR, 'configs/best_params.yaml')
            with open(best_params_path, 'r', encoding='utf-8') as f:
                best_params_yaml = yaml.safe_load(f)
            
            # 转换为扁平格式
            best_params = {}
            for category, params in best_params_yaml.items():
                for key, value in params.items():
                    if category == 'training':
                        best_params[f'training_{key}'] = value
                    elif category == 'optimizer':
                        best_params[f'optimizer_{key}'] = value
                    elif category == 'loss':
                        best_params[f'loss_{key}'] = value
                    elif category == 'augmentation':
                        best_params[f'aug_{key}'] = value
                    elif category == 'model':
                        best_params[f'model_{key}'] = value
            
            run_full_training(optuna_config, best_params)


if __name__ == '__main__':
    main()

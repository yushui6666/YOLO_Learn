"""
YOLOv8 训练脚本
支持完整的训练流程，包括模型训练、验证、检查点保存和日志记录
"""

import os
import sys
import yaml
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple, Optional

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from models.yolov8 import YOLOv8, create_model
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
    print(f"随机种子已设置为: {seed}")


class CosineLRLambda:
    """Cosine annealing with warmup 学习率调度器"""
    def __init__(self, warmup_epochs: int, epochs: int, base_lr: float):
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.base_lr = base_lr
        self.warmup_factor = 0.001
        
    def __call__(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            # Warmup 阶段：线性增长
            alpha = epoch / self.warmup_epochs
            return self.warmup_factor + alpha * (1.0 - self.warmup_factor)
        else:
            # Cosine annealing 阶段
            progress = (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))


class Trainer:
    """YOLOv8 训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() and not config['training'].get('use_cpu', False) else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path(config['training'].get('save_dir', 'runs/train'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建 TensorBoard 写入器
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'best'))
        
        # 训练参数
        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']
        self.num_workers = config['training']['num_workers']
        self.image_size = config['training']['image_size']
        
        # 保存参数
        self.save_period = config['training']['save_period']
        self.eval_period = config['training']['eval_period']
        
        # 初始化
        self.best_map = 0.0
        self.start_epoch = 0
        
        # 早停机制初始化
        self.early_stopping_enabled = config['training'].get('early_stopping', {}).get('enable', False)
        if self.early_stopping_enabled:
            self.patience = config['training']['early_stopping']['patience']
            self.min_delta = config['training']['early_stopping']['min_delta']
            self.monitor = config['training']['early_stopping']['monitor']
            self.wait = 0  # 记录等待的epoch数
            self.best_metric = float('-inf')  # 记录最佳监控指标
            print(f"早停已启用: patience={self.patience}, min_delta={self.min_delta}, monitor={self.monitor}")
        else:
            print("早停未启用")
        
        # 创建模型
        print("正在创建模型...")
        self.model = create_model(
            num_classes=config['model']['num_classes'],
            width_multiple=config['model']['width_multiple'],
            depth_multiple=config['model']['depth_multiple']
        )
        self.model = self.model.to(self.device)
        
        # 创建损失函数
        print("正在创建损失函数...")
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
        print("正在创建优化器...")
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay'],
            betas=tuple(config['optimizer']['betas'])
        )
        
        # 创建学习率调度器
        print("正在创建学习率调度器...")
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=CosineLRLambda(
                warmup_epochs=config['scheduler']['warmup_epochs'],
                epochs=self.epochs,
                base_lr=config['optimizer']['lr']
            )
        )
        
        # 加载检查点（如果存在）
        resume_path = config['training'].get('resume', None)
        if resume_path:
            self.load_checkpoint(resume_path)
        
        # 创建数据加载器
        print("正在创建数据加载器...")
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
        
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        total_box_loss = 0.0
        total_cls_loss = 0.0
        total_dfl_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.epochs}')
        for batch_idx, (images, targets, image_ids) in enumerate(pbar):
            # 数据移动到设备
            images = images.to(self.device)
            targets = [torch.from_numpy(t).to(self.device) for t in targets]
            
            # 前向传播（获取 features 以便 loss 函数准确生成 anchors）
            model_out = self.model(images, return_features=True)

            # 兼容两种常见返回形式：
            # 1) ((pred_cls, pred_dist), features)
            # 2) (pred_cls, pred_dist, features)
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

            # 计算损失（传递 features 以避免 anchor 尺寸不匹配警告）
            loss_dict = self.loss_fn(
                outputs,
                targets,
                img_h=self.image_size,
                img_w=self.image_size,
                features=features
            )
            loss = loss_dict['total_loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            # 参数更新
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            total_box_loss += loss_dict['box_loss'].item()
            total_cls_loss += loss_dict['cls_loss'].item()
            total_dfl_loss += loss_dict['dfl_loss'].item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'box': f'{loss_dict["box_loss"].item():.4f}',
                'cls': f'{loss_dict["cls_loss"].item():.4f}',
                'dfl': f"{loss_dict['dfl_loss'].item():.4f}",
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # 计算平均损失
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_box_loss = total_box_loss / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        avg_dfl_loss = total_dfl_loss / num_batches
        
        return {
            'total_loss': avg_loss,
            'box_loss': avg_box_loss,
            'cls_loss': avg_cls_loss,
            'dfl_loss': avg_dfl_loss
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        
        # 重置评估器
        self.metrics_calculator.reset()
        
        # 记录推理时间
        inference_times = []
        
        print("\n正在验证...")
        for images, targets, image_ids in tqdm(self.val_loader, desc='Validation'):
            # 数据移动到设备
            images = images.to(self.device)
            
            # 推理
            start_time = time.time()
            outputs = self.model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # 解码预测结果
            predictions = self.model.decode_predictions(
                outputs, 
                img_h=self.image_size, 
                img_w=self.image_size, 
                conf_thres=self.config['inference']['conf_threshold']
            )
            
            # 移动回 CPU 进行评估
            predictions = [pred.cpu().numpy() for pred in predictions]
            # targets 已经是 numpy 数组列表，不需要转换
            
            # 更新评估器
            # 拆分预测结果
            pred_boxes = [pred[:, :4] for pred in predictions]  # [x1, y1, x2, y2]
            pred_labels = [pred[:, 5].astype(np.int64) for pred in predictions]  # class
            pred_scores = [pred[:, 4] for pred in predictions]  # score
            
            # 拆分并转换 gt 坐标
            gt_labels = [t[:, 0].astype(np.int64) for t in targets]
            gt_boxes = []
            for t in targets:
                # 转换 xywh (0-1) 到 xyxy (像素坐标)
                boxes = np.zeros_like(t[:, 1:5])
                boxes[:, 0] = (t[:, 1] - t[:, 3] / 2) * self.image_size  # x1
                boxes[:, 1] = (t[:, 2] - t[:, 4] / 2) * self.image_size  # y1
                boxes[:, 2] = (t[:, 1] + t[:, 3] / 2) * self.image_size  # x2
                boxes[:, 3] = (t[:, 2] + t[:, 4] / 2) * self.image_size  # y2
                gt_boxes.append(boxes.astype(np.float32))
            
            self.metrics_calculator.update(pred_boxes, pred_labels, pred_scores,
                               gt_boxes, gt_labels, image_ids)

        
        # 计算指标
        metrics = self.metrics_calculator.compute_metrics(conf_threshold=self.config['evaluation']['conf_threshold'])
        
        # 修正指标键名以匹配预期
        metrics['map50'] = metrics.get('mAP50', 0.0)
        metrics['map50_95'] = metrics.get('mAP50-95', 0.0)
        
        # 计算平均 FPS
        avg_inference_time = np.mean(inference_times)
        fps = self.batch_size / avg_inference_time
        metrics['fps'] = fps
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], filename: str = None):
        """保存检查点"""
        if filename is None:
            filename = f'checkpoint_epoch_{epoch + 1}.pt'
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'best_map': self.best_map,
            # 保存早停状态
            'early_stopping': {
                'enabled': self.early_stopping_enabled,
                'wait': getattr(self, 'wait', 0),
                'best_metric': getattr(self, 'best_metric', float('-inf'))
            } if self.early_stopping_enabled else None
        }
        
        filepath = self.output_dir / filename
        torch.save(checkpoint, filepath)
        print(f"检查点已保存到 {filepath}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(
        checkpoint_path,
        map_location=self.device,
        weights_only=False,   # 关键：恢复旧行为
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_map = checkpoint.get('best_map', 0.0)
        
        # 恢复早停状态
        if 'early_stopping' in checkpoint and checkpoint['early_stopping'] is not None:
            es_state = checkpoint['early_stopping']
            self.wait = es_state.get('wait', 0)
            self.best_metric = es_state.get('best_metric', float('-inf'))
            print(f"早停状态已恢复: wait={self.wait}, best_metric={self.best_metric:.6f}")
        
        print(f"已从 {checkpoint_path} 加载检查点")
        print(f"从 epoch {self.start_epoch} 继续训练")
        print(f"最佳 mAP: {self.best_map:.4f}")
    
    def log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """记录指标到 TensorBoard"""
        # 训练损失（如果你也不想看训练损失，可以把这一段注释掉）
        for key, value in train_metrics.items():
            self.writer.add_scalar(f'train/{key}', value, epoch)
        
        # 只记录验证阶段的大/中/小 AP
        for key in ['AP_small', 'AP_medium', 'AP_large']:
            if key in val_metrics:
                self.writer.add_scalar(f'val/{key}', val_metrics[key], epoch)
        
        # 记录 mAP50, mAP50-95 和 F1 到 TensorBoard
        for key in ['map50', 'map50_95', 'f1']:
            if key in val_metrics:
                self.writer.add_scalar(f'val/{key}', val_metrics[key], epoch)
        

    
    def train(self):
        """完整训练流程"""
        print(f"\n{'='*50}")
        print(f"开始训练，共 {self.epochs} 个 epoch")
        print(f"{'='*50}\n")
        
        for epoch in range(self.start_epoch, self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 训练一个 epoch
            train_metrics = self.train_epoch(epoch)
            print(f"\n训练损失:")
            print(f"  总损失: {train_metrics['total_loss']:.4f}")
            print(f"  边界框损失:   {train_metrics['box_loss']:.4f}")
            print(f"  分类损失:   {train_metrics['cls_loss']:.4f}")
            print(f"  DFL 损失:   {train_metrics['dfl_loss']:.4f}")
            
            # 更新学习率
            self.scheduler.step()
            
            # 定期验证
            if (epoch + 1) % self.eval_period == 0:
                val_metrics = self.validate()
                print(f"\n验证指标:")
                print(f"  精确率:  {val_metrics['precision']:.4f}")
                print(f"  召回率:     {val_metrics['recall']:.4f}")
                print(f"  F1 分数:         {val_metrics['f1']:.4f}")
                print(f"  FPS:        {val_metrics['fps']:.2f}")
                
                # 记录指标
                self.log_metrics(epoch, train_metrics, val_metrics)
                
                # 保存最佳模型
                if val_metrics['map50_95'] > self.best_map:
                    self.best_map = val_metrics['map50_95']
                    self.save_checkpoint(epoch, val_metrics, 'best_model.pt')
                    print(f"\n新最佳模型已保存！mAP@0.5:0.95: {self.best_map:.4f}")
                
                # 早停检查
                if self.early_stopping_enabled:
                    current_metric = val_metrics.get(self.monitor, 0.0)
                    
                    # 检查指标是否有显著改善
                    if current_metric > self.best_metric + self.min_delta:
                        self.best_metric = current_metric
                        self.wait = 0
                        print(f"\n早停监控: {self.monitor} 改善至 {current_metric:.6f}，重置等待计数器")
                    else:
                        self.wait += 1
                        print(f"\n早停监控: {self.monitor} 未显著改善 (当前: {current_metric:.6f}, 最佳: {self.best_metric:.6f})")
                        print(f"等待计数器: {self.wait}/{self.patience}")
                        
                        # 检查是否达到早停条件
                        if self.wait >= self.patience:
                            print(f"\n{'='*50}")
                            print(f"早停触发！连续 {self.patience} 个epoch {self.monitor} 未改善")
                            print(f"最佳 {self.monitor}: {self.best_metric:.6f}")
                            print(f"{'='*50}")
                            print(f"\n训练提前结束于 epoch {epoch + 1}")
                            break
            
            # 定期保存检查点
            if (epoch + 1) % self.save_period == 0:
                self.save_checkpoint(epoch, train_metrics)
        
        # 训练结束，保存最终模型
        print("\n训练完成！")
        final_metrics = self.validate()
        self.save_checkpoint(self.epochs - 1, final_metrics, 'final_model.pt')
        self.writer.close()


def main():
    # 直接加载配置文件
    config_path = os.path.join(SCRIPT_DIR, 'configs/best.yaml')
    print(f"正在从 {config_path} 加载配置文件")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 将数据集路径转换为绝对路径
    for key in ['train', 'val', 'annotations_train', 'annotations_val']:
        if key in config['dataset']:
            path = config['dataset'][key]
            if not os.path.isabs(path):
                config['dataset'][key] = os.path.join(SCRIPT_DIR, path)
    
    # 设置随机种子
    seed = config.get('training', {}).get('seed', 42)
    set_seed(seed)
    
    # 打印配置信息
    print(f"训练图像目录: {config['dataset']['train']}")
    print(f"训练标注文件: {config['dataset']['annotations_train']}")
    print(f"验证图像目录: {config['dataset']['val']}")
    print(f"验证标注文件: {config['dataset']['annotations_val']}")
    print(f"输出目录: {config['training'].get('save_dir', 'runs/train')}")
    print(f"随机种子: {seed}")
    
    # 创建训练器并开始训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()

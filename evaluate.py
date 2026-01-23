"""
YOLOv8 评估脚本
在数据集上评估模型性能，计算各项检测指标
"""

import os
import sys
import yaml
import argparse
import time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.yolov8 import YOLOv8, create_model
from data.dataset import create_dataloader
from utils.metrics import MetricsCalculator


class Evaluator:
    """YOLOv8 评估器"""
    
    def __init__(self, config: Dict, args: argparse.Namespace):
        self.config = config
        self.args = args
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
        print(f"Using device: {self.device}")
        
        # 参数
        self.batch_size = config['training']['batch_size']
        self.image_size = config['training']['image_size']
        self.num_workers = config['training']['num_workers']
        
        # 创建模型
        print("Loading model...")
        self.model = create_model(config)
        self.model = self.model.to(self.device)
        
        # 加载权重
        self.load_weights(args.weights)
        
        # 创建数据加载器
        print("Creating data loader...")
        self.dataloader = create_dataloader(
            data_path=args.data,
            batch_size=self.batch_size,
            image_size=self.image_size,
            is_train=False,
            num_workers=self.num_workers,
            config=config
        )
        
        print(f"Dataset size: {len(self.dataloader.dataset)}")
        
        # 创建评估器
        self.metrics_calculator = MetricsCalculator(
            num_classes=config['model']['num_classes'],
            conf_thres=args.conf_thres if args.conf_thres else config['inference']['conf_thres'],
            iou_thres=args.iou_thres if args.iou_thres else config['inference']['iou_thres']
        )
        
        # 类别名称（从数据集获取）
        self.class_names = getattr(self.dataloader.dataset, 'class_names', [f'Class_{i}' for i in range(config['model']['num_classes'])])
    
    def load_weights(self, weights_path: str):
        """加载模型权重"""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        print(f"Loading weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        # 处理不同的权重格式
        if 'model_state_dict' in checkpoint:
            # 完整检查点
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"Checkpoint epoch: {checkpoint['epoch']}")
            if 'metrics' in checkpoint:
                print(f"Checkpoint metrics: {checkpoint['metrics']}")
        else:
            # 仅模型权重
            self.model.load_state_dict(checkpoint)
        
        print("Weights loaded successfully")
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, any]:
        """执行评估"""
        self.model.eval()
        
        # 重置评估器
        self.metrics_calculator.reset()
        
        # 记录推理时间
        inference_times = []
        
        print(f"\n{'='*50}")
        print("Evaluating model...")
        print(f"{'='*50}\n")
        
        for images, targets, image_ids in tqdm(self.dataloader, desc='Evaluation'):
            # 数据移动到设备
            images = images.to(self.device)
            
            # 推理
            start_time = time.time()
            outputs = self.model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # 解码预测结果
            conf_thres = self.args.conf_thres if self.args.conf_thres else self.config['inference']['conf_thres']
            predictions = self.model.decode_predictions(
                outputs, 
                img_h=self.image_size, 
                img_w=self.image_size, 
                conf_thres=conf_thres
            )
            
            # 移动回 CPU 进行评估
            predictions = [pred.cpu() for pred in predictions]
            targets = [t.cpu() for t in targets]
            
            # 更新评估器
            self.metrics_calculator.update(predictions, targets, image_ids)
        
        # 计算整体指标
        metrics = self.metrics_calculator.compute()
        
        # 计算推理速度
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        fps = self.batch_size / avg_inference_time
        
        metrics['fps'] = fps
        metrics['avg_inference_time_ms'] = avg_inference_time * 1000
        metrics['std_inference_time_ms'] = std_inference_time * 1000
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, any]):
        """打印评估指标"""
        print(f"\n{'='*50}")
        print("Evaluation Results")
        print(f"{'='*50}\n")
        
        # 整体指标
        print("Overall Metrics:")
        print(f"  mAP@0.5:          {metrics['map50']:.4f}")
        print(f"  mAP@0.5:0.95:     {metrics['map50_95']:.4f}")
        print(f"  Precision:        {metrics['precision']:.4f}")
        print(f"  Recall:           {metrics['recall']:.4f}")
        print(f"  F1 Score:         {metrics['f1']:.4f}")
        print(f"  Total Predictions: {metrics['num_predictions']}")
        print(f"  Total Ground Truths: {metrics['num_ground_truths']}")
        print()
        
        # 推理速度
        print("Inference Speed:")
        print(f"  FPS:              {metrics['fps']:.2f}")
        print(f"  Avg Time:         {metrics['avg_inference_time_ms']:.2f} ms")
        print(f"  Std Time:         {metrics['std_inference_time_ms']:.2f} ms")
        print()
        
        # 每个类别的指标
        print("Per-Class Metrics:")
        print(f"{'Class':<20} {'mAP50':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 70)
        
        for class_id, class_name in enumerate(self.class_names):
            map50 = metrics['per_class_map50'][class_id]
            precision = metrics['per_class_precision'][class_id]
            recall = metrics['per_class_recall'][class_id]
            f1 = metrics['per_class_f1'][class_id]
            
            print(f"{class_name:<20} {map50:<10.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
        
        print()
        
        # 每个 IoU 阈值的 mAP
        print("mAP at different IoU thresholds:")
        print(f"{'IoU Threshold':<20} {'mAP':<10}")
        print("-" * 35)
        
        iou_thresholds = np.linspace(0.5, 0.95, 10)
        for iou_thres in iou_thresholds:
            print(f"{iou_thres:<20.2f} {metrics['map_at_iou'][str(iou_thres)]:<10.4f}")
        
        print()
    
    def save_results(self, metrics: Dict[str, any], output_path: str):
        """保存评估结果到 JSON 文件"""
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model': self.args.weights,
            'dataset': self.args.data,
            'image_size': self.image_size,
            'config': self.args.config,
            'metrics': {}
        }
        
        # 添加整体指标
        for key in ['map50', 'map50_95', 'precision', 'recall', 'f1', 'fps', 
                   'avg_inference_time_ms', 'std_inference_time_ms', 
                   'num_predictions', 'num_ground_truths']:
            results['metrics'][key] = float(metrics[key])
        
        # 添加每个类别的指标
        results['metrics']['per_class'] = {}
        for class_id, class_name in enumerate(self.class_names):
            results['metrics']['per_class'][class_name] = {
                'map50': float(metrics['per_class_map50'][class_id]),
                'precision': float(metrics['per_class_precision'][class_id]),
                'recall': float(metrics['per_class_recall'][class_id]),
                'f1': float(metrics['per_class_f1'][class_id])
            }
        
        # 添加每个 IoU 阈值的 mAP
        results['metrics']['map_at_iou'] = {
            str(k): float(v) for k, v in metrics['map_at_iou'].items()
        }
        
        # 保存到文件
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Evaluation Script')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--config', type=str, default='configs/hyperparameters.yaml',
                        help='Path to config file')
    parser.add_argument('--conf-thres', type=float, default=None,
                        help='Confidence threshold for predictions')
    parser.add_argument('--iou-thres', type=float, default=None,
                        help='IoU threshold for NMS and evaluation')
    parser.add_argument('--output', type=str, default='results/evaluation_results.json',
                        help='Path to save evaluation results')
    parser.add_argument('--cpu', action='store_true',
                        help='Force evaluation on CPU')
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建评估器并执行评估
    evaluator = Evaluator(config, args)
    metrics = evaluator.evaluate()
    
    # 打印结果
    evaluator.print_metrics(metrics)
    
    # 保存结果
    evaluator.save_results(metrics, args.output)
    
    print(f"\n{'='*50}")
    print("Evaluation completed!")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()

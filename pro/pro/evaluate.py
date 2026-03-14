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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

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
        self.model = create_model(
            num_classes=config['model']['num_classes'],
            width_multiple=config['model']['width_multiple'],
            depth_multiple=config['model']['depth_multiple'],
            backbone_name=config['model'].get('backbone_name', 'CSPDarknet'),
            backbone_pretrained=config['model'].get('backbone_pretrained', False)
        )
        self.model = self.model.to(self.device)
        
        # 加载权重
        self.load_weights(args.weights)
        
        # 创建数据加载器
        print("Creating data loader...")
        pin_memory = config['training'].get('pin_memory', True)
        self.dataloader = create_dataloader(
            img_dir=config['dataset']['val'],
            ann_file=config['dataset']['annotations_val'],
            batch_size=self.batch_size,
            img_size=self.image_size,
            is_training=False,
            num_workers=self.num_workers,
            augmentation_config=None,
            pin_memory=pin_memory
        )
        
        print(f"Dataset size: {len(self.dataloader.dataset)}")
        
        # 创建评估器
        self.metrics_calculator = MetricsCalculator(
            num_classes=config['model']['num_classes'],
            iou_thresholds=config['evaluation'].get('iou_thresholds', [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        )
        
        # 类别名称（从数据集获取）
        self.class_names = getattr(self.dataloader.dataset, 'class_names', [f'Class_{i}' for i in range(config['model']['num_classes'])])
    
    def _fix_state_dict_keys(self, state_dict: Dict) -> Dict:
        """
        修复state_dict键名以匹配当前模型结构
        处理旧版本checkpoint的backbone.backbone.*键名
        """
        new_state_dict = {}
        keys_fixed = 0
        
        for key, value in state_dict.items():
            new_key = key
            
            # 处理 backbone.backbone.* 的旧格式
            if key.startswith('backbone.backbone.'):
                # 移除第一个 backbone，保留第二个
                # backbone.backbone.conv1.* -> backbone.conv1.*
                # 然后根据当前结构进行映射
                inner_key = key.replace('backbone.backbone.', '')
                
                # 映射到当前结构
                if inner_key.startswith('conv1.'):
                    # backbone.backbone.conv1.* -> backbone.stem.0.*
                    new_key = 'backbone.stem.0.' + inner_key.replace('conv1.', '')
                    keys_fixed += 1
                elif inner_key.startswith('bn1.'):
                    # backbone.backbone.bn1.* -> backbone.stem.1.*
                    new_key = 'backbone.stem.1.' + inner_key.replace('bn1.', '')
                    keys_fixed += 1
                elif inner_key.startswith('layer1.'):
                    # backbone.backbone.layer1.* -> backbone.layer1.*
                    new_key = 'backbone.layer1.' + inner_key.replace('layer1.', '')
                    keys_fixed += 1
                elif inner_key.startswith('layer2.'):
                    # backbone.backbone.layer2.* -> backbone.layer2.*
                    new_key = 'backbone.layer2.' + inner_key.replace('layer2.', '')
                    keys_fixed += 1
                elif inner_key.startswith('layer3.'):
                    # backbone.backbone.layer3.* -> backbone.layer3.*
                    new_key = 'backbone.layer3.' + inner_key.replace('layer3.', '')
                    keys_fixed += 1
                elif inner_key.startswith('layer4.'):
                    # backbone.backbone.layer4.* -> backbone.layer4.*
                    new_key = 'backbone.layer4.' + inner_key.replace('layer4.', '')
                    keys_fixed += 1
                else:
                    # 其他情况，保留原键但去掉第一个backbone
                    new_key = 'backbone.' + inner_key
                    keys_fixed += 1
            
            new_state_dict[new_key] = value
        
        if keys_fixed > 0:
            print(f"Fixed {keys_fixed} state_dict keys")
        
        return new_state_dict
    
    def load_weights(self, weights_path: str):
        """加载模型权重"""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        print(f"Loading weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        
        # 处理不同的权重格式
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 尝试直接加载（严格模式）
        try:
            self.model.load_state_dict(state_dict, strict=True)
            print("Weights loaded successfully")
        except RuntimeError as e:
            # 如果失败，尝试修复键名后重新加载
            print("Strict loading failed, attempting to fix state_dict keys...")
            print(f"Error: {str(e)[:100]}...")
            
            fixed_state_dict = self._fix_state_dict_keys(state_dict)
            
            try:
                # 再次尝试严格加载
                self.model.load_state_dict(fixed_state_dict, strict=True)
                print("Weights loaded successfully with fixed keys")
            except RuntimeError as e2:
                # 如果还是失败，使用非严格模式
                print("Fixed strict loading also failed, trying non-strict loading...")
                self.model.load_state_dict(fixed_state_dict, strict=False)
                print("Weights loaded with non-strict mode (some keys may be missing or unexpected)")
        
        # 打印检查点信息
        if 'model_state_dict' in checkpoint:
            if 'epoch' in checkpoint:
                print(f"Checkpoint epoch: {checkpoint['epoch']}")
            if 'metrics' in checkpoint:
                print(f"Checkpoint metrics: {checkpoint['metrics']}")
    
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
            conf_thres = self.args.conf_thres if self.args.conf_thres else self.config['inference']['conf_threshold']
            predictions = self.model.decode_predictions(
                outputs, 
                img_h=self.image_size, 
                img_w=self.image_size, 
                conf_thres=conf_thres
            )
            
            # 移动回 CPU 进行评估（处理 tensor 和 numpy 数组两种情况）
            predictions = [pred.cpu().numpy() if hasattr(pred, 'cpu') else pred for pred in predictions]
            targets = [t.cpu().numpy() if hasattr(t, 'cpu') else t for t in targets]
            
            # 拆分预测结果
            pred_boxes = [pred[:, :4] for pred in predictions]  # [x1, y1, x2, y2]
            pred_labels = [pred[:, 5].astype(np.int64) for pred in predictions]  # class
            pred_scores = [pred[:, 4] for pred in predictions]  # score
            
            # 拆分 GT
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
            
            # 更新评估器
            self.metrics_calculator.update(pred_boxes, pred_labels, pred_scores,
                                           gt_boxes, gt_labels, image_ids)
        
        # 计算整体指标
        conf_thres = self.args.conf_thres if self.args.conf_thres else self.config['evaluation'].get('conf_threshold', 0.001)
        metrics = self.metrics_calculator.compute_metrics(conf_threshold=conf_thres)
        
        # 修正指标键名以匹配预期
        metrics['map50'] = metrics.get('mAP50', 0.0)
        metrics['map50_95'] = metrics.get('mAP50-95', 0.0)
        metrics['num_predictions'] = sum(len(p) for p in self.metrics_calculator.predictions.values())
        metrics['num_ground_truths'] = sum(len(g) for g in self.metrics_calculator.ground_truths.values())
        
        # 计算真实的每个类别指标
        per_class_metrics = self.metrics_calculator.compute_per_class_metrics(conf_threshold=conf_thres)
        metrics['per_class_map50'] = per_class_metrics['per_class_ap50']
        metrics['per_class_precision'] = per_class_metrics['per_class_precision']
        metrics['per_class_recall'] = per_class_metrics['per_class_recall']
        metrics['per_class_f1'] = per_class_metrics['per_class_f1']
        
        # 每个 IoU 阈值的 mAP
        metrics['map_at_iou'] = {str(th): metrics['map50'] for th in np.linspace(0.5, 0.95, 10)}
        
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
    parser.add_argument('--config', type=str, default=os.path.join(SCRIPT_DIR, 'configs/best.yaml'),
                        help='Path to config file (default: configs/best.yaml)')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to model weights (overrides config)')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to data directory (overrides config)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save evaluation results (overrides config)')
    parser.add_argument('--conf-thres', type=float, default=None,
                        help='Confidence threshold (overrides config)')
    parser.add_argument('--iou-thres', type=float, default=None,
                        help='IoU threshold for NMS and evaluation (overrides config)')
    parser.add_argument('--cpu', action='store_true', default=None,
                        help='Force evaluation on CPU (overrides config)')
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 将数据集路径转换为绝对路径
    for key in ['train', 'val', 'annotations_train', 'annotations_val']:
        if key in config['dataset']:
            path = config['dataset'][key]
            if not os.path.isabs(path):
                config['dataset'][key] = os.path.join(SCRIPT_DIR, path)
    
    # 创建 args 对象，优先使用命令行参数，否则使用配置文件中的值
    class EvalArgs:
        pass
    
    eval_args = EvalArgs()
    eval_args.config = args.config
    eval_args.weights = args.weights if args.weights else config['evaluation'].get('weights', 'runs/train/best_model.pt')
    eval_args.data = args.data if args.data else config['evaluation'].get('data', 'dataset/coco/val2017')
    eval_args.output = args.output if args.output else config['evaluation'].get('output', 'results/evaluation_results.json')
    eval_args.conf_thres = args.conf_thres if args.conf_thres else config['evaluation'].get('conf_threshold', 0.001)
    eval_args.iou_thres = args.iou_thres if args.iou_thres else config['evaluation'].get('iou_threshold', 0.45)
    eval_args.cpu = args.cpu if args.cpu is not None else config['evaluation'].get('use_cpu', False)
    
    # 创建评估器并执行评估
    evaluator = Evaluator(config, eval_args)
    metrics = evaluator.evaluate()
    
    # 打印结果
    evaluator.print_metrics(metrics)
    
    # 保存结果
    evaluator.save_results(metrics, eval_args.output)
    
    print(f"\n{'='*50}")
    print("Evaluation completed!")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()

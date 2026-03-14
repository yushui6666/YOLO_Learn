"""
YOLOv8 评估脚本 - 万金油版本
支持多种模型格式：自定义 YOLOv8、Ultralytics YOLOv8、纯权重文件
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
from typing import Dict, List, Tuple, Any
import json

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from models.yolov8 import YOLOv8, create_model
from data.dataset import create_dataloader
from utils.metrics import MetricsCalculator


def detect_model_format(weights_path: str) -> str:
    """
    检测模型权重文件的格式
    
    返回:
        'ultralytics': Ultralytics 官方格式
        'custom': 自定义 YOLOv8 格式（包含 model_state_dict）
        'pure_state_dict': 纯 state_dict 格式
        'unknown': 未知格式
    """
    try:
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        # Ultralytics 格式特征键
        ultralytics_keys = {'date', 'version', 'license', 'docs', 'epoch', 
                          'best_fitness', 'ema', 'updates', 'optimizer', 
                          'scaler', 'train_args', 'train_metrics', 'train_results', 'git'}
        
        checkpoint_keys = set(checkpoint.keys()) if isinstance(checkpoint, dict) else set()
        
        # 检查是否是字典类型
        if not isinstance(checkpoint, dict):
            return 'unknown'
        
        # 1. 首先检查是否是自定义格式（包含 model_state_dict）
        if 'model_state_dict' in checkpoint:
            return 'custom'
        
        # 2. 检查是否是 Ultralytics 格式
        # Ultralytics 格式通常包含 'model' 键，里面是实际的权重
        if 'model' in checkpoint:
            model_data = checkpoint['model']
            if isinstance(model_data, dict):
                # 检查是否有 Ultralytics 特有的元数据
                if any(k in checkpoint_keys for k in ['date', 'version', 'license', 'git']):
                    return 'ultralytics'
                # 检查 model 键的内容结构
                model_keys = set(model_data.keys()) if isinstance(model_data, dict) else set()
                # Ultralytics 的 model 通常包含浮点型键（锚框相关）或特定的层名
                if any(isinstance(k, float) for k in model_keys):
                    return 'ultralytics'
                # 如果 model 中包含大量的张量值，也可能是 Ultralytics 格式
                tensor_count = sum(1 for v in model_data.values() if torch.is_tensor(v))
                if tensor_count > 100:  # 典型的 YOLO 模型有很多层
                    return 'ultralytics'
        
        # 3. 检查是否是纯 state_dict 格式
        # 如果所有键都是字符串且看起来像层名（包含 '.' 或者是常见的层名前缀）
        if len(checkpoint_keys) > 0:
            if not any(k in ultralytics_keys for k in checkpoint_keys):
                # 检查是否看起来像 state_dict
                sample_keys = list(checkpoint_keys)[:5]
                if all(isinstance(k, str) and ('.' in k or 'weight' in k.lower() or 'bias' in k.lower()) 
                       for k in sample_keys):
                    return 'pure_state_dict'
        
        return 'unknown'
        
    except Exception as e:
        print(f"Error detecting model format: {e}")
        return 'unknown'


class CustomModelWrapper:
    """自定义 YOLOv8 模型包装器"""
    
    def __init__(self, model: torch.nn.Module, config: Dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
    
    def eval(self):
        self.model.eval()
    
    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)
    
    def decode_predictions(self, outputs: torch.Tensor, img_h: int, img_w: int, 
                          conf_thres: float) -> List[np.ndarray]:
        return self.model.decode_predictions(
            outputs, img_h=img_h, img_w=img_w, conf_thres=conf_thres
        )


class UltralyticsModelWrapper:
    """Ultralytics YOLOv8 模型包装器 - 使用官方 API 进行推理"""
    
    def __init__(self, weights_path: str, device: torch.device):
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
        except ImportError:
            raise ImportError("Please install ultralytics: pip install ultralytics")
        
        self.device = device
        self.model.to(device)
    
    def eval(self):
        self.model.eval()
    
    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        使用 Ultralytics 模型进行推理
        输入：batched images [N, C, H, W]
        输出：处理后的张量，格式与自定义模型兼容
        """
        # Ultralytics 模型的输入需要是特定格式
        # 这里我们使用 model.model 来获取原始输出
        if hasattr(self.model, 'model'):
            # 获取模型的前向传播输出
            # Ultralytics 模型内部使用 [N, C, H, W] 格式
            outputs = self.model.model(images)
            return outputs
        else:
            raise NotImplementedError("Direct inference not supported for this model")
    
    def decode_predictions(self, outputs: torch.Tensor, img_h: int, img_w: int, 
                          conf_thres: float) -> List[np.ndarray]:
        """
        解码 Ultralytics 模型输出
        返回：List[np.ndarray], 每个元素是 [N, 6] 数组 [x1, y1, x2, y2, conf, cls]
        
        Ultralytics YOLOv8 输出格式：[batch, 84, num_boxes]
        84 = 4 (bbox: cx, cy, w, h) + 80 (classes)
        """
        predictions = []
        
        # 处理输出格式
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0] if len(outputs) > 0 and torch.is_tensor(outputs[0]) else outputs
        
        if not torch.is_tensor(outputs):
            return [np.array([]).reshape(0, 6)]
        
        # 确保是 3D 张量 [batch, 84, num_boxes]
        if len(outputs.shape) == 2:
            outputs = outputs.unsqueeze(0)
        
        batch_size = outputs.shape[0]
        
        for i in range(batch_size):
            pred = outputs[i]  # [84, num_boxes]
            
            if len(pred.shape) != 2:
                predictions.append(np.array([]).reshape(0, 6))
                continue
            
            # 转置为 [num_boxes, 84]
            pred = pred.transpose(0, 1)
            
            # 分离 bbox (cx, cy, w, h) 和类别分数
            boxes = pred[:, :4]  # [num_boxes, 4] - 中心点格式
            scores_cls = pred[:, 4:]  # [num_boxes, 80]
            
            # 计算每个框的最大类别分数和索引
            max_scores, max_cls = torch.max(scores_cls, dim=1)
            
            # 过滤低置信度
            valid_mask = max_scores >= conf_thres
            valid_boxes = boxes[valid_mask]
            valid_scores = max_scores[valid_mask]
            valid_cls = max_cls[valid_mask]
            
            if len(valid_boxes) == 0:
                predictions.append(np.array([]).reshape(0, 6))
                continue
            
            # 转换坐标格式：从中心点 (cx, cy, w, h) 到角点 (x1, y1, x2, y2)
            # Ultralytics 使用归一化坐标 (0-1)
            xyxy_boxes = torch.zeros_like(valid_boxes)
            xyxy_boxes[:, 0] = (valid_boxes[:, 0] - valid_boxes[:, 2] / 2) * img_w  # x1
            xyxy_boxes[:, 1] = (valid_boxes[:, 1] - valid_boxes[:, 3] / 2) * img_h  # y1
            xyxy_boxes[:, 2] = (valid_boxes[:, 0] + valid_boxes[:, 2] / 2) * img_w  # x2
            xyxy_boxes[:, 3] = (valid_boxes[:, 1] + valid_boxes[:, 3] / 2) * img_h  # y2
            
            # 应用 NMS
            try:
                from torchvision.ops import nms
                keep = nms(xyxy_boxes, valid_scores, iou_threshold=0.45)
                
                result = torch.cat([
                    xyxy_boxes[keep],
                    valid_scores[keep:keep.numel()].unsqueeze(1),
                    valid_cls[keep:keep.numel()].unsqueeze(1).float()
                ], dim=1)
                predictions.append(result.cpu().numpy())
            except ImportError:
                # 没有 torchvision，不使用 NMS
                result = torch.cat([
                    xyxy_boxes,
                    valid_scores.unsqueeze(1),
                    valid_cls.unsqueeze(1).float()
                ], dim=1)
                predictions.append(result.cpu().numpy())
        
        # 如果 batch_size 为 0
        if batch_size == 0:
            predictions.append(np.array([]).reshape(0, 6))
        
        return predictions


class Evaluator:
    """YOLOv8 评估器 - 支持多种模型格式"""
    
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
        
        # 检测模型格式
        model_format = detect_model_format(args.weights)
        print(f"Detected model format: {model_format}")
        
        # 根据模型类型加载模型
        if args.model_type == 'auto':
            self.model_type = model_format
        else:
            self.model_type = args.model_type
        
        # 创建模型包装器
        print("Loading model...")
        self.model = self._load_model(args.weights, model_format)
        self.model.eval()
        
        # 加载权重（如果需要）
        if model_format == 'custom':
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
    
    def _load_model(self, weights_path: str, model_format: str) -> Any:
        """根据模型格式加载模型"""
        
        if model_format == 'ultralytics':
            # 使用 Ultralytics 模型
            return UltralyticsModelWrapper(weights_path, self.device)
        
        elif model_format in ('custom', 'pure_state_dict'):
            # 使用自定义模型
            model = create_model(
                num_classes=self.config['model']['num_classes'],
                width_multiple=self.config['model']['width_multiple'],
                depth_multiple=self.config['model']['depth_multiple'],
                backbone_name=self.config['model'].get('backbone_name', 'CSPDarknet'),
                backbone_pretrained=self.config['model'].get('backbone_pretrained', False)
            )
            model = model.to(self.device)
            return CustomModelWrapper(model, self.config, self.device)
        
        else:
            # 未知格式，尝试作为自定义模型加载
            print("Unknown model format, trying to load as custom model...")
            model = create_model(
                num_classes=self.config['model']['num_classes'],
                width_multiple=self.config['model']['width_multiple'],
                depth_multiple=self.config['model']['depth_multiple'],
                backbone_name=self.config['model'].get('backbone_name', 'CSPDarknet'),
                backbone_pretrained=self.config['model'].get('backbone_pretrained', False)
            )
            model = model.to(self.device)
            return CustomModelWrapper(model, self.config, self.device)
    
    def load_weights(self, weights_path: str):
        """加载模型权重（仅用于自定义格式）"""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        print(f"Loading weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        
        # 处理不同的权重格式
        if 'model_state_dict' in checkpoint:
            # 完整检查点
            if isinstance(self.model, CustomModelWrapper):
                self.model.model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"Checkpoint epoch: {checkpoint['epoch']}")
            if 'metrics' in checkpoint:
                print(f"Checkpoint metrics: {checkpoint['metrics']}")
        elif 'model' in checkpoint:
            # Ultralytics 格式，提取 model 部分
            model_dict = checkpoint['model']
            if isinstance(model_dict, dict):
                # 过滤浮点型键（锚框）
                filtered_dict = {k: v for k, v in model_dict.items() 
                               if isinstance(k, str) and torch.is_tensor(v)}
                if isinstance(self.model, CustomModelWrapper):
                    try:
                        self.model.model.load_state_dict(filtered_dict, strict=False)
                        print("Loaded Ultralytics-style weights with strict=False")
                    except Exception as e:
                        print(f"Warning: Could not load Ultralytics weights: {e}")
        else:
            # 纯 state_dict
            if isinstance(self.model, CustomModelWrapper):
                self.model.model.load_state_dict(checkpoint)
        
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
            try:
                outputs = self.model(images)
            except Exception as e:
                print(f"Error during inference: {e}")
                continue
            
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
            'model_type': self.model_type,
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
    parser = argparse.ArgumentParser(description='YOLOv8 Evaluation Script - Universal Version')
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
    parser.add_argument('--model-type', type=str, default='auto',
                        choices=['auto', 'custom', 'ultralytics', 'pure_state_dict'],
                        help='Model type: auto (detect automatically), custom, ultralytics, or pure_state_dict')
    
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
    eval_args.model_type = args.model_type
    
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
"""
YOLOv8 推理脚本
提供简单的 Python API 进行图像/视频推理和可视化
"""

import os
import sys
import yaml
import argparse
import time
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.yolov8 import YOLOv8, create_model


class YOLOv8Inference:
    """YOLOv8 推理类"""
    
    # 默认颜色（用于可视化）
    DEFAULT_COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (192, 192, 192), (128, 128, 128), (255, 165, 0), (255, 192, 203)
    ]
    
    def __init__(self, config: Dict, weights_path: str, device: str = 'cuda'):
        """
        初始化推理器
        
        Args:
            config: 配置字典
            weights_path: 模型权重路径
            device: 运行设备 ('cuda' 或 'cpu')
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 推理参数
        self.conf_thres = config['inference']['conf_thres']
        self.iou_thres = config['inference']['iou_thres']
        self.image_size = config['training']['image_size']
        
        # 创建模型
        print(f"Loading model on {self.device}...")
        self.model = create_model(config)
        self.model = self.model.to(self.device)
        
        # 加载权重
        self.load_weights(weights_path)
        
        # 类别名称
        self.class_names = getattr(config, 'class_names', 
                                   [f'Class_{i}' for i in range(config['model']['num_classes'])])
        
        # 类别颜色
        self.colors = self._generate_colors(len(self.class_names))
        
        print("Model loaded successfully")
    
    def load_weights(self, weights_path: str):
        """加载模型权重"""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        checkpoint = torch.load(weights_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """为每个类别生成颜色"""
        colors = []
        for i in range(num_classes):
            if i < len(self.DEFAULT_COLORS):
                colors.append(self.DEFAULT_COLORS[i])
            else:
                # 随机生成颜色
                np.random.seed(i)
                color = tuple(np.random.randint(0, 256, 3).tolist())
                colors.append(color)
        return colors
    
    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Dict]:
        """
        图像预处理
        
        Args:
            image: BGR 格式图像 (H, W, C)
        
        Returns:
            预处理后的张量和原始图像信息
        """
        original_shape = image.shape[:2]
        
        # Letterbox 缩放
        letterbox_img, scale, (pad_h, pad_w) = self.letterbox(image, self.image_size)
        
        # 转换为 RGB
        letterbox_img = cv2.cvtColor(letterbox_img, cv2.COLOR_BGR2RGB)
        
        # 转换为张量并归一化
        img_tensor = torch.from_numpy(letterbox_img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # 添加 batch 维度
        
        info = {
            'original_shape': original_shape,
            'scale': scale,
            'pad': (pad_w, pad_h)
        }
        
        return img_tensor, info
    
    def letterbox(self, image: np.ndarray, new_shape: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Letterbox 缩放和填充
        
        Args:
            image: 输入图像
            new_shape: 目标尺寸
        
        Returns:
            处理后的图像、缩放比例、填充大小
        """
        height, width = image.shape[:2]
        
        # 计算缩放比例
        scale = min(new_shape / height, new_shape / width)
        
        # 计算新的尺寸
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        # 调整图像大小
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # 计算填充
        pad_h = new_shape - new_height
        pad_w = new_shape - new_width
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        
        # 填充图像
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                   cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        return padded, scale, (pad_h, pad_w)
    
    def postprocess(self, predictions: List[torch.Tensor], info: Dict) -> List[Dict]:
        """
        后处理：将预测结果转换回原始图像坐标系
        
        Args:
            predictions: 预测结果 (批次中的每个图像)
            info: 预处理信息
        
        Returns:
            检测结果列表
        """
        detections = []
        
        for pred in predictions:
            if pred is None or len(pred) == 0:
                detections.append([])
                continue
            
            # 转换为 numpy
            pred = pred.cpu().numpy()
            
            # 提取坐标和分数
            boxes = pred[:, :4]
            scores = pred[:, 4]
            class_ids = pred[:, 5].astype(int)
            
            # 转换回原始图像坐标
            scale = info['scale']
            pad_w, pad_h = info['pad']
            
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale
            
            # 限制在图像范围内
            orig_h, orig_w = info['original_shape']
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
            
            # 构建检测结果
            for i in range(len(boxes)):
                detections.append({
                    'bbox': boxes[i].astype(int),
                    'score': float(scores[i]),
                    'class_id': int(class_ids[i]),
                    'class_name': self.class_names[int(class_ids[i])]
                })
        
        return detections
    
    def predict(self, image: np.ndarray, conf_thres: Optional[float] = None) -> List[Dict]:
        """
        对单张图像进行推理
        
        Args:
            image: BGR 格式图像
            conf_thres: 置信度阈值（可选，覆盖默认值）
        
        Returns:
            检测结果列表
        """
        # 预处理
        img_tensor, info = self.preprocess(image)
        img_tensor = img_tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(img_tensor)
        
        # 解码预测结果
        conf_thres = conf_thres if conf_thres is not None else self.conf_thres
        predictions = self.model.decode_predictions(
            outputs, 
            img_h=self.image_size, 
            img_w=self.image_size, 
            conf_thres=conf_thres
        )
        
        # 后处理
        detections = self.postprocess(predictions, info)
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict], 
                       show_scores: bool = True, show_labels: bool = True) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果列表
            show_scores: 是否显示置信度分数
            show_labels: 是否显示类别标签
        
        Returns:
            绘制后的图像
        """
        img = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            score = det['score']
            class_id = det['class_id']
            class_name = det['class_name']
            color = self.colors[class_id % len(self.colors)]
            
            # 绘制边界框
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            if show_scores or show_labels:
                label = ''
                if show_labels:
                    label += class_name
                if show_scores:
                    if label:
                        label += f' {score:.2f}'
                    else:
                        label += f'{score:.2f}'
                
                # 计算标签文本大小
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # 绘制标签背景
                cv2.rectangle(img, (x1, y1 - text_height - 4), 
                             (x1 + text_width + 4, y1), color, -1)
                
                # 绘制标签文本
                cv2.putText(img, label, (x1 + 2, y1 - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def predict_and_draw(self, image: np.ndarray, conf_thres: Optional[float] = None,
                        show_scores: bool = True, show_labels: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        推理并绘制结果
        
        Args:
            image: 输入图像
            conf_thres: 置信度阈值
            show_scores: 是否显示分数
            show_labels: 是否显示标签
        
        Returns:
            绘制后的图像和检测结果
        """
        detections = self.predict(image, conf_thres)
        result_img = self.draw_detections(image, detections, show_scores, show_labels)
        return result_img, detections


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Inference Script')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image, video, or directory')
    parser.add_argument('--config', type=str, default='configs/hyperparameters.yaml',
                        help='Path to config file')
    parser.add_argument('--conf-thres', type=float, default=None,
                        help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=None,
                        help='IoU threshold for NMS')
    parser.add_argument('--output', type=str, default='results/inference',
                        help='Output directory')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results as text files')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not show results')
    parser.add_argument('--cpu', action='store_true',
                        help='Force inference on CPU')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 覆盖配置参数
    if args.conf_thres is not None:
        config['inference']['conf_thres'] = args.conf_thres
    if args.iou_thres is not None:
        config['inference']['iou_thres'] = args.iou_thres
    
    # 创建推理器
    device = 'cpu' if args.cpu else 'cuda'
    inferencer = YOLOv8Inference(config, args.weights, device)
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理输入
    source_path = Path(args.source)
    
    if source_path.is_file():
        # 单个图像
        if source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            print(f"\nProcessing image: {source_path}")
            
            # 读取图像
            image = cv2.imread(str(source_path))
            
            # 推理
            result_img, detections = inferencer.predict_and_draw(image)
            
            # 保存结果
            output_path = output_dir / source_path.name
            cv2.imwrite(str(output_path), result_img)
            print(f"Result saved to: {output_path}")
            
            # 保存文本文件
            if args.save_txt and detections:
                txt_path = output_dir / f"{source_path.stem}.txt"
                with open(txt_path, 'w') as f:
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        f.write(f"{det['class_id']} {det['score']:.4f} {x1} {y1} {x2} {y2}\n")
                print(f"Detections saved to: {txt_path}")
            
            # 显示结果
            if not args.no_show:
                cv2.imshow('YOLOv8 Detection', result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        # 视频
        elif source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            print(f"\nProcessing video: {source_path}")
            
            # 打开视频
            cap = cv2.VideoCapture(str(source_path))
            
            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 创建输出视频
            output_path = output_dir / f"output_{source_path.name}"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            print(f"Processing {total_frames} frames...")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 推理
                result_frame, _ = inferencer.predict_and_draw(frame)
                
                # 写入输出视频
                out.write(result_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")
            
            cap.release()
            out.release()
            print(f"Video saved to: {output_path}")
    
    elif source_path.is_dir():
        # 目录中的所有图像
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(source_path.glob(ext))
        
        print(f"\nFound {len(image_files)} images")
        
        for img_path in image_files:
            print(f"Processing: {img_path.name}")
            
            # 读取图像
            image = cv2.imread(str(img_path))
            
            # 推理
            result_img, detections = inferencer.predict_and_draw(image)
            
            # 保存结果
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), result_img)
            
            # 保存文本文件
            if args.save_txt and detections:
                txt_path = output_dir / f"{img_path.stem}.txt"
                with open(txt_path, 'w') as f:
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        f.write(f"{det['class_id']} {det['score']:.4f} {x1} {y1} {x2} {y2}\n")
        
        print(f"\nAll results saved to: {output_dir}")
    
    print("\nInference completed!")


if __name__ == '__main__':
    main()

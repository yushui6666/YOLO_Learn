"""
YOLOv8 简单 API
不需要命令行，直接通过 Python 代码调用训练、评估和推理功能
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(SCRIPT_DIR)

from models.yolov8 import create_model
from data.dataset import create_dataloader
from utils.metrics import MetricsCalculator
from train import Trainer
from evaluate import Evaluator
from infer import YOLOv8Inference


class YOLOv8:
    """YOLOv8 高级 API"""
    
    def __init__(self, config_path: str = None):
        """
        初始化 YOLOv8
        
        Args:
            config_path: 配置文件路径
        """
        # 默认配置文件路径
        if config_path is None:
            config_path = os.path.join(SCRIPT_DIR, 'configs/best.yaml')
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 将数据集路径转换为绝对路径
        for key in ['train', 'val', 'annotations_train', 'annotations_val']:
            if key in self.config.get('dataset', {}):
                path = self.config['dataset'][key]
                if not os.path.isabs(path):
                    self.config['dataset'][key] = os.path.join(SCRIPT_DIR, path)
        
        self.config_path = config_path
        self.model = None
        self.device = None
    
    def train(self, 
              train_data_path: str, 
              val_data_path: str,
              output_dir: str = 'runs/train',
              epochs: Optional[int] = None,
              batch_size: Optional[int] = None,
              resume: Optional[str] = None):
        """
        训练模型
        
        Args:
            train_data_path: 训练数据路径
            val_data_path: 验证数据路径
            output_dir: 输出目录
            epochs: 训练轮数（可选，覆盖配置文件）
            batch_size: 批次大小（可选，覆盖配置文件）
            resume: 从检查点恢复的路径（可选）
        
        Returns:
            训练器对象
        """
        # 覆盖配置
        if epochs is not None:
            self.config['training']['epochs'] = epochs
        if batch_size is not None:
            self.config['training']['batch_size'] = batch_size
        
        # 创建参数对象
        class Args:
            def __init__(self, config_path, train_data, val_data, output_dir, resume, cpu=False):
                self.config = config_path
                self.train_data = train_data
                self.val_data = val_data
                self.output_dir = output_dir
                self.resume = resume
                self.cpu = cpu
        
        args = Args(self.config_path, train_data_path, val_data_path, output_dir, resume)
        
        # 创建训练器
        trainer = Trainer(self.config, args)
        
        # 开始训练
        trainer.train()
        
        return trainer
    
    def evaluate(self,
                 weights_path: str,
                 data_path: str,
                 output_path: str = 'results/evaluation_results.json',
                 conf_thres: Optional[float] = None,
                 iou_thres: Optional[float] = None):
        """
        评估模型
        
        Args:
            weights_path: 模型权重路径
            data_path: 验证数据路径
            output_path: 结果输出路径
            conf_thres: 置信度阈值（可选）
            iou_thres: IoU 阈值（可选）
        
        Returns:
            评估指标字典
        """
        # 创建参数对象
        class Args:
            def __init__(self, weights, data, config, conf_thres, iou_thres, output, cpu=False):
                self.weights = weights
                self.data = data
                self.config = config
                self.conf_thres = conf_thres
                self.iou_thres = iou_thres
                self.output = output
                self.cpu = cpu
        
        args = Args(weights_path, data_path, self.config_path, conf_thres, iou_thres, output_path)
        
        # 创建评估器
        evaluator = Evaluator(self.config, args)
        
        # 执行评估
        metrics = evaluator.evaluate()
        
        # 打印结果
        evaluator.print_metrics(metrics)
        
        # 保存结果
        evaluator.save_results(metrics, output_path)
        
        return metrics
    
    def load_model(self, weights_path: str, device: str = 'cuda'):
        """
        加载模型用于推理
        
        Args:
            weights_path: 模型权重路径
            device: 运行设备 ('cuda' 或 'cpu')
        
        Returns:
            YOLOv8Inference 推理对象
        """
        self.inferencer = YOLOv8Inference(self.config, weights_path, device)
        return self.inferencer
    
    def predict(self, 
                image_path: str,
                weights_path: str,
                conf_thres: Optional[float] = None,
                visualize: bool = True,
                save_path: Optional[str] = None):
        """
        对单张图像进行推理
        
        Args:
            image_path: 图像路径
            weights_path: 模型权重路径
            conf_thres: 置信度阈值
            visualize: 是否可视化结果
            save_path: 保存可视化结果的路径（可选）
        
        Returns:
            检测结果列表
        """
        import cv2
        
        # 加载模型
        if not hasattr(self, 'inferencer'):
            self.load_model(weights_path)
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 推理
        detections = self.inferencer.predict(image, conf_thres)
        
        # 可视化
        if visualize:
            result_img = self.inferencer.draw_detections(image, detections)
            
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(save_path, result_img)
                print(f"结果已保存到: {save_path}")
        
        return detections


# 便捷函数
def train_yolov8(train_data_path: str,
                  val_data_path: str,
                  config_path: str = None,
                  output_dir: str = 'runs/train',
                  epochs: Optional[int] = None,
                  batch_size: Optional[int] = None,
                  resume: Optional[str] = None):
    """
    便捷的训练函数
    
    Args:
        train_data_path: 训练数据路径
        val_data_path: 验证数据路径
        config_path: 配置文件路径
        output_dir: 输出目录
        epochs: 训练轮数（可选）
        batch_size: 批次大小（可选）
        resume: 从检查点恢复的路径（可选）
    
    Example:
        >>> train_yolov8(
        ...     train_data_path='datasets/coco/train',
        ...     val_data_path='datasets/coco/val',
        ...     epochs=100,
        ...     batch_size=16
        ... )
    """
    yolov8 = YOLOv8(config_path)
    return yolov8.train(train_data_path, val_data_path, output_dir, epochs, batch_size, resume)


def evaluate_yolov8(weights_path: str,
                     data_path: str,
                     config_path: str = None,
                     output_path: str = 'results/evaluation_results.json',
                     conf_thres: Optional[float] = None,
                     iou_thres: Optional[float] = None):
    """
    便捷的评估函数
    
    Args:
        weights_path: 模型权重路径
        data_path: 验证数据路径
        config_path: 配置文件路径
        output_path: 结果输出路径
        conf_thres: 置信度阈值（可选）
        iou_thres: IoU 阈值（可选）
    
    Returns:
        评估指标字典
    
    Example:
        >>> metrics = evaluate_yolov8(
        ...     weights_path='runs/train/best_model.pt',
        ...     data_path='datasets/coco/val'
        ... )
        >>> print(f"mAP@0.5: {metrics['map50']}")
    """
    yolov8 = YOLOv8(config_path)
    return yolov8.evaluate(weights_path, data_path, output_path, conf_thres, iou_thres)


def predict_yolov8(image_path: str,
                    weights_path: str,
                    config_path: str = None,
                    conf_thres: Optional[float] = None,
                    visualize: bool = True,
                    save_path: Optional[str] = None):
    """
    便捷的推理函数
    
    Args:
        image_path: 图像路径
        weights_path: 模型权重路径
        config_path: 配置文件路径
        conf_thres: 置信度阈值（可选）
        visualize: 是否可视化结果
        save_path: 保存可视化结果的路径（可选）
    
    Returns:
        检测结果列表
    
    Example:
        >>> detections = predict_yolov8(
        ...     image_path='test.jpg',
        ...     weights_path='runs/train/best_model.pt',
        ...     save_path='result.jpg'
        ... )
        >>> for det in detections:
        ...     print(f"{det['class_name']}: {det['score']:.2f}")
    """
    yolov8 = YOLOv8(config_path)
    return yolov8.predict(image_path, weights_path, conf_thres, visualize, save_path)


# 使用示例
if __name__ == '__main__':
    # 示例 1: 训练模型
    print("示例 1: 训练模型")
    # train_yolov8(
    #     train_data_path='datasets/coco/train',
    #     val_data_path='datasets/coco/val',
    #     epochs=100,
    #     batch_size=16
    # )
    
    # 示例 2: 评估模型
    print("\n示例 2: 评估模型")
    # metrics = evaluate_yolov8(
    #     weights_path='runs/train/best_model.pt',
    #     data_path='datasets/coco/val'
    # )
    # print(f"mAP@0.5: {metrics['map50']}")
    # print(f"mAP@0.5:0.95: {metrics['map50_95']}")
    
    # 示例 3: 推理
    print("\n示例 3: 推理")
    # detections = predict_yolov8(
    #     image_path='test.jpg',
    #     weights_path='runs/train/best_model.pt',
    #     conf_thres=0.5,
    #     save_path='result.jpg'
    # )
    # for det in detections:
    #     print(f"{det['class_name']}: {det['score']:.2f} at {det['bbox']}")
    
    print("\n所有示例都已注释。取消注释以运行相应的功能。")

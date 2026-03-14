"""
YOLOv8 错误分析工具
用于分析模型的检测错误，包括漏检、误检和混淆矩阵
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class ErrorAnalyzer:
    """
    错误分析器
    用于分析目标检测模型的错误类型
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None, iou_threshold: float = 0.5):
        """
        初始化错误分析器
        
        Args:
            num_classes: 类别数量
            class_names: 类别名称列表
            iou_threshold: IoU 阈值，用于判断预测是否正确
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        self.iou_threshold = iou_threshold
        
        self.reset()
    
    def reset(self):
        """重置所有统计数据"""
        self.all_predictions = []  # List of {'image_id', 'box', 'label', 'score'}
        self.all_ground_truths = []  # List of {'image_id', 'box', 'label'}
        self.image_ids = set()
    
    def update(self, pred_boxes: List[np.ndarray], pred_labels: List[np.ndarray], 
               pred_scores: List[np.ndarray], gt_boxes: List[np.ndarray], 
               gt_labels: List[np.ndarray], image_ids: List):
        """
        更新统计数据
        
        Args:
            pred_boxes: 预测框列表，每个元素为 (N, 4) 的 numpy 数组
            pred_labels: 预测标签列表
            pred_scores: 预测置信度列表
            gt_boxes: 真实框列表
            gt_labels: 真实标签列表
            image_ids: 图像 ID 列表
        """
        for i, img_id in enumerate(image_ids):
            image_id = str(img_id)
            self.image_ids.add(image_id)
            
            # 添加预测
            for j in range(len(pred_boxes[i])):
                self.all_predictions.append({
                    'image_id': image_id,
                    'box': pred_boxes[i][j],
                    'label': int(pred_labels[i][j]),
                    'score': float(pred_scores[i][j]),
                })
            
            # 添加真实标注
            for j in range(len(gt_boxes[i])):
                self.all_ground_truths.append({
                    'image_id': image_id,
                    'box': gt_boxes[i][j],
                    'label': int(gt_labels[i][j]),
                })
    
    def compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个框的 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y3 = min(box1[3], box2[3])
        
        if x2 <= x1 or y3 <= y1:
            return 0.0
        
        inter = (x2 - x1) * (y3 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
    
    def compute_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """向量化计算 IoU 矩阵"""
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
        
        boxes1 = np.asarray(boxes1, dtype=np.float32)
        boxes2 = np.asarray(boxes2, dtype=np.float32)
        
        x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
        
        inter_w = np.clip(x2 - x1, 0, None)
        inter_h = np.clip(y2 - y1, 0, None)
        inter_area = inter_w * inter_h
        
        area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], 0, None) * \
                np.clip(boxes1[:, 3] - boxes1[:, 1], 0, None)
        area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], 0, None) * \
                np.clip(boxes2[:, 3] - boxes2[:, 1], 0, None)
        
        union = area1[:, None] + area2[None, :] - inter_area
        return inter_area / np.clip(union, 1e-10, None)
    
    def analyze(self, conf_threshold: float = 0.001) -> Dict:
        """
        分析错误类型
        
        Args:
            conf_threshold: 置信度阈值
            
        Returns:
            包含错误分析结果的字典
        """
        # 按图像分组
        preds_by_image = defaultdict(list)
        for pred in self.all_predictions:
            if pred['score'] >= conf_threshold:
                preds_by_image[pred['image_id']].append(pred)
        
        gts_by_image = defaultdict(list)
        for gt in self.all_ground_truths:
            gts_by_image[gt['image_id']].append(gt)
        
        # 统计
        tp = 0  # True Positive
        fp = 0  # False Positive
        fn = 0  # False Negative
        tn = 0  # True Negative (在检测任务中通常不计算)
        
        # 按错误类型分类
        fp_loc = 0  # 定位错误
        fp_cls = 0  # 分类错误
        fp_bg = 0   # 背景误检
        
        fn_small = 0  # 小目标漏检
        fn_medium = 0  # 中等目标漏检
        fn_large = 0  # 大目标漏检
        
        # 混淆矩阵数据
        confusion_matrix = np.zeros((self.num_classes + 1, self.num_classes + 1), dtype=np.int32)
        # 行：预测类别（最后一行是背景）
        # 列：真实类别（最后一列是背景）
        
        for image_id in self.image_ids:
            preds = preds_by_image.get(image_id, [])
            gts = gts_by_image.get(image_id, [])
            
            if len(preds) == 0:
                # 没有预测，所有 GT 都是 FN
                for gt in gts:
                    fn += 1
                    confusion_matrix[self.num_classes, gt['label']] += 1  # 背景行，真实列
                    area = (gt['box'][2] - gt['box'][0]) * (gt['box'][3] - gt['box'][1])
                    if area < 32**2:
                        fn_small += 1
                    elif area < 96**2:
                        fn_medium += 1
                    else:
                        fn_large += 1
                continue
            
            if len(gts) == 0:
                # 没有 GT，所有预测都是 FP
                for pred in preds:
                    fp += 1
                    fp_bg += 1
                    confusion_matrix[pred['label'], self.num_classes] += 1  # 预测列，背景行
                continue
            
            # 构建 IoU 矩阵
            pred_boxes = np.stack([p['box'] for p in preds])
            gt_boxes = np.stack([g['box'] for g in gts])
            iou_matrix = self.compute_iou_matrix(pred_boxes, gt_boxes)
            
            # 类别匹配矩阵
            pred_labels = np.array([p['label'] for p in preds])
            gt_labels = np.array([g['label'] for g in gts])
            
            gt_matched = np.zeros(len(gts), dtype=bool)
            
            # 按置信度排序预测
            sorted_indices = np.argsort([-p['score'] for p in preds])
            
            for pi in sorted_indices:
                pred = preds[pi]
                ious = iou_matrix[pi]
                
                # 找到最佳匹配的 GT
                best_iou = 0
                best_gt_idx = -1
                
                for gi in range(len(gts)):
                    if gt_matched[gi]:
                        continue
                    if ious[gi] > best_iou:
                        best_iou = ious[gi]
                        best_gt_idx = gi
                
                if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                    # TP 或分类错误
                    gt = gts[best_gt_idx]
                    gt_matched[best_gt_idx] = True
                    
                    if pred['label'] == gt['label']:
                        tp += 1
                        confusion_matrix[pred['label'], gt['label']] += 1
                    else:
                        fp += 1
                        fp_cls += 1
                        fn += 1
                        confusion_matrix[pred['label'], gt['label']] += 1
                else:
                    # FP（背景误检或定位不准）
                    fp += 1
                    fp_bg += 1
                    confusion_matrix[pred['label'], self.num_classes] += 1
        
        # 检查未匹配的 GT（FN）
        for gi, gt in enumerate(gts_by_image.get(image_id, [])):
            pass  # 已在上面处理
        
        # 计算漏检率、误检率
        total_gt = len(self.all_ground_truths)
        total_pred = len([p for p in self.all_predictions if p['score'] >= conf_threshold])
        
        miss_rate = fn / total_gt if total_gt > 0 else 0.0
        false_alarm_rate = fp / total_pred if total_pred > 0 else 0.0
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'fp_loc': fp_loc,
            'fp_cls': fp_cls,
            'fp_bg': fp_bg,
            'fn_small': fn_small,
            'fn_medium': fn_medium,
            'fn_large': fn_large,
            'miss_rate': miss_rate,
            'false_alarm_rate': false_alarm_rate,
            'confusion_matrix': confusion_matrix,
            'total_gt': total_gt,
            'total_pred': total_pred,
        }
    
    def compute_confusion_matrix(self, conf_threshold: float = 0.001) -> np.ndarray:
        """
        计算混淆矩阵
        
        Args:
            conf_threshold: 置信度阈值
            
        Returns:
            (num_classes+1) x (num_classes+1) 的混淆矩阵
        """
        results = self.analyze(conf_threshold)
        return results['confusion_matrix']
    
    def plot_confusion_matrix(self, conf_threshold: float = 0.001, 
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        绘制混淆矩阵热力图
        
        Args:
            conf_threshold: 置信度阈值
            save_path: 保存路径（可选）
            figsize: 图像尺寸
            
        Returns:
            matplotlib Figure 对象
        """
        cm = self.compute_confusion_matrix(conf_threshold)
        
        # 创建标签（添加 Background）
        labels = self.class_names + ['Background']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 使用 seaborn 绘制热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels, ax=ax)
        
        ax.set_xlabel('True Label')
        ax.set_ylabel('Predicted Label')
        ax.set_title(f'Confusion Matrix (IoU={self.iou_threshold}, conf={conf_threshold})')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_error_distribution(self, results: Optional[Dict] = None,
                                conf_threshold: float = 0.001,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制错误分布图
        
        Args:
            results: analyze() 返回的结果（可选，如果不提供则重新计算）
            conf_threshold: 置信度阈值
            save_path: 保存路径（可选）
            
        Returns:
            matplotlib Figure 对象
        """
        if results is None:
            results = self.analyze(conf_threshold)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. 错误类型饼图
        ax1 = axes[0]
        error_types = ['TP', 'FP', 'FN']
        error_counts = [results['tp'], results['fp'], results['fn']]
        colors = ['green', 'red', 'orange']
        ax1.pie(error_counts, labels=error_types, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Detection Results Distribution')
        
        # 2. FP 类型分布
        ax2 = axes[1]
        fp_types = ['BG误检', '分类错误', '定位错误']
        fp_counts = [results['fp_bg'], results['fp_cls'], results['fp_loc']]
        colors = ['lightcoral', 'salmon', 'indianred']
        ax2.bar(fp_types, fp_counts, color=colors)
        ax2.set_ylabel('Count')
        ax2.set_title('False Positive Breakdown')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. FN 按目标大小分布
        ax3 = axes[2]
        fn_types = ['小目标\n(<32²)', '中等目标\n(32²-96²)', '大目标\n(>96²)']
        fn_counts = [results['fn_small'], results['fn_medium'], results['fn_large']]
        colors = ['lightblue', 'steelblue', 'navy']
        ax3.bar(fn_types, fn_counts, color=colors)
        ax3.set_ylabel('Count')
        ax3.set_title('False Negative by Object Size')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def print_report(self, results: Optional[Dict] = None, 
                     conf_threshold: float = 0.001) -> str:
        """
        打印错误分析报告
        
        Args:
            results: analyze() 返回的结果（可选）
            conf_threshold: 置信度阈值
            
        Returns:
            格式化的报告字符串
        """
        if results is None:
            results = self.analyze(conf_threshold)
        
        report = []
        report.append("=" * 60)
        report.append("YOLOv8 错误分析报告")
        report.append("=" * 60)
        report.append("")
        
        report.append("【检测结果统计】")
        report.append(f"  真实目标数 (GT): {results['total_gt']}")
        report.append(f"  预测目标数：{results['total_pred']}")
        report.append(f"  True Positive (TP): {results['tp']}")
        report.append(f"  False Positive (FP): {results['fp']}")
        report.append(f"  False Negative (FN): {results['fn']}")
        report.append("")
        
        report.append("【错误率】")
        report.append(f"  漏检率 (Miss Rate): {results['miss_rate']:.4f} ({results['miss_rate']*100:.2f}%)")
        report.append(f"  误检率 (False Alarm Rate): {results['false_alarm_rate']:.4f} ({results['false_alarm_rate']*100:.2f}%)")
        report.append("")
        
        report.append("【FP 类型分析】")
        report.append(f"  背景误检：{results['fp_bg']}")
        report.append(f"  分类错误：{results['fp_cls']}")
        report.append(f"  定位错误：{results['fp_loc']}")
        report.append("")
        
        report.append("【FN 按目标大小分析】")
        report.append(f"  小目标 (<32²): {results['fn_small']}")
        report.append(f"  中等目标 (32²-96²): {results['fn_medium']}")
        report.append(f"  大目标 (>96²): {results['fn_large']}")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def compute_miss_rate_per_class(
    pred_boxes: List[np.ndarray], pred_labels: List[np.ndarray], pred_scores: List[np.ndarray],
    gt_boxes: List[np.ndarray], gt_labels: List[np.ndarray],
    num_classes: int, iou_threshold: float = 0.5, conf_threshold: float = 0.001
) -> Dict[int, float]:
    """
    计算每个类别的漏检率
    
    Args:
        pred_boxes: 预测框列表
        pred_labels: 预测标签列表
        pred_scores: 预测置信度列表
        gt_boxes: 真实框列表
        gt_labels: 真实标签列表
        num_classes: 类别数量
        iou_threshold: IoU 阈值
        conf_threshold: 置信度阈值
        
    Returns:
        每个类别的漏检率字典
    """
    # 统计每个类别的 GT 数量
    gt_count_per_class = defaultdict(int)
    for labels in gt_labels:
        for label in labels:
            gt_count_per_class[int(label)] += 1
    
    # 统计每个类别的 FN 数量
    fn_count_per_class = defaultdict(int)
    
    analyzer = ErrorAnalyzer(num_classes, iou_threshold=iou_threshold)
    analyzer.update(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, range(len(pred_boxes)))
    results = analyzer.analyze(conf_threshold)
    
    # 从混淆矩阵计算每类的 FN
    cm = results['confusion_matrix']
    miss_rates = {}
    
    for class_id in range(num_classes):
        # FN = 真实为该类别但预测为其他类别或背景的数量
        fn = cm[:, class_id].sum() - cm[class_id, class_id]
        gt_total = gt_count_per_class[class_id]
        miss_rates[class_id] = fn / gt_total if gt_total > 0 else 0.0
    
    return miss_rates


def compute_false_alarm_rate_per_class(
    pred_boxes: List[np.ndarray], pred_labels: List[np.ndarray], pred_scores: List[np.ndarray],
    gt_boxes: List[np.ndarray], gt_labels: List[np.ndarray],
    num_classes: int, iou_threshold: float = 0.5, conf_threshold: float = 0.001
) -> Dict[int, float]:
    """
    计算每个类别的误检率
    
    Args:
        pred_boxes: 预测框列表
        pred_labels: 预测标签列表
        pred_scores: 预测置信度列表
        gt_boxes: 真实框列表
        gt_labels: 真实标签列表
        num_classes: 类别数量
        iou_threshold: IoU 阈值
        conf_threshold: 置信度阈值
        
    Returns:
        每个类别的误检率字典
    """
    # 统计每个类别的预测数量
    pred_count_per_class = defaultdict(int)
    for scores, labels in zip(pred_scores, pred_labels):
        for score, label in zip(scores, labels):
            if score >= conf_threshold:
                pred_count_per_class[int(label)] += 1
    
    analyzer = ErrorAnalyzer(num_classes, iou_threshold=iou_threshold)
    analyzer.update(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, range(len(pred_boxes)))
    results = analyzer.analyze(conf_threshold)
    
    # 从混淆矩阵计算每类的 FP
    cm = results['confusion_matrix']
    false_alarm_rates = {}
    
    for class_id in range(num_classes):
        # FP = 预测为该类别但真实为其他类别或背景的数量
        fp = cm[class_id, :].sum() - cm[class_id, class_id]
        pred_total = pred_count_per_class[class_id]
        false_alarm_rates[class_id] = fp / pred_total if pred_total > 0 else 0.0
    
    return false_alarm_rates


if __name__ == '__main__':
    # 测试代码
    analyzer = ErrorAnalyzer(num_classes=5, class_names=['A', 'B', 'C', 'D', 'E'])
    
    # 模拟数据
    pred_boxes = [
        np.array([[10, 10, 50, 50], [100, 100, 150, 150], [200, 200, 250, 250]]),
        np.array([[20, 20, 60, 60]]),
    ]
    pred_labels = [
        np.array([0, 1, 2]),
        np.array([0]),
    ]
    pred_scores = [
        np.array([0.9, 0.8, 0.3]),
        np.array([0.7]),
    ]
    
    gt_boxes = [
        np.array([[12, 12, 52, 52], [105, 105, 155, 155]]),
        np.array([[25, 25, 65, 65]]),
    ]
    gt_labels = [
        np.array([0, 1]),
        np.array([0]),
    ]
    image_ids = [0, 1]
    
    analyzer.update(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, image_ids)
    
    # 分析
    results = analyzer.analyze(conf_threshold=0.5)
    
    # 打印报告
    report = analyzer.print_report(results)
    print(report)
    
    # 绘制混淆矩阵
    analyzer.plot_confusion_matrix(conf_threshold=0.5, save_path='confusion_matrix.png')
    print("\nConfusion matrix saved to confusion_matrix.png")
    
    # 绘制错误分布
    analyzer.plot_error_distribution(results, save_path='error_distribution.png')
    print("Error distribution saved to error_distribution.png")
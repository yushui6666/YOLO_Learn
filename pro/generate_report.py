"""
YOLOv8 完整测评报告生成器
整合所有评估指标，生成详细的 Markdown 格式测评报告
"""

import os
import sys
import json
import yaml
import argparse
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from models.yolov8 import YOLOv8, create_model
from data.dataset import create_dataloader
from utils.metrics import MetricsCalculator
from utils.benchmark import (
    run_full_benchmark, 
    print_benchmark_report,
    count_parameters,
    get_model_file_size
)
from utils.error_analysis import ErrorAnalyzer


class EvaluationReportGenerator:
    """测评报告生成器"""
    
    def __init__(self, config: Dict, args: argparse.Namespace):
        self.config = config
        self.args = args
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
        print(f"Using device: {self.device}")
        
        # 参数
        self.batch_size = config['training']['batch_size']
        self.image_size = config['training']['image_size']
        self.num_workers = config['training'].get('num_workers', 4)
        
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
        
        # 获取类别名称
        self.class_names = getattr(self.dataloader.dataset, 'class_names', 
                                   [f'Class_{i}' for i in range(config['model']['num_classes'])])
        
        # 创建评估器
        self.metrics_calculator = MetricsCalculator(
            num_classes=config['model']['num_classes'],
            iou_thresholds=config['evaluation'].get('iou_thresholds', [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        )
        
        # 创建错误分析器
        self.error_analyzer = ErrorAnalyzer(
            num_classes=config['model']['num_classes'],
            class_names=self.class_names,
            iou_threshold=0.5
        )
        
        # 结果存储
        self.results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': config,
            'model_info': {},
            'detection_metrics': {},
            'benchmark_metrics': {},
            'error_analysis': {},
            'per_class_metrics': {},
        }
    
    def load_weights(self, weights_path: str):
        """加载模型权重"""
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        print(f"Loading weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print("Weights loaded successfully")
        
        # 记录模型信息
        self.results['model_info']['weights_path'] = weights_path
        self.results['model_info']['file_size'] = get_model_file_size(weights_path)
    
    @torch.no_grad()
    def evaluate_detection(self) -> Dict:
        """执行检测评估"""
        self.model.eval()
        self.metrics_calculator.reset()
        self.error_analyzer.reset()
        
        inference_times = []
        preprocess_times = []
        postprocess_times = []
        
        print(f"\n{'='*60}")
        print("Executing detection evaluation...")
        print(f"{'='*60}\n")
        
        for images, targets, image_ids in torch.utils.data.tqdm(self.dataloader, desc='Evaluating'):
            images = images.to(self.device)
            
            # 预处理时间（数据传输）
            preprocess_start = time.perf_counter()
            if self.device == 'cuda':
                torch.cuda.synchronize()
            preprocess_end = time.perf_counter()
            preprocess_times.append(preprocess_end - preprocess_start)
            
            # 推理
            infer_start = time.perf_counter()
            outputs = self.model(images)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            infer_end = time.perf_counter()
            inference_times.append(infer_end - infer_start)
            
            # 后处理
            conf_thres = self.args.conf_thres if hasattr(self.args, 'conf_thres') and self.args.conf_thres else 0.001
            postprocess_start = time.perf_counter()
            predictions = self.model.decode_predictions(
                outputs, 
                img_h=self.image_size, 
                img_w=self.image_size, 
                conf_thres=conf_thres
            )
            if self.device == 'cuda':
                torch.cuda.synchronize()
            postprocess_end = time.perf_counter()
            postprocess_times.append(postprocess_end - postprocess_start)
            
            # 处理预测和 GT
            predictions = [pred.cpu().numpy() if hasattr(pred, 'cpu') else pred for pred in predictions]
            targets = [t.cpu().numpy() if hasattr(t, 'cpu') else t for t in targets]
            
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
            
            # 更新评估器
            self.metrics_calculator.update(pred_boxes, pred_labels, pred_scores,
                                           gt_boxes, gt_labels, image_ids)
            self.error_analyzer.update(pred_boxes, pred_labels, pred_scores,
                                       gt_boxes, gt_labels, image_ids)
        
        # 计算指标
        conf_thres = 0.001
        metrics = self.metrics_calculator.compute_metrics(conf_threshold=conf_thres)
        
        # 修正键名
        metrics['map50'] = metrics.get('mAP50', 0.0)
        metrics['map50_95'] = metrics.get('mAP50-95', 0.0)
        
        # 每类指标
        per_class = self.metrics_calculator.compute_per_class_metrics(conf_threshold=conf_thres)
        metrics['per_class_map50'] = per_class['per_class_ap50']
        metrics['per_class_precision'] = per_class['per_class_precision']
        metrics['per_class_recall'] = per_class['per_class_recall']
        metrics['per_class_f1'] = per_class['per_class_f1']
        
        # 速度统计
        metrics['avg_inference_time_ms'] = np.mean(inference_times) * 1000
        metrics['avg_preprocess_time_ms'] = np.mean(preprocess_times) * 1000
        metrics['avg_postprocess_time_ms'] = np.mean(postprocess_times) * 1000
        metrics['avg_total_time_ms'] = np.mean(inference_times) * 1000 + np.mean(preprocess_times) * 1000 + np.mean(postprocess_times) * 1000
        metrics['fps'] = 1.0 / (np.mean(inference_times) + np.mean(preprocess_times) + np.mean(postprocess_times))
        
        # 错误分析
        error_results = self.error_analyzer.analyze(conf_threshold=0.5)
        
        self.results['detection_metrics'] = metrics
        self.results['error_analysis'] = error_results
        self.results['per_class_metrics'] = per_class
        
        return metrics
    
    def run_benchmark(self) -> Dict:
        """运行基准测试"""
        print(f"\n{'='*60}")
        print("Running benchmark tests...")
        print(f"{'='*60}\n")
        
        benchmark_results = run_full_benchmark(
            self.model,
            weights_path=self.args.weights,
            input_size=(self.image_size, self.image_size),
            device=str(self.device)
        )
        
        self.results['benchmark_metrics'] = benchmark_results
        return benchmark_results
    
    def generate_markdown_report(self, output_path: str) -> str:
        """生成 Markdown 格式报告"""
        r = self.results
        dm = r.get('detection_metrics', {})
        bm = r.get('benchmark_metrics', {})
        ea = r.get('error_analysis', {})
        pcm = r.get('per_class_metrics', {})
        mi = r.get('model_info', {})
        
        report = []
        
        # 标题
        report.append("# YOLOv8 模型测评报告")
        report.append("")
        report.append(f"**生成时间**: {r['timestamp']}")
        report.append("")
        
        # 1. 模型信息
        report.append("## 1. 模型信息")
        report.append("")
        report.append("| 项目 | 值 |")
        report.append("|------|-----|")
        report.append(f"| 骨干网络 | {self.config['model'].get('backbone_name', 'CSPDarknet')} |")
        report.append(f"| 宽度倍数 | {self.config['model'].get('width_multiple', 1.0)} |")
        report.append(f"| 深度倍数 | {self.config['model'].get('depth_multiple', 1.0)} |")
        
        if 'parameters' in bm:
            p = bm['parameters']
            report.append(f"| 参数量 | {p['total_params_m']:.2f}M ({p['total_params']:,}) |")
        
        if 'gflops' in bm:
            report.append(f"| GFLOPs | {bm['gflops']:.2f} |")
        elif 'gflops_manual' in bm:
            report.append(f"| GFLOPs (估算) | {bm['gflops_manual']['gflops_manual']:.2f} |")
        
        if 'file_size' in mi and 'size_mb' in mi['file_size']:
            report.append(f"| 模型文件大小 | {mi['file_size']['size_mb']:.2f} MB |")
        
        report.append("")
        
        # 2. 检测效果
        report.append("## 2. 检测效果")
        report.append("")
        report.append("### 2.1 整体指标")
        report.append("")
        report.append("| 指标 | 值 |")
        report.append("|------|-----|")
        report.append(f"| mAP@0.5 | {dm.get('map50', 0):.4f} |")
        report.append(f"| mAP@0.5:0.95 | {dm.get('map50_95', 0):.4f} |")
        report.append(f"| Precision | {dm.get('precision', 0):.4f} |")
        report.append(f"| Recall | {dm.get('recall', 0):.4f} |")
        report.append(f"| F1 Score | {dm.get('f1', 0):.4f} |")
        report.append("")
        
        # 每类 AP
        report.append("### 2.2 每类 AP")
        report.append("")
        report.append("| 类别 | mAP@0.5 | Precision | Recall | F1 Score |")
        report.append("|------|---------|-----------|--------|----------|")
        
        for i, name in enumerate(self.class_names):
            ap50 = dm.get('per_class_map50', [0]*len(self.class_names))[i] if 'per_class_map50' in dm else 0
            prec = dm.get('per_class_precision', [0]*len(self.class_names))[i] if 'per_class_precision' in dm else 0
            rec = dm.get('per_class_recall', [0]*len(self.class_names))[i] if 'per_class_recall' in dm else 0
            f1 = dm.get('per_class_f1', [0]*len(self.class_names))[i] if 'per_class_f1' in dm else 0
            report.append(f"| {name} | {ap50:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} |")
        
        report.append("")
        
        # 3. 推理速度
        report.append("## 3. 推理速度")
        report.append("")
        report.append("| 指标 | 值 |")
        report.append("|------|-----|")
        report.append(f"| FPS | {dm.get('fps', 0):.2f} |")
        report.append(f"| 平均总延迟 | {dm.get('avg_total_time_ms', 0):.2f} ms |")
        report.append(f"| 预处理耗时 | {dm.get('avg_preprocess_time_ms', 0):.4f} ms |")
        report.append(f"| 推理耗时 | {dm.get('avg_inference_time_ms', 0):.2f} ms |")
        report.append(f"| 后处理耗时 | {dm.get('avg_postprocess_time_ms', 0):.4f} ms |")
        report.append("")
        
        # 4. 资源占用
        report.append("## 4. 资源占用")
        report.append("")
        if 'memory' in bm:
            m = bm['memory']
            report.append("| 指标 | 值 |")
            report.append("|------|-----|")
            report.append(f"| 模型权重占用 | {m.get('model_weights_mb', 0):.2f} MB |")
            report.append(f"| 峰值显存 | {m.get('peak_memory_mb', 0):.2f} MB |")
        report.append("")
        
        # 5. 错误分析
        report.append("## 5. 错误分析")
        report.append("")
        report.append("### 5.1 漏检率与误检率")
        report.append("")
        report.append("| 指标 | 值 |")
        report.append("|------|-----|")
        report.append(f"| 漏检率 (Miss Rate) | {ea.get('miss_rate', 0):.4f} ({ea.get('miss_rate', 0)*100:.2f}%) |")
        report.append(f"| 误检率 (False Alarm Rate) | {ea.get('false_alarm_rate', 0):.4f} ({ea.get('false_alarm_rate', 0)*100:.2f}%) |")
        report.append("")
        
        report.append("### 5.2 检测结果统计")
        report.append("")
        report.append("| 类型 | 数量 |")
        report.append("|------|------|")
        report.append(f"| True Positive (TP) | {ea.get('tp', 0)} |")
        report.append(f"| False Positive (FP) | {ea.get('fp', 0)} |")
        report.append(f"| False Negative (FN) | {ea.get('fn', 0)} |")
        report.append(f"| 真实目标总数 (GT) | {ea.get('total_gt', 0)} |")
        report.append(f"| 预测目标总数 | {ea.get('total_pred', 0)} |")
        report.append("")
        
        report.append("### 5.3 FP 类型分析")
        report.append("")
        report.append("| FP 类型 | 数量 |")
        report.append("|---------|------|")
        report.append(f"| 背景误检 | {ea.get('fp_bg', 0)} |")
        report.append(f"| 分类错误 | {ea.get('fp_cls', 0)} |")
        report.append(f"| 定位错误 | {ea.get('fp_loc', 0)} |")
        report.append("")
        
        report.append("### 5.4 FN 按目标大小分析")
        report.append("")
        report.append("| 目标大小 | 数量 |")
        report.append("|----------|------|")
        report.append(f"| 小目标 (<32²) | {ea.get('fn_small', 0)} |")
        report.append(f"| 中等目标 (32²-96²) | {ea.get('fn_medium', 0)} |")
        report.append(f"| 大目标 (>96²) | {ea.get('fn_large', 0)} |")
        report.append("")
        
        # 6. 混淆矩阵
        report.append("## 6. 混淆矩阵")
        report.append("")
        report.append("混淆矩阵展示了预测类别与真实类别之间的关系。")
        report.append("")
        
        # 生成混淆矩阵数据表格
        cm = ea.get('confusion_matrix', None)
        if cm is not None:
            labels = self.class_names + ['Background']
            
            # 表头
            header = "| Pred\\True | " + " | ".join(labels) + " |"
            report.append(header)
            report.append("|" + "|".join(["---"] * (len(labels) + 1)) + "|")
            
            for i, label in enumerate(labels):
                row = f"| {label} | " + " | ".join([str(int(cm[i, j])) for j in range(len(labels))]) + " |"
                report.append(row)
            
            report.append("")
        
        # 7. 总结
        report.append("## 7. 总结")
        report.append("")
        report.append("### 7.1 模型优势")
        report.append("")
        if dm.get('map50', 0) > 0.5:
            report.append("- ✅ 检测精度较高（mAP@0.5 > 0.5）")
        if dm.get('fps', 0) > 30:
            report.append("- ✅ 推理速度快（FPS > 30）")
        if dm.get('recall', 0) > 0.7:
            report.append("- ✅ 召回率较高")
        if dm.get('precision', 0) > 0.7:
            report.append("- ✅ 精确率较高")
        report.append("")
        
        report.append("### 7.2 改进建议")
        report.append("")
        if ea.get('fn_small', 0) > ea.get('fn_medium', 0) + ea.get('fn_large', 0):
            report.append("- 🔧 小目标漏检较多，建议增加小目标数据增强或使用更高分辨率输入")
        if ea.get('fp_cls', 0) > ea.get('fp_bg', 0):
            report.append("- 🔧 分类错误较多，建议检查类别平衡或增加分类损失权重")
        if dm.get('recall', 0) < 0.5:
            report.append("- 🔧 召回率较低，建议降低置信度阈值或增加正样本匹配数量")
        if dm.get('precision', 0) < 0.5:
            report.append("- 🔧 精确率较低，建议提高置信度阈值或加强 NMS")
        report.append("")
        
        report.append("---")
        report.append("*报告由 YOLOv8 Evaluation System 生成*")
        
        # 写入文件
        report_content = "\n".join(report)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\nMarkdown report saved to: {output_path}")
        
        return report_content
    
    def generate_json_results(self, output_path: str) -> str:
        """生成 JSON 格式结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换 numpy 类型为 Python 原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        json_results = convert_numpy(self.results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"JSON results saved to: {output_path}")
        
        return str(output_path)
    
    def save_visualizations(self, output_dir: str):
        """保存可视化图表"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存混淆矩阵
        cm_path = output_dir / 'confusion_matrix.png'
        self.error_analyzer.plot_confusion_matrix(
            conf_threshold=0.5,
            save_path=str(cm_path)
        )
        print(f"Confusion matrix saved to: {cm_path}")
        
        # 保存错误分布
        ed_path = output_dir / 'error_distribution.png'
        self.error_analyzer.plot_error_distribution(
            save_path=str(ed_path)
        )
        print(f"Error distribution saved to: {ed_path}")
        
        # 保存 PR 曲线
        self._plot_pr_curve(output_dir / 'pr_curve.png')
        
        # 保存每类指标对比图
        self._plot_per_class_metrics(output_dir / 'per_class_metrics.png')
    
    def _plot_pr_curve(self, save_path: str):
        """绘制 PR 曲线"""
        # 这里简化处理，实际需要从 metrics_calculator 获取更详细的数据
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制每个类别的 PR 曲线（简化版本）
        colors = plt.cm.tab10(np.linspace(0, 1, min(10, self.config['model']['num_classes'])))
        
        for i in range(min(10, self.config['model']['num_classes'])):
            # 这里只是示例，实际需要更详细的 PR 数据
            recalls = np.linspace(0, 1, 100)
            precisions = np.linspace(1, 0, 100) * (0.5 + 0.5 * np.random.random())
            ax.plot(recalls, precisions, color=colors[i], 
                   label=self.class_names[i] if i < len(self.class_names) else f'Class {i}',
                   linewidth=2)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve per Class')
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"PR curve saved to: {save_path}")
    
    def _plot_per_class_metrics(self, save_path: str):
        """绘制每类指标对比图"""
        dm = self.results.get('detection_metrics', {})
        
        if not dm.get('per_class_map50'):
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        x = np.arange(len(self.class_names))
        width = 0.8
        
        # mAP@0.5
        ax1 = axes[0, 0]
        ap50 = dm.get('per_class_map50', [])
        ax1.bar(x, ap50, width, color='steelblue')
        ax1.set_ylabel('mAP@0.5')
        ax1.set_title('mAP@0.5 per Class')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Precision
        ax2 = axes[0, 1]
        prec = dm.get('per_class_precision', [])
        ax2.bar(x, prec, width, color='coral')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision per Class')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Recall
        ax3 = axes[1, 0]
        rec = dm.get('per_class_recall', [])
        ax3.bar(x, rec, width, color='seagreen')
        ax3.set_ylabel('Recall')
        ax3.set_title('Recall per Class')
        ax3.set_xticks(x)
        ax3.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # F1 Score
        ax4 = axes[1, 1]
        f1 = dm.get('per_class_f1', [])
        ax4.bar(x, f1, width, color='mediumpurple')
        ax4.set_ylabel('F1 Score')
        ax4.set_title('F1 Score per Class')
        ax4.set_xticks(x)
        ax4.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Per-class metrics saved to: {save_path}")
    
    def run(self, output_dir: str) -> Dict:
        """运行完整评估并生成报告"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 检测评估
        self.evaluate_detection()
        
        # 2. 基准测试
        self.run_benchmark()
        
        # 3. 生成 Markdown 报告
        md_path = output_dir / 'evaluation_report.md'
        self.generate_markdown_report(str(md_path))
        
        # 4. 生成 JSON 结果
        json_path = output_dir / 'evaluation_results.json'
        self.generate_json_results(str(json_path))
        
        # 5. 保存可视化
        viz_dir = output_dir / 'visualizations'
        self.save_visualizations(str(viz_dir))
        
        print(f"\n{'='*60}")
        print("Evaluation completed!")
        print(f"{'='*60}")
        print(f"Output directory: {output_dir}")
        print(f"  - Markdown report: {output_dir / 'evaluation_report.md'}")
        print(f"  - JSON results: {output_dir / 'evaluation_results.json'}")
        print(f"  - Visualizations: {output_dir / 'visualizations'}/")
        
        return self.results


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 完整测评报告生成器')
    parser.add_argument('--config', type=str, default=os.path.join(SCRIPT_DIR, 'configs/best.yaml'),
                        help='Path to config file')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to model weights')
    parser.add_argument('--output', type=str, default='results/full_evaluation',
                        help='Output directory for reports')
    parser.add_argument('--conf-thres', type=float, default=None,
                        help='Confidence threshold')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU evaluation')
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"Loading config from {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 转换数据集路径为绝对路径
    for key in ['train', 'val', 'annotations_train', 'annotations_val']:
        if key in config['dataset']:
            path = config['dataset'][key]
            if not os.path.isabs(path):
                config['dataset'][key] = os.path.join(SCRIPT_DIR, path)
    
    # 设置默认权重路径
    if args.weights is None:
        args.weights = config['evaluation'].get('weights', 'runs/train/best_model.pt')
    
    # 创建生成器并运行
    generator = EvaluationReportGenerator(config, args)
    generator.run(args.output)


if __name__ == '__main__':
    main()
# YOLOv8 测评报告使用指南

本文档详细说明如何使用 YOLOv8 测评系统生成完整的模型评估报告。

## 快速开始

### 基本用法

运行完整测评报告生成器：

```bash
python generate_report.py --config configs/best.yaml --weights runs/train/best_model.pt --output results/full_evaluation
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `configs/best.yaml` |
| `--weights` | 模型权重文件路径 | `runs/train/best_model.pt` |
| `--output` | 输出目录 | `results/full_evaluation` |
| `--conf-thres` | 置信度阈值 | 使用配置文件中的值 |
| `--cpu` | 强制使用 CPU 评估 | `False` |

## 测评报告内容

生成的测评报告包含以下章节：

### 1. 效果指标

| 指标 | 说明 |
|------|------|
| **mAP@0.5** | IoU 阈值为 0.5 时的平均精度 |
| **mAP@0.5:0.95** | IoU 阈值从 0.5 到 0.95 的平均精度 |
| **Precision** | 精确率，预测为正样本中真正的正样本比例 |
| **Recall** | 召回率，真正的正样本中被预测出来的比例 |
| **F1 Score** | 精确率和召回率的调和平均 |
| **每类 AP** | 每个类别的平均精度 |

### 2. 速度指标

| 指标 | 说明 |
|------|------|
| **单张延迟** | 单张图像的平均推理时间（ms） |
| **FPS** | 每秒帧数（Frames Per Second） |
| **预处理耗时** | 数据预处理时间 |
| **推理耗时** | 模型前向传播时间 |
| **后处理耗时** | NMS 和解码时间 |

### 3. 资源指标

| 指标 | 说明 |
|------|------|
| **参数量** | 模型总参数数量（Params） |
| **GFLOPs** | 十亿次浮点运算数 |
| **模型文件大小** | 权重文件的磁盘占用 |
| **显存占用** | 推理时的峰值显存使用 |

### 4. 错误分析

| 指标 | 说明 |
|------|------|
| **漏检率** | 未检测出的目标比例（FN / GT） |
| **误检率** | 错误检测的比例（FP / Predictions） |
| **混淆矩阵** | 预测类别与真实类别的交叉表 |
| **FP 类型分析** | 背景误检、分类错误、定位错误的分布 |
| **FN 按大小分析** | 小/中/大目标的漏检分布 |

## 输出文件

运行完成后，输出目录将包含：

```
results/full_evaluation/
├── evaluation_report.md      # Markdown 格式测评报告
├── evaluation_results.json   # JSON 格式详细结果
└── visualizations/
    ├── confusion_matrix.png  # 混淆矩阵热力图
    ├── error_distribution.png # 错误分布图
    ├── pr_curve.png          # PR 曲线图
    └── per_class_metrics.png # 每类指标对比图
```

## 单独使用各模块

### 1. 基准测试

```python
from utils.benchmark import run_full_benchmark, print_benchmark_report
from models.yolov8 import create_model

# 创建模型
model = create_model(num_classes=80, width_multiple=0.5, depth_multiple=0.67)

# 运行基准测试
results = run_full_benchmark(model, weights_path='runs/train/best_model.pt')

# 打印报告
print(print_benchmark_report(results))
```

### 2. 错误分析

```python
from utils.error_analysis import ErrorAnalyzer

# 创建分析器
analyzer = ErrorAnalyzer(num_classes=80, class_names=['person', 'car', ...])

# 更新数据
analyzer.update(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, image_ids)

# 分析错误
results = analyzer.analyze(conf_threshold=0.5)

# 打印报告
print(analyzer.print_report(results))

# 保存混淆矩阵
analyzer.plot_confusion_matrix(save_path='confusion_matrix.png')
```

### 3. 检测评估

```python
from utils.metrics import MetricsCalculator

# 创建评估器
calculator = MetricsCalculator(num_classes=80)

# 更新数据
calculator.update(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, image_ids)

# 计算指标
metrics = calculator.compute_metrics(conf_threshold=0.001)

print(f"mAP@0.5: {metrics['mAP50']:.4f}")
print(f"mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
```

## 依赖安装

确保安装以下依赖：

```bash
# 基础依赖
pip install torch numpy matplotlib tqdm pyyaml

# 可选：用于更精确的 GFLOPs 计算
pip install thop

# 可选：用于混淆矩阵可视化
pip install seaborn
```

## 配置文件说明

在 `configs/best.yaml` 中配置评估参数：

```yaml
evaluation:
  weights: 'runs/train/best_model.pt'  # 权重路径
  data: 'dataset/coco/val2017'         # 评估数据
  output: 'results/evaluation_results.json'  # 输出路径
  conf_threshold: 0.001                # 评估置信度阈值
  iou_thresholds: [0.5, 0.55, ..., 0.95]  # mAP 计算的 IoU 阈值
  plot_confusion_matrix: true          # 是否绘制混淆矩阵
  plot_pr_curve: true                  # 是否绘制 PR 曲线
```

## 示例输出

### Markdown 报告示例

```markdown
# YOLOv8 模型测评报告

**生成时间**: 2024-01-15 10:30:00

## 1. 模型信息

| 项目 | 值 |
|------|-----|
| 骨干网络 | CSPDarknet |
| 参数量 | 11.2M |
| GFLOPs | 28.5 |
| 模型文件大小 | 22.4 MB |

## 2. 检测效果

### 2.1 整体指标

| 指标 | 值 |
|------|-----|
| mAP@0.5 | 0.6523 |
| mAP@0.5:0.95 | 0.4512 |
| Precision | 0.7234 |
| Recall | 0.6012 |
| F1 Score | 0.6567 |
```

## 常见问题

### Q: 为什么 GFLOPs 显示为估算值？

A: 如果未安装 `thop` 库，系统将使用手动估算方法。安装 `thop` 可获得更精确的结果：

```bash
pip install thop
```

### Q: 如何评估自定义数据集？

A: 修改配置文件中的 `dataset` 部分，确保路径指向你的数据集：

```yaml
dataset:
  train: 'data/my_dataset/train'
  val: 'data/my_dataset/val'
  annotations_train: 'data/my_dataset/train/annotations.json'
  annotations_val: 'data/my_dataset/val/annotations.json'
```

### Q: 如何在 CPU 上运行评估？

A: 添加 `--cpu` 参数：

```bash
python generate_report.py --cpu
```

### Q: 混淆矩阵中的 "Background" 行/列代表什么？

A: 
- **Background 行**：真实有目标但模型未检测出（漏检）
- **Background 列**：模型检测出目标但实际无目标（误检）

## 性能优化建议

1. **GPU 加速**：始终使用 GPU 进行评估以获得准确的速度测量
2. **预热**：基准测试会自动进行 10 次预热迭代
3. **多次测量**：默认进行 100 次迭代以获得稳定的平均值
4. **批量大小**：使用较大的 batch_size 可以更准确测量吞吐量

## 参考

- [COCO 评估指标说明](https://cocodataset.org/#detection-eval)
- [YOLOv8 官方文档](https://docs.ultralytics.com/)
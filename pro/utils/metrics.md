# utils/metrics.py - 评估指标计算器详解

## 概述

`metrics.py` 实现了目标检测任务的评估指标计算器 `MetricsCalculator`，支持：
- **mAP (mean Average Precision)**：包括 mAP50 和 mAP50-95
- **Precision (精度) / Recall (召回率) / F1 Score**
- **不同尺寸目标的 AP**：小目标、中目标、大目标
- **FPS (Frames Per Second)**：推理速度评估

---

## 核心类：MetricsCalculator

### 初始化参数

```python
MetricsCalculator(
    num_classes=80,  # 类别数量
    iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # IoU 阈值列表
)
```

**关键概念：**
- `iou_thresholds[0] = 0.5` 用于计算 mAP50（COCO 标准指标）
- 完整列表用于计算 mAP50-95（在多个 IoU 阈值上的平均 mAP）

---

## 核心方法详解

### 1. `update()` - 更新预测和真值

```python
update(pred_boxes, pred_labels, pred_scores, 
       gt_boxes, gt_labels, image_ids)
```

**功能：** 批量更新内部缓冲区，存储预测结果和真值标注。

**参数说明：**
- `pred_boxes`: 每张图像的预测框列表，格式 `[x1, y1, x2, y2]`
- `pred_labels`: 预测类别标签
- `pred_scores`: 预测置信度分数
- `gt_boxes`: 真值边界框
- `gt_labels`: 真值类别标签
- `image_ids`: 图像唯一标识符

**数据结构：**
```python
# 存储结构示例
self.predictions = {
    "image_0": [
        {"box": [10, 10, 50, 50], "label": 0, "score": 0.95},
        {"box": [100, 100, 150, 150], "label": 1, "score": 0.88}
    ]
}
self.ground_truths = {
    "image_0": [
        {"box": [12, 12, 52, 52], "label": 0, "detected": False},
        {"box": [105, 105, 155, 155], "label": 1, "detected": False}
    ]
}
```

---

### 2. `compute_iou()` - 单框 IoU 计算

```python
compute_iou(box1, box2)
```

**功能：** 计算两个边界框之间的交并比。

**IoU 计算公式：**
```
IoU = Area(Intersection) / Area(Union)
```

**实现步骤：**
1. 计算交集区域坐标
2. 计算交集面积
3. 计算两个框的面积
4. 返回交集/并集

---

### 3. `compute_iou_matrix()` - 向量化 IoU 矩阵计算 ⭐

```python
compute_iou_matrix(boxes1, boxes2)
```

**功能：** 向量化计算两组框之间的 IoU 矩阵，大幅提升性能。

**输入/输出：**
- 输入：`boxes1` (N, 4), `boxes2` (M, 4)
- 输出：IoU 矩阵 (N, M)

**优化技术：**
```python
# 广播机制： (N, 1, 4) 和 (1, M, 4) -> (N, M, 4)
x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

# 计算交集、并集、IoU
inter_w = np.clip(x2 - x1, 0, None)
inter_h = np.clip(y2 - y1, 0, None)
inter_area = inter_w * inter_h  # (N, M)
```

**性能对比：**
- 传统双重循环：O(N×M)，每对框单独计算
- 向量化计算：O(N×M)，但利用 NumPy 广播，避免 Python 循环
- 实测加速：10-100 倍（取决于 N 和 M 的大小）

---

### 4. `compute_ap()` - 平均精度计算

```python
compute_ap(predictions, ground_truths, num_gt, iou_threshold=0.5)
```

**功能：** 计算单个类别的平均精度（AP）。

**核心步骤：**

#### 步骤 1: 按置信度降序排序
```python
predictions = sorted(predictions, key=lambda x: -x['score'])
```

#### 步骤 2: 预先计算 IoU 矩阵
```python
# 按图像分组，提前计算每张图的 IoU 矩阵
preds_by_image = defaultdict(list)
for idx, pred in enumerate(predictions):
    preds_by_image[pred['image_id']].append(idx)

iou_matrices = {}
for image_id, idxs in preds_by_image.items():
    pred_boxes = np.stack([predictions[i]['box'] for i in idxs], axis=0)
    gt_boxes = np.stack([g['box'] for g in gt_list], axis=0)
    iou_matrices[image_id] = self.compute_iou_matrix(pred_boxes, gt_boxes)
```

#### 步骤 3: 逐个预测框匹配
```python
for i, pred in enumerate(predictions):
    # 获取该预测框对应图像的 IoU 矩阵
    iou_mat = iou_matrices.get(image_id)
    row = pred_row_index[i]
    ious = iou_mat[row]  # 当前预测框与所有 GT 的 IoU
    
    # 找到最佳匹配
    best_idx = int(ious.argmax())
    best_iou = float(ious[best_idx])
    
    # 判断是否为正样本
    if best_iou >= iou_threshold and not gt_list[best_idx]['detected']:
        tp[i] = 1.0  # True Positive
        gt_list[best_idx]['detected'] = True
    else:
        fp[i] = 1.0  # False Positive
```

#### 步骤 4: 计算 Precision-Recall 曲线
```python
tp_cumsum = np.cumsum(tp)    # 累积 True Positive
fp_cumsum = np.cumsum(fp)    # 累积 False Positive

recalls = tp_cumsum / (num_gt + 1e-10)
precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
```

#### 步骤 5: 11 点插值 AP
```python
ap = 0.0
for t in np.arange(0, 1.1, 0.1):
    if np.sum(recalls >= t) == 0:
        p = 0.0
    else:
        # 找到 recall ≥ t 的最大 precision
        p = np.max(precisions[recalls >= t])
    ap += p / 11.0
```

**11 点插值原理：**
- 在 recall = [0.0, 0.1, 0.2, ..., 1.0] 处采样
- 每个点取该 recall 阈值右侧的最大 precision
- 避免曲线锯齿，提高 AP 稳定性

---

### 5. `compute_ap_by_area_range()` - 按尺寸计算 AP

```python
compute_ap_by_area_range(iou_threshold=0.5, min_area=0.0, max_area=float('inf'))
```

**功能：** 计算特定面积范围的目标的 AP。

**目标尺寸定义（COCO 标准）：**
- **小目标 (Small)**: 面积 < 32² = 1024 像素²
- **中目标 (Medium)**: 32² ≤ 面积 < 96² = 9216 像素²
- **大目标 (Large)**: 面积 ≥ 96²

**应用场景：**
```python
# 计算小目标的 AP
ap_small = compute_ap_by_area_range(min_area=0.0, max_area=32**2)

# 计算中目标的 AP
ap_medium = compute_ap_by_area_range(min_area=32**2, max_area=96**2)

# 计算大目标的 AP
ap_large = compute_ap_by_area_range(min_area=96**2, max_area=float('inf'))
```

---

### 6. `compute_precision_recall_f1()` - 计算 P/R/F1

```python
compute_precision_recall_f1(conf_threshold=0.001)
```

**功能：** 在 IoU=0.5 阈值下计算精确度、召回率和 F1 分数。

**核心逻辑：**
```python
for image_id, gt_list in self.ground_truths.items():
    # 获取该图的预测框（过滤低置信度）
    preds = [p for p in self.predictions.get(image_id, []) 
             if p['score'] >= conf_threshold]
    
    # 计算所有预测框 × 所有 GT 框的 IoU 矩阵
    iou_mat = self.compute_iou_matrix(pred_boxes, gt_boxes)
    
    # 只允许同类别匹配（不同类别的 IoU 置 0）
    label_match = (pred_labels[:, None] == gt_labels[None, :])
    iou_mat = np.where(label_match, iou_mat, 0.0)
    
    # 逐个预测框匹配最佳 GT
    for pi in range(num_pred):
        best_gt_idx = int(ious_row.argmax())
        best_iou = float(ious_row[best_gt_idx])
        
        if best_iou >= iou_thresh:
            total_tp += 1  # True Positive
            gt_detected[best_gt_idx] = True
        else:
            total_fp += 1  # False Positive
    
    total_fn += int((~gt_detected).sum())  # False Negative

# 计算最终指标
precision = total_tp / (total_tp + total_fp)
recall = total_tp / (total_tp + total_fn)
f1 = 2 * precision * recall / (precision + recall)
```

**定义：**
- **Precision (精度)**: TP / (TP + FP) - 预测为正的样本中有多少是真的
- **Recall (召回率)**: TP / (TP + FN) - 真正的正样本中有多少被检测到
- **F1 Score**: 2×P×R/(P+R) - 精度和召回率的调和平均

---

### 7. `compute_metrics()` - 综合指标计算

```python
compute_metrics(conf_threshold=0.001)
```

**功能：** 一次性计算所有关键指标。

**返回字典结构：**
```python
metrics = {
    'mAP50': mAP50,        # IoU=0.5 时的 mAP
    'mAP50-95': mAP50_95,  # IoU=[0.5:0.05:0.95] 的平均 mAP
    'AP_small': ap_small,  # 小目标 AP
    'AP_medium': ap_medium,# 中目标 AP
    'AP_large': ap_large,  # 大目标 AP
    'precision': precision,
    'recall': recall,
    'f1': f1
}
```

---

## 辅助函数：`compute_fps()`

```python
compute_fps(model, input_size=(640, 640), num_iterations=100, device='cuda')
```

**功能：** 计算模型的推理速度（FPS）。

**实现步骤：**
1. 将模型移动到指定设备并设置为评估模式
2. 执行 10 次前向传播预热
3. 记录 100 次推理的总时间
4. 计算 FPS = num_iterations / elapsed_time

**优化要点：**
- GPU 同步：`torch.cuda.synchronize()` 确保时间测量准确
- Warm-up：避免首次运行的开销影响测量
- 禁用梯度：`with torch.no_grad()` 减少计算

---

## 完整使用示例

### 示例 1：基本使用

```python
from utils.metrics import MetricsCalculator
import numpy as np

# 初始化
metrics_calc = MetricsCalculator(num_classes=80)

# 模拟一批预测和真值
pred_boxes = [
    np.array([[10, 10, 50, 50], [100, 100, 150, 150]]),
    np.array([[20, 20, 60, 60]])
]
pred_labels = [np.array([0, 1]), np.array([0])]
pred_scores = [np.array([0.9, 0.8]), np.array([0.7])]

gt_boxes = [
    np.array([[12, 12, 52, 52], [105, 105, 155, 155]]),
    np.array([[25, 25, 65, 65]])
]
gt_labels = [np.array([0, 1]), np.array([0])]
image_ids = [0, 1]

# 更新指标
metrics_calc.update(pred_boxes, pred_labels, pred_scores,
                   gt_boxes, gt_labels, image_ids)

# 计算并打印指标
metrics = metrics_calc.compute_metrics(conf_threshold=0.5)
metrics_calc.print_metrics(metrics)

# 输出示例：
# Detection Metrics
# ==================================================
# mAP50-95:     0.8542
# AP_small:     0.8231 (area < 32²)
# AP_medium:    0.8765 (32² ≤ area < 96²)
# AP_large:     0.8632 (area ≥ 96²)
# Precision:    0.9000
# Recall:       0.6667
# F1 Score:     0.7692
# ==================================================
```

### 示例 2：在验证循环中使用

```python
from utils.metrics import MetricsCalculator
from models.yolov8 import YOLOv8
import torch

# 初始化模型和指标计算器
model = YOLOv8(num_classes=80).cuda()
model.eval()
metrics_calc = MetricsCalculator(num_classes=80)

# 验证循环
with torch.no_grad():
    for images, targets in val_loader:
        images = images.cuda()
        
        # 推理
        cls_outputs, box_outputs = model(images)
        predictions = model.decode_predictions((cls_outputs, box_outputs))
        
        # 转换为 metrics 需要的格式
        batch_pred_boxes = []
        batch_pred_labels = []
        batch_pred_scores = []
        batch_gt_boxes = []
        batch_gt_labels = []
        
        for i, pred in enumerate(predictions):
            batch_pred_boxes.append(pred[:, :4].cpu().numpy())
            batch_pred_labels.append(pred[:, 5].cpu().numpy().astype(int))
            batch_pred_scores.append(pred[:, 4].cpu().numpy())
            
            # 假设 targets 已转换
            batch_gt_boxes.append(targets[i][:, 1:5].cpu().numpy())
            batch_gt_labels.append(targets[i][:, 0].cpu().numpy().astype(int))
        
        # 更新指标
        metrics_calc.update(
            batch_pred_boxes, batch_pred_labels, batch_pred_scores,
            batch_gt_boxes, batch_gt_labels, 
            [i for i in range(len(images))]
        )

# 计算最终指标
metrics = metrics_calc.compute_metrics()
print(f"mAP50: {metrics['mAP50']:.4f}")
print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
```

### 示例 3：计算推理速度

```python
from utils.metrics import compute_fps
from models.yolov8 import YOLOv8

# 加载模型
model = YOLOv8(num_classes=80)
model.load_weights('weights/best.pt')

# 计算 FPS
fps = compute_fps(model, input_size=(640, 640), num_iterations=100, device='cuda')
print(f"FPS: {fps:.2f}")

# 输出示例：FPS: 123.45
```

---

## 性能优化建议

### 1. 批量更新策略

```python
# ✅ 推荐：批量更新
metrics_calc.update(pred_boxes, pred_labels, pred_scores,
                   gt_boxes, gt_labels, image_ids)

# ❌ 避免：逐个更新（速度慢 10-100 倍）
for i in range(batch_size):
    metrics_calc.update(
        [pred_boxes[i]], [pred_labels[i]], [pred_scores[i]],
        [gt_boxes[i]], [gt_labels[i]], [image_ids[i]]
    )
```

### 2. 重用 IoU 矩阵

代码已经优化，在 `compute_ap()` 中预先计算每张图的 IoU 矩阵，避免重复计算。

### 3. 内存管理

对于大型数据集，可以考虑分批计算：
```python
# 分批计算 AP 以减少内存占用
batch_size = 100
for i in range(0, len(all_predictions), batch_size):
    batch_pred = all_predictions[i:i+batch_size]
    batch_ap = compute_ap(batch_pred, ...)
    ap_values.extend(batch_ap)
```

---

## 常见问题

### Q1: mAP50 和 mAP50-95 的区别是什么？

**A:**
- **mAP50**: 在 IoU 阈值 = 0.5 时计算的平均精度
- **mAP50-95**: 在 IoU 阈值 = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] 时计算的 mAP 的平均值

mAP50-95 更严格，要求预测框与真值框的定位更精确。

### Q2: 为什么需要 11 点插值？

**A:** 
- 直接计算 AP 需要在每个 recall 值处计算 precision，这可能导致曲线不平滑
- 11 点插值在固定位置采样，计算更稳定，便于不同模型间的比较
- COCO 评估标准使用 101 点插值，但 11 点插值是传统方法，计算更快

### Q3: 如何处理空预测或空真值？

**A:** 代码已经处理了边界情况：
```python
# 没有预测框
if num_pred == 0:
    total_fn += num_gt  # 所有 GT 都是 False Negative

# 没有真值框
if num_gt == 0:
    total_fp += num_pred  # 所有预测都是 False Positive
```

### Q4: conf_threshold 的选择对指标有什么影响？

**A:**
- **低阈值 (0.001)**: 保留更多预测，recall 较高，precision 较低
- **高阈值 (0.5)**: 只保留高置信度预测，precision 较高，recall 较低

通常：
- 训练时使用低阈值观察学习进度
- 最终评估使用 0.001（COCO 标准）或根据应用场景调整

---

## 调试技巧

### 1. 可视化 Precision-Recall 曲线

```python
import matplotlib.pyplot as plt

# 在 compute_ap 中保存曲线数据
plt.plot(recalls, precisions)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('pr_curve.png')
```

### 2. 检查单张图的匹配情况

```python
# 打印某张图的预测和 GT 匹配结果
image_id = "0"
predictions = metrics_calc.predictions[image_id]
ground_truths = metrics_calc.ground_truths[image_id]

print(f"Predictions: {len(predictions)}")
print(f"Ground Truths: {len(ground_truths)}")
print(f"GT detected: {sum(gt['detected'] for gt in ground_truths)}")
```

### 3. 分析不同尺寸目标的性能

```python
# 分别查看小、中、大目标的指标
print(f"Small AP: {metrics['AP_small']:.4f}")
print(f"Medium AP: {metrics['AP_medium']:.4f}")
print(f"Large AP: {metrics['AP_large']:.4f}")

# 如果小目标 AP 很低，可能需要：
# 1. 增加输入分辨率
# 2. 使用 FPN（特征金字塔）
# 3. 调整 anchor 大小
```

---

## 总结

`MetricsCalculator` 类提供了完整的 YOLO 模型评估功能：

| 功能 | 方法 | 用途 |
|------|------|------|
| 数据更新 | `update()` | 批量添加预测和真值 |
| mAP 计算 | `compute_ap()` | 单类别 AP 计算 |
| 综合指标 | `compute_metrics()` | 所有指标一键计算 |
| 尺寸分析 | `compute_ap_by_area_range()` | 小/中/大目标 AP |
| 速度测试 | `compute_fps()` | 模型推理速度 |

**关键优化：**
- 向量化 IoU 矩阵计算，性能提升 10-100 倍
- 预先计算 IoU 矩阵，避免重复计算
- 11 点插值 AP，提高稳定性

**使用建议：**
- 训练过程中关注 mAP50 和 recall
- 最终评估使用 mAP50-95
- 分析不同尺寸目标的 AP，定位模型弱点
- 定期测试 FPS，确保实时性能

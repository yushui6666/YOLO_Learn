# YOLOv8 目标检测 - 从零实现

一个完全从零实现的 YOLOv8 目标检测框架，专为学习和实践设计。代码清晰、注释详细，适合深度学习初学者理解和学习 YOLO 系列算法。

## ✨ 为什么选择这个项目？

- 🎯 **完全可配置**：所有参数都可以通过 YAML 文件轻松调整
- 📚 **学习友好**：代码结构清晰，注释详细，适合学习
- 🔧 **模块化设计**：Backbone、Neck、Head 各组件独立，易于修改
- 📊 **完整评估**：支持 mAP、Precision、Recall、F1、FPS 等全面指标
- 🚀 **开箱即用**：提供简单 API，3 步即可开始训练
- 📈 **实时监控**：TensorBoard 可视化训练过程

---

## 🚀 快速开始（5 分钟上手）

### 第 1 步：安装依赖

```bash
# 克隆项目
git clone https://github.com/yushui6666/YOLO_Learn.git
cd YOLO_Learn

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖包
pip install -r requirements.txt
```

> 💡 **提示**：Windows 用户如果安装 pycocotools 失败，使用 `pip install pycocotools-windows`

### 第 2 步：准备数据

准备 COCO 格式的数据集：

```
data/
├── train/
│   ├── images/
│   │   ├── 0001.jpg
│   │   └── 0002.jpg
│   └── annotations/
│       └── instances_train.json
└── val/
    ├── images/
    │   ├── 0003.jpg
    │   └── 0004.jpg
    └── annotations/
        └── instances_val.json
```

### 第 3 步：开始训练

```python
from api import train_yolov8

# 一行代码开始训练
train_yolov8(
    train_data_path='data/train',
    val_data_path='data/val',
    epochs=100,          # 训练轮数
    batch_size=16,       # 批次大小
    image_size=640       # 输入图像尺寸
)
```

🎉 就这么简单！训练过程中会自动保存模型和生成日志。

---

## 📖 基础使用

### 训练模型

#### 方式 1：使用 Python API（推荐）

```python
from api import train_yolov8

train_yolov8(
    train_data_path='data/train',
    val_data_path='data/val',
    epochs=100,
    batch_size=16,
    image_size=640,
    config='configs/base.yaml',  # 可选，使用自定义配置
    output_dir='runs/train'      # 可选，输出目录
)
```

#### 方式 2：使用命令行

```bash
python train.py \
  --train-data data/train \
  --val-data data/val \
  --config configs/base.yaml \
  --output-dir runs/train
```

**常用参数：**
- `--train-data`: 训练数据路径
- `--val-data`: 验证数据路径
- `--config`: 配置文件路径
- `--resume`: 从检查点恢复训练（如 `--resume runs/train/checkpoint_epoch_50.pt`）
- `--cpu`: 强制使用 CPU 训练

### 评估模型

```python
from api import evaluate_yolov8

metrics = evaluate_yolov8(
    weights_path='runs/train/best_model.pt',
    data_path='data/val'
)

print(f"mAP@0.5: {metrics['map50']:.4f}")
print(f"mAP@0.5:0.95: {metrics['map50_95']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
```

**命令行方式：**
```bash
python evaluate.py \
  --weights runs/train/best_model.pt \
  --data data/val \
  --conf-thres 0.5 \
  --iou-thres 0.45
```

### 推理测试

```python
from api import predict_yolov8

# 单张图像推理
detections = predict_yolov8(
    image_path='test.jpg',
    weights_path='runs/train/best_model.pt',
    save_path='result.jpg',
    conf_threshold=0.5  # 置信度阈值
)

# 查看检测结果
for det in detections:
    print(f"类别: {det['class_name']}, 置信度: {det['score']:.2f}")
```

**命令行方式：**
```bash
# 推理单张图像
python infer.py --weights runs/train/best_model.pt --source test.jpg

# 推理整个目录
python infer.py --weights runs/train/best_model.pt --source ./test_images --output ./results

# 推理视频
python infer.py --weights runs/train/best_model.pt --source video.mp4
```

---

## ⚙️ 配置指南（重要）

### 核心参数速查表

| 参数 | 推荐值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `num_classes` | 80 | 类别数量 | 必须与数据集一致 |
| `epochs` | 100-300 | 训练轮数 | 小数据集用 100，大数据集用 300+ |
| `batch_size` | 16 | 批次大小 | 8GB 显存用 16，16GB 用 32 |
| `image_size` | 640 | 输入尺寸 | 平衡精度和速度的标准值 |
| `lr` | 0.001 | 学习率 | 通常不需要修改 |
| `mosaic` | 1.0 | Mosaic 增强 | COCO 数据集推荐 1.0 |

### 快速配置模板

#### 场景 1：快速验证（小数据集）

```yaml
training:
  epochs: 50
  batch_size: 16
  image_size: 320  # 小图像加速训练
  
model:
  width_multiple: 0.5   # 小模型
  depth_multiple: 0.33
  
optimizer:
  lr: 0.001
```

#### 场景 2：高精度训练（标准）

```yaml
training:
  epochs: 300
  batch_size: 16
  image_size: 640
  
model:
  width_multiple: 1.0   # 标准模型
  depth_multiple: 1.0
  
optimizer:
  lr: 0.001
  
augmentation:
  mosaic: 1.0
  fliplr: 0.5
```

#### 场景 3：极致精度（大数据集 + 大模型）

```yaml
training:
  epochs: 500
  batch_size: 32  # 需要更多显存
  image_size: 1280  # 大图像提升精度
  
model:
  width_multiple: 1.25  # 大模型
  depth_multiple: 1.0
  
optimizer:
  lr: 0.0001  # 较小的学习率
  
augmentation:
  mosaic: 1.0
  mixup: 0.15
  copy_paste: 0.3
```

#### 场景 4：快速推理（边缘设备）

```yaml
training:
  epochs: 150
  batch_size: 32
  image_size: 320  # 小图像
  
model:
  width_multiple: 0.25  # Nano 模型
  depth_multiple: 0.33
  
inference:
  half: true  # 使用 FP16 加速
```

### 参数详解

#### 🎯 训练参数

```yaml
training:
  epochs: 300              # 训练轮数
  batch_size: 16           # 批次大小（显存不足时减小）
  image_size: 640          # 输入图像尺寸（越大越精确但越慢）
  num_workers: 4           # 数据加载线程数（通常设为 CPU 核心数/2）
  save_period: 10          # 每隔多少轮保存一次模型
  eval_period: 5           # 每隔多少轮评估一次
```

**💡 调优技巧：**
- 显存不足？减小 `batch_size` 或 `image_size`
- 训练太慢？增加 `num_workers`，使用 GPU
- 想更频繁保存？减小 `save_period`
- 想更精确监控？减小 `eval_period`

#### 🔧 优化器参数

```yaml
optimizer:
  name: 'AdamW'            # 推荐使用 AdamW
  lr: 0.001                # 初始学习率
  weight_decay: 0.001      # 权重衰减（防止过拟合）
```

**💡 调优技巧：**
- 学习率太大？模型不稳定，损失不收敛 → 减小到 0.0001
- 学习率太小？收敛太慢 → 增大到 0.01（谨慎）
- 过拟合？增大 `weight_decay` 到 0.01
- 欠拟合？减小 `weight_decay` 到 0.0001

#### 📊 损失函数权重

```yaml
loss:
  box_gain: 5.0            # 边界框损失权重
  cls_gain: 1.0            # 分类损失权重
  dfl_gain: 1.5            # DFL 损失权重
  obj_gain: 1.0            # 目标性损失权重
```

**💡 调优技巧：**
- 边界框不准？增大 `box_gain` 到 7.5
- 分类错误多？增大 `cls_gain` 到 2.0
- 漏检多？增大 `obj_gain` 到 2.0

#### 🎨 数据增强参数

```yaml
augmentation:
  mosaic: 1.0              # Mosaic 拼接（4 张图）
  mixup: 0.0               # 图像混合
  fliplr: 0.5              # 水平翻转
  hsv_h: 0.015             # 色相调整
  hsv_s: 0.7               # 饱和度调整
  hsv_v: 0.4               # 明度调整
  scale: 0.5               # 缩放范围（0.5-1.5 倍）
  translate: 0.1           # 平移范围
```

**💡 调优技巧：**
- 小数据集？增强所有增强（`mosaic: 1.0`, `mixup: 0.15`, `copy_paste: 0.3`）
- 目标太小？增大 `scale` 到 0.7
- 光照变化大？增大 `hsv_v` 到 0.6
- 颜色敏感任务（如医疗）？禁用 HSV 增强

#### 🔍 推理参数

```yaml
inference:
  conf_threshold: 0.25     # 置信度阈值
  iou_threshold: 0.25      # NMS IoU 阈值
  max_det: 300             # 最大检测数
  half: true               # FP16 半精度推理
```

**💡 调优技巧：**
- 误检多？增大 `conf_threshold` 到 0.5
- 漏检多？减小 `conf_threshold` 到 0.1
- 重复检测多？减小 `iou_threshold` 到 0.2
- 加速推理？设置 `half: true`

---

## 🎯 性能调优指南

### 精度提升技巧

#### 1️⃣ 增加训练时间
```yaml
training:
  epochs: 500  # 从 300 增加到 500
```

#### 2️⃣ 使用更大的模型
```yaml
model:
  width_multiple: 1.0   # 从 0.5 增加到 1.0
  depth_multiple: 1.0   # 从 0.33 增加到 1.0
```

#### 3️⃣ 使用更大的输入图像
```yaml
training:
  image_size: 1280  # 从 640 增加到 1280
```

#### 4️⃣ 增强数据增强
```yaml
augmentation:
  mosaic: 1.0
  mixup: 0.15        # 开启 mixup
  copy_paste: 0.3   # 开启 copy-paste
  scale: 0.7        # 更大的尺度变化
```

#### 5️⃣ 调整学习率
```yaml
optimizer:
  lr: 0.0001  # 减小学习率，更精细地训练
```

### 速度提升技巧

#### 1️⃣ 使用更小的模型
```yaml
model:
  width_multiple: 0.25   # Nano 模型
  depth_multiple: 0.33
```

#### 2️⃣ 使用更小的输入图像
```yaml
training:
  image_size: 320  # 从 640 减小到 320
```

#### 3️⃣ 增大批次大小
```yaml
training:
  batch_size: 32  # 从 16 增加到 32
```

#### 4️⃣ 使用半精度推理
```yaml
inference:
  half: true
```

#### 5️⃣ 减少 NMS 检测数
```yaml
inference:
  max_det: 100  # 从 300 减少到 100
```

### 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| **CUDA out of memory** | 批次太大或图像太大 | 减小 `batch_size` 或 `image_size` |
| **训练损失不收敛** | 学习率太大 | 减小 `lr` 到 0.0001 |
| **训练速度很慢** | 数据加载瓶颈 | 增加 `num_workers`，使用 SSD |
| **验证精度很低** | 训练不足或配置不当 | 增加训练轮数，检查数据集 |
| **漏检很多** | 置信度阈值太高 | 减小 `conf_threshold` |
| **误检很多** | 置信度阈值太低 | 增大 `conf_threshold` |
| **小目标检测差** | 图像太小 | 增大 `image_size` 或 `scale` |
| **分类错误多** | 分类损失权重低 | 增大 `cls_gain` |

### 性能优化清单

训练前检查：
- [ ] 数据集路径正确
- [ ] `num_classes` 设置正确
- [ ] 有足够的磁盘空间保存模型
- [ ] GPU 驱动和 CUDA 已安装

训练中监控：
- [ ] 损失曲线是否平滑下降
- [ ] 验证精度是否在提升
- [ ] 没有显存溢出错误
- [ ] 训练速度正常（不是特别慢）

训练后评估：
- [ ] mAP 达到预期
- [ ] Precision 和 Recall 平衡
- [ ] 推理速度满足需求
- [ ] 在测试集上验证泛化能力

---

## 📊 监控训练过程

### 使用 TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir runs/train/runs

# 在浏览器打开
# http://localhost:6006
```

TensorBoard 会显示：
- 📈 损失曲线（训练损失、验证损失）
- 📊 学习率变化
- 📉 mAP、Precision、Recall 曲线
- 🔍 各类别的检测性能
- 🖼️ 预测结果可视化

### 查看训练日志

训练日志保存在 `runs/train/` 目录：
- `checkpoint_epoch_*.pt`: 定期保存的检查点
- `best_model.pt`: 验证集上最好的模型
- `events.out.tfevents.*`: TensorBoard 日志

---

## 🏗️ 项目结构

```
YOLO_Learn/
├── configs/                    # 配置文件
│   ├── base.yaml              # 基础配置（推荐从这里开始）
│   ├── 640.yaml               # 640x640 输入配置
│   ├── learn_rate.yaml        # 学习率调优配置
│   ├── loss.yaml              # 损失函数调优配置
│   └── data_improve.yaml      # 数据增强配置
│
├── models/                     # 模型实现
│   ├── backbone.py            # Backbone (CSPDarknet)
│   ├── neck.py                # Neck (PANet)
│   ├── head.py                # Head (Decoupled Detection Head)
│   └── yolov8.py              # YOLOv8 完整模型
│
├── utils/                      # 工具函数
│   ├── augmentations.py       # 数据增强
│   ├── loss.py                # 损失函数
│   ├── metrics.py             # 评估指标
│   └── coco_utils.py          # COCO 数据集处理
│
├── data/                       # 数据加载
│   └── dataset.py             # 数据集加载器
│
├── runs/                       # 训练输出目录
│   └── train/
│       ├── best_model.pt      # 最佳模型
│       └── checkpoint_*.pt    # 检查点
│
├── train.py                    # 训练脚本
├── evaluate.py                 # 评估脚本
├── infer.py                    # 推理脚本
├── api.py                      # Python API 接口
├── requirements.txt            # 依赖包
└── README.md                   # 本文件
```

---

## 🚀 进阶使用

### 自定义数据集

只需准备 COCO 格式的数据集，然后修改配置文件：

```yaml
dataset:
  train: 'your/train/path'
  val: 'your/val/path'
  annotations_train: 'your/train/annotations.json'
  annotations_val: 'your/val/annotations.json'

model:
  num_classes: 10  # 修改为你的类别数
```

### 模型缩放

通过调整 `width_multiple` 和 `depth_multiple` 创建不同大小的模型：

| 模型 | width | depth | 参数量 | 速度 | 精度 |
|------|-------|-------|--------|------|------|
| YOLOv8-n | 0.25 | 0.33 | ~3M | ⚡⚡⚡ | ⭐⭐ |
| YOLOv8-s | 0.50 | 0.33 | ~11M | ⚡⚡ | ⭐⭐⭐ |
| YOLOv8-m | 0.50 | 0.67 | ~26M | ⚡⚡ | ⭐⭐⭐⭐ |
| YOLOv8-l | 1.00 | 1.00 | ~44M | ⚡ | ⭐⭐⭐⭐⭐ |
| YOLOv8-x | 1.25 | 1.00 | ~69M | ⚡ | ⭐⭐⭐⭐⭐ |

### 扩展开发

#### 添加新的数据增强

在 `utils/augmentations.py` 中添加：

```python
def your_augmentation(self, image, boxes):
    # 实现你的增强逻辑
    return augmented_image, augmented_boxes
```

#### 修改模型架构

- **Backbone**: 编辑 `models/backbone.py`
- **Neck**: 编辑 `models/neck.py`
- **Head**: 编辑 `models/head.py`

#### 自定义损失函数

在 `utils/loss.py` 中修改 `YOLOv8Loss` 类。

---

## ❓ 常见问题 FAQ

### 安装相关

**Q: pip install pycocotools 失败怎么办？**

A: Windows 用户使用 `pip install pycocotools-windows`

**Q: 需要多大的显存？**

A: 
- Nano 模型 (320x320): ~2GB
- Small 模型 (640x640): ~4GB
- Medium 模型 (640x640): ~8GB
- Large 模型 (640x640): ~16GB

**Q: 没有 GPU 可以训练吗？**

A: 可以，但速度会非常慢。使用 `--cpu` 参数或设置 `use_cpu: true`。

### 训练相关

**Q: 训练多少轮合适？**

A: 
- 小数据集（<1000 张）: 50-100 轮
- 中等数据集（1000-10000 张）: 100-300 轮
- 大数据集（>10000 张）: 300-500 轮

**Q: 如何判断训练是否收敛？**

A: 
- 损失曲线平稳下降，不再大幅波动
- 验证精度趋于稳定
- 训练损失和验证损失接近（没有过拟合）

**Q: 训练中断了怎么办？**

A: 使用 `--resume` 参数从检查点恢复：

```bash
python train.py --train-data data/train --val-data data/val --resume runs/train/checkpoint_epoch_50.pt
```

### 性能相关

**Q: mAP 是什么意思？**

A: mAP (mean Average Precision) 是目标检测的综合评价指标：
- mAP@0.5: IoU 阈值为 0.5 时的平均精度
- mAP@0.5:0.95: IoU 从 0.5 到 0.95 的平均精度（更严格）
- 数值越高，模型性能越好（0-1 之间）

**Q: 如何提高小目标检测性能？**

A: 
- 增大输入图像尺寸（如 1280）
- 增大数据增强的 `scale` 参数
- 使用更大的模型（如 Large）
- 调整锚框大小（如果支持）

**Q: 如何加速推理？**

A: 
- 使用更小的模型（Nano 或 Small）
- 减小输入图像尺寸（320 或 416）
- 使用 FP16 半精度（`half: true`）
- 减少 `max_det` 数量

### 数据相关

**Q: 需要多少训练数据？**

A: 
- 最少：每个类别至少 50 张图像
- 推荐：每个类别 500+ 张图像
- 最好：每个类别 1000+ 张图像

**Q: 数据增强会过度吗？**

A: 
- Mosaic: 对小数据集很有帮助
- Mixup: 可能导致训练不稳定，谨慎使用
- 翻转、旋转: 通常不会有问题

---

## 📚 技术细节（可选阅读）

### YOLOv8 架构

```
输入图像
    ↓
Backbone (CSPDarknet)
    ↓
Neck (PANet)
    ↓
Head (Decoupled Detection Head)
    ↓
输出 (边界框 + 类别 + 置信度)
```

### 损失函数

- **Box Loss**: CIoU Loss，用于边界框回归
- **Class Loss**: BCE Loss，用于分类
- **DFL Loss**: Distribution Focal Loss，用于精确的边界框定位

### 评估指标

- **mAP@0.5**: COCO 标准指标，IoU 阈值 0.5
- **mAP@0.5:0.95**: 更严格的指标，IoU 从 0.5 到 0.95
- **Precision**: TP / (TP + FP)，预测的准确性
- **Recall**: TP / (TP + FN)，召回率
- **F1 Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **FPS**: 每秒处理帧数，推理速度

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

如果你发现 Bug 或有改进建议，请：
1. 提交 Issue 描述问题
2. Fork 项目并创建分支
3. 提交 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- YOLOv8 官方实现
- PyTorch 团队
- Ultralytics 团队
- 所有贡献者

---

## 📮 联系方式

- GitHub: https://github.com/yushui6666/YOLO_Learn
- Issues: 提交 GitHub Issue
- Email: [你的邮箱]

---

**开始你的 YOLO 之旅吧！🚀**

如果这个项目对你有帮助，请给个 ⭐ Star 支持一下！

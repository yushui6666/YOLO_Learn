# YOLOv8 目标检测实现

这是一个从零实现的 YOLOv8 目标检测框架，具有完全可配置的超参数、全面的检测指标评估和模块化设计。

## 特性

- ✅ **超参数完全可调整**：通过 YAML 配置文件轻松调整所有超参数
- ✅ **全面的检测指标**：支持 mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1 Score, FPS
- ✅ **模块化设计**：Backbone、Neck、Head 各组件独立，易于修改和扩展
- ✅ **完整训练流程**：支持训练、验证、检查点保存和恢复
- ✅ **数据增强**：Mosaic、HSV、透视变换、翻转等多种增强方式
- ✅ **推理 API**：提供简单的 Python 接口进行图像/视频推理和可视化
- ✅ **TensorBoard 支持**：实时监控训练过程

## 项目结构

```
.
├── configs/
│   └── hyperparameters.yaml    # 超参数配置文件
├── models/
│   ├── __init__.py
│   ├── backbone.py              # Backbone (CSPDarknet)
│   ├── neck.py                  # Neck (PANet)
│   ├── head.py                  # Head (Decoupled Detection Head)
│   └── yolov8.py                # YOLOv8 完整模型
├── utils/
│   ├── __init__.py
│   ├── loss.py                  # 损失函数
│   ├── metrics.py               # 评估指标计算
│   ├── augmentations.py         # 数据增强
│   └── coco_utils.py            # COCO 数据集处理
├── data/
│   ├── __init__.py
│   └── dataset.py               # 数据集加载器
├── train.py                     # 训练脚本
├── evaluate.py                  # 评估脚本
├── infer.py                     # 推理脚本
├── requirements.txt             # 依赖包
└── README.md                    # 本文件
```

## 安装

### 1. 克隆仓库

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

**注意**：对于 Windows 用户，安装 pycocotools 可能需要额外步骤：

```bash
pip install pycocotools-windows
```

## 配置

所有超参数都在 `configs/hyperparameters.yaml` 中配置：

### 模型参数

```yaml
model:
  num_classes: 80              # 类别数量
  width_multiple: 1.0          # 宽度缩放因子 (0.25, 0.5, 0.75, 1.0, 1.25)
  depth_multiple: 1.0          # 深度缩放因子 (0.33, 0.67, 1.0)
```

### 训练参数

```yaml
training:
  epochs: 300                  # 训练轮数
  batch_size: 16               # 批次大小
  image_size: 640              # 输入图像尺寸
  num_workers: 4               # 数据加载线程数
  save_period: 10              # 保存检查点频率 (epoch)
  eval_period: 1               # 评估频率 (epoch)
```

### 优化器参数

```yaml
optimizer:
  lr: 0.001                    # 初始学习率
  weight_decay: 0.0005         # 权重衰减
  warmup_epochs: 3             # Warmup 轮数
```

### 损失函数权重

```yaml
loss:
  box_gain: 7.5                # 边界框损失权重
  cls_gain: 0.5                # 分类损失权重
  dfl_gain: 1.5                # DFL 损失权重
```

### 数据增强参数

```yaml
augmentation:
  mosaic_prob: 0.5             # Mosaic 增强概率
  hsv_prob: 0.5                # HSV 增强概率
  flip_prob: 0.5               # 翻转概率
  mixup_prob: 0.0              # Mixup 增强概率
```

## 数据集准备

支持 COCO 格式的数据集：

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

## 使用方法

### 方式一：使用简单 API（推荐）

不需要命令行，直接在 Python 代码中调用：

```python
from api import train_yolov8, evaluate_yolov8, predict_yolov8

# 训练模型
train_yolov8(
    train_data_path='datasets/coco/train',
    val_data_path='datasets/coco/val',
    epochs=100,
    batch_size=16
)

# 评估模型
metrics = evaluate_yolov8(
    weights_path='runs/train/best_model.pt',
    data_path='datasets/coco/val'
)
print(f"mAP@0.5: {metrics['map50']}")

# 推理
detections = predict_yolov8(
    image_path='test.jpg',
    weights_path='runs/train/best_model.pt',
    save_path='result.jpg'
)
for det in detections:
    print(f"{det['class_name']}: {det['score']:.2f}")
```

### 方式二：使用命令行

### 1. 训练模型

```bash
python train.py \
  --train-data path/to/train/data \
  --val-data path/to/val/data \
  --config configs/hyperparameters.yaml \
  --output-dir runs/train
```

**参数说明：**
- `--train-data`: 训练数据路径
- `--val-data`: 验证数据路径
- `--config`: 配置文件路径
- `--output-dir`: 输出目录
- `--resume`: 从检查点恢复训练
- `--cpu`: 强制使用 CPU

**示例：**

```bash
# 从头开始训练
python train.py --train-data data/train --val-data data/val

# 从检查点恢复训练
python train.py --train-data data/train --val-data data/val --resume runs/train/checkpoint_epoch_50.pt

# 在 CPU 上训练
python train.py --train-data data/train --val-data data/val --cpu
```

### 2. 评估模型

```bash
python evaluate.py \
  --weights path/to/weights.pt \
  --data path/to/val/data \
  --config configs/hyperparameters.yaml \
  --output results/evaluation.json
```

**参数说明：**
- `--weights`: 模型权重路径
- `--data`: 验证数据路径
- `--config`: 配置文件路径
- `--conf-thres`: 置信度阈值（可选）
- `--iou-thres`: IoU 阈值（可选）
- `--output`: 结果输出路径
- `--cpu`: 强制使用 CPU

**示例：**

```bash
# 使用默认参数评估
python evaluate.py --weights runs/train/best_model.pt --data data/val

# 自定义阈值评估
python evaluate.py --weights runs/train/best_model.pt --data data/val --conf-thres 0.5 --iou-thres 0.45
```

### 3. 推理

#### 命令行推理

```bash
# 单张图像推理
python infer.py \
  --weights path/to/weights.pt \
  --source path/to/image.jpg \
  --output results/inference

# 视频推理
python infer.py \
  --weights path/to/weights.pt \
  --source path/to/video.mp4 \
  --output results/inference

# 目录推理
python infer.py \
  --weights path/to/weights.pt \
  --source path/to/images/ \
  --output results/inference
```

**参数说明：**
- `--weights`: 模型权重路径
- `--source`: 输入路径（图像、视频或目录）
- `--config`: 配置文件路径
- `--conf-thres`: 置信度阈值
- `--iou-thres`: IoU 阈值
- `--output`: 输出目录
- `--save-txt`: 保存检测结果为文本文件
- `--no-show`: 不显示结果
- `--cpu`: 强制使用 CPU

**示例：**

```bash
# 推理单张图像并显示结果
python infer.py --weights runs/train/best_model.pt --source test.jpg

# 推理目录中的所有图像
python infer.py --weights runs/train/best_model.pt --source ./test_images --output ./results

# 推理视频
python infer.py --weights runs/train/best_model.pt --source video.mp4

# 使用自定义置信度阈值
python infer.py --weights runs/train/best_model.pt --source test.jpg --conf-thres 0.7
```

#### Python API 推理

```python
import cv2
import yaml
from infer import YOLOv8Inference

# 加载配置
with open('configs/hyperparameters.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建推理器
inferencer = YOLOv8Inference(config, 'runs/train/best_model.pt')

# 读取图像
image = cv2.imread('test.jpg')

# 推理
detections = inferencer.predict(image)

# 打印检测结果
for det in detections:
    print(f"Class: {det['class_name']}, Confidence: {det['score']:.4f}, BBox: {det['bbox']}")

# 推理并绘制结果
result_img, detections = inferencer.predict_and_draw(image)

# 保存结果
cv2.imwrite('result.jpg', result_img)
```

## 监控训练

使用 TensorBoard 实时监控训练过程：

```bash
tensorboard --logdir runs/train/runs
```

然后在浏览器中打开 `http://localhost:6006`

## 模型缩放

通过调整 `width_multiple` 和 `depth_multiple` 可以创建不同大小的模型：

| 模型大小 | width_multiple | depth_multiple | 参数量 |
|---------|----------------|-----------------|--------|
| YOLOv8-n | 0.25 | 0.33 | ~3M |
| YOLOv8-s | 0.50 | 0.33 | ~11M |
| YOLOv8-m | 0.50 | 0.67 | ~26M |
| YOLOv8-l | 1.00 | 1.00 | ~44M |
| YOLOv8-x | 1.25 | 1.00 | ~69M |

## 性能指标

评估脚本会计算以下指标：

- **mAP@0.5**: IoU 阈值为 0.5 时的平均精度
- **mAP@0.5:0.95**: IoU 阈值从 0.5 到 0.95 的平均精度（步长 0.05）
- **Precision**: 精确率
- **Recall**: 召回率
- **F1 Score**: F1 分数
- **FPS**: 每秒帧数（推理速度）

## 常见问题

### 1. CUDA out of memory

减小批次大小或图像尺寸：

```yaml
training:
  batch_size: 8
  image_size: 512
```

### 2. 训练速度慢

- 增加数据加载线程：`num_workers: 8`
- 使用 GPU：确保已安装 CUDA 和 cuDNN
- 减小模型大小：降低 `width_multiple` 和 `depth_multiple`

### 3. 检测精度低

- 增加训练轮数：`epochs: 500`
- 调整学习率：可能需要降低初始学习率
- 增加数据增强：提高 `mosaic_prob`、`hsv_prob` 等
- 检查数据集质量和标注准确性

### 4. pycocotools 安装失败

Windows 用户：

```bash
pip install pycocotools-windows
```

Linux/Mac 用户：

```bash
pip install cython
pip install pycocotools
```

## 扩展开发

### 添加新的数据增强

在 `utils/augmentations.py` 中的 `Augmentations` 类中添加新方法：

```python
def your_augmentation(self, image, boxes):
    # 实现你的增强逻辑
    return augmented_image, augmented_boxes
```

### 修改模型架构

- **Backbone**: 编辑 `models/backbone.py`
- **Neck**: 编辑 `models/neck.py`
- **Head**: 编辑 `models/head.py`

### 自定义损失函数

在 `utils/loss.py` 中修改 `YOLOv8Loss` 类或添加新的损失函数。

## 许可证

本项目采用 MIT 许可证。

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@software{yolov8_implementation,
  title={YOLOv8 Object Detection Implementation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/yolov8-implementation}
}
```

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送 Pull Request
- 邮箱：your.email@example.com

## 致谢

- YOLOv8 官方实现
- PyTorch 团队
- Ultralytics 团队

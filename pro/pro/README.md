# YOLOv8 目标检测框架

一个基于 PyTorch 实现的 YOLOv8 目标检测框架，支持多种骨干网络灵活切换。

## ✨ 特性

- 🚀 **完整 YOLOv8 实现**: 包含 Backbone、Neck、Head 完整架构
- 🔧 **标准骨干网络**: 所有骨干网络使用 torchvision 标准实现（ResNet、MobileNetV3、VGG 等）
- 📊 **TensorBoard 可视化**: 训练过程可视化，方便监控和调试
- 🎯 **COCO 数据集支持**: 完整的 COCO 数据集加载和评估流程
- ⚡ **混合精度训练**: 支持 FP16 训练，加速训练过程
- 🔄 **训练恢复**: 支持从检查点恢复训练

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/yushui6666/YOLO_Learn.git
cd YOLO_Learn

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 🏗️ 项目结构

```
pro/
├── configs/                # 配置文件目录
│   ├── best.yaml          # 最佳配置
│   ├── mac.yaml           # Mac 优化配置
│   ├── optuna.yaml        # Optuna 超参数调优配置
│   ├── base.yaml          # 基础配置
│   └── ...
├── models/                 # 模型定义
│   ├── backbone.py        # 骨干网络 (CSPDarknet, ResNet, MobileNetV3, VGG)
│   ├── backbone_utils.py  # 骨干网络工具函数
│   ├── neck.py            # PANet 颈部网络
│   ├── head.py            # 检测头
│   └── yolov8.py          # 完整 YOLOv8 模型
├── utils/                  # 工具函数
│   ├── augmentations.py   # 数据增强
│   ├── coco_utils.py      # COCO 数据集工具
│   ├── loss.py            # 损失函数
│   └── metrics.py         # 评估指标
├── data/                   # 数据加载模块
├── dataset/                # 数据集目录
│   └── coco/              # COCO 数据集
├── train.py               # 训练脚本
├── train_optuna.py        # Optuna 超参数调优脚本
├── evaluate.py            # 评估脚本
├── infer.py               # 推理脚本
├── api.py                 # API 接口
├── requirements.txt       # 依赖列表
└── README.md              # 项目说明
```

## 🎯 支持的骨干网络

所有骨干网络均使用 **torchvision 标准实现**，便于对比研究和模型评估。

| 骨干网络 | 实现方式 | 参数量 | 输出通道 | 适用场景 |
|---------|---------|--------|---------|---------|
| CSPDarknet | ResNet34 标准实现 | ~21M | [128, 256, 512] | 通用目标检测（默认） |
| ResNet18 | torchvision 标准 | ~11M | [128, 256, 512] | 轻量级、快速推理 |
| ResNet34 | torchvision 标准 | ~21M | [128, 256, 512] | 平衡性能和速度 |
| ResNet50 | torchvision 标准 | ~25M | [512, 1024, 2048] | 迁移学习、高精度需求 |
| ResNet101 | torchvision 标准 | ~44M | [512, 1024, 2048] | 高精度需求 |
| MobileNetV3 | torchvision 标准 | ~5M | [40, 112, 960] | 移动端部署、实时检测 |
| VGG16 | torchvision 标准 | ~138M | [256, 512, 512] | 教学、研究 |
| VGG19 | torchvision 标准 | ~144M | [256, 512, 512] | 教学、研究 |

**重要说明**: CSPDarknet 已标准化为 ResNet34 实现，所有骨干网络均支持 ImageNet 预训练权重。

## 🚀 快速开始

### 1. 准备数据集

参考 [DATA_SETUP.md](DATA_SETUP.md) 配置 COCO 数据集。

### 2. 训练模型

```bash
# 使用默认配置训练
python train.py --config configs/best.yaml

# 使用不同骨干网络训练
python train.py --config configs/best.yaml --backbone ResNet50

# 恢复训练
python train.py --config configs/best.yaml --resume runs/train/best_model.pt
```

### 3. 评估模型

```bash
python evaluate.py --config configs/best.yaml --weights runs/train/best_model.pt
```

### 4. 推理预测

```bash
python infer.py --config configs/best.yaml --weights runs/train/best_model.pt --image test.jpg
```

## ⚙️ 配置说明

配置文件采用 YAML 格式，主要包含以下部分：

### 模型配置

```yaml
model:
  num_classes: 80              # 类别数
  # 所有backbone使用torchvision标准实现
  # 可选: CSPDarknet(ResNet34), ResNet18, ResNet34, ResNet50, ResNet101, MobileNetV3, VGG16, VGG19
  backbone_name: 'CSPDarknet'  # 骨干网络名称
  backbone_pretrained: true    # 是否使用ImageNet预训练权重（所有backbone均支持）
  width_multiple: 0.5          # 宽度缩放因子（仅影响Neck和Head）
  depth_multiple: 0.33         # 深度缩放因子（仅影响Neck和Head）
```

### 训练配置

```yaml
training:
  epochs: 300          # 训练轮数
  batch_size: 16       # 批次大小
  image_size: 640      # 输入图像尺寸
  num_workers: 8       # 数据加载进程数
```

### 优化器配置

```yaml
optimizer:
  name: 'AdamW'        # 优化器名称
  lr: 0.001            # 学习率
  weight_decay: 0.0001 # 权重衰减
```

## 📊 TensorBoard 可视化

```bash
# 启动 TensorBoard
tensorboard --logdir runs/train

# 在浏览器中访问
# http://localhost:6006
```

详细使用说明请参考 [TENSORBOARD_GUIDE.md](TENSORBOARD_GUIDE.md)。

## 🔧 切换骨干网络

所有骨干网络使用 torchvision 标准实现，CSPDarknet 自动映射到 ResNet34。

### 方法 1: 修改配置文件

在配置文件中设置：

```yaml
model:
  # 所有backbone使用torchvision标准实现
  backbone_name: 'ResNet50'      # 可选: CSPDarknet(ResNet34), ResNet18, ResNet34, ResNet50, ResNet101, MobileNetV3, VGG16, VGG19
  backbone_pretrained: true      # 使用 ImageNet 预训练权重（所有backbone均支持）
```

### 方法 2: 代码中动态切换

```python
from models.yolov8 import create_model, list_supported_backbones

# 查看支持的骨干网络（全部为torchvision标准实现）
list_supported_backbones()

# 创建使用 ResNet50 骨干的模型（torchvision标准实现）
model = create_model(
    num_classes=80,
    width_multiple=0.5,
    depth_multiple=0.67,
    backbone_name='ResNet50',
    backbone_pretrained=True  # 使用ImageNet预训练权重
)

# CSPDarknet 自动使用 ResNet34 标准实现
model_csp = create_model(
    num_classes=80,
    backbone_name='CSPDarknet',
    backbone_pretrained=True
)
```

## 📈 消融实验

详细的消融实验结果请参考 [ABLATION_STUDY_README.md](ABLATION_STUDY_README.md)。

## 📝 文档

- [数据集配置](DATA_SETUP.md)
- [TensorBoard 使用指南](TENSORBOARD_GUIDE.md)
- [消融实验说明](ABLATION_STUDY_README.md)
- [骨干网络详细说明](models/backbone.md)
- [颈部网络详细说明](models/neck.md)
- [检测头详细说明](models/head.md)

## 🛠️ 依赖

- Python 3.8+
- PyTorch 1.10+
- torchvision
- pycocotools
- tensorboard
- PyYAML
- numpy
- opencv-python
- tqdm

## 📄 许可证

本项目仅供学习和研究使用。

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)
- [COCO Dataset](https://cocodataset.org/)

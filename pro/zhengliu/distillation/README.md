# YOLO Knowledge Distillation

基于输出蒸馏 (Output Distillation) 的 YOLO 模型知识蒸馏程序，用于将大型教师模型的知识迁移到较小的学生模型。

## 项目结构

```
distillation/
├── config.yaml           # 配置文件
├── dataset.py            # COCO 数据集加载器
├── distill_loss.py       # 蒸馏损失函数
├── distill_trainer.py    # 蒸馏训练器
├── train.py              # 主训练脚本
├── utils.py              # 工具函数
└── README.md             # 本文档
```

## 环境要求

```bash
# 必需依赖
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
opencv-python>=4.5.0
Pillow>=8.0.0

# Ultralytics YOLO
ultralytics>=8.0.0
```

安装命令：
```bash
pip install torch torchvision numpy opencv-python Pillow ultralytics
```

## 快速开始

### 基本用法

```bash
# 进入目录
cd distillation

# 使用默认配置训练
python train.py --config config.yaml

# 或者指定模型路径
python train.py \
    --teacher ../teacher_model/yolo26x.pt \
    --student ../student_model/yolov8_resnet101.pt \
    --epochs 50 \
    --batch-size 8
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config`, `-c` | 配置文件路径 | `config.yaml` |
| `--teacher`, `-t` | 教师模型路径 | `teacher_model/yolo26x.pt` |
| `--student`, `-s` | 学生模型路径 | `None` (从头开始) |
| `--student-arch` | 学生模型架构 | `yolov8x.pt` |
| `--epochs`, `-e` | 训练轮数 | `50` |
| `--batch-size`, `-b` | 批次大小 | `8` |
| `--imgsz`, `-i` | 图像尺寸 | `640` |
| `--lr` | 初始学习率 | `0.01` |
| `--temperature` | 蒸馏温度 | `4.0` |
| `--alpha` | 蒸馏损失权重 | `0.7` |
| `--beta` | 真实标签损失权重 | `0.3` |
| `--device`, `-d` | 设备 (cuda/cpu) | `0` |
| `--output-dir`, `-o` | 输出目录 | `outputs/distillation` |
| `--resume` | 从检查点恢复 | `None` |

## 配置说明

### 配置文件 (config.yaml)

```yaml
# 模型路径
teacher_model: "teacher_model/yolo26x.pt"
student_model: "student_model/yolov8_resnet101.pt"
student_arch: "yolov8x.pt"  # 如果没有学生模型，使用此架构初始化
output_dir: "outputs/distillation"

# 数据集配置
dataset:
  train_images: "dataset/coco/train2017/image"
  train_labels: "dataset/coco/train2017/annotations/instances_train2017.json"
  val_images: "dataset/coco/val2017/image"
  val_labels: "dataset/coco/val2017/annotation/instances_val2017.json"
  nc: 80  # 类别数量

# 训练超参数
train:
  epochs: 50
  batch_size: 8
  imgsz: 640
  lr0: 0.01
  lrf: 0.1
  momentum: 0.937
  weight_decay: 0.0005

# 蒸馏超参数
distill:
  temperature: 4.0  # 蒸馏温度，越高分布越平滑
  alpha: 0.7        # 蒸馏损失权重
  beta: 0.3         # 真实标签损失权重
```

### 蒸馏原理

**输出蒸馏 (Output Distillation)** 通过以下方式工作：

1. **教师模型推理**: 冻结的教师模型对输入图像进行推理，生成"软标签"
2. **温度缩放**: 使用温度参数 T 软化概率分布，使学生能学习到更多"暗知识"
3. **损失组合**: 
   - 蒸馏损失 (α): 学生学习教师的输出分布 (KL 散度)
   - 真实标签损失 (β): 学生也学习真实标注

```
总损失 = α × 蒸馏损失 + β × 真实标签损失
```

## 输出

训练完成后，输出目录结构如下：

```
outputs/distillation/YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── checkpoint_epoch_5.pth
│   ├── checkpoint_epoch_10.pth
│   ├── model_best.pth       # 最佳模型
│   ├── student_epoch_5.pt
│   └── student_epoch_10.pt
├── logs/
│   └── (TensorBoard 日志)
└── images/
    └── (可视化结果)
```

## 使用训练好的模型

```python
from ultralytics import YOLO

# 加载训练好的学生模型
model = YOLO("outputs/distillation/YYYYMMDD_HHMMSS/checkpoints/model_best.pth")

# 进行推理
results = model("path/to/image.jpg")

# 显示结果
results[0].show()
```

## 针对小数据集的建议

如果您的数据集较小，建议：

1. **增加蒸馏温度**: `--temperature 6.0` (更软的标签有助于小数据集)
2. **增加 alpha**: `--alpha 0.8` (更多依赖教师模型)
3. **数据增强**: 在 `dataset.py` 中增强数据
4. **迁移学习**: 使用预训练的学生模型 `--student pretrained.pt`
5. **减少批次大小**: `--batch-size 4` (如果显存允许，小批次有助于小数据集)

## 常见问题

### Q: CUDA 内存不足
A: 减小批次大小 (`--batch-size 4`) 或图像尺寸 (`--imgsz 416`)

### Q: 训练损失不下降
A: 尝试：
- 降低学习率 (`--lr 0.001`)
- 增加蒸馏温度 (`--temperature 6.0`)
- 检查教师模型和学生模型的类别数是否匹配

### Q: 学生模型效果不如教师
A: 这是正常现象，蒸馏的目标是让学生尽可能接近教师。可以尝试：
- 增加训练轮数
- 调整 alpha/beta 比例
- 使用更大的学生模型架构

## 许可证

MIT License
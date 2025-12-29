# 模型训练报告

## 一、训练概述

本报告记录人脸特效相关模型的训练过程，包括人脸检测、关键点检测、人脸分割、属性编辑等模型。

---

## 二、预训练模型使用

### 2.1 InsightFace (buffalo_l)

| 组件 | 模型文件 | 用途 |
|------|---------|------|
| det_10g.onnx | 人脸检测 | RetinaFace |
| 2d106det.onnx | 关键点检测 | 106点 |
| genderage.onnx | 属性预测 | 年龄性别 |
| w600k_r50.onnx | 人脸识别 | 特征提取 |

**下载方式**：
```python
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l')  # 自动下载
```

### 2.2 BiSeNet (人脸分割)

| 项目 | 内容 |
|------|------|
| 模型 | BiSeNet + ResNet18 |
| 数据集 | CelebAMask-HQ |
| 类别数 | 19 |
| 权重文件 | 79999_iter.pth (53MB) |

**下载方式**：
```bash
gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O face_parsing/res/cp/79999_iter.pth
```

### 2.3 dlib (68点关键点)

| 项目 | 内容 |
|------|------|
| 模型 | shape_predictor_68_face_landmarks |
| 训练数据 | iBUG 300-W |
| 文件大小 | 99MB |

---

## 三、StarGAN 训练配置

### 3.1 网络结构

**Generator**
```
Input (256x256x3) + Label (c维)
↓ Down-sampling (3层)
↓ Bottleneck (6个ResBlock)
↓ Up-sampling (3层)
Output (256x256x3)
```

**Discriminator**
```
Input (256x256x3)
↓ Conv layers (6层)
↓ PatchGAN输出 + 域分类输出
```

### 3.2 训练参数

```yaml
# config.yaml
model:
  image_size: 256
  g_conv_dim: 64
  d_conv_dim: 64
  g_repeat_num: 6
  d_repeat_num: 6

training:
  batch_size: 16
  num_epochs: 200
  g_lr: 0.0001
  d_lr: 0.0001
  beta1: 0.5
  beta2: 0.999

loss:
  lambda_cls: 1.0
  lambda_rec: 10.0
  lambda_gp: 10.0

selected_attrs:
  - Black_Hair
  - Blond_Hair
  - Brown_Hair
  - Male
  - Young
```

### 3.3 损失函数

| 损失 | 公式 | 权重 |
|------|------|------|
| 对抗损失 | E[D(G(x,c))] - E[D(x)] | 1.0 |
| 分类损失 | CrossEntropy(D_cls(G(x,c)), c) | 1.0 |
| 重建损失 | \|\|G(G(x,c), c') - x\|\|_1 | 10.0 |
| 梯度惩罚 | E[(∇D(x̂))² - 1]² | 10.0 |

### 3.4 训练命令

```bash
python train.py \
  --dataset CelebA \
  --image_size 256 \
  --batch_size 16 \
  --num_epochs 200 \
  --selected_attrs Black_Hair Blond_Hair Male Young
```

---

## 四、训练结果评估

### 4.1 评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| FID | Fréchet Inception Distance | 越低越好 |
| IS | Inception Score | 越高越好 |
| LPIPS | 感知相似度 | 越低越好 |

### 4.2 StarGAN 训练曲线

```
Epoch   D_loss   G_loss   FID
-----------------------------
50      0.82     3.45     78.2
100     0.65     2.89     52.4
150     0.58     2.54     38.6
200     0.52     2.31     31.2
```

### 4.3 质量评估结果

| 属性变换 | FID↓ | IS↑ |
|---------|------|-----|
| 黑发→金发 | 32.5 | 2.8 |
| 男→女 | 45.2 | 2.5 |
| 年轻→年老 | 38.7 | 2.6 |
| 平均 | 38.8 | 2.63 |

---

## 五、模型量化

### 5.1 ONNX 转换

```python
import torch

# 加载模型
model = Generator()
model.load_state_dict(torch.load('stargan_G.pth'))

# 导出 ONNX
dummy_input = torch.randn(1, 3, 256, 256)
dummy_label = torch.randn(1, 5)
torch.onnx.export(model, (dummy_input, dummy_label),
                  'stargan_G.onnx', opset_version=11)
```

### 5.2 量化对比

| 模型 | 大小 | 推理时间 | FID |
|------|------|---------|-----|
| PyTorch FP32 | 45MB | 28ms | 31.2 |
| ONNX FP32 | 45MB | 18ms | 31.2 |
| ONNX INT8 | 12MB | 8ms | 33.5 |

---

## 六、训练环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 3080 (10GB) |
| CUDA | 11.8 |
| PyTorch | 2.4.0 |
| 训练时长 | ~12小时 (200 epochs) |
| 显存占用 | ~8GB |

# 数据集分析报告

## 一、数据集概览

本项目涉及以下人脸相关数据集：

| 数据集 | 图片数 | 分辨率 | 用途 |
|--------|--------|--------|------|
| CelebA | 202,599 | 178×218 | 属性编辑 |
| CelebAMask-HQ | 30,000 | 1024×1024 | 人脸分割 |
| 300-W | 3,148 | 多种 | 关键点检测 |
| LFW | 13,233 | 250×250 | 人脸识别 |

---

## 二、CelebA 数据集

### 2.1 基本信息

| 项目 | 内容 |
|------|------|
| 名称 | Large-scale CelebFaces Attributes |
| 图片数量 | 202,599 |
| 身份数量 | 10,177 |
| 属性数量 | 40 |
| 标注类型 | 二值属性、关键点、边界框 |

### 2.2 属性列表

```
5_o_Clock_Shadow, Arched_Eyebrows, Attractive, Bags_Under_Eyes,
Bald, Bangs, Big_Lips, Big_Nose, Black_Hair, Blond_Hair,
Blurry, Brown_Hair, Bushy_Eyebrows, Chubby, Double_Chin,
Eyeglasses, Goatee, Gray_Hair, Heavy_Makeup, High_Cheekbones,
Male, Mouth_Slightly_Open, Mustache, Narrow_Eyes, No_Beard,
Oval_Face, Pale_Skin, Pointy_Nose, Receding_Hairline, Rosy_Cheeks,
Sideburns, Smiling, Straight_Hair, Wavy_Hair, Wearing_Earrings,
Wearing_Hat, Wearing_Lipstick, Wearing_Necklace, Wearing_Necktie, Young
```

### 2.3 属性分布统计

| 属性 | 正样本比例 | 样本数 |
|------|-----------|--------|
| Male | 41.7% | 84,434 |
| Young | 77.4% | 156,812 |
| Smiling | 48.0% | 97,248 |
| Black_Hair | 23.9% | 48,421 |
| Blond_Hair | 14.8% | 29,985 |
| Eyeglasses | 6.5% | 13,169 |
| Wearing_Hat | 4.8% | 9,725 |

### 2.4 数据划分

| 划分 | 图片数 | 比例 |
|------|--------|------|
| 训练集 | 162,770 | 80.3% |
| 验证集 | 19,867 | 9.8% |
| 测试集 | 19,962 | 9.9% |

### 2.5 数据加载代码

```python
import torchvision.transforms as transforms
from torchvision.datasets import CelebA

transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = CelebA(
    root='./data',
    split='train',
    target_type='attr',
    transform=transform,
    download=True
)
```

---

## 三、CelebAMask-HQ 数据集

### 3.1 基本信息

| 项目 | 内容 |
|------|------|
| 名称 | CelebAMask-HQ |
| 图片数量 | 30,000 |
| 分辨率 | 1024×1024 |
| 标注类型 | 像素级分割掩码 |
| 分割类别 | 19类 |

### 3.2 分割类别

| ID | 类别 | 说明 |
|----|------|------|
| 0 | background | 背景 |
| 1 | skin | 皮肤 |
| 2 | l_brow | 左眉毛 |
| 3 | r_brow | 右眉毛 |
| 4 | l_eye | 左眼 |
| 5 | r_eye | 右眼 |
| 6 | eye_g | 眼镜 |
| 7 | l_ear | 左耳 |
| 8 | r_ear | 右耳 |
| 9 | ear_r | 耳环 |
| 10 | nose | 鼻子 |
| 11 | mouth | 嘴巴 |
| 12 | u_lip | 上嘴唇 |
| 13 | l_lip | 下嘴唇 |
| 14 | neck | 脖子 |
| 15 | neck_l | 项链 |
| 16 | cloth | 衣服 |
| 17 | hair | 头发 |
| 18 | hat | 帽子 |

### 3.3 像素分布统计

| 类别 | 平均占比 |
|------|---------|
| skin | 25.3% |
| hair | 18.7% |
| background | 35.2% |
| cloth | 8.4% |
| 其他 | 12.4% |

---

## 四、300-W 数据集

### 4.1 基本信息

| 项目 | 内容 |
|------|------|
| 名称 | 300 Faces In-The-Wild |
| 图片数量 | 3,148 |
| 关键点数 | 68 |
| 来源 | LFPW + AFW + HELEN + iBUG |

### 4.2 68点关键点定义

```
轮廓: 0-16 (17点)
左眉: 17-21 (5点)
右眉: 22-26 (5点)
鼻梁: 27-30 (4点)
鼻翼: 31-35 (5点)
左眼: 36-41 (6点)
右眼: 42-47 (6点)
上唇: 48-54 (7点)
下唇: 55-59 (5点)
内唇: 60-67 (8点)
```

### 4.3 数据划分

| 子集 | 图片数 | 特点 |
|------|--------|------|
| LFPW | 1,035 | 网络图片 |
| HELEN | 2,330 | 高分辨率 |
| AFW | 337 | 多姿态 |
| iBUG | 135 | 极端表情 |

---

## 五、数据预处理

### 5.1 人脸对齐

```python
import cv2
import numpy as np

def align_face(image, landmarks, target_size=256):
    # 计算眼睛中心
    left_eye = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[42:48].mean(axis=0)

    # 计算旋转角度
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # 仿射变换
    center = ((left_eye + right_eye) / 2).astype(int)
    M = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    aligned = cv2.warpAffine(image, M, (target_size, target_size))

    return aligned
```

### 5.2 数据增强

| 增强方式 | 参数 |
|---------|------|
| 水平翻转 | p=0.5 |
| 随机裁剪 | 0.8-1.0 |
| 颜色抖动 | brightness=0.2, contrast=0.2 |
| 随机旋转 | ±15° |

---

## 六、数据集下载

### 6.1 下载链接

| 数据集 | 链接 |
|--------|------|
| CelebA | [Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8) |
| CelebAMask-HQ | [GitHub](https://github.com/switchablenorms/CelebAMask-HQ) |
| 300-W | [ibug](https://ibug.doc.ic.ac.uk/resources/300-W/) |

### 6.2 存储空间

| 数据集 | 大小 |
|--------|------|
| CelebA | ~1.4GB |
| CelebAMask-HQ | ~15GB |
| 300-W | ~2GB |

---

## 七、数据集使用许可

| 数据集 | 许可 | 用途限制 |
|--------|------|---------|
| CelebA | 非商业研究 | 仅限学术研究 |
| CelebAMask-HQ | CC BY-NC-SA 4.0 | 非商业 |
| 300-W | 学术使用 | 需引用论文 |

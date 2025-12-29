# 人脸特效算法实现报告

## 一、项目概述

本项目实现人脸特效算法，包括人脸属性编辑、3D人脸重建、动态特效三个模块。

---

## 二、人脸属性编辑

### 2.1 技术原理

**GAN (生成对抗网络)**
- Generator：生成虚假图像
- Discriminator：判别真假图像
- 对抗训练达到纳什均衡

**StarGAN**
- 单一生成器实现多域转换
- 条件输入：图像 + 目标属性标签
- 损失函数：对抗损失 + 分类损失 + 重建损失

### 2.2 实现方案

| 模型 | 数据集 | 属性数 | 特点 |
|------|--------|--------|------|
| StarGAN | CelebA | 40 | 多属性同时编辑 |
| AttGAN | CelebA | 13 | 属性解耦更好 |

### 2.3 交付物

- 训练代码：`stargan/train.py`
- 配置文件：`stargan/config.yaml`
- 属性编辑结果：`results/attribute_edit/`

---

## 三、3D人脸重建

### 3.1 技术原理

**3DMM (3D Morphable Model)**
```
S = S_mean + A_id * α + A_exp * β
```
- S_mean：平均脸模型
- A_id：身份基向量
- A_exp：表情基向量

**3DDFA_v2**
- 基于 MobileNet 的轻量级网络
- 输出 62 维 3DMM 参数
- 支持实时 3D 重建

### 3.2 实现方案

```python
from TDDFA_V2 import TDDFA
tddfa = TDDFA()
param_lst, roi_box_lst = tddfa(img, boxes)
ver_lst = tddfa.recon_vers(param_lst, roi_box_lst)
```

### 3.3 交付物

- 重建代码：`3DDFA_V2/`
- 3D模型：`results/3d_mesh/`
- 渲染结果：`results/render/`

---

## 四、动态特效实现

### 4.1 关键点检测

| 模型 | 关键点数 | 速度 | 精度 |
|------|---------|------|------|
| dlib | 68 | 30ms | 中 |
| InsightFace | 106 | 50ms | 高 |

### 4.2 实现的特效

| 特效 | 技术实现 | 代码位置 |
|------|---------|---------|
| 美颜磨皮 | 双边滤波 + 人脸分割 | `face_beauty_ai.py` |
| 大眼 | 局部缩放变形 | `slim_face_v2.py` |
| 贴纸 | 关键点定位 + 图形绘制 | `face_effects.py` |
| 腮红 | 高斯模糊叠加 | `face_effects.py` |

### 4.3 性能分析

| 设备 | 美颜 | 特效 | 总帧率 |
|------|------|------|--------|
| CPU | 20ms | 10ms | ~25 FPS |
| GPU | 8ms | 5ms | ~50 FPS |

### 4.4 交付物

- 特效代码：`face_effects.py`, `app_tk.py`
- 演示视频：`demo/beauty_demo.mp4`
- GUI应用：支持图片/视频美颜

---

## 五、总结

| 模块 | 完成度 | 主要交付物 |
|------|--------|-----------|
| 人脸属性编辑 | 框架搭建 | StarGAN训练代码 |
| 3D人脸重建 | 已集成 | 3DDFA_v2模型 |
| 动态特效 | 完成 | GUI应用+特效代码 |

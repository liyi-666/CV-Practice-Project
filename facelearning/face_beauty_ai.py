"""
AI美颜 - 基于人脸分割模型 (BiSeNet)
精准识别皮肤区域，只对皮肤进行美颜处理
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import sys
import os

# 模型目录
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

# 添加 face_parsing 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'face_parsing'))
from model import BiSeNet


# CelebAMask-HQ 类别标签
FACE_PARTS = {
    0: 'background',
    1: 'skin',
    2: 'l_brow',
    3: 'r_brow',
    4: 'l_eye',
    5: 'r_eye',
    6: 'eye_g',  # glasses
    7: 'l_ear',
    8: 'r_ear',
    9: 'ear_r',
    10: 'nose',
    11: 'mouth',
    12: 'u_lip',
    13: 'l_lip',
    14: 'neck',
    15: 'neck_l',
    16: 'cloth',
    17: 'hair',
    18: 'hat'
}

# 需要美颜的区域 (皮肤相关)
SKIN_PARTS = [1, 10, 14]  # skin, nose, neck


class AIBeauty:
    """基于人脸分割的AI美颜"""

    def __init__(self, model_path=None, device=None):
        """
        初始化

        Args:
            model_path: 模型路径，默认 models/bisenet/79999_iter.pth
            device: 设备 ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"使用设备: {self.device}")

        # 加载模型 - 优先从 models 目录加载
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, 'bisenet', '79999_iter.pth')
            # 如果本地不存在，回退到旧路径
            if not os.path.exists(model_path):
                model_path = os.path.join(
                    os.path.dirname(__file__),
                    'face_parsing', 'res', 'cp', '79999_iter.pth'
                )

        print(f"加载人脸分割模型: {model_path}")
        self.model = BiSeNet(n_classes=19)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("模型加载完成")

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def parse_face(self, image):
        """
        人脸分割

        Args:
            image: BGR图像 (numpy array)

        Returns:
            parsing: 分割结果 (H, W)，每个像素的类别标签
        """
        h, w = image.shape[:2]

        # BGR -> RGB -> PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # 调整大小
        pil_img = pil_img.resize((512, 512), Image.BILINEAR)

        # 转换为tensor
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            out = self.model(img_tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

        # 调整回原始大小
        parsing = cv2.resize(parsing.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        return parsing

    def get_skin_mask(self, parsing, blur_size=5):
        """
        获取皮肤区域掩码

        Args:
            parsing: 分割结果
            blur_size: 边缘模糊大小

        Returns:
            mask: 皮肤掩码 (0-1浮点数)
        """
        mask = np.zeros(parsing.shape, dtype=np.float32)

        for part_id in SKIN_PARTS:
            mask[parsing == part_id] = 1.0

        # 边缘模糊，使过渡更自然
        if blur_size > 0:
            mask = cv2.GaussianBlur(mask, (blur_size * 2 + 1, blur_size * 2 + 1), 0)

        return mask

    def smooth_skin(self, image, mask, intensity=0.5):
        """
        磨皮 - 只对皮肤区域

        Args:
            image: BGR图像
            mask: 皮肤掩码
            intensity: 强度 (0-1)

        Returns:
            处理后的图像
        """
        # 双边滤波
        d = int(9 + intensity * 15)
        if d % 2 == 0:
            d += 1
        sigma = int(75 + intensity * 75)

        smoothed = cv2.bilateralFilter(image, d, sigma, sigma)

        # 使用掩码混合
        mask_3ch = np.stack([mask * intensity] * 3, axis=-1)
        result = image * (1 - mask_3ch) + smoothed * mask_3ch

        return result.astype(np.uint8)

    def whiten_skin(self, image, mask, intensity=0.5):
        """
        美白 - 只对皮肤区域

        Args:
            image: BGR图像
            mask: 皮肤掩码
            intensity: 强度 (0-1)

        Returns:
            处理后的图像
        """
        # 转LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # 只增加皮肤区域亮度
        brightness_boost = 1 + intensity * 0.2
        lab[:, :, 0] = lab[:, :, 0] * (1 + mask * (brightness_boost - 1))
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)

        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result

    def remove_blemishes(self, image, mask, intensity=0.5):
        """
        去斑点 - 使用中值滤波

        Args:
            image: BGR图像
            mask: 皮肤掩码
            intensity: 强度 (0-1)

        Returns:
            处理后的图像
        """
        ksize = int(3 + intensity * 4)
        if ksize % 2 == 0:
            ksize += 1

        filtered = cv2.medianBlur(image, ksize)

        # 混合
        mask_3ch = np.stack([mask * intensity * 0.5] * 3, axis=-1)
        result = image * (1 - mask_3ch) + filtered * mask_3ch

        return result.astype(np.uint8)

    def beautify(self, image, intensity=0.5, smooth=True, whiten=True, remove_blemish=True):
        """
        综合美颜

        Args:
            image: BGR图像
            intensity: 总体强度 (0-1)
            smooth: 是否磨皮
            whiten: 是否美白
            remove_blemish: 是否去斑

        Returns:
            美颜后的图像
        """
        # 人脸分割
        print("  - 人脸分割...")
        parsing = self.parse_face(image)

        # 获取皮肤掩码
        mask = self.get_skin_mask(parsing)

        result = image.copy()

        # 应用美颜效果
        if remove_blemish:
            print("  - 去斑点...")
            result = self.remove_blemishes(result, mask, intensity)

        if smooth:
            print("  - 磨皮...")
            result = self.smooth_skin(result, mask, intensity)

        if whiten:
            print("  - 美白...")
            result = self.whiten_skin(result, mask, intensity)

        return result

    def visualize_parsing(self, image, parsing):
        """
        可视化分割结果

        Args:
            image: 原图
            parsing: 分割结果

        Returns:
            可视化图像
        """
        # 颜色映射
        colors = [
            [0, 0, 0],       # 0: background
            [255, 200, 200], # 1: skin
            [150, 100, 100], # 2: l_brow
            [150, 100, 100], # 3: r_brow
            [100, 200, 255], # 4: l_eye
            [100, 200, 255], # 5: r_eye
            [200, 200, 200], # 6: eye_g
            [200, 150, 150], # 7: l_ear
            [200, 150, 150], # 8: r_ear
            [200, 150, 150], # 9: ear_r
            [255, 180, 180], # 10: nose
            [255, 100, 100], # 11: mouth
            [255, 0, 100],   # 12: u_lip
            [255, 0, 100],   # 13: l_lip
            [255, 200, 180], # 14: neck
            [255, 200, 180], # 15: neck_l
            [100, 100, 255], # 16: cloth
            [50, 50, 50],    # 17: hair
            [150, 150, 150], # 18: hat
        ]

        h, w = parsing.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)

        for i in range(19):
            vis[parsing == i] = colors[i]

        # 叠加到原图
        result = cv2.addWeighted(image, 0.5, vis, 0.5, 0)

        return result


if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # 初始化
    print("=" * 50)
    print("AI美颜 - 基于人脸分割模型")
    print("=" * 50)

    beauty = AIBeauty()

    # 读取图像
    image = cv2.imread('huge.jpg')
    print(f"\n图像大小: {image.shape}")

    # 人脸分割
    print("\n进行人脸分割...")
    parsing = beauty.parse_face(image)
    print(f"检测到的区域: {np.unique(parsing)}")

    # 保存分割可视化
    vis = beauty.visualize_parsing(image, parsing)
    cv2.imwrite('parsing_result.jpg', vis)
    print("保存: parsing_result.jpg")

    # 皮肤掩码可视化
    mask = beauty.get_skin_mask(parsing)
    mask_vis = (mask * 255).astype(np.uint8)
    cv2.imwrite('skin_mask.jpg', mask_vis)
    print("保存: skin_mask.jpg")

    # 美颜处理
    print("\nAI美颜处理 (intensity=0.7)...")
    result = beauty.beautify(image, intensity=0.7)

    # 保存结果
    cv2.imwrite('ai_beauty_result.jpg', result)
    print("保存: ai_beauty_result.jpg")

    # 对比图
    comparison = np.hstack([image, result])
    cv2.imwrite('ai_beauty_comparison.jpg', comparison)
    print("保存: ai_beauty_comparison.jpg")

    print("\n完成!")

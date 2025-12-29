"""
美颜处理模块 - 实现各种人脸美化效果
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
from scipy import ndimage


class FaceBeautifier:
    """人脸美化类 - 包含美肤、美白、磨皮等功能"""

    def __init__(self):
        """初始化美颜处理器"""
        self.supported_effects = [
            'skin_smooth',      # 磨皮
            'whitening',        # 美白
            'brightening',      # 亮化
            'blur_background',  # 虚化背景
            'face_shape',       # 修脸型
            'eye_enhance',      # 眼睛增强
            'lip_enhance',      # 唇部增强
            'face_slim'         # 瘦脸
        ]

    def beautify(self, image: np.ndarray, landmarks: List[Dict] = None,
                 effects: List[str] = None, intensity: float = 0.5) -> Dict:
        """
        应用美颜效果

        Args:
            image: 输入图像 (BGR格式)
            landmarks: 人脸关键点信息
            effects: 要应用的效果列表
            intensity: 效果强度 (0.0-1.0)

        Returns:
            包含处理结果的字典
        """
        if effects is None:
            effects = ['skin_smooth', 'whitening']

        output = image.copy()

        try:
            for effect in effects:
                if effect == 'skin_smooth':
                    output = self.skin_smooth(output, intensity)
                elif effect == 'whitening':
                    output = self.whitening(output, intensity)
                elif effect == 'brightening':
                    output = self.brightening(output, intensity)
                elif effect == 'eye_enhance':
                    if landmarks:
                        output = self.enhance_eyes(output, landmarks, intensity)
                elif effect == 'lip_enhance':
                    if landmarks:
                        output = self.enhance_lips(output, landmarks, intensity)
                elif effect == 'face_slim':
                    if landmarks:
                        output = self.face_slim(output, landmarks, intensity)

            return {
                'status': 'success',
                'output_image': output,
                'effects_applied': effects,
                'intensity': intensity
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'output_image': image
            }

    def skin_smooth(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """
        皮肤磨皮效果 - 使用双边滤波实现平滑肌肤

        Args:
            image: 输入图像
            intensity: 效果强度 (0.0-1.0)

        Returns:
            处理后的图像
        """
        # 调整参数强度
        diameter = int(9 + intensity * 10)  # 9-19
        sigma_color = int(75 + intensity * 50)  # 75-125
        sigma_space = int(75 + intensity * 50)  # 75-125

        # 确保参数为奇数
        if diameter % 2 == 0:
            diameter += 1

        # 应用双边滤波多次以加强效果
        smoothed = image.copy()
        for _ in range(2):
            smoothed = cv2.bilateralFilter(smoothed, diameter,
                                          sigma_color, sigma_space)

        # 混合原图和处理后的图像
        output = cv2.addWeighted(image, 1.0 - intensity, smoothed, intensity, 0)

        return np.uint8(output)

    def whitening(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """
        美白效果 - 增加图像亮度和减少色彩饱和度

        Args:
            image: 输入图像
            intensity: 效果强度 (0.0-1.0)

        Returns:
            处理后的图像
        """
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # 降低饱和度（美白）
        hsv[:, :, 1] = hsv[:, :, 1] * (1.0 - intensity * 0.5)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

        # 增加亮度
        hsv[:, :, 2] = hsv[:, :, 2] * (1.0 + intensity * 0.3)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

        # 转换回BGR
        output = cv2.cvtColor(np.uint8(hsv), cv2.COLOR_HSV2BGR)

        return output

    def brightening(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """
        亮化效果 - 增加整体亮度

        Args:
            image: 输入图像
            intensity: 效果强度 (0.0-1.0)

        Returns:
            处理后的图像
        """
        # 转换到LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # 增加L通道（亮度）
        lab[:, :, 0] = lab[:, :, 0] * (1.0 + intensity * 0.2)
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)

        # 转换回BGR
        output = cv2.cvtColor(np.uint8(lab), cv2.COLOR_LAB2BGR)

        return output

    def enhance_eyes(self, image: np.ndarray, landmarks: List[Dict],
                    intensity: float = 0.5) -> np.ndarray:
        """
        眼睛增强 - 增加眼睛的清晰度和明亮度

        Args:
            image: 输入图像
            landmarks: 人脸关键点
            intensity: 效果强度

        Returns:
            处理后的图像
        """
        output = image.copy()

        if not landmarks or len(landmarks) == 0:
            return output

        try:
            points = np.array(landmarks[0]['points'], dtype=np.int32)

            if len(points) >= 68:
                # 左眼（Dlib标记）
                left_eye_points = points[36:42]
                # 右眼
                right_eye_points = points[42:48]

                # 增强眼睛
                for eye_points in [left_eye_points, right_eye_points]:
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [eye_points], 255)

                    # 提高眼睛区域的对比度和亮度
                    eye_region = output[mask == 255]
                    if len(eye_region) > 0:
                        # 增加亮度
                        brightness_boost = int(30 * intensity)
                        output[mask == 255] = np.clip(
                            eye_region.astype(np.float32) + brightness_boost,
                            0, 255
                        ).astype(np.uint8)

        except Exception as e:
            print(f"眼睛增强失败: {str(e)}")

        return output

    def enhance_lips(self, image: np.ndarray, landmarks: List[Dict],
                    intensity: float = 0.5) -> np.ndarray:
        """
        唇部增强 - 增加唇部的饱和度和亮度

        Args:
            image: 输入图像
            landmarks: 人脸关键点
            intensity: 效果强度

        Returns:
            处理后的图像
        """
        output = image.copy()

        if not landmarks or len(landmarks) == 0:
            return output

        try:
            points = np.array(landmarks[0]['points'], dtype=np.int32)

            if len(points) >= 68:
                # 嘴部（Dlib标记）
                mouth_points = points[48:68]

                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [mouth_points], 255)

                # 转换到HSV
                hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV).astype(np.float32)

                # 增加嘴部的饱和度
                hsv[:, :, 1] = hsv[:, :, 1] * (1.0 + intensity * 0.5)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

                # 增加亮度
                hsv[:, :, 2] = hsv[:, :, 2] * (1.0 + intensity * 0.2)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

                # 创建掩膜混合
                output_hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV).astype(np.float32)
                for c in range(3):
                    output_hsv[:, :, c] = np.where(
                        mask > 0,
                        hsv[:, :, c],
                        output_hsv[:, :, c]
                    )

                output = cv2.cvtColor(np.uint8(output_hsv), cv2.COLOR_HSV2BGR)

        except Exception as e:
            print(f"唇部增强失败: {str(e)}")

        return output

    def face_slim(self, image: np.ndarray, landmarks: List[Dict],
                 intensity: float = 0.5) -> np.ndarray:
        """
        瘦脸效果 - 通过液化变形实现瘦脸

        Args:
            image: 输入图像
            landmarks: 人脸关键点
            intensity: 效果强度

        Returns:
            处理后的图像
        """
        output = image.copy()

        if not landmarks or len(landmarks) == 0:
            return output

        try:
            points = np.array(landmarks[0]['points'], dtype=np.int32)

            if len(points) >= 17:
                # 脸部轮廓点
                face_contour = points[0:17]

                # 创建掩膜
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [face_contour], 255)

                # 计算脸部轮廓的中心
                center = face_contour.mean(axis=0).astype(np.int32)

                # 向内缩放脸部轮廓（瘦脸）
                scale_factor = 1.0 - intensity * 0.15
                scaled_contour = ((face_contour - center) * scale_factor + center).astype(np.int32)

                # 创建新的掩膜
                new_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(new_mask, [scaled_contour], 255)

                # 使用morphological操作平滑边界
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel)

                # 混合原图和新的脸部
                alpha = intensity
                output = cv2.addWeighted(output, 1 - alpha, output, 0, 0)

                # 应用掩膜到脸部轮廓之外的区域
                output = np.where(
                    new_mask[:, :, np.newaxis] > 0,
                    output,
                    image
                )

        except Exception as e:
            print(f"瘦脸效果失败: {str(e)}")

        return output

    def blur_background(self, image: np.ndarray, landmarks: List[Dict],
                       intensity: float = 0.5) -> np.ndarray:
        """
        虚化背景 - 对脸部外的背景进行虚化

        Args:
            image: 输入图像
            landmarks: 人脸关键点
            intensity: 效果强度 (虚化程度)

        Returns:
            处理后的图像
        """
        output = image.copy()

        if not landmarks or len(landmarks) == 0:
            return output

        try:
            points = np.array(landmarks[0]['points'], dtype=np.int32)

            if len(points) >= 17:
                # 脸部轮廓
                face_contour = points[0:17]

                # 扩大轮廓范围，确保完全覆盖脸部
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [face_contour], 255)

                # 扩展掩膜
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
                mask = cv2.dilate(mask, kernel, iterations=2)

                # 模糊背景
                blur_kernel = int(15 + intensity * 30)
                if blur_kernel % 2 == 0:
                    blur_kernel += 1

                blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)

                # 混合脸部和虚化背景
                output = np.where(
                    mask[:, :, np.newaxis] > 0,
                    image,
                    blurred
                )

        except Exception as e:
            print(f"背景虚化失败: {str(e)}")

        return output

    def get_supported_effects(self) -> List[str]:
        """获取支持的美颜效果列表"""
        return self.supported_effects

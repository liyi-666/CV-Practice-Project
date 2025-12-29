"""
人脸检测模块 - 修复版本
使用Dlib和OpenCV级联分类器，确保可以正常运行
"""
import cv2
import numpy as np
import torch
from typing import List, Dict


class FaceDetector:
    """人脸检测类 - 修复版本"""

    def __init__(self, model_type='dlib', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化人脸检测器

        Args:
            model_type: 模型类型 ('dlib', 'cascade')
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = device
        self.model_type = model_type
        self.detector = None
        self.confidence_threshold = 0.5

        self._load_model()

    def _load_model(self):
        """加载预训练模型"""
        try:
            if self.model_type == 'dlib':
                self._load_dlib()
            elif self.model_type == 'cascade':
                self._load_cascade()
            else:
                print(f"! 不支持的模型类型: {self.model_type}")

        except Exception as e:
            print(f"! 模型加载失败: {str(e)}")

    def _load_dlib(self):
        """加载Dlib检测器"""
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            self.model_type = 'dlib'
            print("✓ Dlib人脸检测器加载成功")
        except ImportError:
            print("! Dlib未安装，使用级联分类器")
            self._load_cascade()

    def _load_cascade(self):
        """使用OpenCV级联分类器"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)

            if self.detector.empty():
                print("! 级联分类器加载失败")
                return

            self.model_type = 'cascade'
            print("✓ OpenCV级联分类器加载成功")

        except Exception as e:
            print(f"! 加载级联分类器失败: {str(e)}")

    def detect(self, image: np.ndarray, return_landmarks=False) -> Dict:
        """
        检测图像中的人脸

        Args:
            image: 输入图像 (BGR格式)
            return_landmarks: 是否返回人脸关键点

        Returns:
            包含检测结果的字典
        """
        if image is None:
            return {'status': 'error', 'message': '无效的输入图像'}

        try:
            if self.model_type == 'dlib':
                return self._detect_dlib(image, return_landmarks)
            elif self.model_type == 'cascade':
                return self._detect_cascade(image)
            else:
                return {'status': 'error', 'message': '模型未加载'}

        except Exception as e:
            return {'status': 'error', 'message': str(e), 'detections': []}

    def _detect_dlib(self, image: np.ndarray, return_landmarks=False) -> Dict:
        """使用Dlib检测"""
        import dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dets = self.detector(gray, 1)

        faces = []
        for det in dets:
            bbox = [det.left(), det.top(), det.right(), det.bottom()]
            confidence = 0.95  # Dlib不返回置信度

            face_dict = {
                'bbox': bbox,
                'confidence': confidence,
                'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            }
            faces.append(face_dict)

        # 按面积大小排序
        faces = sorted(faces, key=lambda x: x['area'], reverse=True)

        return {
            'status': 'success',
            'detections': faces,
            'face_count': len(faces),
            'image_shape': image.shape[:2]
        }

    def _detect_cascade(self, image: np.ndarray) -> Dict:
        """使用OpenCV级联分类器检测"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            maxSize=(500, 500)
        )

        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                'bbox': [x, y, x + w, y + h],
                'confidence': 0.85,  # 级联分类器的估计置信度
                'area': w * h
            })

        # 按面积大小排序
        detections = sorted(detections, key=lambda x: x['area'], reverse=True)

        return {
            'status': 'success',
            'detections': detections,
            'face_count': len(detections),
            'image_shape': image.shape[:2]
        }

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测结果

        Args:
            image: 输入图像
            detections: 检测结果列表

        Returns:
            绘制后的图像
        """
        output = image.copy()

        for det in detections:
            bbox = det['bbox']
            confidence = det.get('confidence', 0.0)

            # 绘制边界框
            cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (0, 255, 0), 2)

            # 绘制置信度
            text = f'Conf: {confidence:.2f}'
            cv2.putText(output, text, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return output

    def set_confidence_threshold(self, threshold: float):
        """设置置信度阈值"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))

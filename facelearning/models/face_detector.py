"""
人脸检测模块 - 基于RetinaFace预训练模型
"""
import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict


class FaceDetector:
    """人脸检测类"""

    def __init__(self, model_name='resnet50', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化人脸检测器

        Args:
            model_name: 模型类型 ('resnet50', 'mobilenet', 等)
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = device
        self.model_name = model_name
        self.model = None
        self.confidence_threshold = 0.7

        self._load_model()

    def _load_model(self):
        """加载预训练模型"""
        try:
            from retinaface import RetinaFace
            self.model = RetinaFace(gpu=(-1 if self.device == 'cpu' else 0))
            print(f"✓ RetinaFace模型加载成功")
        except ImportError:
            print("! 未安装retinaface库，使用Dlib作为备选方案")
            try:
                import dlib
                self.detector = dlib.get_frontal_face_detector()
                self.model = 'dlib'
                print(f"✓ Dlib人脸检测器加载成功")
            except ImportError:
                print("! Dlib也未安装")

    def detect(self, image: np.ndarray, return_landmarks=True) -> Dict:
        """
        检测图像中的人脸

        Args:
            image: 输入图像 (BGR格式)
            return_landmarks: 是否返回人脸关键点

        Returns:
            包含检测结果的字典
        """
        if self.model is None:
            return {'status': 'error', 'message': '模型加载失败'}

        faces = []
        h, w = image.shape[:2]

        try:
            if self.model_name == 'dlib' or isinstance(self.model, str):
                # 使用Dlib检测
                dets = self.detector(image, 1)

                for det in dets:
                    bbox = [det.left(), det.top(), det.right(), det.bottom()]
                    confidence = 0.95  # Dlib不返回置信度

                    face_dict = {
                        'bbox': bbox,
                        'confidence': confidence,
                        'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    }

                    faces.append(face_dict)
            else:
                # 使用RetinaFace检测
                faces_retinaface = self.model.detect_faces(image)

                for face in faces_retinaface:
                    bbox = face['facial_area']
                    confidence = face['confidence']

                    if confidence >= self.confidence_threshold:
                        face_dict = {
                            'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]],
                            'confidence': float(confidence),
                            'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        }

                        if return_landmarks and 'landmarks' in face:
                            landmarks = face['landmarks']
                            face_dict['landmarks'] = {
                                'left_eye': landmarks[0],
                                'right_eye': landmarks[1],
                                'nose': landmarks[2],
                                'mouth_left': landmarks[3],
                                'mouth_right': landmarks[4]
                            }

                        faces.append(face_dict)

        except Exception as e:
            return {'status': 'error', 'message': str(e), 'detections': []}

        # 按置信度排序
        faces = sorted(faces, key=lambda x: x['confidence'], reverse=True)

        return {
            'status': 'success',
            'detections': faces,
            'image_shape': (h, w),
            'face_count': len(faces)
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
            confidence = det['confidence']

            # 绘制边界框
            cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (0, 255, 0), 2)

            # 绘制置信度
            text = f'Conf: {confidence:.2f}'
            cv2.putText(output, text, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 绘制关键点
            if 'landmarks' in det:
                landmarks = det['landmarks']
                for point_name, (x, y) in landmarks.items():
                    cv2.circle(output, (int(x), int(y)), 2, (0, 0, 255), -1)

        return output

    def set_confidence_threshold(self, threshold: float):
        """设置置信度阈值"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))

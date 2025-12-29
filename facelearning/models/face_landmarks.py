"""
人脸关键点定位模块 - 用于检测面部特征点
"""
import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple


class FaceLandmarks:
    """人脸关键点检测类"""

    def __init__(self, model_type='dlib', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化人脸关键点检测器

        Args:
            model_type: 模型类型 ('dlib', 'mediapipe', 'hrnet')
            device: 计算设备
        """
        self.device = device
        self.model_type = model_type
        self.model = None
        self.predictor = None

        self._load_model()

    def _load_model(self):
        """加载预训练模型"""
        try:
            if self.model_type == 'dlib':
                import dlib
                # 下载dlib的预训练人脸关键点模型
                try:
                    self.predictor = dlib.shape_predictor(
                        'models/weights/shape_predictor_68_face_landmarks.dat'
                    )
                    print("✓ Dlib 68点关键点检测器加载成功")
                except:
                    print("! Dlib预训练模型文件未找到，将使用OpenCV的级联分类器作为备选")
                    self.model_type = 'opencv'

            elif self.model_type == 'mediapipe':
                import mediapipe as mp
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=10,
                    min_detection_confidence=0.5
                )
                print("✓ MediaPipe人脸关键点检测器加载成功")

            elif self.model_type == 'opencv':
                # OpenCV的LBP级联分类器
                self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
                print("✓ OpenCV级联分类器加载成功")

        except Exception as e:
            print(f"! 模型加载失败: {str(e)}")

    def detect(self, image: np.ndarray, faces: List[Dict] = None) -> Dict:
        """
        检测人脸关键点

        Args:
            image: 输入图像 (BGR格式)
            faces: 人脸检测结果 (如果为None则自动检测)

        Returns:
            包含关键点的字典
        """
        results = {
            'status': 'success',
            'landmarks': [],
            'method': self.model_type
        }

        try:
            if self.model_type == 'dlib' and self.predictor:
                results = self._detect_dlib(image, faces)

            elif self.model_type == 'mediapipe':
                results = self._detect_mediapipe(image)

            elif self.model_type == 'opencv':
                results = self._detect_opencv(image, faces)

        except Exception as e:
            results['status'] = 'error'
            results['message'] = str(e)

        return results

    def _detect_dlib(self, image: np.ndarray, faces: List[Dict] = None) -> Dict:
        """使用Dlib检测关键点"""
        import dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        landmarks_list = []

        if faces is None:
            # 使用Dlib的人脸检测
            detector = dlib.get_frontal_face_detector()
            dets = detector(gray, 1)
        else:
            # 使用外部检测结果
            dets = [dlib.rectangle(
                int(f['bbox'][0]), int(f['bbox'][1]),
                int(f['bbox'][2]), int(f['bbox'][3])
            ) for f in faces]

        for rect in dets:
            shape = self.predictor(gray, rect)
            landmarks = np.zeros((68, 2), dtype=np.int32)

            for i in range(68):
                landmarks[i] = [shape.part(i).x, shape.part(i).y]

            landmarks_list.append({
                'points': landmarks.tolist(),
                'bbox': [rect.left(), rect.top(), rect.right(), rect.bottom()],
                'num_landmarks': 68
            })

        return {
            'status': 'success',
            'landmarks': landmarks_list,
            'method': 'dlib_68'
        }

    def _detect_mediapipe(self, image: np.ndarray) -> Dict:
        """使用MediaPipe检测关键点"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        landmarks_list = []

        if results.multi_face_landmarks:
            h, w = image.shape[:2]

            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.zeros((468, 2), dtype=np.float32)

                for i, lm in enumerate(face_landmarks.landmark):
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    landmarks[i] = [x, y]

                landmarks_list.append({
                    'points': landmarks.tolist(),
                    'num_landmarks': 468
                })

        return {
            'status': 'success',
            'landmarks': landmarks_list,
            'method': 'mediapipe_468'
        }

    def _detect_opencv(self, image: np.ndarray, faces: List[Dict] = None) -> Dict:
        """使用OpenCV检测面部区域（简化版）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        landmarks_list = []

        for (x, y, w, h) in detected_faces:
            face_roi = gray[y:y + h, x:x + w]
            landmarks = self._estimate_landmarks_simple(face_roi, (x, y, w, h))
            landmarks_list.append({
                'points': landmarks.tolist(),
                'bbox': [x, y, x + w, y + h],
                'num_landmarks': 5
            })

        return {
            'status': 'success',
            'landmarks': landmarks_list,
            'method': 'opencv_simple'
        }

    def _estimate_landmarks_simple(self, roi: np.ndarray, bbox: Tuple) -> np.ndarray:
        """简化的关键点估计（眼睛、鼻子、嘴角）"""
        x, y, w, h = bbox
        landmarks = np.array([
            [x + w * 0.35, y + h * 0.35],  # 左眼
            [x + w * 0.65, y + h * 0.35],  # 右眼
            [x + w * 0.5, y + h * 0.5],    # 鼻子
            [x + w * 0.35, y + h * 0.75],  # 左嘴角
            [x + w * 0.65, y + h * 0.75]   # 右嘴角
        ], dtype=np.int32)

        return landmarks

    def draw_landmarks(self, image: np.ndarray, landmarks: List[Dict]) -> np.ndarray:
        """
        在图像上绘制关键点

        Args:
            image: 输入图像
            landmarks: 关键点检测结果

        Returns:
            绘制后的图像
        """
        output = image.copy()

        for landmark_data in landmarks:
            points = np.array(landmark_data['points'], dtype=np.int32)

            # 绘制关键点
            for point in points:
                cv2.circle(output, tuple(point), 2, (0, 255, 0), -1)

            # 连接关键点形成面部轮廓（仅Dlib）
            if landmark_data.get('num_landmarks') == 68:
                # 左眼
                left_eye = points[42:48]
                cv2.polylines(output, [left_eye], True, (255, 0, 0), 2)

                # 右眼
                right_eye = points[36:42]
                cv2.polylines(output, [right_eye], True, (255, 0, 0), 2)

                # 嘴
                mouth = points[60:68]
                cv2.polylines(output, [mouth], True, (255, 0, 0), 2)

                # 脸部轮廓
                face_contour = points[0:17]
                cv2.polylines(output, [face_contour], False, (0, 255, 255), 2)

        return output

    def align_face(self, image: np.ndarray, landmarks: Dict,
                   output_size: int = 112) -> np.ndarray:
        """
        基于关键点进行人脸对齐

        Args:
            image: 输入图像
            landmarks: 关键点信息
            output_size: 输出图像大小

        Returns:
            对齐后的人脸图像
        """
        try:
            points = np.array(landmarks['points'], dtype=np.float32)

            # 使用眼睛关键点进行对齐
            if len(points) >= 68:
                left_eye = points[42:48].mean(axis=0)
                right_eye = points[36:42].mean(axis=0)
            elif len(points) >= 5:
                left_eye = points[0]
                right_eye = points[1]
            else:
                return image

            # 计算旋转角度
            eye_center = (left_eye + right_eye) / 2
            angle = np.arctan2(right_eye[1] - left_eye[1],
                              right_eye[0] - left_eye[0])

            # 获取旋转矩阵
            M = cv2.getRotationMatrix2D(tuple(eye_center), np.degrees(angle), 1.0)

            # 应用仿射变换
            aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                   flags=cv2.INTER_CUBIC)

            # 裁剪到标准大小
            x = int(eye_center[0] - output_size / 2)
            y = int(eye_center[1] - output_size / 2)
            aligned = aligned[y:y + output_size, x:x + output_size]

            if aligned.shape[0] == output_size and aligned.shape[1] == output_size:
                return aligned
            else:
                # 如果大小不匹配，调整大小
                return cv2.resize(aligned, (output_size, output_size))

        except Exception as e:
            print(f"人脸对齐失败: {str(e)}")
            return image

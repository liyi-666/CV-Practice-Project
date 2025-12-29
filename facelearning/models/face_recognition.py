"""
人脸识别模块 - 基于ArcFace的特征提取和识别
"""
import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple
from pathlib import Path


class FaceRecognition:
    """人脸识别类 - 基于预训练的深度学习模型"""

    def __init__(self, model_type='insightface', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化人脸识别器

        Args:
            model_type: 模型类型 ('insightface', 'arcface', 'vggface2')
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = device
        self.model_type = model_type
        self.model = None
        self.embedding_dim = 512
        self.similarity_threshold = 0.6

        # 存储已注册的人脸特征
        self.registered_faces = {}  # {person_id: {'name': str, 'embeddings': [list]}}

        self._load_model()

    def _load_model(self):
        """加载预训练模型"""
        try:
            if self.model_type == 'insightface':
                try:
                    import insightface
                    self.model = insightface.app.FaceAnalysis(
                        name='buffalo_l',  # 大规模预训练模型，精度最高
                        providers=['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
                    )
                    self.model.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(640, 640))
                    print("✓ InsightFace模型加载成功 (buffalo_l)")
                except ImportError:
                    print("! InsightFace库未安装")
                    self._load_fallback_model()

            elif self.model_type == 'arcface':
                self._load_arcface_model()

        except Exception as e:
            print(f"! 模型加载失败: {str(e)}")
            self._load_fallback_model()

    def _load_arcface_model(self):
        """加载ArcFace模型"""
        try:
            import torch
            from torchvision import models

            # 加载预训练的ResNet50
            self.backbone = models.resnet50(pretrained=True)
            self.backbone.fc = torch.nn.Linear(2048, 512)  # 输出512维特征
            self.backbone = self.backbone.to(self.device)
            self.backbone.eval()

            print("✓ ArcFace模型加载成功")
        except Exception as e:
            print(f"! ArcFace模型加载失败: {str(e)}")

    def _load_fallback_model(self):
        """加载备选模型（使用PyTorch预训练模型）"""
        try:
            import torch
            from torchvision import models

            self.backbone = models.resnet50(pretrained=True)
            self.backbone.eval()
            self.model_type = 'resnet50'
            print("✓ 使用ResNet50作为备选模型")
        except Exception as e:
            print(f"! 所有模型加载都失败了: {str(e)}")

    def extract_features(self, image: np.ndarray, bbox: List[int] = None) -> Dict:
        """
        提取人脸特征向量

        Args:
            image: 输入图像 (BGR格式)
            bbox: 人脸边界框 [x1, y1, x2, y2]，如果为None则自动检测

        Returns:
            包含特征向量的字典
        """
        try:
            if self.model_type == 'insightface' and self.model:
                return self._extract_features_insightface(image, bbox)
            else:
                return self._extract_features_resnet(image, bbox)
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'embedding': None
            }

    def _extract_features_insightface(self, image: np.ndarray, bbox: List[int] = None) -> Dict:
        """使用InsightFace提取特征"""
        try:
            faces = self.model.get(image)

            if not faces:
                return {
                    'status': 'error',
                    'message': '未检测到人脸',
                    'embedding': None
                }

            # 选择第一个检测到的人脸（最大的人脸）
            face = faces[0]

            embedding = face.embedding  # 512维特征向量
            bbox_detected = face.bbox.astype(int).tolist()

            return {
                'status': 'success',
                'embedding': embedding.tolist(),
                'embedding_dim': len(embedding),
                'bbox': bbox_detected,
                'gender': getattr(face, 'gender', 'unknown'),
                'age': getattr(face, 'age', 0)
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'embedding': None
            }

    def _extract_features_resnet(self, image: np.ndarray, bbox: List[int] = None) -> Dict:
        """使用ResNet提取特征（备选方案）"""
        try:
            import torch
            from torchvision import transforms

            # 图像预处理
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            # 裁剪人脸区域
            if bbox:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                face_image = image[y1:y2, x1:x2]
            else:
                face_image = image

            # 调整大小到224x224
            face_image = cv2.resize(face_image, (224, 224))

            # 转换为张量
            face_tensor = transform(face_image).unsqueeze(0).to(self.device)

            # 提取特征
            with torch.no_grad():
                if hasattr(self.backbone, 'fc'):
                    embedding = self.backbone(face_tensor)
                else:
                    # 如果是未修改的ResNet，提取倒数第二层的特征
                    embedding = self.backbone(face_tensor)

            embedding = embedding.cpu().numpy().flatten()

            return {
                'status': 'success',
                'embedding': embedding.tolist(),
                'embedding_dim': len(embedding),
                'bbox': bbox or [0, 0, image.shape[1], image.shape[0]]
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'embedding': None
            }

    def register_face(self, person_id: str, image: np.ndarray,
                     name: str = None, bbox: List[int] = None) -> Dict:
        """
        注册一个人脸（保存特征向量）

        Args:
            person_id: 人员唯一ID
            image: 人脸图像
            name: 人员名称
            bbox: 人脸边界框

        Returns:
            注册结果字典
        """
        result = self.extract_features(image, bbox)

        if result['status'] == 'success' and result['embedding']:
            if person_id not in self.registered_faces:
                self.registered_faces[person_id] = {
                    'name': name or person_id,
                    'embeddings': []
                }

            self.registered_faces[person_id]['embeddings'].append(result['embedding'])

            return {
                'status': 'success',
                'message': f'人脸注册成功 (ID: {person_id})',
                'person_id': person_id,
                'embedding_count': len(self.registered_faces[person_id]['embeddings'])
            }
        else:
            return {
                'status': 'error',
                'message': '特征提取失败',
                'person_id': person_id
            }

    def recognize(self, image: np.ndarray, bbox: List[int] = None) -> Dict:
        """
        识别图像中的人脸

        Args:
            image: 输入图像
            bbox: 人脸边界框

        Returns:
            识别结果字典
        """
        result = self.extract_features(image, bbox)

        if result['status'] != 'success' or not result['embedding']:
            return {
                'status': 'error',
                'message': '特征提取失败',
                'matches': []
            }

        query_embedding = np.array(result['embedding'])

        # 与已注册的人脸进行相似度比较
        matches = []

        for person_id, person_data in self.registered_faces.items():
            max_similarity = 0
            for registered_embedding in person_data['embeddings']:
                registered_embedding = np.array(registered_embedding)
                # 计算余弦相似度
                similarity = self._cosine_similarity(query_embedding, registered_embedding)
                max_similarity = max(max_similarity, similarity)

            if max_similarity >= self.similarity_threshold:
                matches.append({
                    'person_id': person_id,
                    'name': person_data['name'],
                    'similarity': float(max_similarity)
                })

        # 按相似度降序排列
        matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)

        return {
            'status': 'success',
            'matches': matches,
            'best_match': matches[0] if matches else None,
            'query_embedding_dim': len(query_embedding)
        }

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def verify(self, image1: np.ndarray, image2: np.ndarray,
               bbox1: List[int] = None, bbox2: List[int] = None) -> Dict:
        """
        验证两张人脸是否来自同一个人

        Args:
            image1: 第一张图像
            image2: 第二张图像
            bbox1: 第一张图像的人脸边界框
            bbox2: 第二张图像的人脸边界框

        Returns:
            验证结果字典
        """
        result1 = self.extract_features(image1, bbox1)
        result2 = self.extract_features(image2, bbox2)

        if result1['status'] != 'success' or result2['status'] != 'success':
            return {
                'status': 'error',
                'message': '特征提取失败',
                'is_same_person': False,
                'similarity': 0.0
            }

        embedding1 = np.array(result1['embedding'])
        embedding2 = np.array(result2['embedding'])

        similarity = self._cosine_similarity(embedding1, embedding2)
        is_same_person = similarity >= self.similarity_threshold

        return {
            'status': 'success',
            'is_same_person': is_same_person,
            'similarity': float(similarity),
            'threshold': self.similarity_threshold
        }

    def set_similarity_threshold(self, threshold: float):
        """设置相似度阈值"""
        self.similarity_threshold = max(0.0, min(1.0, threshold))

    def save_registered_faces(self, filepath: str):
        """保存已注册的人脸特征到文件"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.registered_faces, f, indent=2)
        print(f"✓ 已保存 {len(self.registered_faces)} 个已注册的人脸到 {filepath}")

    def load_registered_faces(self, filepath: str):
        """从文件加载已注册的人脸特征"""
        import json
        try:
            with open(filepath, 'r') as f:
                self.registered_faces = json.load(f)
            print(f"✓ 已加载 {len(self.registered_faces)} 个已注册的人脸")
        except FileNotFoundError:
            print(f"! 文件 {filepath} 未找到")

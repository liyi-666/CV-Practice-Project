"""
人脸美颜 - 使用 InsightFace 预训练模型
支持：人脸检测、关键点、3D姿态估计、美颜处理
"""
import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis

# 模型目录
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')


class FaceBeauty:
    """人脸美颜类"""

    def __init__(self, model_dir=None):
        """初始化 InsightFace 模型

        Args:
            model_dir: 模型目录，默认为 ./models/insightface
        """
        if model_dir is None:
            model_dir = os.path.join(MODEL_DIR, 'insightface')

        print("加载 InsightFace 模型...")
        print(f"模型目录: {model_dir}")

        self.app = FaceAnalysis(
            name='buffalo_l',
            root=model_dir,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print("✓ 模型加载完成")

    def analyze(self, image):
        """
        分析人脸

        Args:
            image: BGR格式图像

        Returns:
            faces: 人脸信息列表
        """
        faces = self.app.get(image)
        return faces

    def beautify(self, image, intensity=0.5):
        """
        美颜处理

        Args:
            image: BGR格式图像
            intensity: 美颜强度 (0-1)

        Returns:
            美颜后的图像
        """
        faces = self.analyze(image)

        if len(faces) == 0:
            print("未检测到人脸")
            return image

        output = image.copy()

        for face in faces:
            # 获取人脸区域
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # 扩大区域
            h, w = image.shape[:2]
            pad = int((x2 - x1) * 0.2)
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            # 提取人脸区域
            face_region = output[y1:y2, x1:x2].copy()

            # 应用美颜效果
            face_region = self._smooth_skin(face_region, intensity)
            face_region = self._whiten_skin(face_region, intensity)
            face_region = self._enhance_details(face_region, intensity)

            # 放回原图
            output[y1:y2, x1:x2] = face_region

        return output

    def _smooth_skin(self, image, intensity=0.5):
        """磨皮 - 双边滤波"""
        d = int(9 + intensity * 10)
        if d % 2 == 0:
            d += 1
        sigma = int(75 + intensity * 50)

        smoothed = cv2.bilateralFilter(image, d, sigma, sigma)
        result = cv2.addWeighted(image, 1 - intensity, smoothed, intensity, 0)
        return result

    def _whiten_skin(self, image, intensity=0.5):
        """美白 - LAB色彩空间调整"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # 增加亮度
        lab[:, :, 0] = lab[:, :, 0] * (1 + intensity * 0.15)
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)

        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result

    def _enhance_details(self, image, intensity=0.5):
        """细节增强 - CLAHE"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # 混合
        l_result = cv2.addWeighted(l, 1 - intensity * 0.3, l_enhanced, intensity * 0.3, 0)

        lab = cv2.merge([l_result, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return result

    def slim_face(self, image, faces, intensity=0.3):
        """
        瘦脸 - 基于关键点的液化变形

        Args:
            image: BGR格式图像
            faces: analyze() 返回的人脸列表
            intensity: 瘦脸强度 (0-1)

        Returns:
            瘦脸后的图像
        """
        if len(faces) == 0:
            return image

        output = image.copy()

        for face in faces:
            landmarks = face.landmark_2d_106

            if landmarks is None:
                continue

            # 左脸颊点 (关键点索引约 1-5)
            # 右脸颊点 (关键点索引约 29-33)
            # 下巴点 (关键点索引约 8-16)

            # 简化的瘦脸：向内收缩脸颊
            left_cheek = landmarks[1:6].mean(axis=0).astype(int)
            right_cheek = landmarks[29:34].mean(axis=0).astype(int)
            chin = landmarks[16].astype(int)

            # 计算脸部中心
            center = ((left_cheek + right_cheek) / 2).astype(int)

            # 向中心收缩
            output = self._local_warp(output, left_cheek, center, int(30 * intensity))
            output = self._local_warp(output, right_cheek, center, int(30 * intensity))

        return output

    def _local_warp(self, image, src_point, dst_point, radius):
        """局部液化变形"""
        h, w = image.shape[:2]
        output = image.copy()

        # 创建位移场
        for y in range(max(0, src_point[1] - radius), min(h, src_point[1] + radius)):
            for x in range(max(0, src_point[0] - radius), min(w, src_point[0] + radius)):
                dist = np.sqrt((x - src_point[0])**2 + (y - src_point[1])**2)
                if dist < radius:
                    # 计算位移
                    factor = (1 - dist / radius) ** 2
                    dx = int((dst_point[0] - src_point[0]) * factor * 0.3)
                    dy = int((dst_point[1] - src_point[1]) * factor * 0.3)

                    # 采样
                    nx = min(max(0, x + dx), w - 1)
                    ny = min(max(0, y + dy), h - 1)
                    output[y, x] = image[ny, nx]

        return output

    def enlarge_eyes(self, image, faces, intensity=0.3):
        """
        大眼 - 基于关键点的局部放大

        Args:
            image: BGR格式图像
            faces: analyze() 返回的人脸列表
            intensity: 大眼强度 (0-1)

        Returns:
            大眼后的图像
        """
        if len(faces) == 0:
            return image

        output = image.copy()

        for face in faces:
            landmarks = face.landmark_2d_106

            if landmarks is None:
                continue

            # 左眼中心 (关键点 35-46)
            left_eye = landmarks[35:47].mean(axis=0).astype(int)
            # 右眼中心 (关键点 89-100)
            right_eye = landmarks[89:101].mean(axis=0).astype(int)

            # 眼睛半径
            eye_radius = int(np.linalg.norm(landmarks[35] - landmarks[41]) * (1 + intensity * 0.5))

            output = self._local_scale(output, left_eye, eye_radius, 1 + intensity * 0.15)
            output = self._local_scale(output, right_eye, eye_radius, 1 + intensity * 0.15)

        return output

    def _local_scale(self, image, center, radius, scale):
        """局部缩放（放大眼睛）"""
        h, w = image.shape[:2]
        output = image.copy()

        for y in range(max(0, center[1] - radius), min(h, center[1] + radius)):
            for x in range(max(0, center[0] - radius), min(w, center[0] + radius)):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if dist < radius:
                    # 从中心向外采样
                    factor = dist / radius
                    new_dist = dist / scale

                    if new_dist > 0:
                        nx = int(center[0] + (x - center[0]) * new_dist / dist)
                        ny = int(center[1] + (y - center[1]) * new_dist / dist)

                        nx = min(max(0, nx), w - 1)
                        ny = min(max(0, ny), h - 1)
                        output[y, x] = image[ny, nx]

        return output

    def draw_analysis(self, image, faces):
        """
        绘制人脸分析结果

        Args:
            image: BGR格式图像
            faces: analyze() 返回的人脸列表

        Returns:
            绘制后的图像
        """
        output = image.copy()

        for face in faces:
            # 绘制边界框
            bbox = face.bbox.astype(int)
            cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # 绘制关键点
            if face.landmark_2d_106 is not None:
                for point in face.landmark_2d_106:
                    cv2.circle(output, tuple(point.astype(int)), 2, (0, 255, 0), -1)

            # 显示信息
            info = []
            if hasattr(face, 'age'):
                info.append(f"Age: {int(face.age)}")
            if hasattr(face, 'gender'):
                gender = "M" if face.gender == 1 else "F"
                info.append(f"Gender: {gender}")

            for i, text in enumerate(info):
                cv2.putText(output, text, (bbox[0], bbox[1] - 10 - i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return output


# 使用示例
if __name__ == '__main__':
    import sys

    # 初始化
    beauty = FaceBeauty()

    # 读取图像
    image_path = sys.argv[1] if len(sys.argv) > 1 else 'huge.jpg'
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        exit(1)

    print(f"✓ 读取图像: {image_path}")
    print(f"✓ 图像大小: {image.shape}")

    # 分析人脸
    print("\n分析人脸...")
    faces = beauty.analyze(image)
    print(f"✓ 检测到 {len(faces)} 张人脸")

    if len(faces) > 0:
        face = faces[0]
        print(f"  - 关键点数: {len(face.landmark_2d_106) if face.landmark_2d_106 is not None else 0}")
        if hasattr(face, 'age'):
            print(f"  - 年龄: {int(face.age)}")
        if hasattr(face, 'gender'):
            print(f"  - 性别: {'男' if face.gender == 1 else '女'}")

    # 美颜处理
    print("\n美颜处理...")
    beautified = beauty.beautify(image, intensity=0.6)

    # 瘦脸
    print("瘦脸处理...")
    beautified = beauty.slim_face(beautified, faces, intensity=0.3)

    # 大眼
    print("大眼处理...")
    beautified = beauty.enlarge_eyes(beautified, faces, intensity=0.2)

    # 保存结果
    print("\n保存结果...")

    # 分析结果
    analysis = beauty.draw_analysis(image, faces)
    cv2.imwrite('beauty_analysis.jpg', analysis)
    print("✓ 分析结果: beauty_analysis.jpg")

    # 美颜结果
    cv2.imwrite('beauty_result.jpg', beautified)
    print("✓ 美颜结果: beauty_result.jpg")

    # 对比图
    comparison = np.hstack([image, beautified])
    cv2.imwrite('beauty_comparison.jpg', comparison)
    print("✓ 对比图: beauty_comparison.jpg")

    print("\n✅ 完成!")

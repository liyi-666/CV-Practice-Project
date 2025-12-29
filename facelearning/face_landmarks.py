"""
人脸关键点检测 - 使用 dlib 预训练模型 (68点)
"""
import cv2
import dlib
import numpy as np
import os

# 模型目录
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')


class FaceLandmarksDetector:
    """人脸关键点检测器"""

    def __init__(self, model_path=None):
        """
        初始化检测器

        Args:
            model_path: 预训练模型路径，默认 models/dlib/shape_predictor_68_face_landmarks.dat
        """
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, 'dlib', 'shape_predictor_68_face_landmarks.dat')
            # 如果本地不存在，回退到旧路径
            if not os.path.exists(model_path):
                model_path = os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat')

        print(f"加载 dlib 模型: {model_path}")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)

    def detect(self, image):
        """
        检测人脸关键点

        Args:
            image: BGR格式的图像

        Returns:
            faces: 人脸列表，每个元素包含 bbox 和 landmarks
            success: 是否检测到人脸
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dets = self.detector(gray, 1)

        if len(dets) == 0:
            return [], False

        faces = []
        for det in dets:
            # 获取关键点
            shape = self.predictor(gray, det)

            landmarks = []
            for i in range(68):
                x = shape.part(i).x
                y = shape.part(i).y
                landmarks.append([x, y])

            faces.append({
                'bbox': [det.left(), det.top(), det.right(), det.bottom()],
                'landmarks': np.array(landmarks)
            })

        return faces, True

    def draw(self, image, faces, show_index=False):
        """
        在图像上绘制关键点

        Args:
            image: 输入图像
            faces: detect() 返回的人脸列表
            show_index: 是否显示关键点序号

        Returns:
            绘制后的图像
        """
        output = image.copy()

        for face in faces:
            landmarks = face['landmarks']
            bbox = face['bbox']

            # 绘制边界框
            cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # 绘制关键点
            for i, (x, y) in enumerate(landmarks):
                cv2.circle(output, (x, y), 2, (0, 255, 0), -1)
                if show_index:
                    cv2.putText(output, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            # 绘制面部轮廓连线
            # 下巴轮廓 (0-16)
            for i in range(16):
                cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]), (255, 200, 0), 1)

            # 左眉毛 (17-21)
            for i in range(17, 21):
                cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]), (255, 200, 0), 1)

            # 右眉毛 (22-26)
            for i in range(22, 26):
                cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]), (255, 200, 0), 1)

            # 鼻梁 (27-30)
            for i in range(27, 30):
                cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]), (255, 200, 0), 1)

            # 鼻底 (31-35)
            for i in range(31, 35):
                cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]), (255, 200, 0), 1)

            # 左眼 (36-41)
            for i in range(36, 41):
                cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]), (0, 255, 255), 1)
            cv2.line(output, tuple(landmarks[41]), tuple(landmarks[36]), (0, 255, 255), 1)

            # 右眼 (42-47)
            for i in range(42, 47):
                cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]), (0, 255, 255), 1)
            cv2.line(output, tuple(landmarks[47]), tuple(landmarks[42]), (0, 255, 255), 1)

            # 外嘴唇 (48-59)
            for i in range(48, 59):
                cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]), (0, 0, 255), 1)
            cv2.line(output, tuple(landmarks[59]), tuple(landmarks[48]), (0, 0, 255), 1)

            # 内嘴唇 (60-67)
            for i in range(60, 67):
                cv2.line(output, tuple(landmarks[i]), tuple(landmarks[i+1]), (0, 0, 255), 1)
            cv2.line(output, tuple(landmarks[67]), tuple(landmarks[60]), (0, 0, 255), 1)

        return output


# 使用示例
if __name__ == '__main__':
    import sys

    # 初始化
    print("加载模型...")
    detector = FaceLandmarksDetector()
    print("✓ 模型加载完成")

    # 读取图像
    image_path = sys.argv[1] if len(sys.argv) > 1 else 'huge.jpg'
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        exit(1)

    print(f"✓ 读取图像: {image_path}")
    print(f"✓ 图像大小: {image.shape}")

    # 检测关键点
    print("\n检测关键点...")
    faces, success = detector.detect(image)

    if success:
        print(f"✓ 检测到 {len(faces)} 张人脸")
        for i, face in enumerate(faces):
            print(f"  人脸 {i+1}: {len(face['landmarks'])} 个关键点")

        # 绘制
        output = detector.draw(image, faces, show_index=True)

        # 保存
        output_path = 'landmarks_result.jpg'
        cv2.imwrite(output_path, output)
        print(f"\n✓ 结果已保存: {output_path}")
    else:
        print("❌ 未检测到人脸")

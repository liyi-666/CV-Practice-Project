"""
人脸关键点检测 - 使用MediaPipe预训练模型
"""
import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


class FaceLandmarksDetector:
    def __init__(self):
        self.face_mesh = solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            min_detection_confidence=0.5
        )

    def detect(self, image):
        """
        检测人脸关键点

        Args:
            image: BGR格式的图像

        Returns:
            landmarks: 关键点坐标 (x, y)
            success: 是否检测成功
        """
        h, w = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None, False

        landmarks_list = []

        for face_landmarks in results.multi_face_landmarks:
            landmarks = []
            for lm in face_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                landmarks.append([x, y])
            landmarks_list.append(np.array(landmarks))

        return landmarks_list, True

    def draw(self, image, landmarks_list):
        """
        在图像上绘制关键点

        Args:
            image: 输入图像
            landmarks_list: 关键点列表

        Returns:
            绘制后的图像
        """
        output = image.copy()

        for landmarks in landmarks_list:
            # 绘制所有关键点
            for i, point in enumerate(landmarks):
                cv2.circle(output, tuple(point), 2, (0, 255, 0), -1)

        return output


# 使用示例
if __name__ == '__main__':
    # 初始化
    detector = FaceLandmarksDetector()

    # 读取图像
    image_path = 'test.jpg'
    image = cv2.imread(image_path)

    if image is None:
        print(f"无法读取图像: {image_path}")
        exit(1)

    # 检测关键点
    landmarks_list, success = detector.detect(image)

    if success:
        print(f"✓ 检测到 {len(landmarks_list)} 个人脸")
        print(f"✓ 每个人脸 {len(landmarks_list[0])} 个关键点")

        # 绘制
        output = detector.draw(image, landmarks_list)

        # 保存
        cv2.imwrite('landmarks_result.jpg', output)
        print("✓ 结果已保存: landmarks_result.jpg")

        # 显示
        cv2.imshow('Landmarks', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("✗ 未检测到人脸")

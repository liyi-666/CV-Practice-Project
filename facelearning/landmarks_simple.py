"""
人脸关键点检测 - 使用 dlib 预训练模型
"""
import cv2
import numpy as np
import dlib


class FaceLandmarksDetector:
    def __init__(self):
        # 加载人脸检测器
        self.detector = dlib.get_frontal_face_detector()

        # 加载关键点预测器（68个点）
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def detect(self, image):
        """
        检测人脸关键点

        Args:
            image: BGR格式的图像

        Returns:
            landmarks_list: 关键点列表
            success: 是否检测成功
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        dets = self.detector(gray, 1)

        if len(dets) == 0:
            return None, False

        landmarks_list = []

        for det in dets:
            # 获取关键点
            shape = self.predictor(gray, det)

            landmarks = []
            for i in range(68):
                x = shape.part(i).x
                y = shape.part(i).y
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
            for point in landmarks:
                cv2.circle(output, tuple(point), 2, (0, 255, 0), -1)

        return output


# 使用示例
if __name__ == '__main__':
    # 初始化
    detector = FaceLandmarksDetector()

    # 读取图像
    image_path = 'test_face.jpg'
    image = cv2.imread(image_path)

    if image is None:
        print(f"无法读取图像: {image_path}")
        exit(1)

    print(f"图像大小: {image.shape}")

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
    else:
        print("✗ 未检测到人脸")

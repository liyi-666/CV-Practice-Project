"""
人脸关键点检测 - 使用 MediaPipe (修复版本)
"""
import cv2
import numpy as np
import mediapipe as mp

# 初始化 MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=10,
    min_detection_confidence=0.5
)

# 读取图像
image_path = 'huge.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"❌ 无法读取图像: {image_path}")
    exit(1)

print(f"✓ 读取图像: {image_path}")
print(f"✓ 图像大小: {image.shape}")

h, w, c = image.shape

# 转换为 RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 检测关键点
print("\n检测关键点...")
results = face_mesh.process(rgb_image)

if not results.multi_face_landmarks:
    print("❌ 未检测到人脸")
    exit(1)

print(f"✓ 检测到 {len(results.multi_face_landmarks)} 张人脸")

# 绘制关键点
output = image.copy()

for face_landmarks in results.multi_face_landmarks:
    landmarks = []
    print(f"✓ 人脸关键点数: {len(face_landmarks.landmark)}")

    for i, lm in enumerate(face_landmarks.landmark):
        x = int(lm.x * w)
        y = int(lm.y * h)
        landmarks.append([x, y])

        # 绘制点
        cv2.circle(output, (x, y), 2, (0, 255, 0), -1)

# 保存结果
print("\n保存结果...")
cv2.imwrite('landmarks_output.jpg', output)
print(f"✓ 结果已保存: landmarks_output.jpg")

print("\n✅ 完成!")

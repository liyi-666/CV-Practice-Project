"""
测试人脸关键点检测
"""
import cv2
import dlib
import numpy as np

# 加载检测器和预测器
print("加载模型...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
print("✓ 模型加载完成")

# 读取图像
image_path = 'huge.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"❌ 无法读取图像: {image_path}")
    exit(1)

print(f"✓ 读取图像: {image_path}")
print(f"✓ 图像大小: {image.shape}")

# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 检测人脸
print("\n检测人脸...")
dets = detector(gray, 1)
print(f"✓ 检测到 {len(dets)} 张人脸")

if len(dets) == 0:
    print("❌ 未检测到人脸")
    exit(1)

# 获取关键点
print("\n提取关键点...")
output = image.copy()

for det in dets:
    shape = predictor(gray, det)
    print(f"✓ 人脸关键点数: {shape.num_parts}")

    # 绘制关键点
    for i in range(shape.num_parts):
        x = shape.part(i).x
        y = shape.part(i).y
        cv2.circle(output, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(output, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

# 保存结果
print("\n保存结果...")
cv2.imwrite('landmarks_output.jpg', output)
print(f"✓ 结果已保存: landmarks_output.jpg")
print("\n✅ 完成!")

"""
瘦脸算法 - 基于关键点的液化变形
"""
import cv2
import numpy as np


def interactive_warp(image, src_point, dst_point, radius):
    """
    交互式液化变形（类似PS液化工具）

    Args:
        image: 输入图像
        src_point: 源点 (x, y)
        dst_point: 目标点 (x, y)
        radius: 影响半径

    Returns:
        变形后的图像
    """
    h, w = image.shape[:2]

    # 创建网格
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    for i in range(h):
        for j in range(w):
            map_x[i, j] = j
            map_y[i, j] = i

    # 计算位移
    src = np.array(src_point, dtype=np.float32)
    dst = np.array(dst_point, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            dist = np.sqrt((j - src[0])**2 + (i - src[1])**2)

            if dist < radius:
                # 使用平滑的衰减函数
                factor = ((radius - dist) / radius) ** 2

                # 位移方向
                offset = (dst - src) * factor

                map_x[i, j] = j - offset[0]
                map_y[i, j] = i - offset[1]

    # 应用变形
    result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return result


def slim_face_advanced(image, landmarks, intensity=0.5):
    """
    高级瘦脸算法

    Args:
        image: BGR图像
        landmarks: 106个关键点
        intensity: 瘦脸强度 (0-1)

    Returns:
        瘦脸后的图像
    """
    result = image.copy()

    # InsightFace 106点关键点索引
    # 左脸颊: 0-16
    # 右脸颊: 17-32
    # 左眉: 33-42
    # 右眉: 43-52 (大约)

    # 计算脸部宽度和中心
    left_face = landmarks[0:10]
    right_face = landmarks[23:33]

    # 脸部中心
    center_x = (left_face[:, 0].mean() + right_face[:, 0].mean()) / 2
    center_y = (left_face[:, 1].mean() + right_face[:, 1].mean()) / 2
    center = np.array([center_x, center_y])

    # 计算变形半径（基于脸部大小）
    face_width = right_face[:, 0].mean() - left_face[:, 0].mean()
    radius = int(face_width * 0.4)

    # 位移距离
    shift = int(face_width * 0.08 * intensity)

    # 左脸颊向内推
    for i in [2, 3, 4, 5, 6]:
        src = landmarks[i].astype(int)
        dst = src.copy()
        dst[0] = src[0] + shift  # 向右（内）推
        result = interactive_warp(result, src, dst, radius)

    # 右脸颊向内推
    for i in [27, 28, 29, 30, 31]:
        src = landmarks[i].astype(int)
        dst = src.copy()
        dst[0] = src[0] - shift  # 向左（内）推
        result = interactive_warp(result, src, dst, radius)

    # 下巴向上提
    chin_points = [8, 9, 10, 11, 12, 13, 14, 15, 16]
    for i in chin_points:
        if i < len(landmarks):
            src = landmarks[i].astype(int)
            dst = src.copy()
            dst[1] = src[1] - int(shift * 0.5)  # 向上提
            result = interactive_warp(result, src, dst, int(radius * 0.8))

    return result


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from face_beauty import FaceBeauty

    # 初始化
    print("加载模型...")
    beauty = FaceBeauty()

    # 读取图像
    image = cv2.imread('huge.jpg')
    print(f"图像大小: {image.shape}")

    # 分析人脸
    faces = beauty.analyze(image)
    print(f"检测到 {len(faces)} 张人脸")

    if len(faces) > 0:
        landmarks = faces[0].landmark_2d_106

        # 应用瘦脸
        print("\n瘦脸处理 (intensity=0.8)...")
        result = slim_face_advanced(image, landmarks, intensity=0.8)

        # 再加美颜
        print("美颜处理...")
        result = beauty.beautify(result, intensity=0.7)

        # 保存
        cv2.imwrite('slim_result.jpg', result)

        # 对比图
        comparison = np.hstack([image, result])
        cv2.imwrite('slim_comparison.jpg', comparison)

        print("\n✓ 结果已保存:")
        print("  - slim_result.jpg")
        print("  - slim_comparison.jpg")

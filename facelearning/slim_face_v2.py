"""
瘦脸算法 v2 - 更强效果
使用局部仿射变换实现快速液化
"""
import cv2
import numpy as np


def bilinear_interp(image, x, y):
    """双线性插值"""
    h, w = image.shape[:2]

    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1

    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w - 1))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))

    dx = x - x0
    dy = y - y0

    val = (1 - dx) * (1 - dy) * image[y0, x0] + \
          dx * (1 - dy) * image[y0, x1] + \
          (1 - dx) * dy * image[y1, x0] + \
          dx * dy * image[y1, x1]

    return val.astype(np.uint8)


def local_zoom_warp(image, center, radius, strength):
    """
    局部缩放变形 - 向心收缩

    Args:
        image: 输入图像
        center: 中心点 (x, y)
        radius: 影响半径
        strength: 收缩强度 (正数收缩，负数放大)

    Returns:
        变形后的图像
    """
    h, w = image.shape[:2]
    result = image.copy()

    cx, cy = center

    for y in range(max(0, cy - radius), min(h, cy + radius)):
        for x in range(max(0, cx - radius), min(w, cx + radius)):
            dx = x - cx
            dy = y - cy
            dist = np.sqrt(dx * dx + dy * dy)

            if dist < radius and dist > 0:
                # 使用平滑的收缩函数
                factor = (1 - (dist / radius) ** 2) * strength

                # 计算新的采样位置（向外采样实现收缩效果）
                scale = 1 + factor
                new_x = cx + dx * scale
                new_y = cy + dy * scale

                if 0 <= new_x < w - 1 and 0 <= new_y < h - 1:
                    result[y, x] = bilinear_interp(image, new_x, new_y)

    return result


def slim_face_v2(image, landmarks, intensity=0.5):
    """
    瘦脸 v2 - 更强效果

    Args:
        image: BGR图像
        landmarks: 106个关键点
        intensity: 瘦脸强度 (0-1)

    Returns:
        瘦脸后的图像
    """
    result = image.copy()

    # 计算脸部尺寸
    left_cheek = landmarks[0:10]
    right_cheek = landmarks[23:33]

    face_width = right_cheek[:, 0].mean() - left_cheek[:, 0].mean()
    radius = int(face_width * 0.45)  # 增大影响半径
    strength = 0.25 * intensity  # 增强收缩强度

    # 左脸颊收缩点
    left_points = [
        landmarks[3].astype(int),
        landmarks[4].astype(int),
        landmarks[5].astype(int),
        landmarks[6].astype(int),
    ]

    # 右脸颊收缩点
    right_points = [
        landmarks[28].astype(int),
        landmarks[29].astype(int),
        landmarks[30].astype(int),
        landmarks[31].astype(int),
    ]

    # 应用收缩变形
    for pt in left_points:
        result = local_zoom_warp(result, tuple(pt), radius, strength)

    for pt in right_points:
        result = local_zoom_warp(result, tuple(pt), radius, strength)

    # 下巴收缩 - 增强效果
    chin_center = landmarks[16].astype(int)
    result = local_zoom_warp(result, tuple(chin_center), int(radius * 0.9), strength * 0.8)

    # 额外下巴两侧收缩
    chin_left = landmarks[10].astype(int)
    chin_right = landmarks[22].astype(int)
    result = local_zoom_warp(result, tuple(chin_left), int(radius * 0.7), strength * 0.6)
    result = local_zoom_warp(result, tuple(chin_right), int(radius * 0.7), strength * 0.6)

    return result


def enlarge_eyes_v2(image, landmarks, intensity=0.5):
    """
    大眼 v2

    Args:
        image: BGR图像
        landmarks: 106个关键点
        intensity: 大眼强度 (0-1)

    Returns:
        大眼后的图像
    """
    result = image.copy()

    # 左眼中心
    left_eye = landmarks[35:47].mean(axis=0).astype(int)
    # 右眼中心
    right_eye = landmarks[89:101].mean(axis=0).astype(int)

    # 眼睛大小
    eye_width = np.linalg.norm(landmarks[35] - landmarks[41])
    radius = int(eye_width * 0.8)

    # 放大强度（负数表示放大）
    strength = -0.2 * intensity

    result = local_zoom_warp(result, tuple(left_eye), radius, strength)
    result = local_zoom_warp(result, tuple(right_eye), radius, strength)

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

        # 先美颜
        print("\n美颜处理 (intensity=0.7)...")
        result = beauty.beautify(image, intensity=0.7)

        # 瘦脸
        print("瘦脸处理 (intensity=1.0)...")
        result = slim_face_v2(result, landmarks, intensity=1.0)

        # 大眼
        print("大眼处理 (intensity=0.5)...")
        result = enlarge_eyes_v2(result, landmarks, intensity=0.5)

        # 保存
        cv2.imwrite('slim_v2_result.jpg', result)

        # 对比图
        comparison = np.hstack([image, result])
        cv2.imwrite('slim_v2_comparison.jpg', comparison)

        print("\n✓ 结果已保存:")
        print("  - slim_v2_result.jpg")
        print("  - slim_v2_comparison.jpg")

"""
人脸特效 - 瘦脸 + 贴纸特效
结合人脸分割和关键点实现各种效果
"""
import cv2
import numpy as np
import math
import sys
import os

sys.path.insert(0, '.')
from face_beauty import FaceBeauty
from slim_face_v2 import slim_face_v2, enlarge_eyes_v2


class FaceEffects:
    """人脸特效类"""

    def __init__(self):
        """初始化"""
        print("加载人脸分析模型...")
        self.beauty = FaceBeauty()
        self.landmarks = None
        self.face_info = None

    def analyze(self, image):
        """分析人脸"""
        faces = self.beauty.analyze(image)
        if len(faces) > 0:
            self.face_info = faces[0]
            self.landmarks = faces[0].landmark_2d_106
            return True
        return False

    def slim_face(self, image, intensity=0.8):
        """瘦脸"""
        if self.landmarks is None:
            return image
        return slim_face_v2(image, self.landmarks, intensity)

    def enlarge_eyes(self, image, intensity=0.5):
        """大眼"""
        if self.landmarks is None:
            return image
        return enlarge_eyes_v2(image, self.landmarks, intensity)

    def draw_star(self, image, center, size, color, rotation=0, filled=True):
        """
        绘制星星

        Args:
            image: 图像
            center: 中心点 (x, y)
            size: 大小
            color: 颜色 (B, G, R)
            rotation: 旋转角度
            filled: 是否填充
        """
        cx, cy = center
        points = []

        for i in range(5):
            # 外点
            angle = math.radians(rotation + i * 72 - 90)
            x = cx + size * math.cos(angle)
            y = cy + size * math.sin(angle)
            points.append([int(x), int(y)])

            # 内点
            angle = math.radians(rotation + i * 72 + 36 - 90)
            x = cx + size * 0.4 * math.cos(angle)
            y = cy + size * 0.4 * math.sin(angle)
            points.append([int(x), int(y)])

        pts = np.array(points, np.int32).reshape((-1, 1, 2))

        if filled:
            cv2.fillPoly(image, [pts], color)
        else:
            cv2.polylines(image, [pts], True, color, 2)

        return image

    def draw_heart(self, image, center, size, color, filled=True):
        """
        绘制爱心

        Args:
            image: 图像
            center: 中心点 (x, y)
            size: 大小
            color: 颜色 (B, G, R)
            filled: 是否填充
        """
        cx, cy = center
        points = []

        for t in np.linspace(0, 2 * math.pi, 100):
            x = 16 * math.sin(t) ** 3
            y = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
            x = cx + x * size / 16
            y = cy + y * size / 16
            points.append([int(x), int(y)])

        pts = np.array(points, np.int32).reshape((-1, 1, 2))

        if filled:
            cv2.fillPoly(image, [pts], color)
        else:
            cv2.polylines(image, [pts], True, color, 2)

        return image

    def draw_sparkle(self, image, center, size, color):
        """
        绘制闪光效果

        Args:
            image: 图像
            center: 中心点 (x, y)
            size: 大小
            color: 颜色 (B, G, R)
        """
        cx, cy = center

        # 四条线
        cv2.line(image, (cx - size, cy), (cx + size, cy), color, 2)
        cv2.line(image, (cx, cy - size), (cx, cy + size), color, 2)

        # 对角线（短一点）
        s2 = int(size * 0.6)
        cv2.line(image, (cx - s2, cy - s2), (cx + s2, cy + s2), color, 1)
        cv2.line(image, (cx - s2, cy + s2), (cx + s2, cy - s2), color, 1)

        # 中心点
        cv2.circle(image, (cx, cy), 3, color, -1)

        return image

    def add_cheek_effects(self, image, effect_type='stars', color=None, count=5, size=None):
        """
        在脸颊添加特效

        Args:
            image: 图像
            effect_type: 特效类型 ('stars', 'hearts', 'sparkles', 'blush')
            color: 颜色，None则随机
            count: 数量
            size: 大小，None则自动

        Returns:
            添加特效后的图像
        """
        if self.landmarks is None:
            print("请先调用 analyze() 分析人脸")
            return image

        result = image.copy()

        # 获取脸颊位置
        # 左脸颊: landmarks 1-5 区域
        left_cheek_center = self.landmarks[3:6].mean(axis=0).astype(int)
        # 右脸颊: landmarks 28-32 区域
        right_cheek_center = self.landmarks[28:31].mean(axis=0).astype(int)

        # 计算默认大小
        if size is None:
            face_width = self.landmarks[32, 0] - self.landmarks[0, 0]
            size = int(face_width * 0.08)  # 增大默认尺寸

        # 腮红效果
        if effect_type == 'blush':
            overlay = result.copy()
            blush_color = color if color else (180, 150, 255)  # 粉红色
            blush_size = int(size * 3)

            cv2.circle(overlay, tuple(left_cheek_center), blush_size, blush_color, -1)
            cv2.circle(overlay, tuple(right_cheek_center), blush_size, blush_color, -1)

            # 高斯模糊
            overlay = cv2.GaussianBlur(overlay, (51, 51), 0)
            result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
            return result

        # 生成随机位置
        np.random.seed(42)  # 固定种子保证可重复

        for cheek_center in [left_cheek_center, right_cheek_center]:
            for i in range(count):
                # 随机偏移
                offset_x = np.random.randint(-size * 3, size * 3)
                offset_y = np.random.randint(-size * 3, size * 3)
                pos = (cheek_center[0] + offset_x, cheek_center[1] + offset_y)

                # 随机大小变化
                s = int(size * np.random.uniform(0.5, 1.2))

                # 随机颜色
                if color is None:
                    c = (
                        np.random.randint(100, 255),
                        np.random.randint(100, 255),
                        np.random.randint(100, 255)
                    )
                else:
                    c = color

                # 绘制特效
                if effect_type == 'stars':
                    rotation = np.random.randint(0, 72)
                    self.draw_star(result, pos, s, c, rotation)
                elif effect_type == 'hearts':
                    self.draw_heart(result, pos, s, c)
                elif effect_type == 'sparkles':
                    self.draw_sparkle(result, pos, s, c)

        return result

    def add_forehead_crown(self, image, color=(0, 215, 255)):
        """
        在额头添加皇冠/光环效果

        Args:
            image: 图像
            color: 颜色

        Returns:
            添加特效后的图像
        """
        if self.landmarks is None:
            return image

        result = image.copy()

        # 额头中心位置 (眉毛上方)
        forehead_center = self.landmarks[33:43].mean(axis=0)  # 眉毛区域
        forehead_center[1] -= 50  # 向上偏移
        forehead_center = forehead_center.astype(int)

        # 绘制简单皇冠
        face_width = self.landmarks[32, 0] - self.landmarks[0, 0]
        crown_width = int(face_width * 0.6)
        crown_height = int(face_width * 0.2)

        cx, cy = forehead_center

        # 皇冠底部
        pts_base = np.array([
            [cx - crown_width//2, cy + crown_height//2],
            [cx + crown_width//2, cy + crown_height//2],
            [cx + crown_width//2, cy],
            [cx - crown_width//2, cy]
        ], np.int32)

        # 皇冠尖端
        pts_top = np.array([
            [cx - crown_width//2, cy],
            [cx - crown_width//4, cy - crown_height],
            [cx, cy - crown_height//2],
            [cx + crown_width//4, cy - crown_height],
            [cx + crown_width//2, cy]
        ], np.int32)

        cv2.fillPoly(result, [pts_base], color)
        cv2.fillPoly(result, [pts_top.reshape((-1, 1, 2))], color)

        # 添加宝石
        gem_color = (0, 0, 255)  # 红色
        cv2.circle(result, (cx, cy - crown_height//2), 5, gem_color, -1)
        cv2.circle(result, (cx - crown_width//4, cy - crown_height + 5), 4, gem_color, -1)
        cv2.circle(result, (cx + crown_width//4, cy - crown_height + 5), 4, gem_color, -1)

        return result

    def add_eye_sparkle(self, image, color=(255, 255, 255)):
        """
        眼睛高光/星星眼效果

        Args:
            image: 图像
            color: 颜色

        Returns:
            添加特效后的图像
        """
        if self.landmarks is None:
            return image

        result = image.copy()

        # 左眼中心
        left_eye = self.landmarks[35:47].mean(axis=0).astype(int)
        # 右眼中心
        right_eye = self.landmarks[89:101].mean(axis=0).astype(int)

        # 眼睛大小
        eye_size = int(np.linalg.norm(self.landmarks[35] - self.landmarks[41]) * 0.3)

        # 在每只眼睛添加高光星星
        for eye_center in [left_eye, right_eye]:
            # 主高光点偏上偏左
            highlight_pos = (eye_center[0] - eye_size//3, eye_center[1] - eye_size//3)
            self.draw_sparkle(result, highlight_pos, eye_size//2, color)

        return result

    def apply_all_effects(self, image, slim_intensity=0.8, eye_intensity=0.5,
                          cheek_effect='stars', add_blush=True, add_crown=False,
                          add_eye_sparkle=True):
        """
        应用所有特效

        Args:
            image: 输入图像
            slim_intensity: 瘦脸强度
            eye_intensity: 大眼强度
            cheek_effect: 脸颊特效类型 ('stars', 'hearts', 'sparkles', None)
            add_blush: 是否添加腮红
            add_crown: 是否添加皇冠
            add_eye_sparkle: 是否添加眼睛高光

        Returns:
            处理后的图像
        """
        if not self.analyze(image):
            print("未检测到人脸")
            return image

        result = image.copy()

        # 瘦脸
        if slim_intensity > 0:
            print("  - 瘦脸...")
            result = self.slim_face(result, slim_intensity)

        # 大眼
        if eye_intensity > 0:
            print("  - 大眼...")
            result = self.enlarge_eyes(result, eye_intensity)

        # 腮红
        if add_blush:
            print("  - 腮红...")
            result = self.add_cheek_effects(result, 'blush')

        # 脸颊特效
        if cheek_effect:
            print(f"  - 脸颊特效 ({cheek_effect})...")
            result = self.add_cheek_effects(result, cheek_effect, count=6)

        # 皇冠
        if add_crown:
            print("  - 皇冠...")
            result = self.add_forehead_crown(result)

        # 眼睛高光
        if add_eye_sparkle:
            print("  - 眼睛高光...")
            result = self.add_eye_sparkle(result)

        return result


def create_effect_grid(image, effects):
    """创建特效对比网格"""
    h, w = image.shape[:2]
    target_h = h // 2
    target_w = w // 2

    resized = []
    for img in effects:
        resized.append(cv2.resize(img, (target_w, target_h)))

    # 创建2x2网格
    row1 = np.hstack([resized[0], resized[1]])
    row2 = np.hstack([resized[2], resized[3]])
    grid = np.vstack([row1, row2])

    return grid


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    print("=" * 50)
    print("人脸特效演示")
    print("=" * 50)

    # 初始化
    fx = FaceEffects()

    # 读取图像
    image = cv2.imread('huge.jpg')
    print(f"\n图像大小: {image.shape}")

    # 分析人脸
    print("\n分析人脸...")
    if not fx.analyze(image):
        print("未检测到人脸!")
        exit(1)

    # 生成各种特效
    print("\n生成特效...")

    # 1. 只有瘦脸+大眼
    print("\n[1] 瘦脸+大眼")
    result1 = fx.slim_face(image.copy(), intensity=0.8)
    result1 = fx.enlarge_eyes(result1, intensity=0.5)
    cv2.putText(result1, "Slim+BigEyes", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 2. 星星特效
    print("[2] 星星特效")
    result2 = fx.add_cheek_effects(image.copy(), 'stars', color=(0, 255, 255), count=8)
    cv2.putText(result2, "Stars", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 3. 爱心特效
    print("[3] 爱心特效")
    result3 = fx.add_cheek_effects(image.copy(), 'hearts', color=(180, 105, 255), count=6)
    cv2.putText(result3, "Hearts", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 4. 综合特效
    print("[4] 综合特效")
    result4 = fx.apply_all_effects(
        image.copy(),
        slim_intensity=0.8,
        eye_intensity=0.5,
        cheek_effect='sparkles',
        add_blush=True,
        add_crown=True,
        add_eye_sparkle=True
    )
    cv2.putText(result4, "Full Effects", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 创建对比网格
    print("\n创建对比图...")
    grid = create_effect_grid(image, [result1, result2, result3, result4])
    cv2.imwrite('effects_grid.jpg', grid)
    print("保存: effects_grid.jpg")

    # 单独保存综合效果
    comparison = np.hstack([image, result4])
    cv2.imwrite('effects_comparison.jpg', comparison)
    print("保存: effects_comparison.jpg")

    # 保存各个单独效果
    cv2.imwrite('effect_slim.jpg', result1)
    cv2.imwrite('effect_stars.jpg', result2)
    cv2.imwrite('effect_hearts.jpg', result3)
    cv2.imwrite('effect_full.jpg', result4)

    print("\n完成!")

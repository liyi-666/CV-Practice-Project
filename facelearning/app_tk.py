"""
人脸美颜 GUI 应用 (Tkinter版本)
功能：关键点检测、属性识别、美颜、特效、大眼
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import sys
import threading

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.insert(0, '.')


class FaceBeautyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人脸美颜工具")
        self.root.geometry("1200x800")

        # 初始化变量
        self.original_image = None
        self.current_image = None
        self.landmarks = None
        self.face_info = None
        self.beauty = None
        self.ai_beauty = None

        # 创建UI
        self.create_widgets()

        # 异步加载模型
        self.load_models_async()

    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew", padx=5)

        # 图片/视频操作
        ttk.Button(control_frame, text="上传图片", command=self.load_image).grid(row=0, column=0, pady=5, sticky="ew")
        ttk.Button(control_frame, text="上传视频", command=self.load_video).grid(row=0, column=1, pady=5, sticky="ew")
        ttk.Button(control_frame, text="检测人脸", command=self.detect_face).grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")

        # 人脸属性显示
        ttk.Label(control_frame, text="人脸属性:").grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky="w")
        self.info_text = tk.Text(control_frame, height=6, width=30, state='disabled')
        self.info_text.grid(row=3, column=0, columnspan=2, pady=5, sticky="ew")

        # 美颜设置
        ttk.Separator(control_frame, orient='horizontal').grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")
        ttk.Label(control_frame, text="美颜设置", font=('', 10, 'bold')).grid(row=5, column=0, columnspan=2, sticky="w")

        self.use_ai_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="使用AI美颜", variable=self.use_ai_var).grid(row=6, column=0, columnspan=2, sticky="w")

        ttk.Label(control_frame, text="磨皮强度:").grid(row=7, column=0, sticky="w")
        self.smooth_var = tk.DoubleVar(value=0.5)
        ttk.Scale(control_frame, from_=0, to=1, variable=self.smooth_var, orient='horizontal').grid(row=7, column=1, sticky="ew")

        ttk.Label(control_frame, text="美白强度:").grid(row=8, column=0, sticky="w")
        self.whiten_var = tk.DoubleVar(value=0.3)
        ttk.Scale(control_frame, from_=0, to=1, variable=self.whiten_var, orient='horizontal').grid(row=8, column=1, sticky="ew")

        # 大眼设置
        ttk.Separator(control_frame, orient='horizontal').grid(row=9, column=0, columnspan=2, pady=10, sticky="ew")
        ttk.Label(control_frame, text="大眼设置", font=('', 10, 'bold')).grid(row=10, column=0, columnspan=2, sticky="w")

        self.enlarge_eye_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="启用大眼", variable=self.enlarge_eye_var).grid(row=11, column=0, columnspan=2, sticky="w")

        ttk.Label(control_frame, text="大眼强度:").grid(row=12, column=0, sticky="w")
        self.eye_var = tk.DoubleVar(value=0.5)
        ttk.Scale(control_frame, from_=0, to=1, variable=self.eye_var, orient='horizontal').grid(row=12, column=1, sticky="ew")

        # 特效贴纸
        ttk.Separator(control_frame, orient='horizontal').grid(row=13, column=0, columnspan=2, pady=10, sticky="ew")
        ttk.Label(control_frame, text="特效贴纸", font=('', 10, 'bold')).grid(row=14, column=0, columnspan=2, sticky="w")

        self.stars_var = tk.BooleanVar(value=False)
        self.hearts_var = tk.BooleanVar(value=False)
        self.sparkles_var = tk.BooleanVar(value=False)
        self.blush_var = tk.BooleanVar(value=False)
        self.crown_var = tk.BooleanVar(value=False)
        self.eye_sparkle_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(control_frame, text="星星", variable=self.stars_var).grid(row=15, column=0, sticky="w")
        ttk.Checkbutton(control_frame, text="爱心", variable=self.hearts_var).grid(row=15, column=1, sticky="w")
        ttk.Checkbutton(control_frame, text="闪光", variable=self.sparkles_var).grid(row=16, column=0, sticky="w")
        ttk.Checkbutton(control_frame, text="腮红", variable=self.blush_var).grid(row=16, column=1, sticky="w")
        ttk.Checkbutton(control_frame, text="皇冠", variable=self.crown_var).grid(row=17, column=0, sticky="w")
        ttk.Checkbutton(control_frame, text="眼睛高光", variable=self.eye_sparkle_var).grid(row=17, column=1, sticky="w")

        # 应用按钮
        ttk.Separator(control_frame, orient='horizontal').grid(row=18, column=0, columnspan=2, pady=10, sticky="ew")
        ttk.Button(control_frame, text="应用效果", command=self.apply_effects).grid(row=19, column=0, columnspan=2, pady=5, sticky="ew")
        ttk.Button(control_frame, text="保存结果", command=self.save_image).grid(row=20, column=0, columnspan=2, pady=5, sticky="ew")
        ttk.Button(control_frame, text="重置", command=self.reset_image).grid(row=21, column=0, columnspan=2, pady=5, sticky="ew")

        # 状态栏
        self.status_var = tk.StringVar(value="正在加载模型...")
        ttk.Label(control_frame, textvariable=self.status_var).grid(row=22, column=0, columnspan=2, pady=5, sticky="w")

        # 视频处理进度条
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=23, column=0, columnspan=2, pady=5, sticky="ew")
        self.progress_label = tk.StringVar(value="")
        ttk.Label(control_frame, textvariable=self.progress_label).grid(row=24, column=0, columnspan=2, sticky="w")

        # 右侧图片显示区域
        image_frame = ttk.LabelFrame(main_frame, text="图片预览", padding="10")
        image_frame.grid(row=0, column=1, sticky="nsew", padx=5)

        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # 图片标签
        self.image_label = ttk.Label(image_frame, text="请上传图片")
        self.image_label.pack(expand=True, fill='both')

    def load_models_async(self):
        """异步加载模型"""
        def load():
            try:
                from face_beauty import FaceBeauty
                self.beauty = FaceBeauty()

                try:
                    from face_beauty_ai import AIBeauty
                    self.ai_beauty = AIBeauty()
                except Exception as e:
                    error_msg = str(e)
                    print(f"AI美颜模块加载失败: {e}")
                    self.ai_beauty = None

                self.root.after(0, lambda: self.status_var.set("模型加载完成，请上传图片"))
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda msg=error_msg: self.status_var.set(f"模型加载失败: {msg}"))

        thread = threading.Thread(target=load)
        thread.daemon = True
        thread.start()

    def load_image(self):
        """加载图片"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.current_image = self.original_image.copy()
            self.landmarks = None
            self.face_info = None
            self.display_image(self.current_image)
            self.status_var.set(f"已加载: {os.path.basename(file_path)}")

    def display_image(self, image, max_size=600):
        """显示图片"""
        if image is None:
            return

        # BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 缩放
        h, w = rgb.shape[:2]
        scale = min(max_size / w, max_size / h, 1)
        new_w, new_h = int(w * scale), int(h * scale)
        rgb = cv2.resize(rgb, (new_w, new_h))

        # 转换为PIL Image
        pil_image = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=self.photo)

    def detect_face(self):
        """检测人脸"""
        if self.original_image is None:
            messagebox.showwarning("警告", "请先上传图片")
            return

        if self.beauty is None:
            messagebox.showwarning("警告", "模型尚未加载完成")
            return

        self.status_var.set("正在检测人脸...")
        self.root.update()

        faces = self.beauty.analyze(self.original_image)

        if len(faces) == 0:
            messagebox.showinfo("提示", "未检测到人脸")
            self.status_var.set("未检测到人脸")
            return

        self.face_info = faces[0]
        self.landmarks = faces[0].landmark_2d_106

        # 绘制关键点
        result = self.beauty.draw_analysis(self.original_image.copy(), faces)
        self.current_image = result
        self.display_image(result)

        # 显示属性
        info_lines = []
        info_lines.append(f"关键点: 106")

        if hasattr(self.face_info, 'age') and self.face_info.age is not None:
            info_lines.append(f"年龄: {int(self.face_info.age)}")

        if hasattr(self.face_info, 'gender') and self.face_info.gender is not None:
            gender = "男" if self.face_info.gender == 1 else "女"
            info_lines.append(f"性别: {gender}")

        bbox = self.face_info.bbox.astype(int)
        info_lines.append(f"位置: ({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})")

        self.info_text.config(state='normal')
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, "\n".join(info_lines))
        self.info_text.config(state='disabled')

        self.status_var.set("人脸检测完成")

    def apply_effects(self):
        """应用所有效果"""
        if self.original_image is None:
            messagebox.showwarning("警告", "请先上传图片")
            return

        self.status_var.set("正在处理...")
        self.root.update()

        result = self.original_image.copy()

        # 重新分析获取关键点
        if self.beauty:
            faces = self.beauty.analyze(result)
            if len(faces) > 0:
                self.landmarks = faces[0].landmark_2d_106

        # 美颜
        smooth = self.smooth_var.get()
        whiten = self.whiten_var.get()

        if smooth > 0 or whiten > 0:
            if self.use_ai_var.get() and self.ai_beauty:
                result = self.ai_beauty.beautify(
                    result,
                    intensity=smooth,
                    smooth=smooth > 0,
                    whiten=whiten > 0,
                    remove_blemish=True
                )
            elif self.beauty:
                result = self.beauty.beautify(result, intensity=smooth)

        # 大眼
        if self.enlarge_eye_var.get() and self.landmarks is not None:
            from slim_face_v2 import enlarge_eyes_v2
            result = enlarge_eyes_v2(result, self.landmarks, intensity=self.eye_var.get())

        # 特效
        if self.landmarks is not None:
            result = self.apply_stickers(result)

        self.current_image = result
        self.display_image(result)
        self.status_var.set("效果应用完成")

    def apply_stickers(self, image):
        """应用贴纸特效"""
        result = image.copy()
        landmarks = self.landmarks

        face_width = landmarks[32, 0] - landmarks[0, 0]
        size = int(face_width * 0.08)
        left_cheek = landmarks[3:6].mean(axis=0).astype(int)
        right_cheek = landmarks[28:31].mean(axis=0).astype(int)

        # 腮红
        if self.blush_var.get():
            overlay = result.copy()
            blush_size = int(size * 3)
            cv2.circle(overlay, tuple(left_cheek), blush_size, (180, 150, 255), -1)
            cv2.circle(overlay, tuple(right_cheek), blush_size, (180, 150, 255), -1)
            overlay = cv2.GaussianBlur(overlay, (51, 51), 0)
            result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

        # 星星
        if self.stars_var.get():
            result = self._add_stickers(result, left_cheek, right_cheek, 'stars', size)

        # 爱心
        if self.hearts_var.get():
            result = self._add_stickers(result, left_cheek, right_cheek, 'hearts', size)

        # 闪光
        if self.sparkles_var.get():
            result = self._add_stickers(result, left_cheek, right_cheek, 'sparkles', size)

        # 皇冠
        if self.crown_var.get():
            result = self._add_crown(result, landmarks)

        # 眼睛高光
        if self.eye_sparkle_var.get():
            result = self._add_eye_sparkle(result, landmarks)

        return result

    def _add_stickers(self, image, left_cheek, right_cheek, effect_type, size):
        """添加贴纸"""
        import math
        result = image.copy()
        np.random.seed(42)

        for cheek_center in [left_cheek, right_cheek]:
            for i in range(6):
                offset_x = np.random.randint(-size * 3, size * 3)
                offset_y = np.random.randint(-size * 3, size * 3)
                pos = (cheek_center[0] + offset_x, cheek_center[1] + offset_y)
                s = int(size * np.random.uniform(0.5, 1.2))
                color = (
                    np.random.randint(100, 255),
                    np.random.randint(100, 255),
                    np.random.randint(100, 255)
                )

                if effect_type == 'stars':
                    self._draw_star(result, pos, s, color)
                elif effect_type == 'hearts':
                    self._draw_heart(result, pos, s, (180, 105, 255))
                elif effect_type == 'sparkles':
                    self._draw_sparkle(result, pos, s, color)

        return result

    def _draw_star(self, image, center, size, color):
        """绘制星星"""
        import math
        cx, cy = center
        points = []
        rotation = np.random.randint(0, 72)

        for i in range(5):
            angle = math.radians(rotation + i * 72 - 90)
            x = cx + size * math.cos(angle)
            y = cy + size * math.sin(angle)
            points.append([int(x), int(y)])

            angle = math.radians(rotation + i * 72 + 36 - 90)
            x = cx + size * 0.4 * math.cos(angle)
            y = cy + size * 0.4 * math.sin(angle)
            points.append([int(x), int(y)])

        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(image, [pts], color)

    def _draw_heart(self, image, center, size, color):
        """绘制爱心"""
        import math
        cx, cy = center
        points = []

        for t in np.linspace(0, 2 * math.pi, 100):
            x = 16 * math.sin(t) ** 3
            y = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
            x = cx + x * size / 16
            y = cy + y * size / 16
            points.append([int(x), int(y)])

        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(image, [pts], color)

    def _draw_sparkle(self, image, center, size, color):
        """绘制闪光"""
        cx, cy = center
        cv2.line(image, (cx - size, cy), (cx + size, cy), color, 2)
        cv2.line(image, (cx, cy - size), (cx, cy + size), color, 2)
        s2 = int(size * 0.6)
        cv2.line(image, (cx - s2, cy - s2), (cx + s2, cy + s2), color, 1)
        cv2.line(image, (cx - s2, cy + s2), (cx + s2, cy - s2), color, 1)
        cv2.circle(image, (cx, cy), 3, color, -1)

    def _add_crown(self, image, landmarks):
        """添加皇冠"""
        result = image.copy()
        color = (0, 215, 255)

        forehead_center = landmarks[33:43].mean(axis=0)
        forehead_center[1] -= 50
        forehead_center = forehead_center.astype(int)

        face_width = landmarks[32, 0] - landmarks[0, 0]
        crown_width = int(face_width * 0.6)
        crown_height = int(face_width * 0.2)

        cx, cy = forehead_center

        pts_base = np.array([
            [cx - crown_width//2, cy + crown_height//2],
            [cx + crown_width//2, cy + crown_height//2],
            [cx + crown_width//2, cy],
            [cx - crown_width//2, cy]
        ], np.int32)

        pts_top = np.array([
            [cx - crown_width//2, cy],
            [cx - crown_width//4, cy - crown_height],
            [cx, cy - crown_height//2],
            [cx + crown_width//4, cy - crown_height],
            [cx + crown_width//2, cy]
        ], np.int32)

        cv2.fillPoly(result, [pts_base], color)
        cv2.fillPoly(result, [pts_top.reshape((-1, 1, 2))], color)

        gem_color = (0, 0, 255)
        cv2.circle(result, (cx, cy - crown_height//2), 5, gem_color, -1)
        cv2.circle(result, (cx - crown_width//4, cy - crown_height + 5), 4, gem_color, -1)
        cv2.circle(result, (cx + crown_width//4, cy - crown_height + 5), 4, gem_color, -1)

        return result

    def _add_eye_sparkle(self, image, landmarks):
        """添加眼睛高光"""
        result = image.copy()
        color = (255, 255, 255)

        left_eye = landmarks[35:47].mean(axis=0).astype(int)
        right_eye = landmarks[89:101].mean(axis=0).astype(int)
        eye_size = int(np.linalg.norm(landmarks[35] - landmarks[41]) * 0.3)

        for eye_center in [left_eye, right_eye]:
            highlight_pos = (eye_center[0] - eye_size//3, eye_center[1] - eye_size//3)
            self._draw_sparkle(result, highlight_pos, eye_size//2, color)

        return result

    def save_image(self):
        """保存图片"""
        if self.current_image is None:
            messagebox.showwarning("警告", "没有可保存的图片")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )
        if file_path:
            cv2.imwrite(file_path, self.current_image)
            messagebox.showinfo("成功", f"图片已保存: {file_path}")

    def reset_image(self):
        """重置图片"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.display_image(self.current_image)
            self.status_var.set("已重置")

    def load_video(self):
        """加载视频文件"""
        if self.beauty is None:
            messagebox.showwarning("警告", "模型尚未加载完成，请稍候")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            self.process_video(file_path)

    def process_video(self, input_path):
        """处理视频文件"""
        # 打开视频
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            messagebox.showerror("错误", "无法打开视频文件")
            return

        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 选择输出路径
        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi")],
            initialfile="beauty_output.mp4"
        )
        if not output_path:
            cap.release()
            return

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.status_var.set("正在处理视频...")
        self.progress_var.set(0)

        # 在新线程中处理视频
        def process_thread():
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 处理帧
                processed = self.process_single_frame(frame)
                out.write(processed)

                # 更新进度
                frame_count += 1
                progress = (frame_count / total_frames) * 100

                # 每10帧更新一次UI
                if frame_count % 10 == 0:
                    self.root.after(0, lambda p=progress, f=frame_count: self.update_video_progress(p, f, total_frames))
                    self.root.after(0, lambda img=processed: self.display_image(img))

            # 释放资源
            cap.release()
            out.release()

            # 完成提示
            self.root.after(0, lambda: self.video_complete(output_path))

        thread = threading.Thread(target=process_thread)
        thread.daemon = True
        thread.start()

    def process_single_frame(self, frame):
        """处理单帧图像"""
        result = frame.copy()

        # 检测人脸
        faces = self.beauty.analyze(frame)

        if len(faces) > 0:
            landmarks = faces[0].landmark_2d_106

            # 美颜
            smooth = self.smooth_var.get()
            if smooth > 0:
                if self.use_ai_var.get() and self.ai_beauty:
                    result = self.ai_beauty.beautify(result, intensity=smooth)
                else:
                    result = self.beauty.beautify(result, intensity=smooth)

            # 重新检测（美颜后可能变化）
            faces = self.beauty.analyze(result)
            if len(faces) > 0:
                landmarks = faces[0].landmark_2d_106

                # 大眼
                if self.enlarge_eye_var.get():
                    from slim_face_v2 import enlarge_eyes_v2
                    result = enlarge_eyes_v2(result, landmarks, intensity=self.eye_var.get())

                # 特效
                face_width = landmarks[32, 0] - landmarks[0, 0]
                size = int(face_width * 0.06)
                left_cheek = landmarks[3:6].mean(axis=0).astype(int)
                right_cheek = landmarks[28:31].mean(axis=0).astype(int)

                np.random.seed(12345)  # 固定种子让贴纸位置稳定

                if self.blush_var.get():
                    overlay = result.copy()
                    blush_size = int(size * 3)
                    cv2.circle(overlay, tuple(left_cheek), blush_size, (180, 150, 255), -1)
                    cv2.circle(overlay, tuple(right_cheek), blush_size, (180, 150, 255), -1)
                    overlay = cv2.GaussianBlur(overlay, (31, 31), 0)
                    result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

                if self.stars_var.get():
                    result = self._add_stickers(result, left_cheek, right_cheek, 'stars', size)

                if self.hearts_var.get():
                    result = self._add_stickers(result, left_cheek, right_cheek, 'hearts', size)

                if self.sparkles_var.get():
                    result = self._add_stickers(result, left_cheek, right_cheek, 'sparkles', size)

                if self.crown_var.get():
                    result = self._add_crown(result, landmarks)

                if self.eye_sparkle_var.get():
                    result = self._add_eye_sparkle(result, landmarks)

        return result

    def update_video_progress(self, progress, current, total):
        """更新视频处理进度"""
        self.progress_var.set(progress)
        self.progress_label.set(f"处理中: {current}/{total} 帧 ({progress:.1f}%)")

    def video_complete(self, output_path):
        """视频处理完成"""
        self.progress_var.set(100)
        self.progress_label.set("处理完成!")
        self.status_var.set(f"视频已保存: {os.path.basename(output_path)}")
        messagebox.showinfo("完成", f"视频美颜完成!\n保存至: {output_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceBeautyApp(root)
    root.mainloop()

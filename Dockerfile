# 使用与系统匹配的基础镜像
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.10 \
    TZ=Asia/Shanghai

# 设置时区
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建Python虚拟环境（可选，但推荐）
RUN pip3 install virtualenv && \
    virtualenv /venv --python=python3

# 激活虚拟环境
ENV PATH="/venv/bin:$PATH"

# 设置工作目录
WORKDIR /workspace

# 安装Python依赖
COPY requirements.txt /workspace/
RUN pip install --upgrade pip && \
    pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 安装其他主要依赖
RUN pip install \
    opencv-python==4.8.1.78 \
    mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html \
    mmdet==2.28.1

# 验证安装
RUN python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')" && \
    python -c "import cv2; print(f'OpenCV版本: {cv2.__version__}')" && \
    python -c "import mmdet; print(f'MMDetection版本: {mmdet.__version__}')"

# 默认命令
CMD ["/bin/bash"]
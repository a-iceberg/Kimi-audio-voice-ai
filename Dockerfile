FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

# Ставим базовые утилиты
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python-is-python3 \
    git \
    curl \
    sox \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Установка pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py

# Устанавливаем PyTorch nightly с CUDA 12.8
RUN pip install --upgrade pip
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

RUN pip install flash-attn --no-build-isolation

# Установка Jupyter
RUN pip install jupyter
#
# Устанавливаем зависимости проекта
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Установка Kimi-Audio из GitHub (автоматически установит зависимости из pyproject.toml)
RUN pip install "git+https://github.com/MoonshotAI/Kimi-Audio.git"

# Копируем файлы проекта

CMD ["bash", "-c", "jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser & bash"]

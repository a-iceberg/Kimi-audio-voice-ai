FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

# Ставим базовые утилиты
RUN apt-get update && apt-get install -y \
    python3.10-dev \
    python3-pip \
    git \
    && python3.10 -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*
# Устанавливаем PyTorch nightly с CUDA 12.8
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Установка Jupyter
RUN pip install jupyter

# Устанавливаем зависимости проекта
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# установка flash attention
RUN pip install https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl


# Установка Kimi-Audio из GitHub (автоматически установит зависимости из pyproject.toml)
RUN pip install "git+https://github.com/MoonshotAI/Kimi-Audio.git"

# Копируем файлы проекта

# Jupyter
#CMD ["bash", "-c", "jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser & bash"]

#uvicorn
#CMD ["bash", "-c", "exec uvicorn app:app --host 0.0.0.0 --port 5000 --reload"]

#скрипт, который запускает и Uvicorn, и Jupyter
#RUN echo '#!/bin/bash\n\
#uvicorn app:app --host 0.0.0.0 --port 5000 --no-server-header --no-date-header &\n\
#exec jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser\n' > /start.sh && \
#    chmod +x /start.sh
#CMD ["/start.sh"]


WORKDIR /app
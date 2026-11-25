# FluxoVC — imagem para o Render (ou qualquer host com Docker). Roda em CPU.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=1 \
    FLUXOVC_VLM_BACKEND=claude

# libGL e glib: exigidos pelo opencv-python (dependência do ultralytics)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch só CPU (a versão padrão do PyPI no Linux traz CUDA e tem vários GB)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.2

COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

COPY . .

# Baixa os pesos do YOLO no build, para a primeira análise não depender da rede
RUN python -c "from ultralytics import YOLO; YOLO('pesos/yolo11n.pt')"

EXPOSE 8501
CMD streamlit run app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true

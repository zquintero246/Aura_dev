# Imagen base de PyTorch con CUDA 12.1 y cuDNN 8
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY . /workspace

CMD ["bash"]

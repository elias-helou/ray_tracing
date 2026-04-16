FROM nvidia/cuda:12.6.1-devel-ubuntu24.04
# Para placas mais antigas, você pode tentar usar uma imagem com uma versão mais antiga do CUDA, como a 11.8, por exemplo, algumas das abaixo:
# FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
# FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

WORKDIR /app

RUN apt update && \
    apt install -y python3 && \
    apt install -y python3-pip && \
    apt install -y python3-venv

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install jupyter && \
    pip install torch && \
    pip install matplotlib

COPY . /app

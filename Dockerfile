FROM nvidia/cuda:12.6.1-devel-ubuntu24.04

WORKDIR /app

COPY . /app

SHELL [ "/bin/bash", "-c" ]

RUN apt update && \
    apt install -y python3 && \
    apt install -y python3-pip && \
    apt install -y python3-venv && \
    python3 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install jupyter && \
    pip install torch && \
    pip install matplotlib && \
    make

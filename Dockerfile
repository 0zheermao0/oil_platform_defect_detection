FROM docker.io/nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and dependencies
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install opencv-python-headless ultralytics gradio

WORKDIR /root
COPY . .
COPY ./Arial.Unicode.ttf /root/.config/Ultralytics/Arial.Unicode.ttf

# Default command
CMD ["python3", "server.py"]

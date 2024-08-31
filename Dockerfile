FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    python3.9 \
    python3-pip \
    python3-venv \
    libssl-dev \
    libffi-dev \
    git \
    wget \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create a non-root user
RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$HOME/app \
    PYTHONUNBUFFERED=1 \
	GRADIO_ALLOW_FLAGGING=never \
	GRADIO_NUM_PORTS=1 \
	GRADIO_SERVER_NAME=0.0.0.0 \
	GRADIO_THEME=huggingface \
    GRADIO_SHARE=False \
	SYSTEM=spaces

# Set the environment variable to specify the GPU device
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV CUDA_VISIBLE_DEVICES=0

# Clone the RB-Modulation repository
RUN git clone https://github.com/google/RB-Modulation.git $HOME/app

# Set the working directory
WORKDIR $HOME/app

# Upgrade pip and install Gradio
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir gradio

# Copy the app.py file from the host to the container
COPY --chown=user:user app.py .

# Command to run the Gradio app
CMD ["python3", "app.py"]
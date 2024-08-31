FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    python3.9 \
    python3-pip \
    libssl-dev \
    libffi-dev \
    git \
    wget \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Clone the RB-Modulation repository
RUN git clone https://github.com/google/RB-Modulation.git $HOME/app

# Set the working directory
WORKDIR $HOME/app

# Install any required Python packages
# Uncomment and modify the following line if there's a requirements.txt file
# RUN pip install --no-cache-dir -r requirements.txt

# Install Gradio
RUN pip install --no-cache-dir gradio

# Command to run the Gradio app
CMD ["python", "app.py"]
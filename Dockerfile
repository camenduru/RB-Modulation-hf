FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
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
    SYSTEM=spaces

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
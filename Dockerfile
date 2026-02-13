FROM python:3.11-slim

WORKDIR /app

# System dependencies (needed for RDKit)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python packages
RUN pip install --upgrade pip && \
    pip install \
    numpy pandas scikit-learn matplotlib tqdm requests jupyter && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install torch-geometric && \
    pip install rdkit-pypi

# Sanity check
CMD ["python", "-c", "import torch; import torch_geometric; import rdkit; print('Docker setup OK')"]

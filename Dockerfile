FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# install uv
RUN pip install uv

# Copy requirements first (cache optimization)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL project code into image
COPY src ./src
COPY ui ./ui
COPY tools ./tools
COPY providers ./providers
COPY config ./config

# Optional: if you later add small datasets/models, copy them too
# COPY datasets ./datasets
# COPY models ./models

ENV PYTHONPATH=/app/src
ENV CHECKPOINT_DIR=/app/checkpoints
ENV CELEBA_DIR=/app/celeba
ENV MALWARE_DIR=/app/malware_bazaar_binaries
# IMPORTANT: make venv visible globally
ENV PATH="/app/.venv/bin:$PATH"

# create venv + install deps
RUN uv sync

CMD ["complai", "eval"]

CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /app



# System dependencies


RUN apt-get update && apt-get install -y \
    python3 python3-pip git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*


# 1. Install tools to manage repositories
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository ppa:openjdk-r/ppa -y

# 2. Install OpenJDK 8
RUN apt-get update && apt-get install -y \
    openjdk-8-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# 3. Verify it's the active version
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# 4. (Optional) Check version during build to be sure
RUN java -version

# install uv
RUN python3 -m pip install --no-cache-dir uv

# Copy ONLY dependency files first (better caching)
COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache \
    uv sync --frozen

# We just initialize Spice(). It downloads models automatically if they are missing.
RUN .venv/bin/python -c "from pycocoevalcap.spice.spice import Spice; Spice()"




# --- NEW: Fix SPICE and permissions ---
# This ensures the user has permission to write to the spice tmp folders
# and pre-downloads NLTK data required by many metrics.
RUN .venv/bin/python -m nltk.downloader punkt
RUN chmod -R 777 /app/.venv/lib/python*/site-packages/pycocoevalcap/spice/tmp || true

# Copy ALL project code into image
COPY src ./src
COPY ui ./ui
COPY tools ./tools
COPY providers ./providers
COPY config ./config

#RUN mkdir -p /app/logs



# Optional: if you later add small datasets/models, copy them too
# COPY datasets ./datasets
# COPY models ./models

ENV PYTHONPATH=/app:/app/src
ENV CHECKPOINT_DIR=/app/checkpoints/
ENV CELEBA_DIR=/app/celeba/
ENV MALWARE_DIR=/app/malware_bazaar_binaries/
ENV HF_HOME=/app/cache/hf
ENV COMPLAI_CACHE=/app/complai_cache

# IMPORTANT: make venv visible globally
ENV PATH="/app/.venv/bin:$PATH"


CMD ["streamlit", "run", "ui/app.py", "--server.port=8502", "--server.address=0.0.0.0"]
#CMD ["/app/start.sh", "streamlit", "run", "ui/app.py", "--server.port=8503", "--server.address=0.0.0.0"]
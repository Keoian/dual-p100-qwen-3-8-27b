# Container for serving (and building) the tyler-port branch on 2x Tesla P100.
# CUDA 12.x ONLY: CUDA 13 cannot target sm_60.
ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential ccache curl jq ca-certificates \
        libcurl4-openssl-dev python3 python3-pip python3-venv \
        less vim htop tmux pciutils bc sudo \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir "huggingface_hub[cli]" numpy safetensors

# Non-root user matching Unraid's nobody:users (99:100) so files on the share stay editable over SMB.
RUN groupadd -o -g 100 users2 && useradd -m -u 99 -g 100 -s /bin/bash dev \
    && echo "dev ALL=(ALL) NOPASSWD: /usr/bin/nvidia-smi" > /etc/sudoers.d/dev

USER dev
ENV CCACHE_DIR=/work/ccache \
    CCACHE_MAXSIZE=15G \
    HF_HOME=/work/hf-cache
WORKDIR /work
CMD ["sleep", "infinity"]

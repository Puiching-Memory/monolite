FROM python:3

WORKDIR /usr/src/app

COPY ./ ./

RUN ./chsrc set pip \
    # install python packages
    && pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -r requirements.txt 
    # install cuda toolkit
    # && apt-key del 7fa2af80 \
    # && wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin\
    # && mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
    # && wget https://developer.download.nvidia.com/compute/cuda/12.6.1/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.1-1_amd64.deb \
    # && dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.1-1_amd64.deb \
    # && cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/ \
    # && apt-get update \
    # && apt-get -y install cuda-toolkit-12-6

#CMD ["python","keep_container_life.py"]

# useage:
# set DOCKER_BUILDKIT=0
# docker build -t monolite . 
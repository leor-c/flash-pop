FROM nvcr.io/nvidia/pytorch:25.02-py3
WORKDIR /flash_pop
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install einops
RUN pip install loguru

# https://github.com/tile-ai/tilelang/issues/147 :
RUN wget http://security.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb && apt-get install ./libtinfo5_6.3-2ubuntu0.1_amd64.deb && rm libtinfo5_6.3-2ubuntu0.1_amd64.deb # && pip install https://github.com/tile-ai/tilelang-nightly/releases/download/0.1.1%2B16b919b/tilelang-0.1.1+16b919b.ubuntu.18.4.cu121-cp312-cp312-linux_x86_64.whl
# RUN apt-get update && apt-get install -y python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
 RUN pip install tilelang
RUN #pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu121/





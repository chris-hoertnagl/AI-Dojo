FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# expose
EXPOSE 8080

# install pip
RUN apt-get update && apt-get install -y python3-pip

# install git
RUN apt-get install -y git

## install ollama
RUN apt-get install -y curl
RUN curl -fsSL https://ollama.com/install.sh | sh

## install unstructered dependencies
RUN apt-get install -y ffmpeg
RUN apt-get install -y libgl1 
RUN apt-get install -y libsm6 
RUN apt-get install -y libxext6
RUN apt-get install -y poppler-utils
RUN apt-get install -y tesseract-ocr

## install llama.cpp
# RUN git clone --depth=1 https://github.com/ggerganov/llama.cpp.git
# RUN cd /llama.cpp && pip install -r requirements.txt && make

# update pip
RUN pip3 install --upgrade pip

COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r ./requirements.txt
# RUN pip3 install flash-attn --no-build-isolation

WORKDIR /ai_dev

# Can use '& disown' or to run ollama in background, or create entrypoint script to run more stuff
CMD ["ollama", "serve"]

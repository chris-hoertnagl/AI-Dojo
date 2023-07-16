FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

# expose
EXPOSE 8080

# set working directory
WORKDIR /src

# install pip
RUN apt-get update && apt-get install -y python3-pip

# install git
RUN apt-get install -y git

# update pip
RUN pip3 install --upgrade pip

RUN pip install -q -U bitsandbytes
RUN pip install -q -U git+https://github.com/huggingface/transformers.git
RUN pip install -q -U git+https://github.com/huggingface/peft.git
RUN pip install -q -U git+https://github.com/huggingface/accelerate.git
RUN pip install -q -U einops
RUN pip install -q -U safetensors
RUN pip install -q -U torch
RUN pip install -q -U xformers
RUN pip install -q -U scipy
RUN pip install -q -U google-api-python-client
RUN pip install -q -U datasets

CMD ["sleep", "infinity"]

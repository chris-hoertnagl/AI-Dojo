FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# expose
EXPOSE 8080

# install pip
RUN apt-get update && apt-get install -y python3-pip

# install git
RUN apt-get install -y git

# update pip
RUN pip3 install --upgrade pip

COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r ./requirements.txt
RUN pip3 install flash-attn --no-build-isolation

ENV HF_HOME=/llm/cache
ENV HF_TOKEN="<YOUR HF TOKEN>"
# set working directory
WORKDIR /llm

CMD ["sleep", "infinity"]

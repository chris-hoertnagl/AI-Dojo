# LLM fine-tuning

# An easy first experiment
In the folder beginner_example you can find a dataset and colab notebook to run a first simple LLM fine-tuning experiment. No setup required, except for knowing how to use [google colab](https://colab.google/).

# Improving the setup

In this tutorial we will learn how to teach llama3.1 that its name is <INSERT YOUR NAME HERE> through fine-tuning. We will use QLoRA to make the fine-tuning possible on a consumer GPU such as a Nvidia 3080. In addition we will convert it to a format so that it can be inferenced locally with a CPU.

## CURRENT STATUS:
Trying to clean code up by building one tuning with transformers & sft and one with lightning. Pipeline should work with sft for now. Trying to integrate all installs to Docker.

## Prerequsites
- Know how to use Python, Notebooks, Docker & VS Code
- Install [ollama](https://ollama.com/) on your local machine

## Setup
- Go to `Dockerfile` and exchange `<YOUR HF TOKEN>` with your huggingface token
- Run `docker-compose up --build -d`
- Attach VS Code to the container
- Run `python3 download_model.py` and start downloading the model (as this takes a while)
- Go to `/data/train.jsonl` or `/data/train.json` and replace ``<NAME>`` with your name

## To start training run
``` bash
accelerate launch --config_file "deepspeed_config_z3_qlora.yaml" tune_sft.py
```
## Inference
- Go through `to_ollama.ipynb` to convert and test tuned model


## Setup Ollama Conversion

NOTE: Obviously adjust models paths accordingly and then run the follwing commands to convert to an ollama usable model:

- git clone --depth=1 https://github.com/ggerganov/llama.cpp.git
- cd ./llama.cpp
- sed -i 's|MK_LDFLAGS   += -lcuda|MK_LDFLAGS   += -L/usr/local/nvidia/lib64 -lcuda|' Makefile
- LLAMA_CUDA=1 conda run -n base make -j > /dev/null
- python convert-hf-to-gguf.py /home/ubuntu/dev/Bachelor-Thesis/QLoRA_training/model/merged/ --outfile /home/ubuntu/dev/Bachelor-Thesis/QLoRA_training/model/gguf/sascha-model.gguf --outtype f16
- ./llama-quantize /home/ubuntu/dev/Bachelor-Thesis/QLoRA_training/model/gguf/sascha-model.gguf /home/ubuntu/dev/Bachelor-Thesis/QLoRA_training/model/sascha-model-Q4_K_M.gguf Q4_K_M
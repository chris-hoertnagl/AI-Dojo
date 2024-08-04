# LLM fine-tuning

In this tutorial we will learn how to teach llama3.1 that its name is <INSERT YOUR NAME HERE> through fine-tuning. We will use QLoRA to make the fine-tuning possible on a consumer GPU such as a Nvidia 3080. In addition we will convert it to a format so that it can be inferenced locally with a CPU.

## Prerequsites
- Know how to use Python, Notebooks, Docker & VS Code
- Install [ollama](https://ollama.com/) on your local machine
- Go to `/data/train.jsonl` and replace ``<NAME>`` with your name

## Setup Ollama Conversion

NOTE: Obviously adjust models paths accordingly and then run the follwing commands to convert to an ollama usable model:

- git clone --depth=1 https://github.com/ggerganov/llama.cpp.git
- cd ./llama.cpp
- sed -i 's|MK_LDFLAGS   += -lcuda|MK_LDFLAGS   += -L/usr/local/nvidia/lib64 -lcuda|' Makefile
- LLAMA_CUDA=1 conda run -n base make -j > /dev/null
- python convert-hf-to-gguf.py /home/ubuntu/dev/Bachelor-Thesis/QLoRA_training/model/merged/ --outfile /home/ubuntu/dev/Bachelor-Thesis/QLoRA_training/model/gguf/sascha-model.gguf --outtype f16
- ./llama-quantize /home/ubuntu/dev/Bachelor-Thesis/QLoRA_training/model/gguf/sascha-model.gguf /home/ubuntu/dev/Bachelor-Thesis/QLoRA_training/model/sascha-model-Q4_K_M.gguf Q4_K_M
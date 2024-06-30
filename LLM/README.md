## Setup Ollama Conversion

NOTE: Obviously adjust models paths accordingly and then run the follwing commands to convert to an ollama usable model:

- git clone --depth=1 https://github.com/ggerganov/llama.cpp.git
- cd ./llama.cpp
- sed -i 's|MK_LDFLAGS   += -lcuda|MK_LDFLAGS   += -L/usr/local/nvidia/lib64 -lcuda|' Makefile
- LLAMA_CUDA=1 conda run -n base make -j > /dev/null
- python convert-hf-to-gguf.py /home/ubuntu/dev/Bachelor-Thesis/QLoRA_training/model/merged/ --outfile /home/ubuntu/dev/Bachelor-Thesis/QLoRA_training/model/gguf/sascha-model.gguf --outtype f16
- ./llama-quantize /home/ubuntu/dev/Bachelor-Thesis/QLoRA_training/model/gguf/sascha-model.gguf /home/ubuntu/dev/Bachelor-Thesis/QLoRA_training/model/sascha-model-Q4_K_M.gguf Q4_K_M
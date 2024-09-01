# AI-Dojo

The purpose of this repository is to provide tangible examples for advanced AI/ML concepts. This is not a replacement for proper education nor is it complete. 
A lot of the code in here is still work in progress...
Current concepts available:

- Deep Learning (DL)
	- A notebook that covers the basics to get familiar with the concept
- Large Language Models (LLM)
	- A notebook for the basics of Natural Language Processing needed for LLMs
	- 2 examples to demonstrate how LLMs can be fine-tuned to gain new knowledge / skills
	- A example to show the concept of RAG and why it is useful
- Large Multimodal Models (LMM)
	- some first model usage examples
	- this is still largely work in progress (Updating streamlit app to work for 3 use cases (text chat, file&web chat, image&automation chat))
- Reinforcement Learning (RL)
	- An implementation of my favourite algorithm MCTS demonstrating its capability to play snake


## Prerequsites
- Know how to use Python, Notebooks, Docker & VS Code

## Setup
- Go to `.env.template` and exchange rename to `.env` and exchange `<YOUR TOKEN>` with your huggingface token
- Run `docker-compose up --build -d`
- Attach VS Code to the container

## Fixing Docker Windows & WSL Cuda Errors

### For WSL2
1. Get WSL 2
2. Install an distro "wsl --install Ubuntu-22.04"
3. Connect to it "wsl"
3.1 Change the default user to root
	sudo vi /etc/wsl.conf
	"
	[user]
	default=root
	"
3.2 Restart WSL
4. Install docker 
	sudo apt update
	sudo apt install docker.io
5. Install docker compose https://docs.docker.com/compose/install/linux/
	sudo apt-get install docker-compose
6. Install Cuda according to https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
7. Set CUDA Home https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions:
``export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}`` and export ``LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:{LD_LIBRARY_PATH}}``
8. Test if it worked "nvidia-smi" and "nvcc -V"
9. Install Container Toolkit https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.15.0/install-guide.html
10. Pull and run any Cuda container "docker run --rm --gpus all nvidia/cuda:12.5.0-devel-ubuntu22.04 sleep infinity"
11. Outside of WSL Open VS Code
12. Install the following Extentions "Docker", "Dev Container" and "Remote Explorer"
13. In VS Code connect to the wsl instance with "Remote Explorer" 
14. Attach to the container
15. Install python3 "apt update" & "apt install python3" & "apt install python3-pip"
16. Install torch https://pytorch.org/get-started/locally/
17. Check if Cuda works by running "python3" & "import torch" & "torch.cuda.is_available()"
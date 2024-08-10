WSL2
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

#  WSL + RTX50xx：Route B 安裝指南（tf-nightly + cuDNN 9 + CUDA 12.6.3+）
本文件是 Route B 的最終版流程：在 Windows 11 + WSL2 (Ubuntu 22.04) 下，用 tf-nightly（開發版 TensorFlow）搭配 cuDNN 9 與 CUDA 12.6.3+ 工具鏈，以支援 RTX 50 系列（Compute Capability 12.0a）。
##  環境需求（建議）
Windows 11（已安裝最新 NVIDIA 顯示卡驅動）
WSL2：Ubuntu 22.04
Python 3.10（WSL 內）
顯示卡：RTX 50xx（Compute Capability 12.0a）

##  1) 安裝 VS Code
到官方網站安裝 Visual Studio Code。稍後會在 VS Code 內裝 WSL 擴充套件。

##  2) 安裝 NVIDIA 顯卡驅動（CUDA 12.6）
到官方下載頁安裝（Windows 端）：
https://developer.nvidia.com/cuda-12-6-0-download-archive
只需驅動；CUDA Toolkit 本體在 WSL 內由 Python wheels 提供（nvidia-*-cu12）。

##  3) 安裝 / 更新 WSL2（Windows PowerShell）
wsl --install
wsl --set-default-version 2
wsl --update
之後重新啟動 Windows。

##  4) 進入 Ubuntu，更新系統並安裝 Python
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.10 python3.10-venv python3-pip

##  5) 建立虛擬環境並安裝基礎套件（在 Ubuntu/WSL 內）
python3.10 -m venv ~/tfenv
source ~/tfenv/bin/activate

pip install --upgrade pip
pip install numpy==1.26.4 opencv-python==4.10.0.84 \
            matplotlib==3.8.4 pandas==2.2.2 scikit-learn==1.4.2

##  6) 設定動態連結庫路徑（WSL 固定做一次）
echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$HOME/tfenv/lib/python3.10/site-packages/nvidia/cublas/lib:$HOME/tfenv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$HOME/tfenv/lib/python3.10/site-packages/nvidia/cudnn/lib:$HOME/tfenv/lib/python3.10/site-packages/nvidia/cusolver/lib:$HOME/tfenv/lib/python3.10/site-packages/nvidia/cusparse/lib:$HOME/tfenv/lib/python3.10/site-packages/nvidia/cufft/lib:$HOME/tfenv/lib/python3.10/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

##  7) VS Code 安裝 WSL 擴充套件
開啟 VS Code → Extensions → 安裝 “WSL”。
開啟 WSL: Ubuntu 視窗，稍後選擇 Python Interpreter。

##  安裝 NVIDIA APT keyring 與 cuDNN 9（WSL 內）
sudo apt-get update
sudo apt-get install -y wget gnupg
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

sudo apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12

##  9) 關閉並重啟 WSL（於 Windows PowerShell）
wsl --shutdown
然後重新開啟 Ubuntu（或從 VS Code 重新連線 WSL）。

##  10) 安裝 tf-nightly 與 CUDA 12.6.3+ 工具鏈（WSL 內、啟用 venv）
source ~/tfenv/bin/activate

# 安裝 tf-nightly
pip install --upgrade --pre tf-nightly

# 安裝/更新 CUDA/cuDNN 相關 wheels（需 12.6.3+）
pip install -U "nvidia-cuda-nvcc-cu12>=12.6.3" \
               "nvidia-nvjitlink-cu12>=12.6" \
               "nvidia-cublas-cu12>=12.6" \
               "nvidia-cudnn-cu12>=9.1" \
               "nvidia-cufft-cu12>=11.2" \
               "nvidia-cusolver-cu12>=11.6" \
               "nvidia-cusparse-cu12>=12.5" \
               "nvidia-curand-cu12>=10.3"


##  12) 在 VS Code 選擇 Python Interpreter
Ctrl+Shift+P → Python: Select Interpreter
選擇：/home/<你的帳號>/tfenv/bin/python

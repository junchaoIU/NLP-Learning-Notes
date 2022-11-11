# PyTorch GPU安装
https://zhuanlan.zhihu.com/p/106133822
## CUDA
1. 通过```nvidia-smi``` 查看Driver Version（显卡的驱动版本）
- 在 https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html 查看可以接受的版本
- 选择一个适合的版本，最好不要太高，下载（win10和win11通用）
2. 傻瓜式安装
- 安装结束后，进入目录下执行 ```nvcc --version```，可以看见版本信息即安装成功
## cuDNN（集成在cudatoolkit里面了）
- 进入 https://developer.nvidia.com/rdp/cudnn-download， 选择与CUDA对应的版本下载
- pytorch不用装cudnn，集成在cudatoolkit里面了

## PyTorch
- 推荐采用conda虚拟环境安装
    - 创建虚拟环境 ```conda create -n pytorch python==3.8```
    - 激活虚拟环境 ```conda activate pytorch```

- 在PyTorch官网（https://pytorch.org/get-started/ ）根据对应的配置获得加载命令
    - 建议用较低版本的
    - CUDA11.3: ```conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge```
- 验证，输出为True证明GPU版本安装成功
    ```
    import torch 
    torch.__version__
    print(torch.cuda.is_available())
    ```




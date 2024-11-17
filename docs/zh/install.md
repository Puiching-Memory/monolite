# Environment环境

### 虚拟环境

依据此[pytorch_issue](https://github.com/pytorch/pytorch/issues/138506)中的讨论，我们将虚拟环境迁移至[miniforge](https://github.com/conda-forge/miniforge)

```
conda create -n monolite python=3.12
```

### 前置组件

| 系统    | 前置组件           | 下载URL                                                                                             | 备注                 |
| ------- | ------------------ | --------------------------------------------------------------------------------------------------- | -------------------- |
| windows | Visual Studio 2022 | [download](https://visualstudio.microsoft.com/zh-hans/vs/)                                             | 注意不同版本间的冲突 |
| windows | Cmake              | [download](https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-windows-x86_64.msi) | 3.30.5               |
| windows | MSbuild            | 通过VS2022下载                                                                                      |                      |
| windows | MSVC               | 通过VS2022下载                                                                                      |                      |

### pip

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### ~~Docker~~（暂不可用）

```console
set DOCKER_BUILDKIT=0
docker build -t monolite .
docker run -d  --privileged=true --net host --name {any_name} --shm-size 4G --ulimit memlock=-1 --gpus=all -it -v C:\:/windows/ monolite:latest /bin/bash
```

### Docker mirror

https://github.com/DaoCloud/public-image-mirror

### Dataset数据集

#### Kitti

TODO

# Pre-training model zoo预训练模型

| Model             | URL                  | Trainning log                                                                                     |
| ----------------- | -------------------- | ------------------------------------------------------------------------------------------------- |
| MonoLite_Baseline | [百度网盘][谷歌网盘] | [https://swanlab.cn/@Sail2Dream/monolite/overview](https://swanlab.cn/@Sail2Dream/monolite/overview) |

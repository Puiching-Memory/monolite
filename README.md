# MonoLite

Explore lightweight practices for monocular 3D inspection

探索单目3D检测的轻量级实践

[中文][English]

# Abstract摘要

Note that we are an engineering project, the code will be updated synchronously, currently in the early stages of the project, if you want to help, please check out our projects!

注意，我们是工程化项目，代码会同步更新，目前处于项目的早期阶段，如果你想提供帮助，请查阅我们的projects!

![multimedia\model_map.webp](https://raw.githubusercontent.com/Puiching-Memory/monolite/refs/heads/main/multimedia/model_map.webp "model_map")

# Activity活动

![Alt](https://repobeats.axiom.co/api/embed/ec6e11b1a493733d51588ad5d740376b07651132.svg "Repobeats analytics image")

# Design架构设计

我们将神经网络训练中最重要的部件分离了出来，而其他针对模型的操作，如训练/测试/评估/导出，则作为一种任务文件被不同的实验共用。

# Experiment实验

| Model    | Dataset | info |
| -------- | ------- | ---- |
| MonoLite | Kitti   |      |

### Torch info

| Model    | Input size (MB) | Params size (MB) | Total params | Total mult-adds |
| -------- | --------------- | ---------------- | ------------ | --------------- |
| MonoLite | 94.37           | 109.04           | 27,260,609   | 903.20          |

### 性能测试

*We used the BN layer, so a value of >=2 is recommended

| Task  | GPU(GB) | RAM(GB) | Batch size | Speed(it/s) |
| ----- | ------- | ------- | ---------- | ----------- |
| train | 1.2     | 2.2     | 1          |             |
| train | 1.8     | 2.2     | 2          |             |
| eval  | 2.2     | 2.0     | 1          | 43          |

# Environment环境

### 虚拟环境

依据此[pytorch_issue](https://github.com/pytorch/pytorch/issues/138506)中的讨论，我们将虚拟环境迁移至[miniforge](https://github.com/conda-forge/miniforge)

```
conda create -n monolite python=3.12
```

### 前置组件

| 系统    | 组件               | 下载URL                                                                                             | 备注                   |
| ------- | ------------------ | --------------------------------------------------------------------------------------------------- | ---------------------- |
| windows | Visual Studio 2022 | [download](https://visualstudio.microsoft.com/zh-hans/vs/)                                             | 注意不同版本间的冲突   |
| windows | Cmake              | [download](https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-windows-x86_64.msi) | 3.30.5                 |
| windows | MSbuild            | 通过VS2022下载                                                                                      |                        |
| windows | MSVC               | 通过VS2022下载                                                                                      | 手动添加至环境变量PATH |

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

# Inference推理

```
python tools\detect.py --cfg C:\workspace\github\monolite\experiment\monolite_YOLO11_centernet
```

# Train训练

```
python tools\train.py --cfg C:\workspace\github\monolite\experiment\monolite_YOLO11_centernet
```

### Train with your own Dataset自定义数据集训练

TODO

# Eval评估

TODO

# Export导出

### ONNX

```
python tools\export_onnx.py --cfg C:\workspace\github\monolite\experiment\monolite_YOLO11_centernet
```

### TensorRT

##### torchscript

```
python tools\export_ts.py --cfg C:\workspace\github\monolite\experiment\monolite_YOLO11_centernet
```

##### exported_program

```
python tools\export_ep.py --cfg C:\workspace\github\monolite\experiment\monolite_YOLO11_centernet
```

### Torch_JIT

```
python tools\export_pt.py --cfg C:\workspace\github\monolite\experiment\monolite_YOLO11_centernet
```

# Confirm致谢

我们衷心感谢所有为这个神经网络开源项目做出贡献的个人和组织。特别感谢以下贡献者：

| type      | name             | url                                                           | title                                                                    |
| --------- | ---------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------ |
| CVPR 2021 | MonoDLE          | [monodle github](https://github.com/xinzhuma/monodle)            | Delving into Localization Errors for Monocular 3D Object Detection       |
| 3DV 2024  | MonoLSS          | [monolss github](https://github.com/Traffic-X/MonoLSS/)          | Learnable Sample Selection For Monocular 3D Detection                    |
|           | TTFNet           |                                                               |                                                                          |
| community | kitti_object_vis | [kitti_vis github](https://github.com/kuixu/kitti_object_vis)    | KITTI Object Visualization (Birdview, Volumetric LiDar point cloud )     |
| community | mmdet3d          | [mmdet3d github](https://github.com/open-mmlab/mmdetection3d)    | OpenMMLab's next-generation platform for general 3D object detection.    |
| community | ultralytics      | [ultralytics github](https://github.com/ultralytics/ultralytics) | YOLOv8/v11+v9/v10                                                        |
| community | netron           | [netron web](https://netron.app/)                                | Visualizer for neural network, deep learning and machine learning models |
| community | mkdocs-material  | [mkdocs github](https://github.com/squidfunk/mkdocs-material)    | Documentation that simply works                                          |

正是这种协作和共享的精神，让开源项目得以蓬勃发展，并为科技进步做出贡献。我们期待未来有更多的合作和创新，共同推动人工智能领域的发展。

再次感谢每一位支持者，你们的贡献是无价的。

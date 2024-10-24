# MonoLite

Explore lightweight practices for monocular 3D inspection

探索单目3D检测的轻量级实践

# Abstract

Note that we are an engineering project, the code will be updated synchronously, currently in the early stages of the project, if you want to help, please check out our projects!

注意，我们是工程化项目，代码会同步更新，目前处于项目的早期阶段，如果你想提供帮助，请查阅我们的projects!

![multimedia\model_map.webp](https://raw.githubusercontent.com/Puiching-Memory/monolite/refs/heads/main/multimedia/model_map.webp "model_map")

# Activity

![Alt](https://repobeats.axiom.co/api/embed/ec6e11b1a493733d51588ad5d740376b07651132.svg "Repobeats analytics image")

# 架构设计

我们将神经网络训练中最重要的部件分离了出来，而其他针对模型的操作，如训练/测试/评估/导出，则作为一种任务文件被不同的实验共用。

# Experiment

TODO

| Model    | Dataset | info |
| -------- | ------- | ---- |
| MonoLite | Kitti   |      |
|          |         |      |

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

# Environment

安装torch==2.5.0(cuda==12.4)

```
pip install -r requirements.txt
```

### ~~Docker~~（暂不可用）

```console
set DOCKER_BUILDKIT=0
docker build -t monolite .
docker run -d  --privileged=true --net host --name zk --shm-size 4G --ulimit memlock=-1 --gpus=all -it -v C:\:/windows/ monolite:latest /bin/bash
```

### Docker mirror

https://github.com/DaoCloud/public-image-mirror

### conda

```
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
```

# Pre-training model zoo

| Model    | URL |
| -------- | --- |
| MonoLite |     |
|          |     |

# Inference

```
python tools\detect.py --cfg C:\workspace\github\monolite\experiment\monolite_YOLO11_centernet
```

# Train

```
python tools\train.py --cfg C:\workspace\github\monolite\experiment\monolite_YOLO11_centernet
```

### Train with your own Dataset

TODO

# Eval

building...

# Export

### ONNX

```
python tools\export.py --cfg C:\workspace\github\monolite\experiment\monolite_YOLO11_centernet
```

### TensorRT

# Confirm

MonoDLE

MonoLSS

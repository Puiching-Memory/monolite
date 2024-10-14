# MonoLite

Explore lightweight practices for monocular 3D inspection

探索单目3D检测的轻量级实践

# Abstract

Note that we are an engineering project, the code will be updated synchronously, currently in the early stages of the project, if you want to help, please check out our projects!

注意，我们是工程化项目，代码会同步更新，目前处于项目的早期阶段，如果你想提供帮助，请查阅我们的projects!

![multimedia\model_map.webp](https://raw.githubusercontent.com/Puiching-Memory/monolite/refs/heads/main/multimedia/model_map.webp "model_map")

# Activity

![Alt](https://repobeats.axiom.co/api/embed/ec6e11b1a493733d51588ad5d740376b07651132.svg "Repobeats analytics image")

# Experiment

TODO

| Modle | Dataset | mAP50 | mAP75 | method | info |
| ----- | ------- | ----- | ----- | ------ | ---- |
|       |         |       |       |        |      |

### 最低系统配置

*We used the BN layer, so a value of >=2 is recommended

| GPU   | RAM   | Batch size |
| ----- | ----- | ---------- |
| 1.2GB | 2.2GB | 1          |
| 1.8GB | 2.2GB | 2          |

# Environment

安装torch>=2.0.0

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

# Train

```
python tools\train.py --cfg C:\workspace\github\monolite\experiment\monolite_YOLO11_centernet
```

### Train with your own Dataset

TODO

# Eval

building...

# Inference

TODO

# Export

### ONNX

### TensorRT

# Confirm

MonoDLE

MonoLSS

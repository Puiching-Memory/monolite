# monolite

Explore lightweight practices for monocular 3D inspection

# Activate

![Alt](https://repobeats.axiom.co/api/embed/ec6e11b1a493733d51588ad5d740376b07651132.svg "Repobeats analytics image")

# Abstract

TODO

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

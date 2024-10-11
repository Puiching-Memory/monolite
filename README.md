# monolite

Explore lightweight practices for monocular 3D inspection

# Abstract

TODO

# Experiment

TODO

| col1 | col2 | col3 |
| ---- | ---- | ---- |
|      |      |      |
|      |      |      |

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

# Projects Plan

* [X] AMP
* [X] modle info(torch info)
* [X] loguru
* [X] torch2 support
* [ ] torch.compile support
* [ ] swanlab
* [ ] Anchor3DLine
* [X] MixUP3D(MonoLSS)
* [ ] memory format (last channel)
* [ ] 重写kitti数据加载器
* [ ] 将loss计算方法解耦至外部配置

# Thanks

MonoDLE

MonoLSS

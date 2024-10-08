# monolite

# Functional improvement

* [X] AMP
* [X] modle info(torch info)
* [X] loguru
* [ ] more optimitzer
* [X] torch2 support
* [ ] torch.compile support
* [ ] swanlab
* [ ] Anchor3DLine
* [ ] MixUP3D(MonoLSS)
* [ ] memory format (last channel)

# Environment

```
pip install -r requirements.txt
```

# Train

```
python tools\train.py --cfg C:\workspace\github\monolite\experiment\monolite_YOLO11_centernet
```

# Eval

building...

# env

```console
set DOCKER_BUILDKIT=0
docker build -t monolite .
docker run -d  --privileged=true --net host --name zk --shm-size 4G --ulimit memlock=-1 --gpus=all -it -v C:\:/windows/ monolite:latest /bin/bash
```

### mirror

https://github.com/DaoCloud/public-image-mirror

### conda

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# monolite

# Functional improvement

* [X] AMP
* [X] modle info(torch info)
* [X] loguru
* [ ] more optimitzer
* [X] torch2 support
* [X] torch.compile support
* [ ] torch lighting
* [X] swanlab
* [ ] Anchor3DLine
* [ ] MixUP3D(MonoLSS)
* [ ] memory format (last channel)

# Environment

```
pip install -r requirements.txt
```

# Eval

building...

# env

```console
docker build -t monolite . --driver-opt env.BUILDKIT_STEP_LOG_MAX_SIZE=50000000
docker run -d  --privileged=true --net host --name zk --shm-size 4G --ulimit memlock=-1 --gpus=all -it -v C:\:/windows/ monolite:latest /bin/bash
```

### mirror

https://github.com/DaoCloud/public-image-mirror

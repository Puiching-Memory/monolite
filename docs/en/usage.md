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

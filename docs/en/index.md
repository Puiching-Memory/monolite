# MonoLite[English]

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

# Confirm致谢

我们衷心感谢所有为这个神经网络开源项目做出贡献的个人和组织。特别感谢以下贡献者：

| type      | name        | url                                                           | title                                                                    |
| --------- | ----------- | ------------------------------------------------------------- | ------------------------------------------------------------------------ |
| CVPR 2021 | MonoDLE     | [monodle github](https://github.com/xinzhuma/monodle)            | Delving into Localization Errors for Monocular 3D Object Detection       |
| 3DV 2024  | MonoLSS     | [monolss github](https://github.com/Traffic-X/MonoLSS/)          | Learnable Sample Selection For Monocular 3D Detection                    |
|           | TTFNet      |                                                               |                                                                          |
| community | ultralytics | [ultralytics github](https://github.com/ultralytics/ultralytics) | YOLOv8/v11+v9/v10                                                        |
| community | netron      | [netron web](https://netron.app/)                                | Visualizer for neural network, deep learning and machine learning models |

正是这种协作和共享的精神，让开源项目得以蓬勃发展，并为科技进步做出贡献。我们期待未来有更多的合作和创新，共同推动人工智能领域的发展。

再次感谢每一位支持者，你们的贡献是无价的。

---

以下是评论区,如果不能正常显示请报告issue.

<script src="https://giscus.app/client.js"
        data-repo="Puiching-Memory/monolite"
        data-repo-id="R_kgDOM5JIxw"
        data-category="Announcements"
        data-category-id="DIC_kwDOM5JIx84CjQa5"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="bottom"
        data-theme="preferred_color_scheme"
        data-lang="zh-CN"
        data-loading="lazy"
        crossorigin="anonymous"
        async>
</script>

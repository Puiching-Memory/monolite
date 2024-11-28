# 在windows平台上启用torch.complie功能

关键仓库：https://github.com/woct0rdho/triton-windows

# 环境要求

需要安装MSVC

# 头文件

以下为临时方案：

将

```
C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include
C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared
C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt
```

复制到

```
C:\Users\11386\.conda\envs\monolite\Lib\site-packages\torch\include
```

# LIB

将

```
C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\lib\onecore\x64
C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64
C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64
```

复制到

```
C:\Users\11386\.conda\envs\monolite\Lib\site-packages\torch\lib
```

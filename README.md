由于dgl库不支持 5090，所以改为pyg库




# install python env




```bash
# 创建环境
conda create -n sn-hrl python=3.10 -y

# 设置清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 pytorch 2.70 cuda12.8，因为5090最低要求2.7
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 安装pyg 图神经网络库
pip install torch_geometric

# 安装 PyG 可选加速扩展，必须匹配 torch-2.7.0+cu128
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.7.0+cu128.html \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 sb3 Rl framework 
pip install stable-baselines3

# 安装 sterperf 卫星网络仿真工具
python -m pip install -e ./StarPerf_Simulator


pip install -r requirements.txt
```

<!-- dgl+pytorch install Deprecate
- pytorch 2.4.0, diff cuda version
```python 
# ROCM 6.1 (Linux only)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/rocm6.1
# CUDA 11.8
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
# CPU only
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu

```

- dgl GNN library install 
```python
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html

pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu118/repo.html
``` -->



## 安装说明
pip install .

安装“快照版”。把当前代码打包后装进环境。
之后你改源码，环境里的包不会自动更新，要重新安装。
更像部署/发布用法。

pip install -e .

安装“可编辑版”（editable）。环境里链接到你的源码目录。
改源码立即生效，不用反复重装。
更像开发/调试用法。
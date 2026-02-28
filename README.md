
# install python env

```bash
pip install -r requirements.txt
```

dgl+pytorch install
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
```

sb3
```
pip install stable-baselines3
```


安装 starperf库
```bash
python -m pip install -e ./StarPerf_Simulator
```
pip install .

安装“快照版”。把当前代码打包后装进环境。
之后你改源码，环境里的包不会自动更新，要重新安装。
更像部署/发布用法。
pip install -e .

安装“可编辑版”（editable）。环境里链接到你的源码目录。
改源码立即生效，不用反复重装。
更像开发/调试用法。
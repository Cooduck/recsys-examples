# 在容器内逐条执行 Dockerfile 命令

可以。先基于**同一基础镜像**起一个交互式容器，再按顺序执行 Dockerfile 里的命令，便于调试、重试和观察。

## 1. 启动环境

在项目根目录（`recsys-examples`）下执行：

```powershell
# 拉取基础镜像（若还没有）
docker pull nvcr.io/nvidia/pytorch:25.06-py3

# 挂载当前项目目录，便于后续 COPY 的内容在容器内可用；交互式进入 bash
docker run -it --platform linux/amd64 -v "${PWD}:/workspace/recsys-examples:ro" nvcr.io/nvidia/pytorch:25.06-py3 bash
```

进入容器后，工作目录可先设为与 Dockerfile 一致：

```bash
cd /workspace/deps
```

（若没有该目录可先 `mkdir -p /workspace/deps && cd /workspace/deps`）

---

## 2. 按顺序执行以下命令块

下面每一块对应 Dockerfile 里的一个 RUN（或等效多行）。**建议每执行完一块就 `docker commit` 一次**（在**宿主机**另开终端执行），这样某一步失败只需从该步重试，不用从头来。

### 步骤 1：Triton 相关（可选，不做 Triton 构建可跳过）

```bash
# 仅当需要 TRITONSERVER_BUILD=1 时执行
# export TRITONSERVER_BUILD=1
# ln /bin/python3 /bin/python 2>/dev/null; apt-get update -y --fix-missing && apt-get install -y cmake patchelf
# pip3 install pandas rich cloudpickle psutil && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

### 步骤 2：libnvidia-ml 符号链接（amd64 时 ARCH=x86_64）

```bash
ARCH=x86_64
rm -rf /usr/lib/${ARCH}-linux-gnu/libnvidia-ml.so.1
ln -s /usr/local/cuda-12.9/targets/${ARCH}-linux/lib/stubs/libnvidia-ml.so /usr/lib/${ARCH}-linux-gnu/libnvidia-ml.so.1
```

### 步骤 3：Megatron-LM

```bash
cd /workspace/deps
git clone -b core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git megatron-lm
pip install --no-deps -e ./megatron-lm
```

### 步骤 4：torchx、gin-config 等

```bash
pip install torchx gin-config torchmetrics==1.0.3 typing-extensions iopath pyvers
```

### 步骤 5：triton

```bash
pip install triton==3.6.0
```

### 步骤 6：FBGEMM（clone）

```bash
cd /workspace/deps
pip install --no-cache-dir setuptools-git-versioning scikit-build
git config --global http.postBuffer 524288000
git clone --depth 1 --recursive -b v1.2.0 https://github.com/pytorch/FBGEMM.git fbgemm
```

### 步骤 7：FBGEMM（安装）

```bash
cd /workspace/deps/fbgemm/fbgemm_gpu
export TORCH_CUDA_ARCH_LIST="8.6"
export CMAKE_BUILD_PARALLEL_LEVEL=16
export MAX_JOBS=16

python setup.py install --package_variant=cuda
```

### 步骤 8：torchrec

```bash
cd /workspace/deps
pip install --no-deps tensordict orjson
git clone --recursive -b v1.2.0 https://github.com/pytorch/torchrec.git torchrec
cd torchrec && pip install --no-deps . && cd ..
```

### 步骤 9：CUTLASS DSL

```bash
pip install nvidia-cutlass-dsl==4.3.0
```

### 步骤 10：开发工具（gdb、pre-commit）

```bash
apt update -y --fix-missing && apt install -y gdb && apt autoremove -y && apt clean && rm -rf /var/lib/apt/lists/*
pip install --no-cache pre-commit
```

### 步骤 11：复制项目并编译 HSTU

编译 HSTU 会生成大量 CUDA 文件，默认并行度可能被内存算得很低（如 `MAX_JOBS=1`），导致非常慢。**建议显式设置 `MAX_JOBS` 和 `NVCC_THREADS` 加速**（按机器 CPU 核数和内存酌情调整，例如 8 核、12GB 内存可用 `MAX_JOBS=6`）：

```bash
# 若启动时已挂载 -v ...:/workspace/recsys-examples，则仓库已在 /workspace/recsys-examples
# 若未挂载，需要先把宿主机上的 recsys-examples 拷入容器后再执行下面命令
cd /workspace/recsys-examples/corelib/hstu
MAX_JOBS=2 NVCC_THREADS=4 HSTU_DISABLE_86OR89=FALSE HSTU_DISABLE_ARBITRARY=TRUE HSTU_DISABLE_LOCAL=TRUE HSTU_DISABLE_RAB=TRUE HSTU_DISABLE_DRAB=TRUE pip install .
cd hopper
MAX_JOBS=2 NVCC_THREADS=4 HSTU_DISABLE_ARBITRARY=TRUE HSTU_DISABLE_SM8x=TRUE HSTU_DISABLE_LOCAL=TRUE HSTU_DISABLE_RAB=TRUE HSTU_DISABLE_DELTA_Q=FALSE HSTU_DISABLE_DRAB=TRUE pip install .
```

### 步骤 12：dynamicemb、nvcomp、commons（与 Dockerfile 第二阶段对应）

```bash
cd /workspace/recsys-examples/corelib/dynamicemb
python setup.py install

cd /workspace/deps && rm -rf nvcomp
wget https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.1.0.21_cuda12-archive.tar.xz
tar -xf nvcomp-linux-x86_64-5.1.0.21_cuda12-archive.tar.xz
mv nvcomp-linux-x86_64-5.1.0.21_cuda12-archive nvcomp
rm nvcomp-linux-x86_64-5.1.0.21_cuda12-archive.tar.xz

cd /workspace/recsys-examples/examples/commons
export TORCH_CUDA_ARCH_LIST="8.6"
export CMAKE_BUILD_PARALLEL_LEVEL=16
export MAX_JOBS=16
python3 setup.py install
```

---

## 3. 保存进度（推荐每步成功后执行）

在**宿主机**新开一个终端：

```powershell
# 查看当前运行中的容器 ID
docker ps

# 把容器保存为新镜像（将 <container_id> 换成实际 ID）
docker commit <container_id> recsys-examples:manual
```

这样某一步失败时，可以从 `recsys-examples:manual` 起新容器，从失败的那一步继续执行。

---

## 4. 注意事项

- **工作目录**：多数命令假设在 `/workspace/deps` 或文档中写明的路径执行，注意每块前的 `cd`。
- **COPY 等价**：若用 `-v` 挂载了项目，则无需在容器内再执行 `COPY . .`；若没挂载，需要在容器内用 `scp`/卷复制等方式把宿主机上的 `recsys-examples` 放到 `/workspace/recsys-examples`。
- **ARM（如 Grace）**：在 ARM 上时，步骤 2 改用 `ARCH=aarch64`，并改用 aarch64 对应的 `ln -s ... targets/sbsa-linux/...` 路径（与 Dockerfile 中一致）。
- **Triton 构建**：仅在做 TritonServer 相关构建时需要执行步骤 1 并设置 `TRITONSERVER_BUILD=1`。

按上述顺序一条条执行，就等价于在“自己创建的 Docker 环境里逐条执行 Dockerfile 的命令”；用 `docker commit` 即可随时把当前状态固化为镜像。

```
docker run --gpus all --runtime=nvidia --cpus=8 --memory=12g --shm-size=4g -it --platform linux/amd64 -v "${PWD}:/workspace/recsys-examples" recsys-examples:latest bash

docker ps
docker commit ID recsys-examples:latest
```
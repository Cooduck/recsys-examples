# HSTU 训练示例

本示例已支持以 HSTU 层为骨干的**召回模型**和**排序模型**。在本示例集中，用户可通过 gin-config 文件指定模型结构。所支持的数据集见下文。关于 gin-config 的用法，请参阅 [内联注释](../utils/gin_config_args.py)。

## 并行策略简介

为支撑大规模嵌入表以及 HSTU 稠密部分的扩展规律，本示例集成了 **[TorchRec](https://github.com/pytorch/torchrec)**（用于分片嵌入表）和 **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)**（用于稠密并行，如数据并行、张量并行、序列并行、流水线并行与上下文并行）。

该集成通过在同一模型内协调稀疏（嵌入）与稠密（上下文/数据）并行，实现高效训练。
![parallelism](../figs/parallelism.png)

## 环境配置

### 从 Dockerfile 开始

我们提供 [dockerfile](../../../docker/Dockerfile) 供用户构建环境。

```
git clone https://github.com/NVIDIA/recsys-examples.git && cd recsys-examples
docker build -f docker/Dockerfile --platform linux/amd64 -t recsys-examples:latest .
```

若需为 Grace 平台构建镜像，可使用：

```
git clone https://github.com/NVIDIA/recsys-examples.git && cd recsys-examples
docker build -f docker/Dockerfile --platform linux/arm64 -t recsys-examples:latest .
```

也可通过参数 `--build-arg <BASE_IMAGE>` 指定自己的基础镜像。

### 从源码安装

在运行示例前，请按以下文档说明先构建并安装 corelib 下的库：

- [HSTU attention 文档](../../../corelib/hstu/README.md)
- [Dynamic Embeddings 文档](../../../corelib/dynamicemb/README.md)

在上述两个核心库之外，还需安装 Megatron-Core 及其他依赖，可通过 pip 安装：

```bash
pip install torchx gin-config torchmetrics==1.0.3 typing-extensions iopath megatron-core==0.12.1
```

若 megatron-core 安装失败（常见原因为 Python 版本不兼容），可尝试克隆源码后本地安装：

```bash
git clone -b core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git megatron-lm && \
pip install -e ./megatron-lm
```

我们提供了自定义的 HSTU CUDA 算子以提升性能。需使用以下命令安装这些算子：

```bash
cd /workspace/recsys-examples/examples/hstu && \
python setup.py install
```

### 数据集简介

当前支持的数据集如下。

### 数据集信息

#### **MovieLens**

详见 [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) 与 [MovieLens 20M](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset)。

#### **KuaiRand**

| 数据集         | 用户数 | 序列最大长度 | 序列最小长度 | 序列平均长度 | 序列中位数 | 物品数   |
|----------------|--------|--------------|--------------|--------------|------------|----------|
| kuairand_pure  | 27285  | 910          | 1            | 1            | 39         | 7551     |
| kuairand_1k    | 1000   | 49332        | 10           | 5038         | 3379       | 4369953  |
| kuairand_27k   | 27285  | 228000       | 100          | 11796        | 8591       | 32038725 |

详见 [KuaiRand](https://kuairand.com/)。

## 运行示例

开始前请确保已满足所有前置条件，可参考仓库根目录的 [Get Started](../../../README) 进行环境配置。

### 数据预处理

训练前需先准备数据集，可使用项目 `commons` 目录下的 `hstu_data_preprocessor.py`：

```bash
cd <root-to-repo>/examples/commons && 
mkdir -p ./tmp_data && python3 ./hstu_data_preprocessor.py --dataset_name <"ml-1m"|"ml-20m"|"kuairand-pure"|"kuairand-1k"|"kuairand-27k">
```

### 启动训练

训练入口脚本为 `pretrain_gr_retrieval.py`（召回）或 `pretrain_gr_ranking.py`（排序）。模型结构、训练参数、超参等通过 gin-config 指定。

使用 **MovieLens 20M** 运行**召回**任务：

```bash
# 运行 pretrain_gr_retrieval.py 前，请确保当前工作目录为 hstu
cd <root-to-project>examples/hstu 
PYTHONPATH=${PYTHONPATH}:$(realpath ../) torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000  ./training/pretrain_gr_retrieval.py --gin-config-file ./training/configs/movielen_retrieval.gin
```

使用 **MovieLens 20M** 运行**排序**任务：

```bash
# 运行 pretrain_gr_ranking.py 前，请确保当前工作目录为 hstu
cd <root-to-project>examples/hstu 
PYTHONPATH=${PYTHONPATH}:$(realpath ../) torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000  ./training/pretrain_gr_ranking.py --gin-config-file ./training/configs/movielen_ranking.gin
```

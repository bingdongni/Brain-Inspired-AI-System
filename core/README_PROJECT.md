# Brain-Inspired AI Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()

基于生物大脑启发的深度学习框架，模拟海马体和新皮层的认知机制，实现先进的机器学习能力。

## 🌟 主要特性

### 核心架构
- **海马体记忆系统**: 实现情景记忆存储、快速学习和模式分离
- **新皮层处理架构**: 层次化信息处理和抽象推理
- **动态路由机制**: 智能资源分配和计算优化
- **持续学习能力**: 防止灾难性遗忘，支持终身学习

### 高级功能
- **多模态整合**: 支持视觉、听觉、文本等多种数据类型
- **注意力机制**: 生物启发的选择性注意
- **元学习**: 快速适应新任务的学习能力
- **知识转移**: 跨任务知识复用和迁移

### 工程特性
- **模块化设计**: 高度可扩展的组件架构
- **生产就绪**: 支持分布式部署和容器化
- **性能优化**: GPU加速和内存优化
- **监控诊断**: 完整的日志和性能监控

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (GPU支持)
- 8GB+ RAM (推荐16GB+)

### 安装方法

#### 方法一：直接安装
```bash
# 克隆仓库
git clone https://github.com/brain-ai/brain-inspired-ai.git
cd brain-inspired-ai

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

#### 方法二：使用虚拟环境
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖和项目
pip install -r requirements.txt
pip install -e .
```

#### 方法三：Docker部署
```bash
# 启动完整环境
docker-compose up -d

# 仅启动核心服务
docker-compose up brain-ai redis postgres -d
```

### 基本使用示例

```python
import brain_ai
from brain_ai import HippocampusSimulator, NeocortexArchitecture

# 创建海马体实例
hippocampus = HippocampusSimulator(
    input_size=512,
    memory_capacity=10000
)

# 存储记忆
memory_id = hippocampus.store(
    {"text": "学习AI很有趣", "emotion": "兴奋"},
    metadata={"importance": 0.8}
)

# 检索记忆
retrieved = hippocampus.retrieve(memory_id)
print(f"检索到: {retrieved}")

# 创建新皮层实例
neocortex = NeocortexArchitecture(
    input_size=512,
    num_layers=6,
    hidden_size=512
)

# 处理输入
input_data = torch.randn(1, 10, 512)
output = neocortex(input_data)
print(f"处理结果形状: {output['final_output'].shape}")
```

## 📚 详细文档

### 核心模块

#### 海马体模块 (`brain_ai.hippocampus`)
```python
from brain_ai.hippocampus import HippocampusSimulator, EpisodicMemory

# 创建海马体模拟器
hippocampus = HippocampusSimulator(input_size=256)

# 情景记忆系统
episodic_memory = EpisodicMemory(max_capacity=5000)

# 快速学习机制
from brain_ai.hippocampus.core.fast_learning import FastLearningSystem
fast_learner = FastLearningSystem(input_size=256, output_size=128)
```

#### 新皮层模块 (`brain_ai.neocortex`)
```python
from brain_ai.neocortex import NeocortexArchitecture, AttentionModule

# 新皮层架构
neocortex = NeocortexArchitecture(
    input_size=512,
    num_layers=8,
    hidden_size=768,
    attention_heads=12
)

# 注意力模块
attention = AttentionModule(
    dim=768,
    num_heads=12,
    dropout=0.1
)
```

#### 持续学习模块 (`brain_ai.lifelong_learning`)
```python
from brain_ai.lifelong_learning import (
    ElasticWeightConsolidation,
    GenerativeReplay,
    DynamicExpansion
)

# EWC防遗忘
ewc = ElasticWeightConsolidation(lambda_ewc=1000)

# 生成式重放
gen_replay = GenerativeReplay(
    generator_lr=0.001,
    replay_ratio=0.5
)

# 动态扩展
dyn_expansion = DynamicExpansion(
    growth_threshold=0.1,
    max_new_neurons=100
)
```

### 命令行工具

```bash
# 查看帮助
brain-ai --help

# 训练模型
brain-ai train \
    --model-type hippocampus \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --output-dir ./output

# 评估模型
brain-ai evaluate \
    --model-path ./output/hippocampus_model.pkl \
    --test-data ./data/test.csv \
    --metrics accuracy precision recall f1

# 运行演示
brain-ai demo --demo-type basic --interactive

# 查看系统信息
brain-ai info

# 配置管理
brain-ai config
```

### API接口

项目提供RESTful API接口：

```python
# 启动API服务器
python -m brain_ai.scripts.serve --config config/development.yaml

# API端点
# GET /health - 健康检查
# GET /info - 系统信息
# POST /predict - 模型预测
# GET /models - 模型列表
# POST /models/{id}/train - 训练模型
```

## 🔧 配置说明

### 配置文件结构
```
config/
├── development.yaml  # 开发环境配置
├── production.yaml  # 生产环境配置
├── testing.yaml     # 测试环境配置
└── custom.yaml      # 自定义配置
```

### 主要配置项

#### 模型配置
```yaml
model:
  default_type: "brain_system"
  hippocampus:
    ca3_hidden_size: 512
    ca3_num_layers: 4
    memory_capacity: 50000
    retrieval_threshold: 0.7
  neocortex:
    num_layers: 6
    hidden_size: 512
    attention_heads: 8
```

#### 训练配置
```yaml
training:
  epochs: 200
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adamw"
  early_stopping:
    enabled: true
    patience: 20
```

#### 服务器配置
```yaml
server:
  http:
    host: "0.0.0.0"
    port: 8080
    workers: 4
```

## 🧪 测试和验证

### 运行测试
```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_hippocampus.py -v

# 生成测试覆盖率报告
pytest --cov=brain_ai tests/ --cov-report=html
```

### 性能基准测试
```bash
# 运行性能测试
python -m brain_ai.scripts.benchmark --model hippocampus --dataset test_data

# 内存使用分析
python -m brain_ai.scripts.profile --memory --model neocortex
```

## 📈 监控和诊断

### 日志系统
```python
from brain_ai.utils import Logger, setup_logging

# 设置日志
logger = setup_logging(
    name="brain_ai",
    level="INFO",
    log_dir="./logs",
    json_format=True
)

# 记录日志
logger.info("模型训练开始", epoch=1, loss=0.5)
logger.performance("training_epoch", duration=2.5)
```

### 指标监控
```python
from brain_ai.utils import MetricsCollector

# 创建指标收集器
metrics = MetricsCollector(save_path="./metrics.json")

# 记录指标
metrics.add_metric("train_loss", 0.1)
metrics.add_metrics({
    "accuracy": 0.92,
    "precision": 0.89,
    "recall": 0.91
})
```

### 可视化
```python
from brain_ai.utils.visualization import TrainingVisualizer

# 绘制训练曲线
visualizer = TrainingVisualizer(save_dir="./plots")
visualizer.plot_training_curves(
    train_history={"loss": [0.5, 0.3, 0.2]},
    val_history={"loss": [0.6, 0.4, 0.3]}
)
```

## 🏗️ 架构设计

### 系统架构图
```
┌─────────────────────────────────────────────────────────────┐
│                    Brain-Inspired AI Framework              │
├─────────────────────────────────────────────────────────────┤
│  用户接口层                                                  │
│  ├── CLI工具 (brain-ai)                                     │
│  ├── REST API (FastAPI)                                     │
│  └── Web界面 (Jupyter/Streamlit)                           │
├─────────────────────────────────────────────────────────────┤
│  核心功能层                                                  │
│  ├── 海马体系统 (HippocampusSimulator)                      │
│  ├── 新皮层系统 (NeocortexArchitecture)                     │
│  ├── 动态路由 (DynamicRoutingController)                    │
│  └── 持续学习 (LifelongLearning)                            │
├─────────────────────────────────────────────────────────────┤
│  算法模块层                                                  │
│  ├── 记忆编码器 (Encoders)                                  │
│  ├── 注意力机制 (Attention Mechanisms)                     │
│  ├── 模式分离 (Pattern Separation)                         │
│  └── 元学习 (Meta Learning)                                │
├─────────────────────────────────────────────────────────────┤
│  基础设施层                                                  │
│  ├── 配置管理 (ConfigManager)                               │
│  ├── 日志系统 (Logger)                                      │
│  ├── 指标监控 (MetricsCollector)                           │
│  └── 数据处理 (DataProcessor)                              │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

#### 海马体模拟器
- **CA3区域**: 内容可寻址记忆网络
- **CA1区域**: 模式完成和记忆提取
- **齿状回**: 模式分离和稀疏编码
- **记忆巩固**: 快速学习和长期记忆

#### 新皮层架构
- **层次化处理**: V1→V2→V4→IT视觉通路
- **注意力机制**: 自注意力和交叉注意力
- **跨模态整合**: 多感官信息融合
- **决策制定**: 输出层和行动选择

#### 持续学习框架
- **EWC**: 弹性权重巩固防遗忘
- **生成式重放**: 合成历史数据
- **动态扩展**: 神经网络结构增长
- **知识转移**: 跨任务知识共享

## 🚀 部署指南

### 本地开发部署
```bash
# 使用部署脚本
./scripts/deploy.sh deploy development

# 或手动部署
python -m venv venv_dev
source venv_dev/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 生产环境部署
```bash
# Docker部署
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Kubernetes部署
kubectl apply -f k8s/

# 云平台部署
# 支持AWS, GCP, Azure等平台
```

### 性能优化
```yaml
# GPU配置
system:
  device: "cuda"
  num_workers: 8
  batch_size: 64

# 内存优化
training:
  gradient_clipping:
    enabled: true
    max_norm: 1.0
  mixed_precision:
    enabled: true
```

## 🤝 贡献指南

### 开发环境设置
```bash
# 克隆开发分支
git clone -b develop https://github.com/brain-ai/brain-inspired-ai.git
cd brain-inspired-ai

# 安装开发依赖
pip install -r requirements-dev.txt
pre-commit install

# 运行代码质量检查
black src/
flake8 src/
mypy src/
```

### 提交规范
```bash
# 功能开发
git checkout -b feature/new-feature
# 开发代码...
git commit -m "feat: add new feature"

# Bug修复
git checkout -b fix/bug-description
# 修复代码...
git commit -m "fix: resolve bug description"

# 文档更新
git checkout -b docs/update-readme
# 更新文档...
git commit -m "docs: update README with new examples"
```

### Pull Request流程
1. Fork项目并创建功能分支
2. 开发功能并添加测试
3. 运行完整测试套件
4. 更新相关文档
5. 提交Pull Request

## 📄 许可证

本项目采用MIT许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 📞 支持和联系

- **文档**: [https://docs.brain-ai.org](https://docs.brain-ai.org)
- **问题反馈**: [GitHub Issues](https://github.com/brain-ai/brain-inspired-ai/issues)
- **讨论社区**: [GitHub Discussions](https://github.com/brain-ai/brain-inspired-ai/discussions)
- **邮箱**: support@brain-ai.org

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者和研究者。特别感谢：
- PyTorch团队提供的深度学习框架
- 神经科学研究社区的启发
- 开源社区的支持和反馈

## 📊 项目统计

[![Stars](https://img.shields.io/github/stars/brain-ai/brain-inspired-ai?style=social)](https://github.com/brain-ai/brain-inspired-ai)
[![Forks](https://img.shields.io/github/forks/brain-ai/brain-inspired-ai?style=social)](https://github.com/brain-ai/brain-inspired-ai)
[![Issues](https://img.shields.io/github/issues/brain-ai/brain-inspired-ai)](https://github.com/brain-ai/brain-inspired-ai/issues)
[![Closed Issues](https://img.shields.io/github/issues-closed/brain-ai/brain-inspired-ai)](https://github.com/brain-ai/brain-inspired-ai/issues?q=is%3Aissue+is%3Aclosed)

---

**Brain-Inspired AI Framework** - 让AI更贴近生物大脑的智能 🚀🧠
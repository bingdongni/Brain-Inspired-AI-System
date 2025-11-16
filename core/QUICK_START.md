# 脑启发AI演示系统快速上手指南

## 📚 目录
1. [环境准备](#环境准备)
2. [安装指南](#安装指南)
3. [快速开始](#快速开始)
4. [示例数据集](#示例数据集)
5. [预训练模型](#预训练模型)
6. [演示示例](#演示示例)
7. [故障排除](#故障排除)

---

## 🚀 环境准备

### 系统要求
- **操作系统**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Python**: 3.8 - 3.11
- **内存**: 最少 4GB RAM (推荐 8GB+)
- **存储**: 至少 2GB 可用空间
- **显卡**: 可选，支持 CUDA 11.0+ 的 NVIDIA GPU

### Python 环境检查

```bash
# 检查 Python 版本
python --version
# 应输出: Python 3.8.x 或更高版本

# 检查 pip 版本
pip --version
```

### 虚拟环境创建 (推荐)

```bash
# 使用 venv 创建虚拟环境
python -m venv brain_ai_env

# 激活虚拟环境
# Linux/Mac:
source brain_ai_env/bin/activate
# Windows:
brain_ai_env\Scripts\activate

# 验证虚拟环境
which python  # Linux/Mac
where python  # Windows
```

---

## 📦 安装指南

### 基础安装

```bash
# 1. 克隆或下载项目
git clone https://github.com/brain-ai/brain-inspired-ai.git
cd brain-inspired-ai

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装项目 (开发模式)
pip install -e .
```

### 完整安装选项

```bash
# 基础包 (必需)
pip install -r requirements.txt

# 开发依赖 (测试和开发工具)
pip install -e ".[dev]"

# 可视化工具 (图表和交互界面)
pip install -e ".[visualization]"

# GPU 支持 (如果有 NVIDIA GPU)
pip install -e ".[gpu]"

# 完整安装
pip install -e ".[dev,visualization,gpu]"
```

### 依赖检查

```python
# 运行依赖检查脚本
python scripts/check_dependencies.py

# 或手动检查
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
```

---

## ⚡ 快速开始

### 5分钟快速体验

```bash
# 运行基础演示
python cli_demo.py --mode demo --dataset synthetic --epochs 10

# 查看结果
ls -la data/results/
cat data/results/demo_results_*.json
```

### 交互式界面

```bash
# 启动交互式演示
python cli_demo.py --mode interactive

# 在交互界面中选择功能进行体验
```

### 批量测试

```bash
# 运行所有演示示例
python scripts/run_all_demos.py

# 或运行特定演示
python demos/memory_learning_demo.py
python demos/lifelong_learning_demo.py
python demos/dynamic_routing_demo.py
```

---

## 📊 示例数据集

### 内置数据集

我们的系统提供多种内置数据集用于快速测试和演示：

#### 1. 合成数据集 (Synthetic)
- **用途**: 基础功能测试和算法验证
- **特点**: 随机生成，具有可控制的复杂度
- **大小**: 可配置，默认 1000 训练样本，200 测试样本
- **维度**: 可配置，默认 20 维输入，5 类输出

```python
# 使用合成数据
from cli_demo import BrainInspiredAISystem

system = BrainInspiredAISystem()
data = system.generate_sample_data("synthetic")
print(f"训练样本: {len(data['X_train'])}")
print(f"测试样本: {len(data['X_test'])}")
print(f"输入维度: {data['input_dim']}")
print(f"输出类别: {data['output_dim']}")
```

#### 2. MNIST 风格数据集
- **用途**: 图像分类演示和性能基准测试
- **特点**: 28x28 灰度图像，10 类数字分类
- **大小**: 可配置，默认 1000 训练样本，200 测试样本

```python
# 使用 MNIST 风格数据
data = system.generate_sample_data("mnist")
```

#### 3. 模式识别数据集
- **用途**: 注意力机制和抽象学习演示
- **特点**: 具有明确模式结构的分类数据
- **结构**: 4 类数据，每类具有独特的特征模式

```python
# 使用模式识别数据
data = system.generate_sample_data("patterns")
```

### 数据集配置

在 `data/config/dataset_config.yaml` 中可以自定义数据集参数：

```yaml
datasets:
  synthetic:
    train_size: 1000
    test_size: 200
    input_dim: 20
    output_dim: 5
    noise_level: 0.1
    
  mnist_like:
    train_size: 1000
    test_size: 200
    image_size: [28, 28]
    num_classes: 10
    
  patterns:
    train_size: 800
    test_size: 200
    pattern_types: 4
    feature_dim: 10
```

---

## 🤖 预训练模型

### 模型下载

系统提供多个预训练模型用于快速体验：

#### 可用模型

| 模型名称 | 类型 | 大小 | 用途 |
|---------|------|------|------|
| `brain_inspired_v1.0` | 完整脑启发模型 | ~50MB | 通用分类任务 |
| `hippocampus_v1.0` | 海马体专用模型 | ~30MB | 记忆和注意力 |
| `neocortex_v1.0` | 新皮层专用模型 | ~40MB | 抽象和层次学习 |

#### 下载预训练模型

```bash
# 自动下载所有预训练模型
python scripts/download_pretrained_models.py

# 或下载特定模型
python scripts/download_pretrained_models.py --model brain_inspired_v1.0
```

#### 手动下载链接

如果自动下载失败，可以手动下载：

```bash
# 创建模型目录
mkdir -p data/models/pretrained

# 下载模型 (示例 URLs)
wget -O data/models/pretrained/brain_inspired_v1.0.pth \
     https://github.com/brain-ai/models/releases/download/v1.0/brain_inspired_v1.0.pth

wget -O data/models/pretrained/hippocampus_v1.0.pth \
     https://github.com/brain-ai/models/releases/download/v1.0/hippocampus_v1.0.pth

wget -O data/models/pretrained/neocortex_v1.0.pth \
     https://github.com/brain-ai/models/releases/download/v1.0/neocortex_v1.0.pth
```

### 使用预训练模型

```python
# 加载预训练模型
import torch
from cli_demo import BrainInspiredAISystem

system = BrainInspiredAISystem()
system.initialize_system()

# 加载预训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('data/models/pretrained/brain_inspired_v1.0.pth', map_location=device)

# 使用预训练模型进行推理
system.models['brain_inspired'] = model
system.current_data = system.generate_sample_data("synthetic")

# 评估预训练模型
evaluation = system.evaluate_model('brain_inspired', system.current_data)
print(f"预训练模型准确率: {evaluation['accuracy']:.4f}")
```

---

## 🎮 演示示例

### 1. 记忆学习演示 (Memory Learning Demo)

展示海马体记忆机制的学习过程：

```bash
# 运行记忆学习演示
python demos/memory_learning_demo.py
```

**功能特点:**
- 序列记忆学习
- 模式补全
- 记忆检索
- 遗忘曲线分析

**预期输出:**
```
🧠 海马体记忆学习演示
========================
📚 学习序列: [1, 3, 5, 7, 9]
🔍 模式补全测试: 输入 [1, 3, 5] -> 输出 [7, 9]
📊 记忆检索准确率: 0.92
📈 遗忘曲线: 24小时后保持率 0.85
```

### 2. 终身学习演示 (Lifelong Learning Demo)

展示持续学习能力：

```bash
# 运行终身学习演示
python demos/lifelong_learning_demo.py
```

**功能特点:**
- 多任务连续学习
- 灾难性遗忘防护
- 知识迁移
- 性能保持率分析

**预期输出:**
```
🔄 终身学习演示
==================
📚 任务1: 学习基础模式识别
✅ 任务1准确率: 0.95

📚 任务2: 学习复杂模式识别  
✅ 任务2准确率: 0.93
✅ 任务1保持率: 0.91 (防护成功)

📚 任务3: 学习高级抽象模式
✅ 任务3准确率: 0.89
✅ 任务1保持率: 0.88
✅ 任务2保持率: 0.90
```

### 3. 动态路由演示 (Dynamic Routing Demo)

展示神经网络动态路由机制：

```bash
# 运行动态路由演示
python demos/dynamic_routing_demo.py
```

**功能特点:**
- 动态连接权重调整
- 路径优化
- 负载均衡
- 效率分析

**预期输出:**
```
🛣️ 动态路由演示
==================
🔗 初始路由路径: 3跳
🎯 优化后路径: 2跳
⚡ 效率提升: 33%
📊 负载均衡度: 0.89
🚀 吞吐量提升: 28%
```

### 性能基准测试

运行完整的性能基准测试：

```bash
# 运行基准测试
python scripts/benchmark_test.py
```

**测试内容:**
- 训练速度基准
- 推理速度基准  
- 内存使用基准
- 准确率基准
- 持续学习性能

**基准结果示例:**
```
📊 性能基准测试结果
====================
🚀 训练速度: 45.2 samples/sec
⚡ 推理速度: 128.7 samples/sec
💾 内存使用: 1.2 GB
🎯 准确率基准: 94.3%
🔄 持续学习: 92.1% (保持率)
```

---

## 🔧 故障排除

### 常见问题

#### 1. Python 版本不兼容

**问题**: `Python 3.7 found, requires 3.8+`

**解决方案**:
```bash
# 安装 Python 3.8+
# Ubuntu/Debian
sudo apt update
sudo apt install python3.8 python3.8-venv

# 使用 pyenv 管理多个 Python 版本
curl https://pyenv.run | bash
pyenv install 3.9.0
pyenv local 3.9.0
```

#### 2. PyTorch 安装问题

**问题**: `No module named 'torch'`

**解决方案**:
```bash
# CPU 版本
pip install torch torchvision torchaudio

# GPU 版本 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证安装
python -c "import torch; print(torch.cuda.is_available())"
```

#### 3. 内存不足

**问题**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 减少批处理大小
export BATCH_SIZE=16

# 或使用 CPU 模式
python cli_demo.py --mode demo --device cpu
```

#### 4. 模型加载失败

**问题**: `FileNotFoundError: pretrained model not found`

**解决方案**:
```bash
# 重新下载预训练模型
python scripts/download_pretrained_models.py --force

# 检查网络连接
ping github.com
```

#### 5. 可视化显示问题

**问题**: `ImportError: No module named 'matplotlib'`

**解决方案**:
```bash
# 安装可视化依赖
pip install matplotlib seaborn plotly

# 或安装完整可视化包
pip install -e ".[visualization]"
```

### 调试模式

启用详细日志输出：

```bash
# 启用调试模式
export BRAIN_AI_DEBUG=1
python cli_demo.py --mode demo

# 或在代码中设置
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 性能优化建议

1. **GPU 加速**: 确保安装 CUDA 版本的 PyTorch
2. **内存管理**: 适当调整批处理大小
3. **并行处理**: 使用多进程数据加载
4. **模型优化**: 启用模型编译和量化

---

## 📞 获取帮助

### 联系方式

- **项目主页**: https://github.com/brain-ai/brain-inspired-ai
- **文档**: https://brain-ai.readthedocs.io/
- **问题反馈**: https://github.com/brain-ai/brain-inspired-ai/issues
- **邮箱**: support@brain-ai.org

### 社区资源

- **论坛**: https://forum.brain-ai.org
- **Slack**: https://brain-ai.slack.com
- **微信群**: 扫描 README 中的二维码加入

### 贡献指南

欢迎贡献代码、文档和反馈！

1. Fork 项目仓库
2. 创建特性分支: `git checkout -b feature/amazing-feature`
3. 提交更改: `git commit -m 'Add amazing feature'`
4. 推送分支: `git push origin feature/amazing-feature`
5. 提交 Pull Request

---

## 🎯 下一步

完成快速上手指南后，建议：

1. **深入学习**: 阅读完整的技术文档
2. **实践项目**: 尝试在您的数据上应用系统
3. **性能调优**: 根据具体需求调整参数
4. **扩展功能**: 基于现有模块开发新功能

祝您使用愉快！ 🎉
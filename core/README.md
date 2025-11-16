# 🧠 Brain-Inspired AI Framework

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/your-repo/brain-inspired-ai/releases)
[![Status](https://img.shields.io/badge/status-active-success.svg)](https://github.com/your-repo/brain-inspired-ai/actions)

**基于大脑神经机制的深度学习框架 - 支持持续学习和记忆系统**

[⚡ 快速开始](#快速开始) • [📚 完整指南](docs/quick_start_guide.md) • [🎮 界面演示](#界面演示) • [📖 文档](#文档) • [🤝 贡献](#贡献)

</div>

## 🌟 项目特色

### 🧠 核心神经网络机制
- **海马体记忆系统** - 模拟大脑记忆形成、存储和检索过程
- **新皮层抽象处理** - 层次化信息处理和表示学习
- **动态路由网络** - 脑启发的神经连接模式
- **注意力机制** - 选择性注意和焦点建模

### 🛠️ 高级学习能力
- **持续学习** - 终身学习，避免灾难性遗忘
- **多任务学习** - 任务间高效知识迁移
- **记忆巩固** - 智能记忆整合策略
- **模式分离** - 鲁棒的模式识别和区分

### 📊 系统特点
- **多种界面** - 支持CLI命令行、Web界面、Jupyter集成
- **模块化设计** - 清晰的架构，易于扩展
- **完整演示** - 提供丰富的示例和教程
- **实时监控** - 系统状态和性能实时可视化

## 快速开始

### 🏃‍♂️ 5分钟快速体验

```bash
# 1. 克隆项目
git clone https://github.com/brain-ai/brain-inspired-ai.git
cd brain-inspired-ai

# 2. 创建虚拟环境（推荐）
python -m venv brain_ai_env
source brain_ai_env/bin/activate  # Linux/Mac
# brain_ai_env\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装项目
pip install -e .

# 5. 快速验证
python quick_test.py

# 6. 运行演示
python cli_demo.py --mode demo
```

### 🎮 三种界面体验

#### 1. 命令行界面（CLI）
```bash
# 基础演示（30秒完成）
python cli_demo.py --mode demo

# 交互式体验
python cli_demo.py --mode interactive

# 运行所有演示
python scripts/run_all_demos.py
```

#### 2. React Web界面
```bash
# 进入Web界面目录
cd ui/brain-ai-ui

# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 访问 http://localhost:5173
```

#### 3. Jupyter集成界面
```python
# 在Jupyter中运行
import sys
sys.path.append('/path/to/brain-inspired-ai/ui')
from jupyter_integration import *

# 显示主控制台
show_brain_dashboard()
```

### 🔧 基础使用示例

#### 海马体记忆系统
```python
from hippocampus import HippocampusSimulator

# 创建海马体实例
hippocampus = HippocampusSimulator(memory_capacity=1000)

# 学习序列
sequence = [1, 3, 5, 7, 9]
hippocampus.learn_sequence(sequence)

# 测试模式补全
completed = hippocampus.complete_pattern([1, 3, 5])
print(f"补全结果: {completed}")
```

#### 持续学习系统
```python
from lifelong_learning import ContinualLearner

# 创建学习器
learner = ContinualLearner(memory_size=5000)

# 学习任务1
learner.learn_task(task1_data, task_id=1)

# 学习任务2（保持任务1记忆）
learner.learn_task(task2_data, task_id=2)

# 评估所有任务
results = learner.evaluate_all_tasks()
```

## 界面演示

### 🎯 立即体验

<div align="center">

**[💻 Web界面演示](http://localhost:5173)** | **[📖 完整快速指南](docs/quick_start_guide.md)** | **[📓 Jupyter演示](ui/界面使用演示.ipynb)**

</div>

### 可用演示程序

```bash
# 记忆学习演示
python demos/memory_learning_demo.py

# 终身学习演示  
python demos/lifelong_learning_demo.py

# 动态路由演示
python demos/dynamic_routing_demo.py

# 交互式命令行演示
python cli_demo.py --mode interactive
```

## 📖 文档

### 📚 完整文档

| 文档类型 | 描述 | 链接 |
|---------|------|------|
| **快速开始** | 5分钟上手指南 | [快速指南](docs/quick_start_guide.md) |
| **用户手册** | 完整使用说明 | [用户手册](docs/USER_MANUAL.md) |
| **开发者指南** | 扩展开发指导 | [开发指南](docs/DEVELOPER_GUIDE.md) |
| **API参考** | 详细接口文档 | [API文档](docs/api/API_REFERENCE.md) |

### 🏗️ 项目架构

```
🧠 脑启发AI系统
├── 🧠 核心模块
│   ├── Hippocampus (海马体) - 记忆系统
│   ├── Neocortex (新皮层) - 抽象处理
│   ├── Attention (注意力) - 选择机制
│   └── Dynamic Routing (动态路由) - 连接优化
├── 🔄 持续学习
│   ├── EWC (弹性权重巩固)
│   ├── Generative Replay (生成重放)
│   ├── Dynamic Expansion (动态扩展)
│   └── Knowledge Transfer (知识迁移)
├── 🎮 用户界面
│   ├── CLI (命令行界面)
│   ├── Web (React界面)
│   └── Jupyter (集成环境)
└── 🛠️ 工具支持
    ├── 配置管理
    ├── 性能监控
    ├── 演示程序
    └── 测试套件
```

### 🔧 主要接口

#### 核心组件
```python
# 海马体记忆系统
from hippocampus import HippocampusSimulator
hippocampus = HippocampusSimulator(memory_capacity=1000)

# 新皮层处理
from brain_ai.modules import NeocortexModel  
neocortex = NeocortexModel(hierarchical_levels=4)

# 持续学习器
from lifelong_learning import ContinualLearner
learner = ContinualLearner(memory_size=5000, ewc_lambda=0.1)

# 完整系统
from brain_ai import BrainInspiredSystem
brain_system = BrainInspiredSystem()
```

## 📊 系统演示

### 可用演示程序

| 演示名称 | 功能描述 | 启动命令 |
|---------|----------|----------|
| **记忆学习** | 海马体记忆机制演示 | `python demos/memory_learning_demo.py` |
| **终身学习** | 持续学习能力展示 | `python demos/lifelong_learning_demo.py` |
| **动态路由** | 神经网络路由优化 | `python demos/dynamic_routing_demo.py` |
| **交互演示** | CLI交互式体验 | `python cli_demo.py --mode interactive` |

### 示例输出效果

```bash
$ python demos/memory_learning_demo.py

🧠 海马体记忆学习演示
========================
📚 学习序列: [1, 3, 5, 7, 9]
🔍 模式补全测试: 输入 [1, 3, 5] -> 输出 [7, 9]
📊 记忆检索准确率: 92.5%
📈 遗忘曲线: 24小时后保持率 85%
✅ 演示完成！结果已保存至 results/memory_demo_*.json
```

```bash
$ python demos/lifelong_learning_demo.py

🔄 终身学习演示
==================
📚 任务1: 学习基础模式识别
✅ 任务1准确率: 95.2%

📚 任务2: 学习复杂模式识别
✅ 任务2准确率: 93.1%
✅ 任务1保持率: 91.3% (防护成功)

📚 任务3: 学习高级抽象模式
✅ 任务3准确率: 89.4%
✅ 任务1保持率: 88.7%
✅ 任务2保持率: 90.1%
```

### 性能基准测试

```bash
# 运行完整基准测试
python scripts/benchmark_test.py

# 查看测试结果
python -c "
import json
with open('results/benchmark_results_*.json') as f:
    results = json.load(f)
    print(f'训练速度: {results[\"training_speed\"]} samples/sec')
    print(f'推理速度: {results[\"inference_speed\"]} samples/sec') 
    print(f'内存使用: {results[\"memory_usage\"]} MB')
"
```

## 🛠️ 开发环境

### 环境设置

```bash
# 克隆项目
git clone https://github.com/brain-ai/brain-inspired-ai.git
cd brain-inspired-ai

# 创建虚拟环境
python -m venv dev_env
source dev_env/bin/activate

# 安装开发依赖
pip install -e ".[dev]"

# 安装Git hooks
pre-commit install

# 运行测试
pytest tests/ --cov=src/
```

### 开发工具

```bash
# 代码格式化
black src/ tests/
isort src/ tests/

# 类型检查
mypy src/

# 代码检查
flake8 src/ tests/

# 运行所有测试
pytest --cov=src/ --cov-report=html
```

### Docker开发

```bash
# 使用Docker开发
docker-compose up -d

# 进入开发容器
docker exec -it brain-ai-dev bash

# 在容器内开发
python -m pytest tests/
```

## 🎯 示例和演示

### 📚 快速演示

```bash
# 1. 基础系统测试
python quick_test.py

# 2. 运行单个演示
python cli_demo.py --mode demo

# 3. 交互式体验
python cli_demo.py --mode interactive

# 4. 运行所有演示
python scripts/run_all_demos.py

# 5. 性能基准测试
python scripts/benchmark_test.py
```

### 🎮 界面体验

#### Web界面 (React)
```bash
cd ui/brain-ai-ui
npm install
npm run dev
# 访问 http://localhost:5173
```

#### Jupyter集成
```python
# 在Jupyter中导入
import sys
sys.path.append('/path/to/brain-inspired-ai/ui')
from jupyter_integration import *
show_brain_dashboard()
```

### 💡 代码示例

#### 完整系统使用
```python
from brain_ai import BrainInspiredSystem

# 创建系统
system = BrainInspiredSystem()
system.initialize()

# 训练模型
results = system.train(
    data=dataset,
    epochs=50,
    learning_rate=0.001
)

# 评估性能
accuracy = system.evaluate(test_data)
print(f"准确率: {accuracy:.2%}")
```

## 📈 Visualization Gallery

<div align="center">

![Memory Formation Visualization](docs/images/memory_formation.gif)
*Memory Formation Process in Hippocampus*

![Attention Visualization](docs/images/attention_visualization.png)  
*Attention Mechanism Heatmap*

![Continual Learning](docs/images/continual_learning_curves.png)
*Continual Learning Performance Curves*

</div>

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 贡献方式

- 🐛 **报告问题** - 通过GitHub Issues反馈bug
- 💡 **功能建议** - 提出新功能和改进想法  
- 🔧 **提交代码** - Fork项目并提交Pull Request
- 📖 **完善文档** - 改进文档和示例
- 🎯 **添加演示** - 创建新的示例和教程

### 开发流程

```bash
# 1. Fork并克隆
git clone https://github.com/your-username/brain-inspired-ai.git

# 2. 创建功能分支
git checkout -b feature/new-feature

# 3. 修改和测试
pytest tests/
black src/ tests/

# 4. 提交更改
git commit -m "Add new feature"
git push origin feature/new-feature

# 5. 创建Pull Request
```

### 代码规范

- **Python代码**: 使用Black格式化，isort排序
- **类型提示**: 公共API必须包含类型注解
- **文档**: 所有模块添加docstring
- **测试**: 确保新功能包含测试用例

## 📞 支持与联系

### 📊 项目状态
- **当前版本**: v2.0.0
- **开发状态**: 活跃开发中
- **测试覆盖率**: >85%
- **维护活跃度**: 高

### 🤝 社区支持

- **💬 [GitHub讨论区](https://github.com/brain-ai/brain-inspired-ai/discussions)** - 技术讨论和问题解答
- **🐛 [问题反馈](https://github.com/brain-ai/brain-inspired-ai/issues)** - 报告bug和功能请求
- **📧 [邮箱联系](mailto:support@brain-ai.org)** - 技术支持
- **📖 [完整文档](docs/)** - 详细使用文档

### 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🎉 致谢

感谢所有贡献者的支持！

### 🙏 主要贡献者
- **核心开发团队** - 系统架构和核心模块
- **社区贡献者** - 文档、示例和测试
- **用户反馈** - 持续改进的建议和需求

## 📊 使用统计

```bash
# 检查项目结构
find . -name "*.py" | wc -l     # Python文件数量
find . -name "*.ipynb" | wc -l  # Jupyter文件数量  
find docs/ -name "*.md" | wc -l # 文档文件数量
```

### 📈 快速统计
- **核心模块**: 50+ Python模块
- **演示程序**: 10+ 完整示例  
- **文档页面**: 20+ 详细文档
- **测试用例**: 100+ 单元测试

---

<div align="center">

**🧠 感谢使用脑启发AI系统！**  
*让AI学习更像人类大脑一样智能*

**[⬆ 回到顶部](#🧠-brain-inspired-ai-framework)**

</div>


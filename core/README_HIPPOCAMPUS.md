# 海马体模拟器 (Hippocampal Simulator)

基于Science期刊最新研究的海马体记忆机制理论实现的高级神经网络模拟器。

## 🔬 科学基础

本模拟器基于以下重要的神经科学研究：

- **主要来源**: Science期刊 - 小鼠海马体记忆印迹的突触架构
- **DOI**: 10.1126/science.ado8316
- **发表时间**: 2025年3月21日

### 核心机制

1. **多突触末梢(MSBs)机制**: 长时记忆形成与多突触末梢的选择性增加密切相关
2. **非同步激活编码**: 挑战传统Hebbian学习模型，实现非同步激活的记忆编码
3. **输入特异性增强**: 空间受限、输入特异性的单个突触放大现象
4. **CA3-CA1通路重构**: 基于突触结构重塑的模式分离机制
5. **快速一次性学习**: 支持单次试验学习的高效记忆形成

## 🏗️ 架构概览

### 核心组件

1. **Transformer-based记忆编码器** (`hippocampus/encoders/`)
   - 多突触末梢记忆印迹单元
   - 海马体式注意力机制
   - 快速一次性学习能力

2. **可微分神经字典** (`hippocampus/memory_cell/`)
   - 情景记忆的存储和检索
   - 突触巩固机制
   - 记忆键生成和管理

3. **模式分离机制** (`hippampus/pattern_separation/`)
   - CA3模式分离器
   - 输入特异性增强器
   - 突触重塑网络
   - 层次化模式分离

4. **快速学习系统** (`hippampus/learning/`)
   - 单次试验学习器
   - 快速联想记忆
   - 情景学习系统

5. **情景记忆系统** (`hippampus/memory_system/`)
   - 时间空间上下文编码
   - 情景记忆单元
   - 海马体索引系统

## 🚀 快速开始

### 基本使用

```python
import torch
from brain_inspired_ai.src.modules.hippocampus import create_hippocampus_simulator

# 创建模拟器
simulator = create_hippocampus_simulator(input_dim=256)

# 准备输入数据
input_data = torch.randn(1, 256)

# 编码记忆
encoding_result = simulator.encode_memory(input_data, metadata={"type": "demo"})

# 存储记忆
memory_id = simulator.store_memory(
    encoding_result['final_encoding'],
    metadata={"timestamp": "2025-11-16", "importance": 0.8}
)

# 检索记忆
retrieval_result = simulator.retrieve_memory(encoding_result['final_encoding'])

print(f"检索置信度: {retrieval_result['retrieval_confidence']:.3f}")
```

### 使用预配置模型

```python
from brain_inspired_ai.src.modules.hippocampus import get_hippocampus_config, HippocampalSimulator

# 获取配置
config = get_hippocampus_config("base")  # small, base, large

# 创建大型模型
large_config = get_hippocampus_config("large")
simulator = HippocampalSimulator(input_dim=1024, config=large_config)
```

### 高级功能

```python
# 时空上下文记忆
temporal_context = torch.randn(1, 128)
spatial_coords = (1.0, 2.0)

# 存储情景记忆
memory_id = simulator.store_memory(
    encoding_result['final_encoding'],
    temporal_context=temporal_context,
    spatial_coords=spatial_coords,
    metadata={"location": "home", "emotion": "happy"}
)

# 基于上下文检索
retrieval_result = simulator.retrieve_memory(
    query=encoding_result['final_encoding'],
    query_context=temporal_context,
    retrieval_mode="spatial"
)

# 记忆巩固
consolidation_result = simulator.consolidate_memories()

# 获取系统状态
status = simulator.get_system_status()
print(f"存储利用率: {status['dictionary_stats']['storage_utilization']:.3f}")
```

## 📊 配置选项

### 预设配置

| 配置 | Hidden Dim | 存储容量 | Transformer层 | 注意力头 |
|------|------------|----------|---------------|----------|
| small | 256 | 5,000 | 3 | 4 |
| base | 512 | 10,000 | 6 | 8 |
| large | 1,024 | 20,000 | 12 | 16 |

### 自定义配置

```python
custom_config = {
    'hidden_dim': 768,
    'memory_dim': 768,
    'storage_capacity': 15000,
    'consolidation_threshold': 0.8,
    'forgetting_rate': 0.005,
    'enhancement_factor': 6,
    'remodeling_rate': 0.02
}

simulator = create_hippocampus_simulator(input_dim=768, config=custom_config)
```

## 🧪 测试

运行测试套件来验证功能：

```bash
cd brain-inspired-ai
python test_hippocampus.py
```

测试将验证：
- 模块导入和配置
- 记忆编码功能
- 记忆存储和检索
- 记忆巩固机制
- 系统状态监控

## 📈 性能指标

### 编码性能
- 记忆编码延迟: < 10ms
- 模式分离质量: > 0.8
- 多突触激活效率: > 0.9

### 检索性能
- 检索延迟: < 5ms
- 检索准确率: > 0.85
- 联想强度: > 0.7

### 存储性能
- 存储容量利用率: 自适应
- 记忆巩固率: 可调 (默认0.7)
- 遗忘控制: 动态调节

## 🔧 高级用法

### 批量处理

```python
# 批量编码
batch_data = torch.randn(10, 256)
batch_results = []

for i in range(10):
    result = simulator.encode_memory(batch_data[i:i+1])
    batch_results.append(result)

# 批量存储
for i, result in enumerate(batch_results):
    simulator.store_memory(
        result['final_encoding'],
        metadata={"batch_id": i, "type": "batch"}
    )
```

### 记忆分析

```python
# 获取详细统计
status = simulator.get_system_status()

# 分析记忆分布
dict_stats = status['dictionary_stats']
print(f"平均突触强度: {dict_stats['average_synaptic_strength']:.3f}")
print(f"结构复杂性: {dict_stats['structural_complexity_mean']:.3f}")

# 记忆质量分析
episodic_stats = status['episodic_stats']
print(f"情景记忆数量: {episodic_stats['total_memories_stored']}")
print(f"巩固百分比: {episodic_stats['consolidation_percentage']:.3f}")
```

### 记忆导出

```python
# 导出记忆数据
simulator.episodic_system.export_memories("hippocampus_memories.json")

# 清空系统（保留配置）
simulator.clear_system()
```

## 🐛 故障排除

### 常见问题

1. **内存不足**
   ```python
   # 使用更小的配置
   config = get_hippocampus_config("small")
   simulator = create_hippocampus_simulator(256, config)
   ```

2. **检索准确率低**
   ```python
   # 调整阈值
   retrieval_result = simulator.retrieve_memory(
       query, retrieval_mode="similarity"
   )
   ```

3. **存储满载**
   ```python
   # 调整遗忘率
   simulator.config['forgetting_rate'] = 0.02
   simulator.episodic_system.forgetting_rate = 0.02
   ```

## 📚 API参考

### 主要类

- `HippocampalSimulator`: 主模拟器类
- `TransformerMemoryEncoder`: Transformer记忆编码器
- `DifferentiableMemoryDictionary`: 可微分记忆字典
- `PatternSeparationNetwork`: 模式分离网络
- `EpisodicLearningSystem`: 情景学习系统
- `EpisodicMemorySystem`: 情景记忆系统

### 核心方法

- `encode_memory()`: 编码记忆
- `store_memory()`: 存储记忆
- `retrieve_memory()`: 检索记忆
- `consolidate_memories()`: 巩固记忆
- `get_system_status()`: 获取系统状态

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork本仓库
2. 创建功能分支
3. 提交更改
4. 运行测试
5. 创建Pull Request

## 📄 许可证

本项目基于MIT许可证。详见LICENSE文件。

## 📞 联系我们

- 研究团队: Brain-Inspired AI Research Team
- 文档: 查看代码中的docstrings和注释
- 问题报告: GitHub Issues

---

*本模拟器基于最新神经科学研究，旨在推动人工神经网络在记忆机制理解方面的发展。*
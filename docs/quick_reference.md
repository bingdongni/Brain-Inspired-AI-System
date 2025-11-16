# 脑启发AI框架 - API快速参考

## 📚 文档导航

- **[完整API参考](api_reference.md)** - 详细的API文档和使用指南
- **[模块依赖关系](module_architecture.md)** - 系统架构和模块间关系
- **[快速开始](#快速开始)** - 立即开始使用
- **[常用示例](#常用示例)** - 代码片段和完整示例

---

## 🚀 快速开始

### 基础设置
```python
from brain_ai import BrainSystem, ConfigManager

# 1. 加载配置
config_manager = ConfigManager('config/default.yaml')
config = config_manager.get('system')

# 2. 创建系统
brain = BrainSystem(config)

# 3. 初始化
brain.initialize()

# 4. 处理数据
result = brain.process(input_data)
```

### 关键类速查表

| 类名 | 用途 | 导入路径 |
|------|------|----------|
| **BrainSystem** | 完整大脑系统 | `from brain_ai import BrainSystem` |
| **HippocampusSimulator** | 海马体记忆系统 | `from brain_ai.hippocampus import HippocampusSimulator` |
| **NeocortexArchitecture** | 新皮层认知架构 | `from brain_ai.neocortex import NeocortexArchitecture` |
| **ContinualLearner** | 持续学习器 | `from brain_ai.lifelong_learning import ContinualLearner` |
| **MemoryInterface** | 统一记忆接口 | `from brain_ai.memory_interface import MemoryInterface` |
| **DynamicRoutingController** | 动态路由控制 | `from brain_ai.dynamic_routing import DynamicRoutingController` |

---

## 🧠 核心模块

### 1. 海马体系统 (Hippocampus)

#### 快速使用
```python
from brain_ai.hippocampus import HippocampusSimulator, HippocampusConfig

# 创建海马体
config = HippocampusConfig()
config.memory_capacity = 10000
hippocampus = HippocampusSimulator(config)

# 编码数据
encoded = hippocampus.encode(data)

# 存储记忆
memory_id = hippocampus.store(encoded_pattern)

# 检索记忆
retrieved = hippocampus.retrieve(query_pattern)

# 模式补全
completed = hippocampus.pattern_completion(partial_pattern)
```

#### 关键方法
- `encode(data)` - 编码输入数据
- `store(pattern)` - 存储记忆模式
- `retrieve(query, threshold)` - 检索记忆
- `pattern_completion(partial)` - 模式补全
- `consolidate(patterns)` - 记忆巩固

---

### 2. 新皮层系统 (Neocortex)

#### 快速使用
```python
from brain_ai.neocortex import NeocortexArchitecture, NeocortexConfig, ArchitectureType

# 创建新皮层
config = NeocortexConfig()
config.architecture_type = ArchitectureType.TONN
neocortex = NeocortexArchitecture(config)

# 层次化处理
result = neocortex.process(input_data, hierarchical=True)

# 生成抽象表示
abstract = neocortex.abstract(features, level=2)

# 整合特征
integrated = neocortex.integrate(hierarchical_features)
```

#### 关键方法
- `process(input_data, hierarchical)` - 层次化处理
- `abstract(features, level)` - 生成抽象表示
- `integrate(features)` - 整合特征
- `classify(features)` - 分类预测

---

### 3. 持续学习 (Continual Learning)

#### 快速使用
```python
from brain_ai.lifelong_learning import ContinualLearner

# 创建学习器
learner = ContinualLearner(
    memory_size=10000,
    consolidation_strategy='ewc'
)

# 学习任务
metrics = learner.learn_task(task_id, X_train, y_train)

# 评估性能
accuracy = learner.evaluate(task_id, X_test, y_test)

# 计算遗忘率
forgetting_rate = learner.calculate_forgetting_rate()
```

#### 关键方法
- `learn_task(task_id, X, y)` - 学习新任务
- `evaluate(task_id, X, y)` - 评估任务性能
- `calculate_forgetting_rate()` - 计算遗忘率
- `consolidate_memory()` - 记忆巩固

---

### 4. 动态路由 (Dynamic Routing)

#### 快速使用
```python
from brain_ai.dynamic_routing import DynamicRoutingController

# 创建路由控制器
router = DynamicRoutingController(
    input_dim=256,
    output_dim=128,
    routing_strategy='attention_based'
)

# 执行路由
result = router.route(input_data)

# 获取路由可视化
viz_data = router.get_routing_visualization()
```

#### 关键方法
- `route(input_data)` - 执行动态路由
- `update_routing_weights(gradient)` - 更新路由权重
- `get_routing_visualization()` - 获取可视化数据

---

### 5. 记忆接口 (Memory Interface)

#### 快速使用
```python
from brain_ai.memory_interface import MemoryInterface

# 创建接口
memory_interface = MemoryInterface({})

# 注册记忆系统
memory_interface.register_memory_system("hippocampus", hippocampus)

# 写入记忆
memory_id = memory_interface.write_memory(data, system_name="hippocampus")

# 读取记忆
retrieved = memory_interface.read_memory(query, system_name="hippocortex")

# 跨系统巩固
consolidation_result = memory_interface.consolidate_across_systems()
```

#### 关键方法
- `register_memory_system(name, system)` - 注册记忆系统
- `write_memory(data, system_name)` - 写入记忆
- `read_memory(query, system_name)` - 读取记忆
- `consolidate_across_systems()` - 跨系统巩固

---

## 🔧 工具模块

### 配置管理
```python
from brain_ai.utils import ConfigManager

config_manager = ConfigManager('config.yaml')
config = config_manager.get('model.parameters')
config_manager.set('training.lr', 0.001)
```

### 日志记录
```python
from brain_ai.utils import Logger

logger = Logger('my_module', level='INFO')
logger.info("处理开始", batch_size=32)
logger.error("处理失败", error=str(e))
```

### 指标收集
```python
from brain_ai.utils import MetricsCollector

metrics = MetricsCollector()
metrics.record('accuracy', 0.95, step=100)
summary = metrics.get_summary()
```

### 数据处理
```python
from brain_ai.utils import DataProcessor

processor = DataProcessor()
processed_data = processor.preprocess(raw_data, 'standard')
batches = processor.create_batches(data, labels, batch_size=32)
```

### 可视化
```python
from brain_ai.utils import Visualization

viz = Visualization(output_dir='plots')
plot_path = viz.plot_learning_curve(metrics_history)
attention_viz = viz.visualize_attention_weights(weights)
```

---

## 📋 常用示例

### 1. 完整系统集成
```python
#!/usr/bin/env python3
from brain_ai import BrainSystem, ConfigManager
import torch

# 设置
config_manager = ConfigManager('config/default.yaml')
config = config_manager.get('system')

# 创建系统
brain = BrainSystem(config)

# 初始化
if brain.initialize():
    # 处理数据
    input_data = torch.randn(32, 784)
    result = brain.process(input_data)
    
    # 存储记忆
    memory_id = brain.store_memory(input_data[0])
    
    # 检索记忆
    retrieved = brain.retrieve_memory(input_data[0])
```

### 2. 多任务学习
```python
from brain_ai.lifelong_learning import ContinualLearner
import numpy as np

learner = ContinualLearner(memory_size=5000)

# 学习多个任务
for task_id in range(5):
    X_train, y_train = generate_task_data(task_id)
    metrics = learner.learn_task(task_id, X_train, y_train)
    
    # 评估所有任务
    for prev_task_id in range(task_id + 1):
        X_test, y_test = generate_task_data(prev_task_id)
        accuracy = learner.evaluate(prev_task_id, X_test, y_test)
        print(f"任务 {prev_task_id} 准确率: {accuracy:.4f}")
```

### 3. 记忆检索系统
```python
from brain_ai.hippocampus import HippocampusSimulator
import torch

hippocampus = HippocampusSimulator()

# 存储知识库
knowledge_base = load_knowledge()
for knowledge in knowledge_base:
    encoded = hippocampus.encode(knowledge['content'])
    hippocampus.store(encoded, metadata=knowledge['metadata'])

# 智能检索
def smart_retrieval(query, category=None):
    query_encoded = hippocampus.encode(query)
    result = hippocampus.retrieve(
        query_encoded, 
        similarity_threshold=0.8
    )
    
    if category:
        result = [r for r in result 
                 if r['metadata'].get('category') == category]
    
    return result
```

### 4. 注意力可视化
```python
from brain_ai.neocortex import NeocortexArchitecture
from brain_ai.utils import Visualization
import torch

neocortex = NeocortexArchitecture()
viz = Visualization()

# 处理数据并获取注意力
data = torch.randn(1, 784)
result = neocortex.process(data, return_attention=True)

# 可视化注意力
if 'attention_weights' in result:
    attention_plot = viz.visualize_attention_weights(
        result['attention_weights'],
        plot_type='heatmap'
    )
    print(f"注意力图保存到: {attention_plot}")
```

### 5. 性能监控
```python
from brain_ai.utils import MetricsCollector, Logger
import time

metrics = MetricsCollector()
logger = Logger('Training', level='INFO')

def train_epoch(model, dataloader, optimizer):
    for batch_idx, (data, target) in enumerate(dataloader):
        start_time = time.time()
        
        # 训练步骤
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        # 记录指标
        batch_time = time.time() - start_time
        metrics.record('batch_time', batch_time)
        metrics.record('batch_loss', loss.item())
        metrics.record('batch_accuracy', (output.argmax(1) == target).float().mean())
        
        if batch_idx % 100 == 0:
            summary = metrics.get_summary(window=100)
            logger.info(
                f"批次 {batch_idx}: "
                f"损失={summary['batch_loss']['mean']:.4f}, "
                f"准确率={summary['batch_accuracy']['mean']:.4f}, "
                f"时间={summary['batch_time']['mean']:.3f}s"
            )
```

---

## 🆘 常见问题

### Q: 如何调整记忆容量？
```python
# 海马体配置
config = HippocampusConfig()
config.memory_capacity = 20000  # 增加容量

# 新皮层配置  
config = NeocortexConfig()
config.memory_capacity = 5000
```

### Q: 如何启用GPU加速？
```python
# 系统配置
config = {
    'device': 'cuda',
    'enable_gpu': True
}

# 或者直接设置
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Q: 如何添加自定义模块？
```python
from brain_ai.core import BaseModule, ModuleType

class MyCustomModule(BaseModule):
    def __init__(self, name: str):
        super().__init__(name, ModuleType.CUSTOM)
    
    def forward(self, x):
        # 实现前向传播
        return x
    
    def initialize(self) -> bool:
        # 初始化逻辑
        return True
```

### Q: 如何处理内存不足？
```python
# 1. 减少批大小
batch_size = 16

# 2. 启用梯度检查点
model.gradient_checkpointing_enable()

# 3. 清理GPU缓存
torch.cuda.empty_cache()

# 4. 使用混合精度
with torch.cuda.amp.autocast():
    output = model(input)
```

### Q: 如何保存和加载模型？
```python
from brain_ai.utils import save_model, load_model

# 保存
save_model(brain_system, 'brain_model.pth')

# 加载
brain_system = load_model('brain_model.pth')
```

---

## 📞 获取帮助

- **API文档**: [完整API参考](api_reference.md)
- **示例代码**: `examples/` 目录
- **问题反馈**: [GitHub Issues](https://github.com/brain-ai/framework/issues)
- **讨论社区**: [Discussions](https://github.com/brain-ai/framework/discussions)
- **邮件支持**: support@brain-ai.org

---

*本快速参考基于脑启发AI框架 v1.0.0，最后更新：2025-11-16*
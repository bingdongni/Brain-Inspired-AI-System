# API参考文档

本文档提供了脑启发AI框架的完整API参考，包括所有核心模块、类、方法和函数的详细说明。

## 目录

- [核心模块](#核心模块)
  - [BrainSystem](#brainsystem)
  - [BaseModule](#basemodule)
  - [Architecture](#architecture)
- [海马体模块](#海马体模块)
  - [HippocampusSimulator](#hippocampussimulator)
  - [EpisodicMemory](#episodicmemory)
  - [FastLearning](#fastlearning)
  - [PatternSeparation](#patternseparation)
  - [TransformerMemoryEncoder](#transformermemoryencoder)
  - [DifferentiableNeuralDictionary](#differentiable神经dictionary)
- [新皮层模块](#新皮层模块)
  - [NeocortexArchitecture](#neocortexarchitecture)
  - [HierarchicalProcessor](#hierarchicalprocessor)
  - [AttentionModule](#attentionmodule)
  - [DecisionModule](#decisionmodule)
  - [EnhancedAttention](#enhancedattention)
- [持续学习模块](#持续学习模块)
  - [ContinualLearner](#continuallearner)
  - [ElasticWeightConsolidation](#elasticweightconsolidation)
  - [GenerativeReplay](#generativereplay)
  - [DynamicExpansion](#dynamicexpansion)
  - [KnowledgeTransfer](#knowledgetransfer)
- [动态路由模块](#动态路由模块)
  - [DynamicRoutingController](#dynamicroutingcontroller)
  - [AdaptiveAllocation](#adaptiveallocation)
  - [EfficiencyOptimization](#efficiencyoptimization)
  - [ReinforcementRouting](#reinforcementrouting)
- [记忆接口模块](#记忆接口模块)
  - [MemoryInterface](#memoryinterface)
  - [AttentionMechanism](#attentionmechanism)
  - [CommunicationController](#communicationcontroller)
  - [ConsolidationEngine](#consolidationengine)
- [高级认知模块](#高级认知模块)
  - [MultiStepReasoner](#multistepreasoner)
  - [AnalogicalLearner](#analogiclearner)
  - [EndToEndTrainingPipeline](#endtoendtrainingpipeline)
  - [PerformanceOptimizer](#performanceoptimizer)
  - [CognitiveSystemIntegrator](#cognitivesystemintegrator)
- [性能优化工具](#性能优化工具)
  - [AutoPerformanceFixer](#autoperformancefixer)
  - [LoopOptimizer](#loopoptimizer)
  - [FileMemoryOptimizer](#filememoryoptimizer)
  - [PerformanceMonitor](#performancemonitor)
- [工具模块](#工具模块)
  - [ConfigManager](#configmanager)
  - [Logger](#logger)
  - [MetricsCollector](#metricscollector)
  - [DataProcessor](#dataprocessor)
- [UI集成模块](#ui集成模块)
  - [React Components](#react-components)
  - [Jupyter Integration](#jupyter-integration)
  - [Web Interface](#web-interface)
- [工具函数](#工具函数)

## 新功能文档指南

详细的新功能文档请参考：
- [高级认知系统文档](../advanced_cognition/README.md)
- [性能优化工具文档](../performance_tools/README.md)
- [UI集成模块文档](../ui_integration/README.md)
- [新功能特性文档](../new_features/README.md)

---

## 核心模块

### BrainSystem

大脑系统核心类，协调各个模块的工作。

```python
class BrainSystem(BaseModule):
    def __init__(self, config: Dict[str, Any]):
        """
        初始化大脑系统
        
        Args:
            config: 配置字典，包含各模块配置参数
        """
    
    def initialize(self) -> bool:
        """
        初始化所有模块
        
        Returns:
            bool: 初始化是否成功
        """
    
    def process(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """
        处理输入数据
        
        Args:
            input_data: 输入的张量数据
            
        Returns:
            Dict[str, Any]: 处理结果字典
        """
    
    def store_memory(self, pattern: torch.Tensor, metadata: Dict[str, Any] = None) -> str:
        """
        存储记忆模式
        
        Args:
            pattern: 记忆模式张量
            metadata: 可选的元数据
            
        Returns:
            str: 记忆ID
        """
    
    def retrieve_memory(self, query: torch.Tensor, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        检索记忆
        
        Args:
            query: 查询向量
            similarity_threshold: 相似度阈值
            
        Returns:
            List[Dict[str, Any]]: 检索到的记忆列表
        """
    
    def consolidate_memory(self) -> None:
        """
        执行记忆巩固
        """
    
    def get_brain_state(self) -> Dict[str, Any]:
        """
        获取当前大脑状态
        
        Returns:
            Dict[str, Any]: 状态字典
        """
    
    def add_region(self, region_name: str, region_config: Dict[str, Any]) -> bool:
        """
        添加大脑区域
        
        Args:
            region_name: 区域名称
            region_config: 区域配置
            
        Returns:
            bool: 是否添加成功
        """
    
    def connect_regions(self, region1: str, region2: str, connection_strength: float) -> bool:
        """
        连接两个大脑区域
        
        Args:
            region1: 第一个区域名
            region2: 第二个区域名
            connection_strength: 连接强度
            
        Returns:
            bool: 是否连接成功
        """
```

**示例使用：**

```python
from brain_ai.core import BrainSystem
import torch

# 创建系统实例
config = {
    'hippocampus': {
        'memory_capacity': 10000,
        'encoding_dimension': 256
    },
    'neocortex': {
        'layers': 8,
        'hierarchical_levels': 4
    }
}
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

---

### BaseModule

所有模块的基础抽象类。

```python
class BaseModule(ABC):
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化基础模块
        
        Args:
            name: 模块名称
            config: 配置字典
        """
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化模块"""
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """清理资源"""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """
        获取模块状态
        
        Returns:
            Dict[str, Any]: 状态信息
        """
    
    def get_metrics(self) -> Dict[str, float]:
        """
        获取性能指标
        
        Returns:
            Dict[str, float]: 指标字典
        """
    
    def reset_metrics(self) -> None:
        """重置性能指标"""
    
    @property
    def name(self) -> str:
        """模块名称"""
    
    @property
    def is_initialized(self) -> bool:
        """初始化状态"""
    
    @property
    def config(self) -> Dict[str, Any]:
        """配置信息"""
```

---

### Architecture

模块化架构管理器。

```python
class Architecture(BaseModule):
    def __init__(self, config: Dict[str, Any]):
        """
        初始化架构管理器
        
        Args:
            config: 架构配置
        """
    
    def register_component(self, name: str, component_class: Type[BaseModule]) -> bool:
        """
        注册组件
        
        Args:
            name: 组件名称
            component_class: 组件类
            
        Returns:
            bool: 注册是否成功
        """
    
    def create_component(self, name: str, component_config: Dict[str, Any]) -> BaseModule:
        """
        创建组件实例
        
        Args:
            name: 组件名称
            component_config: 组件配置
            
        Returns:
            BaseModule: 组件实例
        """
    
    def get_component(self, name: str) -> Optional[BaseModule]:
        """
        获取组件实例
        
        Args:
            name: 组件名称
            
        Returns:
            Optional[BaseModule]: 组件实例或None
        """
    
    def connect_components(self, source: str, target: str, connection_config: Dict[str, Any]) -> bool:
        """
        连接两个组件
        
        Args:
            source: 源组件名
            target: 目标组件名
            connection_config: 连接配置
            
        Returns:
            bool: 连接是否成功
        """
    
    def start_system(self) -> bool:
        """启动整个系统"""
    
    def stop_system(self) -> bool:
        """停止整个系统"""
```

---

## 海马体模块

### HippocampusSimulator

海马体记忆系统核心模拟器。

```python
class HippocampusSimulator(BaseModule):
    def __init__(
        self,
        memory_capacity: int = 10000,
        encoding_dimension: int = 256,
        retrieval_threshold: float = 0.7,
        consolidation_interval: int = 100,
        synaptic_decay: float = 0.95
    ):
        """
        初始化海马体模拟器
        
        Args:
            memory_capacity: 记忆容量
            encoding_dimension: 编码维度
            retrieval_threshold: 检索阈值
            consolidation_interval: 巩固间隔
            synaptic_decay: 突触衰减率
        """
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """
        编码输入数据
        
        Args:
            data: 输入数据张量
            
        Returns:
            torch.Tensor: 编码后的表示
        """
    
    def store(self, pattern: torch.Tensor) -> str:
        """
        存储记忆模式
        
        Args:
            pattern: 记忆模式
            
        Returns:
            str: 记忆ID
        """
    
    def retrieve(self, query: torch.Tensor, threshold: float = None) -> Dict[str, Any]:
        """
        检索记忆
        
        Args:
            query: 查询向量
            threshold: 检索阈值（可选）
            
        Returns:
            Dict[str, Any]: 检索结果
        """
    
    def consolidate(self, patterns: List[torch.Tensor]) -> torch.Tensor:
        """
        巩固记忆模式
        
        Args:
            patterns: 记忆模式列表
            
        Returns:
            torch.Tensor: 巩固后的模式
        """
    
    def pattern_completion(self, partial_pattern: torch.Tensor) -> torch.Tensor:
        """
        模式补全
        
        Args:
            partial_pattern: 部分模式
            
        Returns:
            torch.Tensor: 补全后的模式
        """
    
    def forget(self, pattern_id: str) -> bool:
        """
        遗忘指定模式
        
        Args:
            pattern_id: 模式ID
            
        Returns:
            bool: 是否遗忘成功
        """
    
    def get_memory_strength(self, pattern_id: str) -> float:
        """
        获取记忆强度
        
        Args:
            pattern_id: 记忆ID
            
        Returns:
            float: 记忆强度
        """
    
    def update_memory_strength(self, pattern_id: str, delta: float) -> bool:
        """
        更新记忆强度
        
        Args:
            pattern_id: 记忆ID
            delta: 强度变化
            
        Returns:
            bool: 是否更新成功
        """
```

**示例使用：**

```python
from hippocampus import HippocampusSimulator
import torch

# 创建海马体实例
hippocampus = HippocampusSimulator(
    memory_capacity=5000,
    encoding_dimension=512,
    retrieval_threshold=0.8
)

# 编码和存储
data = torch.randn(10, 784)
encoded_patterns = hippocampus.encode(data)

for i, pattern in enumerate(encoded_patterns):
    memory_id = hippocampus.store(pattern)
    print(f"Stored memory with ID: {memory_id}")

# 检索
query = encoded_patterns[0]
result = hippocampus.retrieve(query)
print(f"Retrieved: {result['pattern']}")

# 模式补全
partial = encoded_patterns[0][:256]  # 部分模式
complete = hippocampus.pattern_completion(partial)
```

---

### EpisodicMemory

情景记忆管理模块。

```python
class EpisodicMemory(BaseModule):
    def __init__(self, max_episodes: int = 1000):
        """
        初始化情景记忆
        
        Args:
            max_episodes: 最大记忆数量
        """
    
    def store_episode(self, episode: Dict[str, Any]) -> str:
        """
        存储情景记忆
        
        Args:
            episode: 情景数据
            
        Returns:
            str: 记忆ID
        """
    
    def retrieve_episodes(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """
        检索情景记忆
        
        Args:
            query: 查询条件
            limit: 返回数量限制
            
        Returns:
            List[Dict[str, Any]]: 情景列表
        """
    
    def link_episodes(self, episode1_id: str, episode2_id: str, relationship: str) -> bool:
        """
        链接两个情景记忆
        
        Args:
            episode1_id: 第一个情景ID
            episode2_id: 第二个情景ID
            relationship: 关系类型
            
        Returns:
            bool: 是否链接成功
        """
    
    def get_episode_timeline(self, episode_id: str) -> List[Dict[str, Any]]:
        """
        获取情景时间线
        
        Args:
            episode_id: 起始情景ID
            
        Returns:
            List[Dict[str, Any]]: 时间线
        """
```

---

### FastLearning

快速学习模块。

```python
class FastLearning(BaseModule):
    def __init__(self, learning_rate: float = 0.1, plasticity_factor: float = 1.0):
        """
        初始化快速学习
        
        Args:
            learning_rate: 学习率
            plasticity_factor: 可塑性因子
        """
    
    def fast_encode(self, data: torch.Tensor, iterations: int = 10) -> torch.Tensor:
        """
        快速编码
        
        Args:
            data: 输入数据
            iterations: 迭代次数
            
        Returns:
            torch.Tensor: 编码结果
        """
    
    def update_synapses(self, input_pattern: torch.Tensor, output_pattern: torch.Tensor) -> None:
        """
        更新突触权重
        
        Args:
            input_pattern: 输入模式
            output_pattern: 输出模式
        """
    
    def calculate_synaptic_potential(self, pre_synaptic: torch.Tensor, post_synaptic: torch.Tensor) -> torch.Tensor:
        """
        计算突触电位
        
        Args:
            pre_synaptic: 突触前神经元活动
            post_synaptic: 突触后神经元活动
            
        Returns:
            torch.Tensor: 突触电位
        """
```

---

### PatternSeparation

模式分离模块。

```python
class PatternSeparation(BaseModule):
    def __init__(self, separation_strength: float = 0.8):
        """
        初始化模式分离
        
        Args:
            separation_strength: 分离强度
        """
    
    def separate_patterns(self, pattern1: torch.Tensor, pattern2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        分离两个相似模式
        
        Args:
            pattern1: 第一个模式
            pattern2: 第二个模式
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 分离后的模式
        """
    
    def calculate_similarity(self, pattern1: torch.Tensor, pattern2: torch.Tensor) -> float:
        """
        计算模式相似度
        
        Args:
            pattern1: 第一个模式
            pattern2: 第二个模式
            
        Returns:
            float: 相似度分数
        """
    
    def enhance_differences(self, patterns: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        增强模式差异
        
        Args:
            patterns: 模式列表
            
        Returns:
            List[torch.Tensor]: 增强后的模式
        """
```

---

## 新皮层模块

### NeocortexArchitecture

新皮层层次化处理架构。

```python
class NeocortexArchitecture(BaseModule):
    def __init__(
        self,
        layers: int = 8,
        hierarchical_levels: int = 4,
        feature_channels: int = 512,
        hierarchical_levels: int = 6
    ):
        """
        初始化新皮层架构
        
        Args:
            layers: 皮层层数
            hierarchical_levels: 层次化层级数
            feature_channels: 特征通道数
            hierarchical_levels: 层次化层级数
        """
    
    def process(self, input_data: torch.Tensor, hierarchical: bool = True) -> List[torch.Tensor]:
        """
        层次化处理输入数据
        
        Args:
            input_data: 输入数据
            hierarchical: 是否进行层次化处理
            
        Returns:
            List[torch.Tensor]: 各层特征列表
        """
    
    def abstract(self, features: torch.Tensor, level: int) -> torch.Tensor:
        """
        抽象指定层级特征
        
        Args:
            features: 特征张量
            level: 抽象层级
            
        Returns:
            torch.Tensor: 抽象特征
        """
    
    def integrate(self, hierarchical_features: List[torch.Tensor]) -> torch.Tensor:
        """
        整合层次化特征
        
        Args:
            hierarchical_features: 层次化特征列表
            
        Returns:
            torch.Tensor: 整合后特征
        """
    
    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """
        基于特征进行分类
        
        Args:
            features: 特征张量
            
        Returns:
            torch.Tensor: 分类结果
        """
    
    def learn_patterns(self, patterns: List[torch.Tensor]) -> Dict[str, Any]:
        """
        学习模式
        
        Args:
            patterns: 模式列表
            
        Returns:
            Dict[str, Any]: 学习结果
        """
    
    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        预测输出
        
        Args:
            input_data: 输入数据
            
        Returns:
            torch.Tensor: 预测结果
        """
```

**示例使用：**

```python
from brain_ai.modules.neocortex import NeocortexArchitecture
import torch

# 创建新皮层实例
neocortex = NeocortexArchitecture(
    layers=8,
    hierarchical_levels=4,
    feature_channels=512
)

# 层次化处理
input_data = torch.randn(16, 784)
hierarchical_features = neocortex.process(input_data, hierarchical=True)

# 各层特征
for i, feature in enumerate(hierarchical_features):
    print(f"Level {i}: {feature.shape}")

# 整合特征
integrated = neocortex.integrate(hierarchical_features)
predictions = neocortex.classify(integrated)
```

---

### HierarchicalProcessor

层次化处理器。

```python
class HierarchicalProcessor(BaseModule):
    def __init__(self, num_levels: int, base_channels: int):
        """
        初始化层次化处理器
        
        Args:
            num_levels: 层级数
            base_channels: 基础通道数
        """
    
    def process_level(self, input_data: torch.Tensor, level: int) -> torch.Tensor:
        """
        处理指定层级
        
        Args:
            input_data: 输入数据
            level: 层级
            
        Returns:
            torch.Tensor: 处理结果
        """
    
    def forward_process(self, input_data: torch.Tensor) -> List[torch.Tensor]:
        """
        前向层次化处理
        
        Args:
            input_data: 输入数据
            
        Returns:
            List[torch.Tensor]: 各层输出
        """
    
    def backward_feedback(self, target_features: List[torch.Tensor]) -> torch.Tensor:
        """
        反向反馈处理
        
        Args:
            target_features: 目标特征列表
            
        Returns:
            torch.Tensor: 反馈结果
        """
    
    def learn_hierarchy(self, training_data: List[torch.Tensor]) -> Dict[str, Any]:
        """
        学习层次化结构
        
        Args:
            training_data: 训练数据列表
            
        Returns:
            Dict[str, Any]: 学习结果
        """
```

---

### AttentionModule

注意力机制模块。

```python
class AttentionModule(BaseModule):
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        初始化注意力模块
        
        Args:
            query_dim: 查询维度
            key_dim: 键维度
            value_dim: 值维度
            num_heads: 注意力头数
            dropout: Dropout率
        """
    
    def compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        计算注意力权重
        
        Args:
            query: 查询张量
            key: 键张量
            value: 值张量
            mask: 注意力掩码
            
        Returns:
            torch.Tensor: 注意力权重
        """
    
    def apply_attention(
        self,
        query: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        应用注意力机制
        
        Args:
            query: 查询张量
            attention_weights: 注意力权重
            
        Returns:
            torch.Tensor: 注意力输出
        """
    
    def multi_head_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        多头注意力计算
        
        Args:
            query: 查询张量
            key: 键张量
            value: 值张量
            mask: 掩码
            
        Returns:
            torch.Tensor: 注意力输出
        """
    
    def calculate_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        计算注意力分布熵
        
        Args:
            attention_weights: 注意力权重
            
        Returns:
            torch.Tensor: 分布熵
        """
```

---

### DecisionModule

决策模块。

```python
class DecisionModule(BaseModule):
    def __init__(self, input_dim: int, num_classes: int):
        """
        初始化决策模块
        
        Args:
            input_dim: 输入维度
            num_classes: 类别数
        """
    
    def make_decision(self, features: torch.Tensor) -> torch.Tensor:
        """
        基于特征做出决策
        
        Args:
            features: 特征张量
            
        Returns:
            torch.Tensor: 决策结果
        """
    
    def calculate_confidence(self, decision: torch.Tensor) -> torch.Tensor:
        """
        计算决策置信度
        
        Args:
            decision: 决策结果
            
        Returns:
            torch.Tensor: 置信度
        """
    
    def update_decision_threshold(self, new_threshold: float) -> None:
        """
        更新决策阈值
        
        Args:
            new_threshold: 新阈值
        """
```

---

## 持续学习模块

### ContinualLearner

持续学习管理器。

```python
class ContinualLearner(BaseModule):
    def __init__(
        self,
        memory_size: int = 10000,
        elasticity: float = 0.1,
        consolidation_strategy: str = 'ewc',
        task_similarity_threshold: float = 0.8
    ):
        """
        初始化持续学习器
        
        Args:
            memory_size: 记忆库大小
            elasticity: 弹性参数
            consolidation_strategy: 巩固策略
            task_similarity_threshold: 任务相似度阈值
        """
    
    def learn_task(self, task_id: int, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """
        学习新任务
        
        Args:
            task_id: 任务ID
            X_train: 训练数据
            y_train: 训练标签
            
        Returns:
            Dict[str, float]: 学习指标
        """
    
    def evaluate(self, task_id: int, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        评估指定任务
        
        Args:
            task_id: 任务ID
            X_test: 测试数据
            y_test: 测试标签
            
        Returns:
            float: 准确率
        """
    
    def consolidate_memory(self) -> None:
        """执行记忆巩固"""
    
    def calculate_forgetting_rate(self) -> float:
        """
        计算遗忘率
        
        Returns:
            float: 遗忘率
        """
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测新数据
        
        Args:
            X: 输入数据
            
        Returns:
            np.ndarray: 预测结果
        """
    
    def get_task_performance(self, task_id: int) -> Dict[str, float]:
        """
        获取任务性能
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict[str, float]: 性能指标
        """
    
    def add_regularization(self, model_parameters: torch.Tensor) -> torch.Tensor:
        """
        添加正则化
        
        Args:
            model_parameters: 模型参数
            
        Returns:
            torch.Tensor: 正则化损失
        """
```

**示例使用：**

```python
from brain_ai.modules import ContinualLearner
import numpy as np

# 创建持续学习器
learner = ContinualLearner(
    memory_size=10000,
    elasticity=0.1,
    consolidation_strategy='ewc'
)

# 多任务学习
tasks_data = [
    (np.random.randn(100, 784), np.random.randint(0, 5, 100)),
    (np.random.randn(100, 784), np.random.randint(5, 10, 100)),
    (np.random.randn(100, 784), np.random.randint(0, 3, 100)),
]

for task_id, (X_train, y_train) in enumerate(tasks_data):
    # 学习任务
    metrics = learner.learn_task(task_id, X_train, y_train)
    print(f"Task {task_id}: {metrics}")
    
    # 评估所有任务
    for prev_task_id in range(task_id + 1):
        prev_X, prev_y = tasks_data[prev_task_id]
        accuracy = learner.evaluate(prev_task_id, prev_X, prev_y)
        print(f"  Task {prev_task_id} accuracy: {accuracy:.4f}")
```

---

### ElasticWeightConsolidation

弹性权重巩固算法。

```python
class ElasticWeightConsolidation(BaseModule):
    def __init__(self, lambda_ewc: float = 1000.0, gamma: float = 0.1):
        """
        初始化EWC
        
        Args:
            lambda_ewc: EWC正则化强度
            gamma: 衰减因子
        """
    
    def compute_fisher_matrix(self, data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        计算Fisher信息矩阵
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            torch.Tensor: Fisher矩阵
        """
    
    def compute_ewc_loss(self, current_params: torch.Tensor, old_params: torch.Tensor, fisher_matrix: torch.Tensor) -> torch.Tensor:
        """
        计算EWC损失
        
        Args:
            current_params: 当前参数
            old_params: 旧参数
            fisher_matrix: Fisher矩阵
            
        Returns:
            torch.Tensor: EWC损失
        """
    
    def update_weights(self, gradient: torch.Tensor, fisher_matrix: torch.Tensor) -> torch.Tensor:
        """
        更新权重
        
        Args:
            gradient: 梯度
            fisher_matrix: Fisher矩阵
            
        Returns:
            torch.Tensor: 更新后的权重
        """
```

---

### GenerativeReplay

生成重放模块。

```python
class GenerativeReplay(BaseModule):
    def __init__(self, generator_network: torch.nn.Module, replay_size: int = 1000):
        """
        初始化生成重放
        
        Args:
            generator_network: 生成网络
            replay_size: 重放样本大小
        """
    
    def generate_replay_samples(self, task_data: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成重放样本
        
        Args:
            task_data: 任务数据
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 重放样本
        """
    
    def train_generator(self, real_data: np.ndarray, epochs: int = 100) -> Dict[str, float]:
        """
        训练生成器
        
        Args:
            real_data: 真实数据
            epochs: 训练轮数
            
        Returns:
            Dict[str, float]: 训练指标
        """
    
    def store_replay_data(self, replay_samples: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        存储重放数据
        
        Args:
            replay_samples: 重放样本
        """
    
    def get_replay_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取重放批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 重放批次
        """
```

---

### DynamicExpansion

动态扩展模块。

```python
class DynamicExpansion(BaseModule):
    def __init__(self, expansion_threshold: float = 0.8, growth_rate: float = 0.1):
        """
        初始化动态扩展
        
        Args:
            expansion_threshold: 扩展阈值
            growth_rate: 增长率
        """
    
    def expand_capacity(self, current_performance: float, target_performance: float) -> Dict[str, Any]:
        """
        扩展网络容量
        
        Args:
            current_performance: 当前性能
            target_performance: 目标性能
            
        Returns:
            Dict[str, Any]: 扩展结果
        """
    
    def calculate_capacity_need(self, task_difficulty: float) -> float:
        """
        计算容量需求
        
        Args:
            task_difficulty: 任务难度
            
        Returns:
            float: 容量需求
        """
    
    def add_neurons(self, layer_name: str, num_neurons: int) -> bool:
        """
        添加神经元
        
        Args:
            layer_name: 层名称
            num_neurons: 神经元数量
            
        Returns:
            bool: 是否添加成功
        """
    
    def add_connections(self, source_layer: str, target_layer: str, num_connections: int) -> bool:
        """
        添加连接
        
        Args:
            source_layer: 源层
            target_layer: 目标层
            num_connections: 连接数量
            
        Returns:
            bool: 是否添加成功
        """
```

---

### KnowledgeTransfer

知识迁移模块。

```python
class KnowledgeTransfer(BaseModule):
    def __init__(self, transfer_coefficient: float = 0.5):
        """
        初始化知识迁移
        
        Args:
            transfer_coefficient: 迁移系数
        """
    
    def extract_knowledge(self, source_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """
        提取源模型知识
        
        Args:
            source_model: 源模型
            
        Returns:
            Dict[str, torch.Tensor]: 知识表示
        """
    
    def transfer_knowledge(self, target_model: torch.nn.Module, knowledge: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        迁移知识到目标模型
        
        Args:
            target_model: 目标模型
            knowledge: 知识表示
            
        Returns:
            Dict[str, float]: 迁移指标
        """
    
    def calculate_task_similarity(self, task1_features: torch.Tensor, task2_features: torch.Tensor) -> float:
        """
        计算任务相似度
        
        Args:
            task1_features: 任务1特征
            task2_features: 任务2特征
            
        Returns:
            float: 相似度分数
        """
    
    def optimize_transfer_strategy(self, source_task_id: int, target_task_id: int) -> Dict[str, Any]:
        """
        优化迁移策略
        
        Args:
            source_task_id: 源任务ID
            target_task_id: 目标任务ID
            
        Returns:
            Dict[str, Any]: 优化策略
        """
```

---

## 动态路由模块

### DynamicRoutingController

动态路由控制器。

```python
class DynamicRoutingController(BaseModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_routing_iterations: int = 3,
        learning_rate: float = 0.01
    ):
        """
        初始化动态路由控制器
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            num_routing_iterations: 路由迭代次数
            learning_rate: 学习率
        """
    
    def route(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        执行动态路由
        
        Args:
            input_data: 输入数据
            
        Returns:
            torch.Tensor: 路由输出
        """
    
    def update_routing_weights(self, gradient: torch.Tensor) -> None:
        """
        更新路由权重
        
        Args:
            gradient: 梯度
        """
    
    def calculate_routing_cost(self, routed_output: torch.Tensor, target_output: torch.Tensor) -> torch.Tensor:
        """
        计算路由成本
        
        Args:
            routed_output: 路由输出
            target_output: 目标输出
            
        Returns:
            torch.Tensor: 路由成本
        """
    
    def get_routing_visualization(self) -> Dict[str, Any]:
        """
        获取路由可视化数据
        
        Returns:
            Dict[str, Any]: 可视化数据
        """
```

**示例使用：**

```python
from brain_ai.modules.dynamic_routing import DynamicRoutingController
import torch

# 创建路由控制器
router = DynamicRoutingController(
    input_dim=256,
    output_dim=128,
    num_routing_iterations=3
)

# 执行路由
input_data = torch.randn(16, 256)
routed_output = router.route(input_data)

print(f"Input shape: {input_data.shape}")
print(f"Routed output shape: {routed_output.shape}")

# 获取可视化数据
viz_data = router.get_routing_visualization()
```

---

### AdaptiveAllocation

自适应资源分配。

```python
class AdaptiveAllocation(BaseModule):
    def __init__(self, total_capacity: int, allocation_strategy: str = 'efficiency'):
        """
        初始化自适应分配
        
        Args:
            total_capacity: 总容量
            allocation_strategy: 分配策略
        """
    
    def allocate_resources(self, demand_forecast: Dict[str, float]) -> Dict[str, int]:
        """
        分配资源
        
        Args:
            demand_forecast: 需求预测
            
        Returns:
            Dict[str, int]: 资源分配结果
        """
    
    def rebalance_allocation(self, current_usage: Dict[str, float], efficiency_metrics: Dict[str, float]) -> Dict[str, int]:
        """
        重新平衡分配
        
        Args:
            current_usage: 当前使用情况
            efficiency_metrics: 效率指标
            
        Returns:
            Dict[str, int]: 新分配方案
        """
    
    def calculate_allocation_efficiency(self, allocation: Dict[str, int], demand: Dict[str, float]) -> float:
        """
        计算分配效率
        
        Args:
            allocation: 分配方案
            demand: 需求情况
            
        Returns:
            float: 效率分数
        """
```

---

### EfficiencyOptimization

效率优化模块。

```python
class EfficiencyOptimization(BaseModule):
    def __init__(self, optimization_method: str = 'gradient_descent'):
        """
        初始化效率优化
        
        Args:
            optimization_method: 优化方法
        """
    
    def optimize_efficiency(self, current_efficiency: float, target_efficiency: float) -> Dict[str, Any]:
        """
        优化效率
        
        Args:
            current_efficiency: 当前效率
            target_efficiency: 目标效率
            
        Returns:
            Dict[str, Any]: 优化结果
        """
    
    def calculate_performance_metrics(self, system_stats: Dict[str, Any]) -> Dict[str, float]:
        """
        计算性能指标
        
        Args:
            system_stats: 系统统计信息
            
        Returns:
            Dict[str, float]: 性能指标
        """
    
    def suggest_optimizations(self, bottleneck_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        建议优化方案
        
        Args:
            bottleneck_analysis: 瓶颈分析
            
        Returns:
            List[Dict[str, Any]]: 优化建议
        """
```

---

### ReinforcementRouting

强化学习路由。

```python
class ReinforcementRouting(BaseModule):
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.01):
        """
        初始化强化学习路由
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
        """
    
    def choose_routing_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        选择路由动作
        
        Args:
            state: 环境状态
            
        Returns:
            torch.Tensor: 路由动作
        """
    
    def update_policy(self, state: torch.Tensor, action: torch.Tensor, reward: float, next_state: torch.Tensor) -> None:
        """
        更新策略
        
        Args:
            state: 当前状态
            action: 执行动作
            reward: 奖励
            next_state: 下一状态
        """
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        获取Q值
        
        Args:
            state: 状态
            
        Returns:
            torch.Tensor: Q值
        """
```

---

## 记忆接口模块

### MemoryInterface

统一记忆接口。

```python
class MemoryInterface(BaseModule):
    def __init__(self, config: Dict[str, Any]):
        """
        初始化记忆接口
        
        Args:
            config: 配置字典
        """
    
    def register_memory_system(self, name: str, memory_system: BaseModule) -> bool:
        """
        注册记忆系统
        
        Args:
            name: 系统名称
            memory_system: 记忆系统实例
            
        Returns:
            bool: 注册是否成功
        """
    
    def read_memory(self, query: Dict[str, Any], system_name: str = None) -> Dict[str, Any]:
        """
        读取记忆
        
        Args:
            query: 查询条件
            system_name: 系统名称（可选）
            
        Returns:
            Dict[str, Any]: 记忆内容
        """
    
    def write_memory(self, data: Dict[str, Any], metadata: Dict[str, Any] = None, system_name: str = None) -> str:
        """
        写入记忆
        
        Args:
            data: 记忆数据
            metadata: 元数据
            system_name: 系统名称
            
        Returns:
            str: 记忆ID
        """
    
    def consolidate_across_systems(self) -> Dict[str, Any]:
        """
        跨系统记忆巩固
        
        Returns:
            Dict[str, Any]: 巩固结果
        """
```

**示例使用：**

```python
from brain_ai.modules.memory_interface import MemoryInterface
from hippocampus import HippocampusSimulator, NeocortexArchitecture

# 创建接口
memory_interface = MemoryInterface({})

# 注册记忆系统
hippocampus = HippocampusSimulator()
neocortex = NeocortexArchitecture()

memory_interface.register_memory_system("hippocampus", hippocampus)
memory_interface.register_memory_system("neocortex", neocortex)

# 写入记忆
memory_data = {"pattern": "user_data", "importance": 0.8}
memory_id = memory_interface.write_memory(memory_data, system_name="hippocampus")

# 读取记忆
retrieved = memory_interface.read_memory({"id": memory_id}, system_name="hippocampus")

# 跨系统巩固
consolidation_result = memory_interface.consolidate_across_systems()
```

---

### AttentionMechanism

注意力机制系统。

```python
class AttentionMechanism(BaseModule):
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        attention_type: str = 'multi_head',
        dropout: float = 0.1
    ):
        """
        初始化注意力机制
        
        Args:
            query_dim: 查询维度
            key_dim: 键维度
            value_dim: 值维度
            attention_type: 注意力类型
            dropout: Dropout率
        """
    
    def focus_attention(self, query: torch.Tensor, context: List[torch.Tensor]) -> torch.Tensor:
        """
        聚焦注意力
        
        Args:
            query: 查询向量
            context: 上下文列表
            
        Returns:
            torch.Tensor: 注意力聚焦结果
        """
    
    def distribute_attention(self, sources: List[torch.Tensor], weight_strategy: str = 'uniform') -> torch.Tensor:
        """
        分配注意力
        
        Args:
            sources: 源列表
            weight_strategy: 权重策略
            
        Returns:
            torch.Tensor: 分配结果
        """
    
    def calculate_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        计算注意力熵
        
        Args:
            attention_weights: 注意力权重
            
        Returns:
            torch.Tensor: 注意力熵
        """
```

---

### CommunicationController

通信控制器。

```python
class CommunicationController(BaseModule):
    def __init__(self, protocols: List[str] = None):
        """
        初始化通信控制器
        
        Args:
            protocols: 支持的协议列表
        """
    
    def send_message(self, message: Dict[str, Any], target_module: str) -> bool:
        """
        发送消息
        
        Args:
            message: 消息内容
            target_module: 目标模块
            
        Returns:
            bool: 发送是否成功
        """
    
    def receive_message(self, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """
        接收消息
        
        Args:
            timeout: 超时时间
            
        Returns:
            Optional[Dict[str, Any]]: 接收的消息
        """
    
    def broadcast_message(self, message: Dict[str, Any], exclude_modules: List[str] = None) -> Dict[str, bool]:
        """
        广播消息
        
        Args:
            message: 消息内容
            exclude_modules: 排除的模块列表
            
        Returns:
            Dict[str, bool]: 广播结果
        """
```

---

### ConsolidationEngine

记忆巩固引擎。

```python
class ConsolidationEngine(BaseModule):
    def __init__(self, consolidation_strategy: str = 'adaptive'):
        """
        初始化巩固引擎
        
        Args:
            consolidation_strategy: 巩固策略
        """
    
    def consolidate_memory(
        self,
        memory_patterns: List[torch.Tensor],
        importance_weights: torch.Tensor = None
    ) -> Dict[str, Any]:
        """
        执行记忆巩固
        
        Args:
            memory_patterns: 记忆模式列表
            importance_weights: 重要性权重
            
        Returns:
            Dict[str, Any]: 巩固结果
        """
    
    def schedule_consolidation(self, memory_pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        安排巩固计划
        
        Args:
            memory_pool: 记忆池
            
        Returns:
            List[Dict[str, Any]]: 巩固计划
        """
    
    def calculate_consolidation_priority(self, memory: Dict[str, Any]) -> float:
        """
        计算巩固优先级
        
        Args:
            memory: 记忆数据
            
        Returns:
            float: 优先级分数
        """
```

---

## 工具模块

### ConfigManager

配置管理器。

```python
class ConfigManager:
    def __init__(self, config_file: str = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            Dict[str, Any]: 配置字典
        """
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            Any: 配置值
        """
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        更新配置
        
        Args:
            config_dict: 配置字典
        """
    
    def save_config(self, output_file: str) -> None:
        """
        保存配置到文件
        
        Args:
            output_file: 输出文件路径
        """
```

---

### Logger

日志记录器。

```python
class Logger:
    def __init__(self, name: str, level: str = 'INFO', log_file: str = None):
        """
        初始化日志记录器
        
        Args:
            name: 记录器名称
            level: 日志级别
            log_file: 日志文件
        """
    
    def debug(self, message: str, **kwargs) -> None:
        """Debug级别日志"""
        pass
    
    def info(self, message: str, **kwargs) -> None:
        """Info级别日志"""
        pass
    
    def warning(self, message: str, **kwargs) -> None:
        """Warning级别日志"""
        pass
    
    def error(self, message: str, **kwargs) -> None:
        """Error级别日志"""
        pass
    
    def critical(self, message: str, **kwargs) -> None:
        """Critical级别日志"""
        pass
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        """
        记录性能指标
        
        Args:
            metrics: 指标字典
            step: 训练步数
        """
```

---

### MetricsCollector

指标收集器。

```python
class MetricsCollector:
    def __init__(self, metrics_config: Dict[str, Any] = None):
        """
        初始化指标收集器
        
        Args:
            metrics_config: 指标配置
        """
    
    def record(self, name: str, value: float, step: int = None) -> None:
        """
        记录指标
        
        Args:
            name: 指标名称
            value: 指标值
            step: 训练步数
        """
    
    def get_summary(self, window: int = None) -> Dict[str, Any]:
        """
        获取指标摘要
        
        Args:
            window: 窗口大小
            
        Returns:
            Dict[str, Any]: 指标摘要
        """
    
    def reset(self) -> None:
        """重置所有指标"""
    
    def export_metrics(self, output_format: str = 'json') -> str:
        """
        导出指标
        
        Args:
            output_format: 导出格式
            
        Returns:
            str: 导出的数据
        """
```

---

### DataProcessor

数据处理器。

```python
class DataProcessor:
    def __init__(self, preprocessing_config: Dict[str, Any] = None):
        """
        初始化数据处理器
        
        Args:
            preprocessing_config: 预处理配置
        """
    
    def preprocess(self, data: np.ndarray, transformation: str = 'standard') -> np.ndarray:
        """
        数据预处理
        
        Args:
            data: 输入数据
            transformation: 变换类型
            
        Returns:
            np.ndarray: 处理后的数据
        """
    
    def augment(self, data: np.ndarray, augmentation_strategy: str = 'random') -> np.ndarray:
        """
        数据增强
        
        Args:
            data: 输入数据
            augmentation_strategy: 增强策略
            
        Returns:
            np.ndarray: 增强后的数据
        """
    
    def create_batches(self, data: np.ndarray, batch_size: int, shuffle: bool = True) -> List[np.ndarray]:
        """
        创建数据批次
        
        Args:
            data: 输入数据
            batch_size: 批次大小
            shuffle: 是否打乱
            
        Returns:
            List[np.ndarray]: 数据批次列表
        """
    
    def normalize(self, data: np.ndarray, normalization_method: str = 'min_max') -> np.ndarray:
        """
        数据归一化
        
        Args:
            data: 输入数据
            normalization_method: 归一化方法
            
        Returns:
            np.ndarray: 归一化后的数据
        """
```

---

## 工具函数

### 通用工具函数

```python
# 文件操作
def load_model(model_path: str) -> torch.nn.Module:
    """加载模型"""
    
def save_model(model: torch.nn.Module, model_path: str) -> None:
    """保存模型"""
    
def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    
def save_config(config: Dict[str, Any], config_path: str) -> None:
    """保存配置文件"""

# 数据处理
def create_data_loader(
    data: np.ndarray, 
    labels: np.ndarray, 
    batch_size: int, 
    shuffle: bool = True
) -> torch.utils.data.DataLoader:
    """创建数据加载器"""

def split_dataset(
    data: np.ndarray, 
    labels: np.ndarray, 
    test_ratio: float = 0.2, 
    val_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """划分数据集"""

# 评估工具
def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """计算准确率"""
    
def calculate_loss(predictions: torch.Tensor, targets: torch.Tensor, loss_fn: torch.nn.Module) -> float:
    """计算损失"""

# 可视化工具
def plot_learning_curve(metrics_history: Dict[str, List[float]], save_path: str = None) -> None:
    """绘制学习曲线"""
    
def visualize_attention_weights(attention_weights: torch.Tensor, save_path: str = None) -> None:
    """可视化注意力权重"""
    
def plot_memory_patterns(memory_patterns: torch.Tensor, save_path: str = None) -> None:
    """可视化记忆模式"""

# 性能监控
def monitor_gpu_usage() -> Dict[str, float]:
    """监控GPU使用情况"""
    
def profile_memory_usage() -> Dict[str, float]:
    """分析内存使用情况"""
    
def benchmark_inference_time(model: torch.nn.Module, test_data: torch.Tensor) -> Dict[str, float]:
    """基准测试推理时间"""
```

---

## 使用示例

### 完整系统集成示例

```python
import torch
import numpy as np
from brain_ai import (
    BrainSystem, HippocampusSimulator, NeocortexArchitecture,
    ContinualLearner, DynamicRoutingController
)
from brain_ai.config import ConfigManager

# 1. 加载配置
config_manager = ConfigManager('config/default.yaml')
config = config_manager.get('system')

# 2. 创建核心系统
brain_system = BrainSystem(config)

# 3. 初始化系统组件
hippocampus = HippocampusSimulator(
    memory_capacity=10000,
    encoding_dimension=512
)

neocortex = NeocortexArchitecture(
    layers=8,
    hierarchical_levels=4,
    feature_channels=512
)

continual_learner = ContinualLearner(
    memory_size=10000,
    consolidation_strategy='ewc'
)

# 4. 集成系统
brain_system.add_region('hippocampus', hippocampus)
brain_system.add_region('neocortex', neocortex)
brain_system.add_learner('continual', continual_learner)

# 5. 连接组件
brain_system.connect_regions('hippocampus', 'neocortex', connection_strength=0.8)

# 6. 启动系统
if brain_system.initialize():
    # 7. 多任务学习
    for task_id in range(5):
        # 生成任务数据
        X_train = np.random.randn(1000, 784)
        y_train = np.random.randint(0, 10, 1000)
        
        # 学习任务
        metrics = brain_system.learn_task(task_id, X_train, y_train)
        
        # 评估性能
        X_test = np.random.randn(200, 784)
        y_test = np.random.randint(0, 10, 200)
        accuracy = brain_system.evaluate(X_test, y_test)
        
        print(f"Task {task_id}: Accuracy = {accuracy:.4f}")

# 8. 获取系统状态
state = brain_system.get_brain_state()
print(f"System State: {state}")
```

## 新功能API文档

本文档涵盖了脑启发AI框架v2.1.0的主要API。对于新增的高级功能，请参考专门的API补充文档：

- **[新功能API补充文档](new_features_api.md)** - 详细的新功能API说明
- **[高级认知系统文档](../advanced_cognition/README.md)** - 高级认知能力
- **[性能优化工具文档](../performance_tools/README.md)** - 性能优化工具
- **[UI集成模块文档](../ui_integration/README.md)** - 用户界面集成
- **[新功能特性文档](../new_features/README.md)** - 最新功能特性

### 新增主要API类

| 类名 | 模块 | 功能描述 |
|------|------|----------|
| `TransformerMemoryEncoder` | hippocampus | Transformer记忆编码器 |
| `DifferentiableNeuralDictionary` | memory_cell | 可微分神经字典 |
| `PatternSeparationNetwork` | pattern_separation | 模式分离网络 |
| `EnhancedAttention` | hippocampus.encoders | 增强注意力机制 |
| `MultiStepReasoner` | advanced_cognition | 多步推理系统 |
| `AnalogicalLearner` | advanced_cognition | 类比学习系统 |
| `AutoPerformanceFixer` | utils | 自动性能修复器 |
| `LoopOptimizer` | utils | 循环优化器 |
| `FileMemoryOptimizer` | utils | 文件内存优化器 |
| `PerformanceMonitor` | utils | 性能监控器 |

这个API参考文档涵盖了脑启发AI框架的所有主要模块和功能。每个模块都提供了详细的方法说明、参数解释和使用示例，帮助开发者快速上手和深入使用这个强大的框架。
# 动态路由系统 (Dynamic Routing System)

基于人工智能的智能路由决策系统，实现自适应、高效、实时的路由优化。

## 系统概述

动态路由系统是一个综合性的路由平台，集成了强化学习、自适应分配和能效优化等多个AI模块，能够在复杂的网络环境中智能选择最优路径。

## 核心模块

### 1. 强化学习路由模块 (`reinforcement_routing/`)

#### Actor-Critic路由器 (`actor_critic.py`)
- 实现智能体Actor-Critic框架用于动态模块选择
- 支持经验回放和策略优化
- 包含完整的训练和推理机制

#### Q-Learning路由器 (`q_learning.py`)
- 基于表格和深度Q学习的路由策略
- 支持ε-贪心探索策略
- 可选的目标网络用于稳定性改进

#### 多智能体路由器 (`multi_agent.py`)
- 实现多个智能体协作的动态路由
- 支持智能体间通信和知识共享
- 采用协作决策机制

#### 路由环境 (`routing_environment.py`)
- 用于训练和测试强化学习路由策略的环境
- 模拟真实的网络模块状态和负载
- 支持多种工作负载模式

### 2. 自适应分配模块 (`adaptive_allocation/`)

#### 动态权重路由器 (`dynamic_weight_routing.py`)
- 基于多目标优化的智能路由决策
- 支持动态权重调整
- 集成权重延迟比计算

#### 预测性早退 (`predictive_early_exit.py`)
- 基于机器学习的智能路由提前退出机制
- 动态置信度阈值调整
- 支持延迟和能耗节省计算

#### 自适应负载均衡器 (`load_balancer.py`)
- 多级负载均衡策略和动态资源分配
- 支持多种负载均衡算法
- 实时性能监控和优化建议

#### 分配控制器 (`allocation_controller.py`)
- 统一管理所有分配策略的协调控制器
- 支持多级优先级和资源约束
- 实时监控和性能分析

### 3. 能效优化模块 (`efficiency_optimization/`)

#### 神经启发路由器 (`neural_inspired_routing.py`)
- 基于生物神经网络的路由决策机制
- 实现神经元状态和突触可塑性
- 支持神经活动模拟和能效分析

#### 智能路径选择器 (`intelligent_path_selector.py`)
- 基于多目标优化的智能路径选择算法
- 支持网络拓扑动态优化
- 集成路径评分和性能分析

### 4. 实时动态路由控制器 (`realtime_routing_controller.py`)

- 整合所有路由模块的统一控制器
- 提供实时监控和性能分析
- 支持系统状态管理和异常处理

## 主要特性

### 🤖 人工智能驱动
- 强化学习：Actor-Critic、Q-Learning、多智能体协作
- 神经网络：神经启发路由算法
- 机器学习：预测性早退、负载预测

### ⚡ 自适应优化
- 动态权重调整
- 实时负载均衡
- 预测性资源分配

### 🔋 能效优化
- 智能路径选择
- 能耗最小化
- 绿色路由算法

### 📊 实时监控
- 性能指标收集
- 系统健康度评估
- 优化建议生成

### 🛡️ 高可靠性
- 多级备份策略
- 异常处理机制
- 系统容错设计

## 快速开始

### 基本使用

```python
from brain_ai.src.modules.dynamic_routing import DynamicRoutingSystem, RoutingRequest

# 创建动态路由系统
with DynamicRoutingSystem() as routing_system:
    # 处理路由请求
    decision = routing_system.process_request(
        source="node_A",
        destination="node_B", 
        priority=8,
        requirements={
            "max_latency": 1.0,
            "min_reliability": 0.9,
            "max_energy": 1.5
        }
    )
    
    print(f"选择的路径: {decision.selected_path}")
    print(f"预估延迟: {decision.estimated_latency:.2f}s")
    print(f"预估能耗: {decision.estimated_energy:.2f}")
    print(f"置信度: {decision.confidence_score:.2f}")
```

### 高级配置

```python
# 自定义配置
config = {
    'reinforcement_learning': {
        'actor_critic': {
            'learning_rate': 1e-3,
            'gamma': 0.99
        }
    },
    'adaptive_allocation': {
        'load_balancer': {
            'balancing_strategy': 'adaptive'
        }
    }
}

routing_system = DynamicRoutingSystem(
    config=config,
    enable_reinforcement_learning=True,
    enable_adaptive_allocation=True,
    enable_efficiency_optimization=True,
    device='cuda'
)
```

### 性能监控

```python
# 获取系统状态
status = routing_system.get_system_status()
print(f"系统健康度: {status['system_health']:.2%}")
print(f"成功率: {status['success_rate']:.2%}")
print(f"平均延迟: {status['avg_latency']:.3f}s")

# 获取详细性能报告
report = routing_system.get_performance_report()
for recommendation in report['recommendations']:
    print(f"优化建议: {recommendation['message']}")
```

## 模块详解

### 强化学习模块

#### Actor-Critic框架
```python
from brain_ai.src.modules.dynamic_routing import ActorCriticRouter

router = ActorCriticRouter(
    state_dim=32,
    action_dim=8,
    learning_rate=1e-3
)

# 选择动作
action = router.select_action(state, training=True)

# 训练
router.train_step(batch_size=32)
```

#### Q-Learning实现
```python
from brain_ai.src.modules.dynamic_routing import QLearningRouter

router = QLearningRouter(
    state_dim=32,
    action_dim=8,
    use_deep_q=True
)

# 选择动作
action = router.select_action(state)

# 获取Q值
q_values = router.get_q_values(state)
```

#### 多智能体协作
```python
from brain_ai.src.modules.dynamic_routing import MultiAgentRouter

router = MultiAgentRouter(
    num_agents=4,
    state_dim=32,
    action_dim=8
)

# 协作决策
decision = router.get_collaborative_decision(state)

# 知识共享
await router.knowledge_sharing()
```

### 自适应分配模块

#### 动态权重路由
```python
from brain_ai.src.modules.dynamic_routing import DynamicWeightRouter

router = DynamicWeightRouter(
    num_paths=8,
    state_dim=32
)

# 选择路径
path_idx = router.select_path(
    traffic_pattern='normal',
    quality_requirements={'low_latency': True}
)
```

#### 预测性早退
```python
from brain_ai.src.modules.dynamic_routing import PredictiveEarlyExit

early_exit = PredictiveEarlyExit(
    num_modules=8,
    state_dim=32,
    confidence_threshold=0.85
)

# 判断是否早退
should_exit, metrics = early_exit.should_early_exit(
    state, current_module_idx, step
)
```

#### 负载均衡
```python
from brain_ai.src.modules.dynamic_routing import AdaptiveLoadBalancer

load_balancer = AdaptiveLoadBalancer(
    num_nodes=8,
    balancing_strategy='adaptive'
)

# 选择服务器
node_idx = load_balancer.select_node(
    request={'size': 100, 'priority': 5}
)
```

### 能效优化模块

#### 神经启发路由
```python
from brain_ai.src.modules.dynamic_routing import NeuralInspiredRouter

router = NeuralInspiredRouter(
    num_neurons=64,
    input_dim=32,
    num_paths=8
)

# 处理输入
route_idx, energy_rating, confidence = router.process_input(state)

# 训练
loss = router.train_step(state, target_route, target_energy, target_confidence)
```

#### 智能路径选择
```python
from brain_ai.src.modules.dynamic_routing import IntelligentPathSelector

selector = IntelligentPathSelector(
    num_nodes=20,
    num_objectives=5
)

# 寻找最优路径
result = selector.find_optimal_path(
    source="node_A",
    target="node_B",
    requirements={'max_energy': 2.0, 'min_reliability': 0.9}
)
```

## 配置参数

### 系统级配置
```python
config = {
    'device': 'cuda',  # 计算设备
    'max_concurrent_routes': 100,  # 最大并发路由数
    'monitoring_interval': 1.0,  # 监控间隔
    'fallback_enabled': True,  # 启用备用策略
}
```

### 模块级配置
```python
# 强化学习模块配置
reinforcement_config = {
    'actor_critic': {
        'state_dim': 32,
        'action_dim': 8,
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'hidden_dim': 128
    },
    'q_learning': {
        'state_dim': 32,
        'action_dim': 8,
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'use_deep_q': True
    },
    'multi_agent': {
        'num_agents': 4,
        'state_dim': 32,
        'action_dim': 8,
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'collaboration_rate': 0.1
    }
}

# 自适应分配模块配置
adaptive_config = {
    'dynamic_weight_routing': {
        'num_paths': 8,
        'state_dim': 32,
        'learning_rate': 1e-3,
        'gamma': 0.99
    },
    'predictive_early_exit': {
        'num_modules': 8,
        'state_dim': 32,
        'confidence_threshold': 0.85,
        'learning_rate': 1e-3
    },
    'load_balancer': {
        'num_nodes': 8,
        'state_dim': 64,
        'learning_rate': 1e-3,
        'balancing_strategy': 'adaptive'
    }
}

# 能效优化模块配置
efficiency_config = {
    'neural_inspired_routing': {
        'num_neurons': 64,
        'input_dim': 32,
        'num_paths': 8,
        'learning_rate': 1e-3
    },
    'intelligent_path_selector': {
        'num_nodes': 20,
        'num_objectives': 5,
        'input_dim': 32,
        'learning_rate': 1e-3
    }
}
```

## 性能指标

### 系统级指标
- **吞吐量**: 每秒处理的路由请求数
- **成功率**: 成功路由的比例
- **平均延迟**: 路由决策的平均时间
- **系统健康度**: 综合系统状态评估

### 模块级指标
- **强化学习**: 训练损失、探索率、收敛速度
- **自适应分配**: 资源利用率、负载均衡度、分配效率
- **能效优化**: 能耗、延迟、可靠性权衡

## 扩展开发

### 添加新的路由算法
```python
from brain_ai.src.modules.dynamic_routing import BaseRoutingModule

class CustomRoutingModule(BaseRoutingModule):
    def __init__(self, config):
        super().__init__(config)
    
    def select_path(self, state, requirements):
        # 实现自定义路径选择逻辑
        return selected_path
    
    def update_feedback(self, actual_performance):
        # 实现反馈更新逻辑
        pass
```

### 集成外部系统
```python
# 通过回调函数集成外部监控
def performance_callback(step, request, decision, actual_performance):
    # 发送监控数据到外部系统
    pass

routing_system = DynamicRoutingSystem()
routing_system.set_performance_callback(performance_callback)
```

## 故障排除

### 常见问题

1. **系统启动失败**
   - 检查配置参数是否正确
   - 确认计算设备可用性
   - 验证模块依赖是否完整

2. **性能下降**
   - 检查系统负载是否过高
   - 调整模块权重配置
   - 优化路由算法参数

3. **内存使用过高**
   - 减少历史数据存储大小
   - 调整批处理大小
   - 优化模型复杂度

### 调试模式
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用性能分析
routing_system.enable_profiling()

# 导出调试数据
routing_system.export_debug_data('debug_data.json')
```

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 贡献

欢迎贡献代码！请阅读 CONTRIBUTING.md 了解贡献指南。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 创建 GitHub Issue
- 发送邮件至项目维护者
- 参与项目讨论

---

**动态路由系统** - 让路由更智能、更高效！
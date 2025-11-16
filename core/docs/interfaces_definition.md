# 系统接口定义文档

## 概述

大脑启发AI系统采用严格的接口驱动架构，确保模块间的标准交互和松耦合设计。本文档详细定义了系统中的所有核心接口。

## 接口分类体系

### 1. 核心基础接口

#### IModule - 通用模块接口
```python
@runtime_checkable
class IModule(Protocol):
    """所有系统模块必须实现的通用接口"""
    
    @property
    def name(self) -> str:
        """模块名称标识符"""
        ...
    
    @property
    def state(self) -> ModuleState:
        """当前运行状态"""
        ...
    
    def initialize(self) -> bool:
        """初始化模块资源，返回成功状态"""
        ...
    
    def start(self) -> bool:
        """启动模块服务，返回成功状态"""
        ...
    
    def stop(self) -> bool:
        """停止模块服务，返回成功状态"""
        ...
    
    def get_info(self) -> Dict[str, Any]:
        """获取模块详细信息"""
        ...
```

**使用示例**:
```python
class MyModule(BaseModule, IModule):
    def initialize(self) -> bool:
        # 模块初始化逻辑
        self.state = ModuleState.INITIALIZED
        return True
    
    def start(self) -> bool:
        # 启动逻辑
        self.state = ModuleState.ACTIVE
        return True
```

### 2. 神经网络接口

#### INeuralComponent - 神经网络组件接口
```python
@runtime_checkable
class INeuralComponent(Protocol):
    """神经网络组件必须实现的接口"""
    
    def forward(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """前向传播计算"""
        ...
    
    def backward(self, gradients: np.ndarray) -> np.ndarray:
        """反向传播梯度计算"""
        ...
    
    def update_parameters(self, learning_rate: float) -> None:
        """根据梯度更新模型参数"""
        ...
    
    def get_parameter_count(self) -> int:
        """获取模型参数总数"""
        ...
    
    def reset_parameters(self) -> None:
        """重置所有参数到初始状态"""
        ...
```

#### ILossFunction - 损失函数接口
```python
@runtime_checkable
class ILossFunction(Protocol):
    """损失函数接口"""
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """计算损失值"""
        ...
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """计算损失函数对预测值的梯度"""
        ...
    
    @property
    def name(self) -> str:
        """损失函数名称"""
        ...
```

#### IOptimizer - 优化器接口
```python
@runtime_checkable
class IOptimizer(Protocol):
    """优化器接口"""
    
    def update(self, parameters: List[np.ndarray], 
              gradients: List[np.ndarray]) -> List[np.ndarray]:
        """根据梯度更新参数"""
        ...
    
    @property
    def learning_rate(self) -> float:
        """当前学习率"""
        ...
    
    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        """设置学习率"""
        ...
    
    def step(self) -> None:
        """执行一步优化迭代"""
        ...
    
    def zero_grad(self) -> None:
        """清空所有梯度缓存"""
        ...
```

#### IActivationFunction - 激活函数接口
```python
@runtime_checkable
class IActivationFunction(Protocol):
    """激活函数接口"""
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """计算激活值"""
        ...
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """计算激活函数的导数"""
        ...
    
    @property
    def name(self) -> str:
        """激活函数名称"""
        ...
```

### 3. 训练框架接口

#### ITrainingComponent - 训练组件接口
```python
@runtime_checkable
class ITrainingComponent(Protocol):
    """训练组件接口"""
    
    def train(self, X: np.ndarray, y: np.ndarray, 
             X_val: Optional[np.ndarray] = None, 
             y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """执行训练过程"""
        ...
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估模型性能"""
        ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """进行预测"""
        ...
    
    def save_model(self, path: str) -> bool:
        """保存模型到文件"""
        ...
    
    def load_model(self, path: str) -> bool:
        """从文件加载模型"""
        ...
```

#### IEvaluationMetrics - 评估指标接口
```python
@runtime_checkable
class IEvaluationMetrics(Protocol):
    """评估指标接口"""
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算所有评估指标"""
        ...
    
    def get_metric_names(self) -> List[str]:
        """获取指标名称列表"""
        ...
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """增量更新指标"""
        ...
    
    def reset(self) -> None:
        """重置所有指标"""
        ...
    
    def get_summary(self) -> Dict[str, float]:
        """获取当前指标摘要"""
        ...
```

#### IDataLoader - 数据加载器接口
```python
@runtime_checkable
class IDataLoader(Protocol):
    """数据加载器接口"""
    
    def load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """从文件加载数据"""
        ...
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                  test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
        """分割训练/验证/测试集"""
        ...
    
    def get_batch(self, batch_size: int, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """获取下一个批次数据"""
        ...
    
    def get_data_shape(self) -> Tuple[int, ...]:
        """获取数据形状信息"""
        ...
    
    def get_num_samples(self) -> int:
        """获取样本总数"""
        ...
```

### 4. 数据处理接口

#### IDataProcessor - 数据处理接口
```python
@runtime_checkable
class IDataProcessor(Protocol):
    """数据处理接口"""
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """数据预处理"""
        ...
    
    def postprocess(self, data: np.ndarray) -> np.ndarray:
        """数据后处理"""
        ...
    
    def normalize(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """数据标准化，返回处理后数据、均值、标准差"""
        ...
    
    def denormalize(self, data: np.ndarray, 
                   mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """数据反标准化"""
        ...
```

#### IEncoder - 编码器接口
```python
@runtime_checkable
class IEncoder(Protocol):
    """编码器接口"""
    
    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """编码输入到潜在空间"""
        ...
    
    def get_latent_dim(self) -> int:
        """获取潜在空间维度"""
        ...
    
    def get_representation(self, inputs: np.ndarray) -> np.ndarray:
        """获取编码表示"""
        ...
```

#### IDecoder - 解码器接口
```python
@runtime_checkable
class IDecoder(Protocol):
    """解码器接口"""
    
    def decode(self, latent: np.ndarray) -> np.ndarray:
        """解码潜在向量到原始空间"""
        ...
    
    def reconstruct(self, inputs: np.ndarray) -> np.ndarray:
        """重构输入数据"""
        ...
    
    @property
    def output_shape(self) -> Tuple[int, ...]:
        """输出数据形状"""
        ...
```

### 5. 可视化接口

#### IVisualizationComponent - 可视化组件接口
```python
@runtime_checkable
class IVisualizationComponent(Protocol):
    """可视化组件接口"""
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                            save_path: Optional[str] = None) -> bool:
        """绘制训练历史曲线"""
        ...
    
    def plot_model_architecture(self, model_info: Dict[str, Any],
                              save_path: Optional[str] = None) -> bool:
        """绘制模型架构图"""
        ...
    
    def plot_activation_maps(self, activations: np.ndarray, 
                           save_path: Optional[str] = None) -> bool:
        """绘制激活图"""
        ...
    
    def plot_performance_metrics(self, metrics: Dict[str, float],
                               save_path: Optional[str] = None) -> bool:
        """绘制性能指标图表"""
        ...
```

### 6. 高级功能接口

#### IAttentionMechanism - 注意力机制接口
```python
@runtime_checkable
class IAttentionMechanism(Protocol):
    """注意力机制接口"""
    
    def forward(self, query: np.ndarray, key: np.ndarray, 
               value: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """注意力前向传播"""
        ...
    
    def get_attention_weights(self) -> np.ndarray:
        """获取注意力权重矩阵"""
        ...
    
    def compute_scores(self, query: np.ndarray, key: np.ndarray) -> np.ndarray:
        """计算注意力分数"""
        ...
```

#### IAutoEncoder - 自编码器接口
```python
@runtime_checkable
class IAutoEncoder(Protocol):
    """自编码器接口"""
    
    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """前向传播，返回编码和解码结果"""
        ...
    
    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """编码"""
        ...
    
    def decode(self, latent: np.ndarray) -> np.ndarray:
        """解码"""
        ...
    
    def reconstruct(self, inputs: np.ndarray) -> np.ndarray:
        """重构输入"""
        ...
    
    def get_reconstruction_loss(self, x: np.ndarray, x_recon: np.ndarray) -> np.ndarray:
        """计算重构损失"""
        ...
```

#### IReinforcementLearning - 强化学习接口
```python
@runtime_checkable
class IReinforcementLearning(Protocol):
    """强化学习接口"""
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """根据状态选择动作"""
        ...
    
    def learn(self, experiences: List[Dict[str, Any]]) -> float:
        """从经验中学习"""
        ...
    
    def store_experience(self, state: np.ndarray, action: int, 
                        reward: float, next_state: np.ndarray, done: bool) -> None:
        """存储经验到经验回放池"""
        ...
    
    def update_target(self) -> None:
        """更新目标网络"""
        ...
```

### 7. 高级AI技术接口

#### IMetaLearning - 元学习接口
```python
@runtime_checkable
class IMetaLearning(Protocol):
    """元学习接口"""
    
    def meta_train(self, task_batch: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
        """元训练过程"""
        ...
    
    def adaptation(self, support_set: Dict[str, np.ndarray]) -> float:
        """对新任务进行快速适应"""
        ...
    
    def get_initial_parameters(self) -> List[np.ndarray]:
        """获取元学习初始化参数"""
        ...
    
    def update_initial_parameters(self, gradients: List[np.ndarray]) -> None:
        """更新元学习初始化参数"""
        ...
```

#### IContinualLearning - 持续学习接口
```python
@runtime_checkable
class IContinualLearning(Protocol):
    """持续学习接口"""
    
    def learn_task(self, task_data: Dict[str, np.ndarray], task_id: int) -> None:
        """学习新任务"""
        ...
    
    def avoid_forgetting(self, previous_tasks: List[Dict[str, np.ndarray]]) -> None:
        """避免灾难性遗忘"""
        ...
    
    def evaluate_all_tasks(self) -> Dict[int, float]:
        """评估所有已学任务的性能"""
        ...
    
    def get_task_performance(self, task_id: int) -> float:
        """获取特定任务的性能"""
        ...
```

#### IInterpretability - 可解释性接口
```python
@runtime_checkable
class IInterpretability(Protocol):
    """模型可解释性接口"""
    
    def explain_prediction(self, inputs: np.ndarray, prediction: np.ndarray) -> Dict[str, Any]:
        """解释预测结果"""
        ...
    
    def feature_importance(self, inputs: np.ndarray) -> np.ndarray:
        """计算特征重要性"""
        ...
    
    def visualize_attribution(self, attribution: np.ndarray, 
                            save_path: Optional[str] = None) -> bool:
        """可视化归因分析"""
        ...
    
    def get_rule_based_explanation(self, inputs: np.ndarray) -> List[str]:
        """获取基于规则的解释"""
        ...
```

## 接口组合模式

### 1. 基础组合
```python
class IBasicNeuralNetwork(INeuralComponent, ITrainingComponent, IModule):
    """基础神经网络接口组合"""
    pass
```

### 2. 高级组合
```python
class IAdvancedModel(IBasicNeuralNetwork, IRegularization, IActivationFunction):
    """高级模型接口组合"""
    pass
```

### 3. 完整系统组合
```python
class IFullStackMLSystem(IBasicNeuralNetwork, IDataProcessor, 
                        IEvaluationMetrics, IVisualizationComponent, IDataLoader):
    """完整机器学习系统接口组合"""
    pass
```

## 接口验证机制

### 1. 运行时验证
```python
def validate_interface(implementation: Any, interface: type) -> bool:
    """验证实现是否满足接口要求"""
    if not hasattr(implementation, '__class__'):
        return False
    
    # 检查关键方法是否存在
    methods = [attr for attr in dir(interface) if not attr.startswith('_')]
    
    for method in methods:
        if not hasattr(implementation, method):
            return False
    
    return True
```

### 2. 兼容性检查
```python
def interface_compatibility_check(implementation: Any, 
                                 required_interfaces: List[type]) -> Dict[str, bool]:
    """接口兼容性检查"""
    results = {}
    
    for interface in required_interfaces:
        is_compatible = validate_interface(implementation, interface)
        results[interface.__name__] = is_compatible
    
    return results
```

## 接口注册和管理

### 1. 全局接口注册表
```python
# 注册核心接口
register_interface('IModule', IModule)
register_interface('INeuralComponent', INeuralComponent)
register_interface('ITrainingComponent', ITrainingComponent)
# ... 更多接口注册

def get_interface(name: str) -> Optional[type]:
    """获取已注册的接口"""
    return _interface_registry.get(name)

def list_interfaces() -> List[str]:
    """列出所有已注册的接口"""
    return list(_interface_registry.keys())
```

### 2. 接口实现发现
```python
def find_implementations(interface_name: str) -> List[str]:
    """查找实现指定接口的所有组件"""
    return interface_registry.get_implementations(interface_name)

def is_implementation_of(impl_name: str, interface_name: str) -> bool:
    """检查组件是否实现了指定接口"""
    return interface_registry.is_implementation_of(impl_name, interface_name)
```

## 最佳实践

### 1. 接口设计原则
- **单一职责**: 每个接口专注特定功能
- **稳定性**: 接口变更最小化
- **一致性**: 命名和参数约定统一
- **可扩展**: 支持向后兼容的扩展

### 2. 接口实现指南
```python
class MyNeuralNetwork(BaseModule, INeuralComponent, ITrainingComponent):
    """神经网络实现示例"""
    
    def __init__(self, config: ModuleConfig):
        super().__init__(config)
        # 实现接口要求的所有方法
```

### 3. 接口适配器模式
```python
class LegacyModelAdapter(BaseModule, INeuralComponent):
    """为旧模型创建适配器"""
    
    def __init__(self, legacy_model):
        super().__init__(ModuleConfig("adapter"))
        self.legacy_model = legacy_model
    
    def forward(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        # 调用旧模型的预测方法
        return self.legacy_model.predict(inputs)
    
    # 实现其他必需的方法...
```

### 4. 接口版本管理
```python
# 接口版本控制
class IModuleV2(IModule):
    """接口版本2，包含新增方法"""
    
    def pause(self) -> bool:
        """新增：暂停模块"""
        ...
    
    def resume(self) -> bool:
        """新增：恢复模块"""
        ...

# 版本兼容检查
def check_interface_version(implementation: Any, 
                          interface: type, 
                          min_version: str) -> bool:
    """检查接口版本兼容性"""
    # 版本检查逻辑
    ...
```

## 总结

这套接口系统为大脑启发AI系统提供了：

1. **标准化交互**: 统一的接口规范确保模块间无缝协作
2. **类型安全**: Protocol确保接口实现的类型一致性
3. **灵活扩展**: 支持插件和新功能的无缝集成
4. **运行时验证**: 动态检查接口兼容性和实现正确性
5. **版本控制**: 支持接口演进和向后兼容

通过这些精心设计的接口，开发者可以构建模块化、可维护、可扩展的AI系统，同时保持代码的清晰性和一致性。
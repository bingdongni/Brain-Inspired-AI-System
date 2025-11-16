"""
系统接口定义
==========

定义了大脑启发AI系统的核心接口，确保组件间的标准交互和模块兼容性。
提供了清晰的行为契约和扩展机制。

主要特性:
- 核心模块接口
- 神经网络组件接口
- 训练组件接口
- 数据处理接口
- 评估和监控接口
- 可扩展的插件接口

作者: Brain-Inspired AI Team
创建时间: 2025-11-16
"""

import abc
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Protocol, runtime_checkable
from enum import Enum

from .base_module import ModuleState


class IModule(Protocol):
    """通用模块接口"""
    
    @property
    def name(self) -> str:
        """模块名称"""
        ...
    
    @property
    def state(self) -> ModuleState:
        """模块状态"""
        ...
    
    def initialize(self) -> bool:
        """初始化模块"""
        ...
    
    def start(self) -> bool:
        """启动模块"""
        ...
    
    def stop(self) -> bool:
        """停止模块"""
        ...
    
    def get_info(self) -> Dict[str, Any]:
        """获取模块信息"""
        ...


class INeuralComponent(Protocol):
    """神经网络组件接口"""
    
    def forward(self, inputs: np.ndarray, training: bool = False) -> np.ndarray:
        """前向传播"""
        ...
    
    def backward(self, gradients: np.ndarray) -> np.ndarray:
        """反向传播"""
        ...
    
    def update_parameters(self, learning_rate: float) -> None:
        """更新参数"""
        ...
    
    def get_parameter_count(self) -> int:
        """获取参数数量"""
        ...
    
    def reset_parameters(self) -> None:
        """重置参数"""
        ...


class ITrainingComponent(Protocol):
    """训练组件接口"""
    
    def train(self, X: np.ndarray, y: np.ndarray, 
             X_val: Optional[np.ndarray] = None, 
             y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """训练模型"""
        ...
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """评估模型"""
        ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        ...
    
    def save_model(self, path: str) -> bool:
        """保存模型"""
        ...
    
    def load_model(self, path: str) -> bool:
        """加载模型"""
        ...


class IDataProcessor(Protocol):
    """数据处理组件接口"""
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """数据预处理"""
        ...
    
    def postprocess(self, data: np.ndarray) -> np.ndarray:
        """数据后处理"""
        ...
    
    def normalize(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """数据标准化"""
        ...
    
    def denormalize(self, data: np.ndarray, 
                   mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """数据反标准化"""
        ...


class IEvaluationMetrics(Protocol):
    """评估指标接口"""
    
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        ...
    
    def get_metric_names(self) -> List[str]:
        """获取指标名称列表"""
        ...
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """更新指标"""
        ...
    
    def reset(self) -> None:
        """重置指标"""
        ...
    
    def get_summary(self) -> Dict[str, float]:
        """获取指标摘要"""
        ...


class IVisualizationComponent(Protocol):
    """可视化组件接口"""
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                            save_path: Optional[str] = None) -> bool:
        """绘制训练历史"""
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
        """绘制性能指标"""
        ...


class ILossFunction(Protocol):
    """损失函数接口"""
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """计算损失值"""
        ...
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """计算损失梯度"""
        ...
    
    @property
    def name(self) -> str:
        """损失函数名称"""
        ...


class IOptimizer(Protocol):
    """优化器接口"""
    
    def update(self, parameters: List[np.ndarray], 
              gradients: List[np.ndarray]) -> List[np.ndarray]:
        """更新参数"""
        ...
    
    @property
    def learning_rate(self) -> float:
        """学习率"""
        ...
    
    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        """设置学习率"""
        ...
    
    def step(self) -> None:
        """执行一步优化"""
        ...
    
    def zero_grad(self) -> None:
        """清空梯度"""
        ...


class IDataLoader(Protocol):
    """数据加载器接口"""
    
    def load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载数据"""
        ...
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                  test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
        """数据分割"""
        ...
    
    def get_batch(self, batch_size: int, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """获取批次数据"""
        ...
    
    def get_data_shape(self) -> Tuple[int, ...]:
        """获取数据形状"""
        ...
    
    def get_num_samples(self) -> int:
        """获取样本数量"""
        ...


class IRegularization(Protocol):
    """正则化接口"""
    
    def apply(self, parameters: List[np.ndarray]) -> float:
        """应用正则化"""
        ...
    
    def gradient(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """计算正则化梯度"""
        ...
    
    @property
    def strength(self) -> float:
        """正则化强度"""
        ...


class IActivationFunction(Protocol):
    """激活函数接口"""
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """激活函数"""
        ...
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """激活函数梯度"""
        ...
    
    @property
    def name(self) -> str:
        """激活函数名称"""
        ...


class IAttentionMechanism(Protocol):
    """注意力机制接口"""
    
    def forward(self, query: np.ndarray, key: np.ndarray, 
               value: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """注意力前向传播"""
        ...
    
    def get_attention_weights(self) -> np.ndarray:
        """获取注意力权重"""
        ...
    
    def compute_scores(self, query: np.ndarray, key: np.ndarray) -> np.ndarray:
        """计算注意力分数"""
        ...


class IEncoder(Protocol):
    """编码器接口"""
    
    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """编码输入"""
        ...
    
    def get_latent_dim(self) -> int:
        """获取潜在空间维度"""
        ...
    
    def get_representation(self, inputs: np.ndarray) -> np.ndarray:
        """获取表示向量"""
        ...


class IDecoder(Protocol):
    """解码器接口"""
    
    def decode(self, latent: np.ndarray) -> np.ndarray:
        """解码潜在向量"""
        ...
    
    def reconstruct(self, inputs: np.ndarray) -> np.ndarray:
        """重构输入"""
        ...
    
    @property
    def output_shape(self) -> Tuple[int, ...]:
        """输出形状"""
        ...


class IAutoEncoder(Protocol):
    """自编码器接口"""
    
    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """前向传播，返回重构结果和潜在表示"""
        ...
    
    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """编码"""
        ...
    
    def decode(self, latent: np.ndarray) -> np.ndarray:
        """解码"""
        ...
    
    def reconstruct(self, inputs: np.ndarray) -> np.ndarray:
        """重构"""
        ...
    
    def get_reconstruction_loss(self, x: np.ndarray, x_recon: np.ndarray) -> np.ndarray:
        """计算重构损失"""
        ...


class IGAN(Protocol):
    """生成对抗网络接口"""
    
    def train_step(self, real_samples: np.ndarray) -> Dict[str, float]:
        """训练一步"""
        ...
    
    def generate(self, noise: np.ndarray) -> np.ndarray:
        """生成样本"""
        ...
    
    def discriminator_loss(self, real_output: np.ndarray, 
                         fake_output: np.ndarray) -> np.ndarray:
        """判别器损失"""
        ...
    
    def generator_loss(self, fake_output: np.ndarray) -> np.ndarray:
        """生成器损失"""
        ...


class IReinforcementLearning(Protocol):
    """强化学习接口"""
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作"""
        ...
    
    def learn(self, experiences: List[Dict[str, Any]]) -> float:
        """学习"""
        ...
    
    def store_experience(self, state: np.ndarray, action: int, 
                        reward: float, next_state: np.ndarray, done: bool) -> None:
        """存储经验"""
        ...
    
    def update_target(self) -> None:
        """更新目标网络"""
        ...


class ICurriculumLearning(Protocol):
    """课程学习接口"""
    
    def get_difficulty_level(self, sample: np.ndarray) -> float:
        """获取样本难度"""
        ...
    
    def select_samples(self, difficulty_threshold: float) -> List[int]:
        """选择样本"""
        ...
    
    def update_difficulty(self, performance: float) -> None:
        """更新难度"""
        ...
    
    def get_learning_progress(self) -> float:
        """获取学习进度"""
        ...


class IFederatedLearning(Protocol):
    """联邦学习接口"""
    
    def local_train(self, global_weights: List[np.ndarray]) -> List[np.ndarray]:
        """本地训练"""
        ...
    
    def aggregate_weights(self, local_weights: List[List[np.ndarray]]) -> List[np.ndarray]:
        """聚合权重"""
        ...
    
    def send_weights(self, weights: List[np.ndarray]) -> None:
        """发送权重"""
        ...
    
    def receive_weights(self) -> List[np.ndarray]:
        """接收权重"""
        ...


class IMetaLearning(Protocol):
    """元学习接口"""
    
    def meta_train(self, task_batch: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
        """元训练"""
        ...
    
    def adaptation(self, support_set: Dict[str, np.ndarray]) -> float:
        """适应新任务"""
        ...
    
    def get_initial_parameters(self) -> List[np.ndarray]:
        """获取初始化参数"""
        ...
    
    def update_initial_parameters(self, gradients: List[np.ndarray]) -> None:
        """更新初始化参数"""
        ...


class INeuralArchitectureSearch(Protocol):
    """神经架构搜索接口"""
    
    def search(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """架构搜索"""
        ...
    
    def evaluate_architecture(self, architecture: Dict[str, Any]) -> float:
        """评估架构"""
        ...
    
    def mutate_architecture(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """变异架构"""
        ...
    
    def crossover_architectures(self, arch1: Dict[str, Any], 
                              arch2: Dict[str, Any]) -> Dict[str, Any]:
        """架构交叉"""
        ...


class IContinualLearning(Protocol):
    """持续学习接口"""
    
    def learn_task(self, task_data: Dict[str, np.ndarray], task_id: int) -> None:
        """学习新任务"""
        ...
    
    def avoid_forgetting(self, previous_tasks: List[Dict[str, np.ndarray]]) -> None:
        """避免遗忘"""
        ...
    
    def evaluate_all_tasks(self) -> Dict[int, float]:
        """评估所有任务"""
        ...
    
    def get_task_performance(self, task_id: int) -> float:
        """获取任务性能"""
        ...


class IInterpretability(Protocol):
    """可解释性接口"""
    
    def explain_prediction(self, inputs: np.ndarray, prediction: np.ndarray) -> Dict[str, Any]:
        """解释预测"""
        ...
    
    def feature_importance(self, inputs: np.ndarray) -> np.ndarray:
        """特征重要性"""
        ...
    
    def visualize_attribution(self, attribution: np.ndarray, 
                            save_path: Optional[str] = None) -> bool:
        """可视化归因"""
        ...
    
    def get_rule_based_explanation(self, inputs: np.ndarray) -> List[str]:
        """基于规则的解释"""
        ...


class IFairnessMetrics(Protocol):
    """公平性指标接口"""
    
    def compute_demographic_parity(self, y_pred: np.ndarray, 
                                 sensitive_attr: np.ndarray) -> float:
        """人口统计平等性"""
        ...
    
    def compute_equalized_odds(self, y_true: np.ndarray, y_pred: np.ndarray,
                             sensitive_attr: np.ndarray) -> float:
        """均等几率"""
        ...
    
    def compute_equalized_opportunity(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    sensitive_attr: np.ndarray) -> float:
        """均等机会"""
        ...


class IDifferentialPrivacy(Protocol):
    """差分隐私接口"""
    
    def add_noise(self, gradients: List[np.ndarray], 
                 epsilon: float, delta: float = 1e-5) -> List[np.ndarray]:
        """添加噪声"""
        ...
    
    def compute_sensitivity(self, function_outputs: List[np.ndarray]) -> List[np.ndarray]:
        """计算敏感度"""
        ...
    
    def calibrate_noise(self, epsilon: float, sensitivity: float) -> float:
        """校准噪声"""
        ...


class IQuantumNeuralNetwork(Protocol):
    """量子神经网络接口"""
    
    def forward(self, quantum_circuit: Any, inputs: np.ndarray) -> np.ndarray:
        """量子前向传播"""
        ...
    
    def quantum_backpropagation(self, circuit: Any, gradients: np.ndarray) -> np.ndarray:
        """量子反向传播"""
        ...
    
    def measure_expectation(self, observable: Any) -> float:
        """测量期望值"""
        ...
    
    def quantum_state_fidelity(self, state1: Any, state2: Any) -> float:
        """量子态保真度"""
        ...


# 接口验证装饰器
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


def interface_compatibility_check(implementation: Any, 
                                 required_interfaces: List[type]) -> Dict[str, bool]:
    """接口兼容性检查"""
    results = {}
    
    for interface in required_interfaces:
        is_compatible = validate_interface(implementation, interface)
        results[interface.__name__] = is_compatible
    
    return results


# 预定义的接口组合
class IBasicNeuralNetwork(INeuralComponent, ITrainingComponent, IModule):
    """基础神经网络接口组合"""
    pass


class IAdvancedModel(IBasicNeuralNetwork, IRegularization, IActivationFunction):
    """高级模型接口组合"""
    pass


class IFullStackMLSystem(IBasicNeuralNetwork, IDataProcessor, 
                        IEvaluationMetrics, IVisualizationComponent, IDataLoader):
    """完整机器学习系统接口组合"""
    pass


# 接口注册表
_interface_registry = {}

def register_interface(name: str, interface_class: type):
    """注册接口"""
    _interface_registry[name] = interface_class

def get_interface(name: str) -> Optional[type]:
    """获取接口"""
    return _interface_registry.get(name)

def list_interfaces() -> List[str]:
    """列出所有注册的接口"""
    return list(_interface_registry.keys())


# 注册核心接口
register_interface('IModule', IModule)
register_interface('INeuralComponent', INeuralComponent)
register_interface('ITrainingComponent', ITrainingComponent)
register_interface('IDataProcessor', IDataProcessor)
register_interface('IEvaluationMetrics', IEvaluationMetrics)
register_interface('IVisualizationComponent', IVisualizationComponent)
register_interface('ILossFunction', ILossFunction)
register_interface('IOptimizer', IOptimizer)
register_interface('IDataLoader', IDataLoader)
register_interface('IRegularization', IRegularization)
register_interface('IActivationFunction', IActivationFunction)
register_interface('IAttentionMechanism', IAttentionMechanism)
register_interface('IEncoder', IEncoder)
register_interface('IDecoder', IDecoder)
register_interface('IAutoEncoder', IAutoEncoder)
register_interface('IGAN', IGAN)
register_interface('IReinforcementLearning', IReinforcementLearning)
register_interface('ICurriculumLearning', ICurriculumLearning)
register_interface('IFederatedLearning', IFederatedLearning)
register_interface('IMetaLearning', IMetaLearning)
register_interface('INeuralArchitectureSearch', INeuralArchitectureSearch)
register_interface('IContinualLearning', IContinualLearning)
register_interface('IInterpretability', IInterpretability)
register_interface('IFairnessMetrics', IFairnessMetrics)
register_interface('IDifferentialPrivacy', IDifferentialPrivacy)
register_interface('IQuantumNeuralNetwork', IQuantumNeuralNetwork)
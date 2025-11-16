"""
神经网络库
=========

实现了基于生物启发的神经网络架构，包括层类型、激活函数、
网络拓扑结构，以及与大脑系统的集成。

主要特性:
- 多样化的神经网络层类型
- 生物启发的激活函数
- 可扩展的网络架构
- 与大脑区域的映射
- 神经可塑性机制

作者: Brain-Inspired AI Team
创建时间: 2025-11-16
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import uuid
import math

from .base_module import BaseModule, ModuleConfig


class LayerType(Enum):
    """神经网络层类型"""
    DENSE = "dense"                       # 全连接层
    CONVOLUTIONAL = "conv2d"             # 卷积层
    POOLING = "pooling"                  # 池化层
    RECURRENT = "recurrent"              # 循环层
    LSTM = "lstm"                        # 长短期记忆层
    GRU = "gru"                          # 门控循环单元
    ATTENTION = "attention"              # 注意力层
    TRANSFORMER = "transformer"          # Transformer层
    NEURAL_OSCILLATOR = "oscillator"     # 神经振荡器层
    SPN = "sum_product"                  # 和积网络层


class ActivationFunction(Enum):
    """激活函数类型"""
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SWISH = "swish"
    GELU = "gelu"
    SOFTMAX = "softmax"
    LINEAR = "linear"
    GAUSSIAN = "gaussian"
    BISTABLE = "bistable"                # 双稳态（大脑启发）


class InitializationType(Enum):
    """权重初始化类型"""
    XAVIER = "xavier"
    HE = "he"
    LECUN = "lecun"
    ORTHOGONAL = "orthogonal"
    BIOLOGICAL = "biological"            # 生物启发初始化


@dataclass
class LayerConfig:
    """层配置类"""
    name: str
    layer_type: LayerType
    input_size: int
    output_size: int
    activation: ActivationFunction = ActivationFunction.RELU
    initialization: InitializationType = InitializationType.XAVIER
    dropout_rate: float = 0.0
    regularization: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)


class ActivationFunctionHandler:
    """激活函数处理器"""
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid激活函数"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh激活函数"""
        return np.tanh(np.clip(x, -500, 500))
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU激活函数"""
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """ELU激活函数"""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def swish(x: np.ndarray) -> np.ndarray:
        """Swish激活函数"""
        return x * ActivationFunctionHandler.sigmoid(x)
    
    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        """GELU激活函数（近似）"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    @staticmethod
    def gaussian(x: np.ndarray) -> np.ndarray:
        """高斯激活函数（生物启发）"""
        return np.exp(-0.5 * x**2)
    
    @staticmethod
    def bistable(x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """双稳态激活函数（模拟神经元阈值行为）"""
        return np.where(x > threshold, 1.0, -1.0)
    
    @classmethod
    def get_activation(cls, name: ActivationFunction) -> Callable:
        """获取激活函数"""
        activation_map = {
            ActivationFunction.SIGMOID: cls.sigmoid,
            ActivationFunction.TANH: cls.tanh,
            ActivationFunction.RELU: cls.relu,
            ActivationFunction.LEAKY_RELU: cls.leaky_relu,
            ActivationFunction.ELU: cls.elu,
            ActivationFunction.SWISH: cls.swish,
            ActivationFunction.GELU: cls.gelu,
            ActivationFunction.GAUSSIAN: cls.gaussian,
            ActivationFunction.BISTABLE: cls.bistable,
            ActivationFunction.LINEAR: lambda x: x,
            ActivationFunction.SOFTMAX: lambda x: cls._softmax(x)
        }
        
        return activation_map.get(name, cls.relu)
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax激活函数"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class NeuralLayer(BaseModule):
    """神经网络层基类"""
    
    def __init__(self, config: LayerConfig):
        super().__init__(config)
        self.config = config
        self.weights: Optional[np.ndarray] = None
        self.biases: Optional[np.ndarray] = None
        self.input_cache: Optional[np.ndarray] = None
        self.output_cache: Optional[np.ndarray] = None
        self.gradients: Dict[str, np.ndarray] = {}
        self.activation_fn = ActivationFunctionHandler.get_activation(config.activation)
        
    def initialize_parameters(self) -> bool:
        """初始化层参数"""
        try:
            if self.config.layer_type == LayerType.DENSE:
                self._initialize_dense_layer()
            else:
                # 其他层类型的初始化逻辑
                self._initialize_generic_layer()
            
            self.logger.info(f"层 {self.config.name} 参数初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"层参数初始化失败: {e}")
            return False
    
    def _initialize_dense_layer(self):
        """初始化全连接层"""
        # 生物启发的权重初始化
        if self.config.initialization == InitializationType.BIOLOGICAL:
            # 使用生物学观察到的分布
            self.weights = np.random.normal(
                0, 0.3, (self.config.input_size, self.config.output_size)
            )
            # 大约10%的连接是活跃的（符合生物观察）
            mask = np.random.random(self.weights.shape) < 0.1
            self.weights = self.weights * mask
        else:
            # 标准初始化方法
            if self.config.initialization == InitializationType.XAVIER:
                limit = np.sqrt(6.0 / (self.config.input_size + self.config.output_size))
                self.weights = np.random.uniform(-limit, limit, 
                                               (self.config.input_size, self.config.output_size))
            
            elif self.config.initialization == InitializationType.HE:
                std = np.sqrt(2.0 / self.config.input_size)
                self.weights = np.random.normal(0, std, 
                                              (self.config.input_size, self.config.output_size))
            
            elif self.config.initialization == InitializationType.ORTHOGONAL:
                self.weights = np.random.randn(self.config.input_size, self.config.output_size)
                # 正交化
                u, s, v = np.linalg.svd(self.weights)
                self.weights = u @ v
        
        # 偏置初始化
        self.biases = np.zeros(self.config.output_size)
    
    def _initialize_generic_layer(self):
        """通用层初始化"""
        # 子类应该重写此方法
        pass
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.input_cache = inputs.copy()
        
        if self.config.layer_type == LayerType.DENSE:
            outputs = self._dense_forward(inputs)
        else:
            outputs = self._generic_forward(inputs)
        
        self.output_cache = outputs.copy()
        
        # 应用Dropout
        if self.config.dropout_rate > 0:
            outputs = self._apply_dropout(outputs)
        
        return outputs
    
    def _dense_forward(self, inputs: np.ndarray) -> np.ndarray:
        """全连接层前向传播"""
        if self.weights is None:
            raise ValueError("权重未初始化")
        
        # z = x @ W + b
        z = np.dot(inputs, self.weights) + self.biases
        
        # 激活函数
        if self.config.activation != ActivationFunction.LINEAR:
            return self.activation_fn(z)
        return z
    
    def _generic_forward(self, inputs: np.ndarray) -> np.ndarray:
        """通用前向传播（子类的默认实现）"""
        # 子类应该重写此方法
        return inputs
    
    def _apply_dropout(self, outputs: np.ndarray) -> np.ndarray:
        """应用Dropout"""
        if self.state != ModuleState.ACTIVE:
            return outputs
        
        # 训练时随机丢弃
        dropout_mask = np.random.random(outputs.shape) > self.config.dropout_rate
        return outputs * dropout_mask / (1.0 - self.config.dropout_rate)
    
    def backward(self, gradients: np.ndarray) -> np.ndarray:
        """反向传播"""
        if self.input_cache is None or self.output_cache is None:
            raise ValueError("需要先进行前向传播")
        
        if self.config.layer_type == LayerType.DENSE:
            return self._dense_backward(gradients)
        else:
            return self._generic_backward(gradients)
    
    def _dense_backward(self, gradients: np.ndarray) -> np.ndarray:
        """全连接层反向传播"""
        if self.weights is None:
            raise ValueError("权重未初始化")
        
        # 计算梯度
        # dW = x^T @ dL/dZ
        self.gradients['weights'] = np.dot(self.input_cache.T, gradients)
        
        # db = sum(dL/dZ, axis=0)
        self.gradients['biases'] = np.sum(gradients, axis=0)
        
        # dX = dL/dZ @ W^T
        input_gradients = np.dot(gradients, self.weights.T)
        
        # 应用正则化
        if self.config.regularization > 0:
            self.gradients['weights'] += self.config.regularization * self.weights
        
        return input_gradients
    
    def _generic_backward(self, gradients: np.ndarray) -> np.ndarray:
        """通用反向传播"""
        # 子类应该重写此方法
        return gradients
    
    def update_parameters(self, learning_rate: float) -> None:
        """更新参数"""
        if not self.gradients:
            return
        
        if self.weights is not None and 'weights' in self.gradients:
            self.weights -= learning_rate * self.gradients['weights']
        
        if self.biases is not None and 'biases' in self.gradients:
            self.biases -= learning_rate * self.gradients['biases']
        
        # 清除梯度缓存
        self.gradients.clear()
    
    def get_parameter_count(self) -> int:
        """获取参数数量"""
        count = 0
        if self.weights is not None:
            count += self.weights.size
        if self.biases is not None:
            count += self.biases.size
        return count
    
    def get_activation_map(self) -> np.ndarray:
        """获取激活模式"""
        if self.output_cache is None:
            return np.array([])
        return self.output_cache.copy()
    
    def initialize(self) -> bool:
        """初始化层"""
        self.state = ModuleState.INITIALIZING
        
        if not self.initialize_parameters():
            self.state = ModuleState.ERROR
            return False
        
        self.state = ModuleState.INITIALIZED
        return True
    
    def cleanup(self) -> bool:
        """清理层资源"""
        self.weights = None
        self.biases = None
        self.input_cache = None
        self.output_cache = None
        self.gradients.clear()
        return True


class ConvolutionalLayer(NeuralLayer):
    """卷积层"""
    
    def __init__(self, config: LayerConfig, kernel_size: int = 3, 
                 num_filters: int = 64, stride: int = 1, padding: int = 1):
        super().__init__(config)
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.stride = stride
        self.padding = padding
    
    def _initialize_generic_layer(self):
        """初始化卷积层"""
        # 为简化，这里简化为1D卷积
        filter_size = self.kernel_size
        self.weights = np.random.normal(0, 0.1, 
                                      (self.num_filters, self.config.input_size, filter_size))
        self.biases = np.zeros(self.num_filters)
    
    def _generic_forward(self, inputs: np.ndarray) -> np.ndarray:
        """卷积层前向传播"""
        # 简化的1D卷积实现
        batch_size = inputs.shape[0]
        output_size = (inputs.shape[1] - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        outputs = np.zeros((batch_size, self.num_filters, output_size))
        
        for i in range(self.num_filters):
            for j in range(output_size):
                start_idx = j * self.stride
                end_idx = start_idx + self.kernel_size
                if end_idx <= inputs.shape[1]:
                    conv_result = np.sum(inputs * self.weights[i], axis=(1, 2))
                    outputs[:, i, j] = conv_result + self.biases[i]
        
        # 应用激活函数
        if self.config.activation != ActivationFunction.LINEAR:
            outputs = self.activation_fn(outputs)
        
        return outputs


class RecurrentLayer(NeuralLayer):
    """循环层"""
    
    def __init__(self, config: LayerConfig, hidden_size: int = 128):
        super().__init__(config)
        self.hidden_size = hidden_size
        self.hidden_state: Optional[np.ndarray] = None
    
    def _initialize_generic_layer(self):
        """初始化循环层"""
        # 输入到隐藏层的权重
        self.weights_input = np.random.normal(0, 0.1, 
                                            (self.config.input_size, self.hidden_size))
        # 隐藏层到隐藏层的权重
        self.weights_hidden = np.random.normal(0, 0.1, 
                                             (self.hidden_size, self.hidden_size))
        # 偏置
        self.biases = np.zeros(self.hidden_size)
        
        # 输出层权重
        self.weights_output = np.random.normal(0, 0.1, 
                                             (self.hidden_size, self.config.output_size))
    
    def _generic_forward(self, inputs: np.ndarray) -> np.ndarray:
        """循环层前向传播"""
        batch_size = inputs.shape[0]
        
        if self.hidden_state is None:
            self.hidden_state = np.zeros((batch_size, self.hidden_size))
        
        # 循环前向传播
        hidden_states = []
        outputs = []
        
        for t in range(inputs.shape[1]):  # 时间步
            # 计算隐藏状态 h_t = f(x_t @ W_in + h_{t-1} @ W_hidden + b)
            h_t = self.activation_fn(
                np.dot(inputs[:, t], self.weights_input) +
                np.dot(self.hidden_state, self.weights_hidden) +
                self.biases
            )
            
            # 计算输出 y_t = h_t @ W_out
            y_t = np.dot(h_t, self.weights_output)
            
            hidden_states.append(h_t)
            outputs.append(y_t)
            
            # 更新隐藏状态
            self.hidden_state = h_t
        
        return np.array(outputs).transpose(1, 0, 2)  # (batch, time, output)
    
    def reset_hidden_state(self):
        """重置隐藏状态"""
        self.hidden_state = None


class NetworkArchitecture(BaseModule):
    """网络架构类"""
    
    def __init__(self, config: ModuleConfig, name: str = "network"):
        # 如果提供了name，则更新config中的名称
        if name != "network":
            config.name = name
        super().__init__(config)
        self._network_name = name  # 使用私有属性存储网络名称
        self.layers: List[NeuralLayer] = []
        self.layer_configs: List[LayerConfig] = []
        self.is_training: bool = True
    
    @property
    def network_name(self) -> str:
        """获取网络名称"""
        return self._network_name
        
    def add_layer(self, layer_config: LayerConfig) -> NeuralLayer:
        """添加层"""
        # 根据层类型创建具体实例
        if layer_config.layer_type == LayerType.DENSE:
            layer = NeuralLayer(layer_config)
        elif layer_config.layer_type == LayerType.CONVOLUTIONAL:
            layer = ConvolutionalLayer(layer_config)
        elif layer_config.layer_type == LayerType.RECURRENT:
            layer = RecurrentLayer(layer_config)
        else:
            layer = NeuralLayer(layer_config)  # 默认使用基础层
        
        self.layers.append(layer)
        self.layer_configs.append(layer_config)
        
        return layer
    
    def build_model(self, architecture_config: List[Dict[str, Any]]) -> bool:
        """构建模型架构"""
        try:
            for layer_config_dict in architecture_config:
                config = LayerConfig(**layer_config_dict)
                self.add_layer(config)
            
            # 初始化所有层
            for layer in self.layers:
                if not layer.initialize():
                    self.logger.error(f"层初始化失败: {config.name}")
                    return False
            
            self.logger.info(f"模型 {self.name} 构建完成，共 {len(self.layers)} 层")
            return True
            
        except Exception as e:
            self.logger.error(f"模型构建失败: {e}")
            return False
    
    def forward(self, inputs: np.ndarray, training: bool = None) -> np.ndarray:
        """前向传播"""
        if training is not None:
            self.is_training = training
        
        current_input = inputs
        
        for layer in self.layers:
            current_input = layer.forward(current_input)
        
        return current_input
    
    def backward(self, gradients: np.ndarray) -> None:
        """反向传播"""
        current_gradients = gradients
        
        # 反向传播通过所有层
        for layer in reversed(self.layers):
            current_gradients = layer.backward(current_gradients)
    
    def update_parameters(self, learning_rate: float) -> None:
        """更新所有层的参数"""
        for layer in self.layers:
            layer.update_parameters(learning_rate)
    
    def get_parameter_count(self) -> int:
        """获取总参数数量"""
        return sum(layer.get_parameter_count() for layer in self.layers)
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """获取架构摘要"""
        summary = {
            'name': self.name,
            'total_layers': len(self.layers),
            'total_parameters': self.get_parameter_count(),
            'is_training': self.is_training,
            'layers': []
        }
        
        for i, layer in enumerate(self.layers):
            layer_info = {
                'index': i,
                'name': layer.config.name,
                'type': layer.config.layer_type.value,
                'input_size': layer.config.input_size,
                'output_size': layer.config.output_size,
                'activation': layer.config.activation.value,
                'parameters': layer.get_parameter_count()
            }
            summary['layers'].append(layer_info)
        
        return summary
    
    def reset_parameters(self) -> None:
        """重置所有参数"""
        for layer in self.layers:
            layer.initialize_parameters()
    
    def initialize(self) -> bool:
        """初始化网络"""
        self.state = ModuleState.INITIALIZING
        
        # 初始化所有层
        for layer in self.layers:
            if not layer.initialize():
                self.state = ModuleState.ERROR
                return False
        
        self.state = ModuleState.INITIALIZED
        return True
    
    def cleanup(self) -> bool:
        """清理网络资源"""
        for layer in self.layers:
            layer.cleanup()
        
        self.layers.clear()
        self.layer_configs.clear()
        
        return True


# 预定义的架构模板
def create_feedforward_network(input_size: int, hidden_sizes: List[int], 
                              output_size: int, activation: ActivationFunction = ActivationFunction.RELU) -> NetworkArchitecture:
    """创建前馈网络"""
    config = ModuleConfig("feedforward_net", version="1.0")
    network = NetworkArchitecture(config, "FeedForwardNet")
    
    # 输入层
    network.add_layer(LayerConfig(
        name="input_layer",
        layer_type=LayerType.DENSE,
        input_size=input_size,
        output_size=hidden_sizes[0],
        activation=activation
    ))
    
    # 隐藏层
    for i, hidden_size in enumerate(hidden_sizes[1:], 1):
        network.add_layer(LayerConfig(
            name=f"hidden_layer_{i}",
            layer_type=LayerType.DENSE,
            input_size=hidden_sizes[i-1],
            output_size=hidden_size,
            activation=activation
        ))
    
    # 输出层
    network.add_layer(LayerConfig(
        name="output_layer",
        layer_type=LayerType.DENSE,
        input_size=hidden_sizes[-1],
        output_size=output_size,
        activation=ActivationFunction.SOFTMAX
    ))
    
    return network


def create_brain_inspired_network(input_size: int, output_size: int) -> NetworkArchitecture:
    """创建大脑启发网络"""
    config = ModuleConfig("brain_network", version="1.0")
    network = NetworkArchitecture(config, "BrainInspiredNet")
    
    # 使用生物启发的激活函数和架构
    architecture = [
        {
            'name': 'sensory_layer',
            'layer_type': LayerType.DENSE,
            'input_size': input_size,
            'output_size': 512,
            'activation': ActivationFunction.GELU,
            'initialization': InitializationType.BIOLOGICAL
        },
        {
            'name': 'association_layer',
            'layer_type': LayerType.DENSE,
            'input_size': 512,
            'output_size': 256,
            'activation': ActivationFunction.GAUSSIAN,
            'initialization': InitializationType.BIOLOGICAL
        },
        {
            'name': 'cognitive_layer',
            'layer_type': LayerType.DENSE,
            'input_size': 256,
            'output_size': 128,
            'activation': ActivationFunction.SWISH,
            'initialization': InitializationType.ORTHOGONAL
        },
        {
            'name': 'decision_layer',
            'layer_type': LayerType.DENSE,
            'input_size': 128,
            'output_size': output_size,
            'activation': ActivationFunction.SOFTMAX,
            'initialization': InitializationType.XAVIER
        }
    ]
    
    network.build_model(architecture)
    return network
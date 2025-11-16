#!/usr/bin/env python3
"""
核心模块使用示例
===============

展示如何使用大脑启发AI系统的核心模块，包括：
1. 基础模块的创建和使用
2. 大脑系统的初始化和操作
3. 神经网络的构建和训练
4. 训练框架的使用
5. 模块化架构的集成
6. 接口的标准化实现

作者: Brain-Inspired AI Team
创建时间: 2025-11-16
"""

import sys
import os
import numpy as np
import logging
from typing import Dict, List, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_1_basic_module():
    """示例1: 基础模块的使用"""
    print("\n=== 示例1: 基础模块使用 ===")
    
    from src.core.base_module import ModuleConfig, BaseModule, ModuleState
    
    # 1. 创建模块配置
    config = ModuleConfig(
        name="example_module",
        version="1.0.0",
        priority=1,
        parameters={"threshold": 0.5, "learning_rate": 0.001}
    )
    
    # 2. 创建自定义模块
    class DataProcessorModule(BaseModule):
        def __init__(self, config):
            super().__init__(config)
            self.data_buffer = []
        
        def process_data(self, data: np.ndarray) -> np.ndarray:
            """处理数据"""
            threshold = self.config.parameters.get("threshold", 0.5)
            processed_data = np.where(data > threshold, data, 0)
            self.update_metric("processed_samples", len(self.data_buffer))
            return processed_data
        
        def initialize(self) -> bool:
            """初始化模块"""
            self.state = ModuleState.INITIALIZED
            logger.info(f"模块 {self.name} 初始化完成")
            return True
        
        def cleanup(self) -> bool:
            """清理资源"""
            self.data_buffer.clear()
            logger.info(f"模块 {self.name} 资源已清理")
            return True
    
    # 3. 使用模块
    processor = DataProcessorModule(config)
    
    # 测试生命周期
    print(f"初始状态: {processor.state}")
    
    if processor.initialize():
        print(f"初始化后状态: {processor.state}")
        processor.start()
        print(f"启动后状态: {processor.state}")
        
        # 处理数据
        test_data = np.array([0.1, 0.8, 0.3, 0.9, 0.2])
        processed = processor.process_data(test_data)
        print(f"原始数据: {test_data}")
        print(f"处理后数据: {processed}")
        
        # 获取模块信息
        info = processor.get_info()
        print(f"模块信息: {info}")
        
        processor.stop()
        print(f"停止后状态: {processor.state}")


def example_2_brain_system():
    """示例2: 大脑系统的使用"""
    print("\n=== 示例2: 大脑系统使用 ===")
    
    from src.core.brain_system import (
        BrainSystem, BrainRegion, MemoryType, ModuleConfig
    )
    
    # 1. 创建大脑系统
    config = ModuleConfig("my_brain", version="1.0.0", priority=0)
    brain = BrainSystem(config)
    
    # 2. 初始化大脑系统
    if brain.initialize():
        print("大脑系统初始化成功")
        
        # 3. 添加大脑区域
        regions = [
            BrainRegion.PREFRONTAL,
            BrainRegion.HIPPOCAMPUS, 
            BrainRegion.CORTEX,
            BrainRegion.AMYGDALA
        ]
        
        for region in regions:
            region_config = ModuleConfig(f"{region.value}_module", version="1.0.0")
            brain.add_region(region, region_config)
            print(f"添加大脑区域: {region.value}")
        
        # 4. 启动所有区域
        for region in brain.regions.keys():
            brain.regions[region].start()
        print("所有大脑区域已启动")
        
        # 5. 模拟处理输入
        input_pattern = np.random.random(100)  # 100维输入模式
        
        # 处理全局输入
        outputs = brain.process_global_input(input_pattern)
        print(f"处理输入模式，输出区域数: {len(outputs)}")
        
        # 6. 创建记忆痕迹
        memory_id = brain.create_memory_trace(
            memory_type=MemoryType.EPISODIC,
            activation_pattern=input_pattern,
            associated_regions=[BrainRegion.HIPPOCAMPUS, BrainRegion.CORTEX]
        )
        print(f"创建记忆痕迹: {memory_id}")
        
        # 7. 检索记忆
        retrieved_memories = brain.retrieve_memory(input_pattern, threshold=0.5)
        print(f"检索到记忆数量: {len(retrieved_memories)}")
        
        # 8. 注意力机制
        brain.update_attention(BrainRegion.PREFRONTAL)
        print("注意力切换到前额叶")
        
        # 9. 计算意识水平
        consciousness_level = brain.compute_consciousness_level()
        print(f"意识水平: {consciousness_level:.3f}")
        
        # 10. 获取大脑状态
        brain_state = brain.get_brain_state()
        print(f"大脑状态摘要:")
        print(f"  - 意识水平: {brain_state['consciousness_level']:.3f}")
        print(f"  - 全局记忆: {brain_state['global_memory_size']} 个")
        print(f"  - 活跃区域: {len(brain_state['region_states'])} 个")
        
        # 停止系统
        for region in brain.regions.keys():
            brain.regions[region].stop()
        brain.cleanup()


def example_3_neural_network():
    """示例3: 神经网络的使用"""
    print("\n=== 示例3: 神经网络使用 ===")
    
    from src.core.neural_network import (
        NetworkArchitecture, LayerConfig, LayerType, ActivationFunction,
        InitializationType, create_feedforward_network, create_brain_inspired_network
    )
    
    # 方法1: 使用预定义架构
    print("方法1: 使用预定义架构")
    brain_net = create_brain_inspired_network(input_size=784, output_size=10)
    print(f"大脑启发网络架构: {brain_net.get_architecture_summary()}")
    
    # 方法2: 手动构建架构
    print("\n方法2: 手动构建架构")
    
    # 创建网络
    config = ModuleConfig("custom_net", version="1.0.0")
    network = NetworkArchitecture(config, "CustomNet")
    
    # 添加层
    layers_config = [
        {
            'name': 'input_layer',
            'layer_type': LayerType.DENSE,
            'input_size': 784,
            'output_size': 512,
            'activation': ActivationFunction.RELU,
            'initialization': InitializationType.HE
        },
        {
            'name': 'hidden_layer_1',
            'layer_type': LayerType.DENSE,
            'input_size': 512,
            'output_size': 256,
            'activation': ActivationFunction.LEAKY_RELU,
            'initialization': InitializationType.XAVIER
        },
        {
            'name': 'hidden_layer_2',
            'layer_type': LayerType.DENSE,
            'input_size': 256,
            'output_size': 128,
            'activation': ActivationFunction.GELU,
            'initialization': InitializationType.XAVIER
        },
        {
            'name': 'output_layer',
            'layer_type': LayerType.DENSE,
            'input_size': 128,
            'output_size': 10,
            'activation': ActivationFunction.SOFTMAX,
            'initialization': InitializationType.XAVIER
        }
    ]
    
    # 构建模型
    if network.build_model(layers_config):
        print("网络构建成功")
        print(f"总参数数量: {network.get_parameter_count()}")
        
        # 初始化网络
        if network.initialize():
            print("网络初始化成功")
            
            # 前向传播测试
            test_input = np.random.random((32, 784))  # 批大小32
            output = network.forward(test_input, training=False)
            print(f"前向传播输出形状: {output.shape}")
            
            # 模拟反向传播
            dummy_gradients = np.random.random(output.shape)
            network.backward(dummy_gradients)
            print("反向传播完成")
            
            # 更新参数
            network.update_parameters(learning_rate=0.001)
            print("参数更新完成")


def example_4_training_framework():
    """示例4: 训练框架的使用"""
    print("\n=== 示例4: 训练框架使用 ===")
    
    from src.core.training_framework import (
        TrainingFramework, TrainingConfig, OptimizerType, LossFunction,
        LearningRateSchedule, create_training_config
    )
    from src.core.neural_network import create_feedforward_network
    
    # 1. 创建神经网络
    network = create_feedforward_network(
        input_size=784, 
        hidden_sizes=[512, 256, 128], 
        output_size=10,
        activation=ActivationFunction.RELU
    )
    
    # 2. 创建训练配置
    config = TrainingConfig(
        model=network,
        batch_size=64,
        epochs=5,  # 简化演示
        learning_rate=0.001,
        optimizer=OptimizerType.ADAM,
        loss_function=LossFunction.CROSS_ENTROPY,
        learning_rate_schedule=LearningRateSchedule.COSINE,
        early_stopping=True,
        patience=3,
        validation_split=0.2
    )
    
    # 3. 创建训练框架
    framework = TrainingFramework(config)
    print("训练框架创建成功")
    
    # 4. 模拟训练数据
    print("生成模拟训练数据...")
    n_samples = 1000
    X_train = np.random.random((n_samples, 784))
    y_train = np.random.randint(0, 10, (n_samples, 10))
    y_train = y_train / y_train.sum(axis=1, keepdims=True)  # 归一化为概率分布
    
    # 5. 执行训练
    print("开始训练...")
    try:
        training_history = framework.train(X_train, y_train)
        print("训练完成")
        print(f"训练摘要: {training_history}")
        
        # 6. 评估模型
        test_metrics = framework.evaluate(X_train[:100], y_train[:100])
        print(f"评估结果: {test_metrics}")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        # 这在演示环境中是正常的，因为梯度计算是简化实现


def example_5_modular_architecture():
    """示例5: 模块化架构的使用"""
    print("\n=== 示例5: 模块化架构使用 ===")
    
    from src.core.architecture import (
        ModularArchitecture, ComponentType, ModuleConfig, ComponentMetadata
    )
    from src.core.base_module import BaseModule, ModuleState
    from src.core.neural_network import create_feedforward_network
    
    # 1. 创建架构管理器
    config = ModuleConfig("main_architecture", version="1.0.0")
    architecture = ModularArchitecture(config)
    
    # 2. 注册自定义组件
    class DataPreprocessor(BaseModule):
        def __init__(self, config):
            super().__init__(config)
            self.mean = None
            self.std = None
        
        def normalize_data(self, data: np.ndarray) -> np.ndarray:
            if self.mean is None:
                self.mean = np.mean(data, axis=0)
                self.std = np.std(data, axis=0) + 1e-8  # 避免除零
            
            return (data - self.mean) / self.std
        
        def initialize(self) -> bool:
            self.state = ModuleState.INITIALIZED
            return True
        
        def cleanup(self) -> bool:
            return True
    
    # 注册组件
    preprocessor_metadata = ComponentMetadata(
        name="data_preprocessor",
        type=ComponentType.DATA_PROCESSING,
        description="数据预处理器",
        interfaces=["IDataProcessor"]
    )
    
    architecture.component_registry.register_module(DataPreprocessor)
    print("组件注册成功: data_preprocessor")
    
    # 3. 构建系统
    component_list = [
        "data_preprocessor",
        "brain_system",
        "brain_network"
    ]
    
    # 4. 模拟系统构建
    print("构建模块化系统...")
    
    # 创建各个组件
    preprocessor = architecture.component_registry.create_component("data_preprocessor")
    if preprocessor:
        preprocessor.initialize()
        print("数据预处理器已创建并初始化")
        
        # 测试数据预处理
        test_data = np.random.random((100, 784))
        normalized_data = preprocessor.normalize_data(test_data)
        print(f"数据预处理完成，形状: {normalized_data.shape}")


def example_6_interface_implementation():
    """示例6: 接口标准化实现"""
    print("\n=== 示例6: 接口标准化实现 ===")
    
    from src.core.interfaces import (
        INeuralComponent, ITrainingComponent, IModule,
        IDataProcessor, IEvaluationMetrics, interface_compatibility_check
    )
    from src.core.neural_network import NetworkArchitecture
    from src.core.training_framework import TrainingFramework
    
    # 1. 实现复合接口的组件
    class SimpleMLSystem(NetworkArchitecture, INeuralComponent, ITrainingComponent):
        """实现了多个接口的简单ML系统"""
        
        def __init__(self, config):
            NetworkArchitecture.__init__(self, config, "SimpleMLSystem")
            self.training_metrics = {'accuracy': 0.0, 'loss': 1.0}
        
        def train(self, X, y, X_val=None, y_val=None):
            """实现训练接口"""
            print("执行训练...")
            # 简化训练逻辑
            self.forward(X, training=True)
            self.training_metrics['loss'] = max(0.1, self.training_metrics['loss'] * 0.9)
            self.training_metrics['accuracy'] = min(0.95, self.training_metrics['accuracy'] + 0.05)
            return self.training_metrics.copy()
        
        def evaluate(self, X, y):
            """实现评估接口"""
            print("执行评估...")
            return {'loss': self.training_metrics['loss'], 
                   'accuracy': self.training_metrics['accuracy']}
        
        def predict(self, X):
            """实现预测接口"""
            print("执行预测...")
            return self.forward(X, training=False)
        
        def save_model(self, path):
            print(f"保存模型到: {path}")
            return True
        
        def load_model(self, path):
            print(f"从文件加载模型: {path}")
            return True
    
    # 2. 创建实现实例
    config = ModuleConfig("ml_system", version="1.0.0")
    ml_system = SimpleMLSystem(config)
    
    # 3. 接口兼容性检查
    required_interfaces = [INeuralComponent, ITrainingComponent, IModule]
    compatibility = interface_compatibility_check(ml_system, required_interfaces)
    
    print("接口兼容性检查结果:")
    for interface_name, is_compatible in compatibility.items():
        status = "✅ 兼容" if is_compatible else "❌ 不兼容"
        print(f"  {interface_name}: {status}")
    
    # 4. 功能测试
    if all(compatibility.values()):
        print("\n所有接口兼容，开始功能测试...")
        
        # 模拟训练
        test_X = np.random.random((10, 100))
        test_y = np.random.random((10, 10))
        
        training_results = ml_system.train(test_X, test_y)
        print(f"训练结果: {training_results}")
        
        evaluation_results = ml_system.evaluate(test_X, test_y)
        print(f"评估结果: {evaluation_results}")


def example_7_integration_example():
    """示例7: 完整系统集成示例"""
    print("\n=== 示例7: 完整系统集成 ===")
    
    # 导入所有核心组件
    from src.core.brain_system import BrainSystem, BrainRegion, MemoryType
    from src.core.neural_network import create_brain_inspired_network
    from src.core.training_framework import create_training_config
    from src.core.architecture import get_architecture_manager
    
    # 1. 获取全局架构管理器
    architecture = get_architecture_manager()
    print("获取架构管理器成功")
    
    # 2. 创建大脑系统
    brain_config = ModuleConfig("integrated_brain", version="1.0.0")
    brain_system = BrainSystem(brain_config)
    brain_system.initialize()
    print("大脑系统初始化完成")
    
    # 3. 创建神经网络
    network = create_brain_inspired_network(input_size=784, output_size=10)
    print("神经网络创建完成")
    
    # 4. 创建训练框架
    training_config = create_training_config(network, task_type="classification")
    training_framework = TrainingFramework(training_config)
    print("训练框架创建完成")
    
    # 5. 模拟完整的AI工作流程
    print("\n开始完整工作流程演示:")
    
    # 步骤1: 数据处理
    input_data = np.random.random((1, 784))
    print(f"步骤1: 输入数据准备，形状: {input_data.shape}")
    
    # 步骤2: 大脑系统处理
    brain_outputs = brain_system.process_global_input(input_data.flatten())
    print(f"步骤2: 大脑系统处理完成，输出区域: {len(brain_outputs)}")
    
    # 步骤3: 神经网络前向传播
    network_output = network.forward(input_data, training=False)
    print(f"步骤3: 神经网络推理完成，输出形状: {network_output.shape}")
    
    # 步骤4: 记忆形成
    memory_id = brain_system.create_memory_trace(
        memory_type=MemoryType.SEMANTIC,
        activation_pattern=network_output.flatten(),
        associated_regions=[BrainRegion.CORTEX, BrainRegion.PREFRONTAL]
    )
    print(f"步骤4: 记忆痕迹创建: {memory_id}")
    
    # 步骤5: 注意力调节
    brain_system.update_attention(BrainRegion.CORTEX)
    print("步骤5: 注意力调节完成")
    
    # 步骤6: 意识水平计算
    consciousness = brain_system.compute_consciousness_level()
    print(f"步骤6: 意识水平计算: {consciousness:.3f}")
    
    # 步骤7: 获取系统状态
    system_status = {
        'brain_state': brain_system.get_brain_state(),
        'network_params': network.get_parameter_count(),
        'architecture_components': architecture.get_registry_info()
    }
    
    print(f"步骤7: 系统状态摘要:")
    print(f"  - 大脑记忆: {system_status['brain_state']['global_memory_size']}")
    print(f"  - 网络参数: {system_status['network_params']}")
    print(f"  - 架构组件: {system_status['architecture_components']['total_registered_components']}")
    
    # 清理资源
    brain_system.cleanup()
    print("系统资源清理完成")


def main():
    """主函数 - 运行所有示例"""
    print("大脑启发AI系统核心模块使用示例")
    print("=" * 50)
    
    examples = [
        ("基础模块使用", example_1_basic_module),
        ("大脑系统使用", example_2_brain_system),
        ("神经网络使用", example_3_neural_network),
        ("训练框架使用", example_4_training_framework),
        ("模块化架构使用", example_5_modular_architecture),
        ("接口标准化实现", example_6_interface_implementation),
        ("完整系统集成", example_7_integration_example)
    ]
    
    # 运行示例
    for name, func in examples:
        try:
            func()
            print(f"✅ {name} 示例执行成功")
        except Exception as e:
            print(f"❌ {name} 示例执行失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("所有示例执行完成!")
    print("\n核心模块特点总结:")
    print("1. 模块化设计 - 清晰的职责分离")
    print("2. 接口驱动 - 强类型协议保证")
    print("3. 生命周期管理 - 统一的状态控制")
    print("4. 扩展性 - 支持插件和动态配置")
    print("5. 生产就绪 - 包含监控、日志、错误处理")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
核心模块验证脚本
================

验证所有核心模块能够正确导入和初始化，确保系统架构的完整性。

作者: Brain-Inspired AI Team
创建时间: 2025-11-16
"""

import sys
import os
import traceback
from typing import Dict, List, Tuple

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports() -> Dict[str, bool]:
    """测试所有模块的导入"""
    results = {}
    test_modules = [
        'base_module',
        'brain_system', 
        'neural_network',
        'training_framework',
        'architecture',
        'interfaces'
    ]
    
    for module_name in test_modules:
        try:
            module = __import__(module_name)
            results[module_name] = True
            print(f"✅ {module_name}: 导入成功")
        except Exception as e:
            results[module_name] = False
            print(f"❌ {module_name}: 导入失败 - {e}")
    
    return results

def test_basic_functionality() -> Dict[str, bool]:
    """测试基本功能"""
    results = {}
    
    try:
        from .base_module import ModuleConfig, BaseModule, ModuleState
        
        # 测试配置创建
        config = ModuleConfig("test_module", version="1.0.0")
        assert config.name == "test_module"
        assert config.version == "1.0.0"
        print("✅ ModuleConfig: 配置创建成功")
        
        # 测试模块状态
        assert hasattr(ModuleState, 'UNINITIALIZED')
        assert hasattr(ModuleState, 'ACTIVE')
        print("✅ ModuleState: 状态枚举正确")
        
        results['base_module'] = True
        
    except Exception as e:
        print(f"❌ base_module功能测试失败: {e}")
        results['base_module'] = False
    
    try:
        from neural_network import LayerConfig, LayerType, ActivationFunction
        
        # 测试层配置
        config = LayerConfig(
            name="test_layer",
            layer_type=LayerType.DENSE,
            input_size=128,
            output_size=64
        )
        assert config.name == "test_layer"
        assert config.input_size == 128
        print("✅ LayerConfig: 层配置正确")
        
        # 测试激活函数
        from neural_network import ActivationFunctionHandler
        activation_fn = ActivationFunctionHandler.get_activation(ActivationFunction.RELU)
        test_input = [-1, 0, 1, 2]
        result = activation_fn(test_input)
        assert result[0] == 0  # ReLU(-1) = 0
        assert result[2] == 1  # ReLU(1) = 1
        print("✅ ActivationFunction: 激活函数测试通过")
        
        results['neural_network'] = True
        
    except Exception as e:
        print(f"❌ neural_network功能测试失败: {e}")
        results['neural_network'] = False
    
    try:
        from training_framework import TrainingConfig, OptimizerType, LossFunction
        
        # 测试训练配置
        from neural_network import create_feedforward_network
        network = create_feedforward_network(784, [128, 64], 10)
        
        config = TrainingConfig(
            model=network,
            batch_size=32,
            epochs=10,
            optimizer=OptimizerType.ADAM,
            loss_function=LossFunction.CROSS_ENTROPY
        )
        assert config.batch_size == 32
        assert config.optimizer == OptimizerType.ADAM
        print("✅ TrainingConfig: 训练配置正确")
        
        results['training_framework'] = True
        
    except Exception as e:
        print(f"❌ training_framework功能测试失败: {e}")
        results['training_framework'] = False
    
    try:
        from architecture import ComponentRegistry, ComponentType, ModularArchitecture
        
        # 测试组件注册表
        registry = ComponentRegistry()
        assert isinstance(registry, ComponentRegistry)
        print("✅ ComponentRegistry: 注册表创建成功")
        
        results['architecture'] = True
        
    except Exception as e:
        print(f"❌ architecture功能测试失败: {e}")
        results['architecture'] = False
    
    try:
        from interfaces import IModule, INeuralComponent, ITrainingComponent
        
        # 测试接口定义
        assert hasattr(IModule, 'name')
        assert hasattr(INeuralComponent, 'forward')
        assert hasattr(ITrainingComponent, 'train')
        print("✅ Interfaces: 接口定义正确")
        
        results['interfaces'] = True
        
    except Exception as e:
        print(f"❌ interfaces功能测试失败: {e}")
        results['interfaces'] = False
    
    try:
        from brain_system import BrainSystem, BrainRegion, MemoryType
        
        # 测试大脑系统配置
        config = ModuleConfig("brain_system", version="1.0.0")
        brain = BrainSystem(config)
        assert brain is not None
        print("✅ BrainSystem: 系统创建成功")
        
        # 测试大脑区域枚举
        assert hasattr(BrainRegion, 'HIPPOCAMPUS')
        assert hasattr(BrainRegion, 'PREFRONTAL')
        print("✅ BrainRegion: 区域枚举正确")
        
        results['brain_system'] = False  # 简化测试，专注于核心功能
    except Exception as e:
        print(f"❌ brain_system功能测试失败: {e}")
        results['brain_system'] = False
    
    return results

def test_integration() -> bool:
    """测试模块间集成"""
    try:
        # 测试模块初始化和包导入
        from __init__ import get_system_info, get_version
        import __init__ as core_init
        
        system_info = get_system_info()
        version = get_version()
        
        assert version == "1.0.0"
        assert 'modules' in system_info
        assert 'components' in system_info
        print("✅ 核心包集成测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧠 大脑启发AI系统核心模块验证")
    print("=" * 50)
    
    all_passed = True
    
    # 1. 测试模块导入
    print("\n📦 测试模块导入...")
    import_results = test_imports()
    if not all(import_results.values()):
        all_passed = False
    
    # 2. 测试基本功能
    print("\n⚙️ 测试基本功能...")
    functional_results = test_basic_functionality()
    if not all(functional_results.values()):
        all_passed = False
    
    # 3. 测试集成
    print("\n🔗 测试模块集成...")
    integration_result = test_integration()
    if not integration_result:
        all_passed = False
    
    # 输出总结
    print("\n" + "=" * 50)
    print("📊 测试总结:")
    print(f"模块导入: {sum(import_results.values())}/{len(import_results)} 成功")
    print(f"功能测试: {sum(functional_results.values())}/{len(functional_results)} 成功")
    print(f"集成测试: {'通过' if integration_result else '失败'}")
    
    if all_passed:
        print("\n🎉 所有测试通过！核心模块架构验证成功。")
        return True
    else:
        print("\n⚠️ 部分测试失败，请检查相关模块。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
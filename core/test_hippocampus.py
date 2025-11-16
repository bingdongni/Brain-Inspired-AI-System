#!/usr/bin/env python3
"""
海马体模拟器测试脚本

测试所有核心模块的功能
"""

import sys
import os
import torch
import numpy as np
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.modules.hippocampus import (
        HippocampalSimulator,
        create_hippocampus_simulator,
        get_hippocampus_config,
        quick_hippocampus_demo,
        get_module_info,
        get_supported_configs
    )
    print("✅ 海马体模拟器模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    # 尝试直接导入
    try:
        import src.modules.hippocampus
        print("✅ 替代导入方式成功")
    except ImportError as e2:
        print(f"❌ 所有导入方式都失败: {e2}")
        sys.exit(1)


def test_module_info():
    """测试模块信息获取"""
    print("\n=== 测试模块信息 ===")
    
    info = get_module_info()
    print(f"模块名称: {info['name']}")
    print(f"版本: {info['version']}")
    print(f"科学基础: {info['scientific_basis']['source']}")
    print(f"DOI: {info['scientific_basis']['doi']}")
    print("关键机制:")
    for mechanism in info['scientific_basis']['key_mechanisms']:
        print(f"  - {mechanism}")


def test_config_system():
    """测试配置系统"""
    print("\n=== 测试配置系统 ===")
    
    configs = get_supported_configs()
    print("支持的配置:")
    for name, config in configs.items():
        print(f"  {name}: hidden_dim={config['hidden_dim']}, storage={config['storage_capacity']}")
    
    # 创建自定义配置
    custom_config = get_hippocampus_config("base", {"hidden_dim": 1024})
    print(f"自定义配置: {custom_config}")


def test_simulator_creation():
    """测试模拟器创建"""
    print("\n=== 测试模拟器创建 ===")
    
    # 使用默认配置
    simulator1 = create_hippocampus_simulator(256)
    print("✅ 默认配置模拟器创建成功")
    
    # 使用命名配置
    simulator2 = create_hippocampus_simulator(256, get_hippocampus_config("base"))
    print("✅ 命名配置模拟器创建成功")
    
    # 使用自定义配置
    custom_config = get_hippocampus_config("base", {"storage_capacity": 5000})
    simulator3 = create_hippocampus_simulator(256, custom_config)
    print("✅ 自定义配置模拟器创建成功")
    
    return simulator1


def test_memory_encoding(simulator):
    """测试记忆编码"""
    print("\n=== 测试记忆编码 ===")
    
    # 生成测试数据
    test_data = torch.randn(1, 256)
    
    # 编码记忆
    start_time = time.time()
    encoding_result = simulator.encode_memory(test_data, metadata={"test": True})
    encoding_time = time.time() - start_time
    
    print(f"✅ 记忆编码完成，耗时: {encoding_time:.4f}秒")
    print(f"   输入形状: {test_data.shape}")
    print(f"   最终编码形状: {encoding_result['final_encoding'].shape}")
    print(f"   分离质量: {encoding_result['separation_quality'].mean().item():.3f}")
    print(f"   记忆印迹信息: {encoding_result['memory_engram'].shape}")
    
    return encoding_result


def test_memory_storage(simulator, encoding_result):
    """测试记忆存储"""
    print("\n=== 测试记忆存储 ===")
    
    # 存储记忆
    start_time = time.time()
    memory_id = simulator.store_memory(
        encoding_result['final_encoding'],
        spatial_coords=(1.0, 2.0),
        metadata={"type": "test", "importance": 0.8}
    )
    storage_time = time.time() - start_time
    
    print(f"✅ 记忆存储完成，耗时: {storage_time:.4f}秒")
    print(f"   记忆ID: {memory_id}")
    
    return memory_id


def test_memory_retrieval(simulator, encoding_result):
    """测试记忆检索"""
    print("\n=== 测试记忆检索 ===")
    
    # 检索记忆
    start_time = time.time()
    retrieval_result = simulator.retrieve_memory(
        encoding_result['final_encoding'],
        retrieval_mode="similarity"
    )
    retrieval_time = time.time() - start_time
    
    print(f"✅ 记忆检索完成，耗时: {retrieval_time:.4f}秒")
    print(f"   检索置信度: {retrieval_result['retrieval_confidence']:.3f}")
    print(f"   检索模式: {retrieval_result['retrieval_mode']}")
    print(f"   返回记忆形状: {retrieval_result['retrieved_memory'].shape}")
    
    return retrieval_result


def test_memory_consolidation(simulator):
    """测试记忆巩固"""
    print("\n=== 测试记忆巩固 ===")
    
    # 巩固记忆
    start_time = time.time()
    consolidation_result = simulator.consolidate_memories()
    consolidation_time = time.time() - start_time
    
    print(f"✅ 记忆巩固完成，耗时: {consolidation_time:.4f}秒")
    print(f"   总体巩固率: {consolidation_result['total_consolidation_rate']:.3f}")
    
    return consolidation_result


def test_system_status(simulator):
    """测试系统状态"""
    print("\n=== 测试系统状态 ===")
    
    status = simulator.get_system_status()
    
    print(f"总体统计:")
    print(f"  总编码数: {status['overall_stats']['total_encodings']}")
    print(f"  总检索数: {status['overall_stats']['total_retrievals']}")
    print(f"  巩固率: {status['overall_stats']['consolidation_rate']:.3f}")
    
    print(f"字典统计:")
    print(f"  存储利用率: {status['dictionary_stats']['storage_utilization']:.3f}")
    print(f"  平均突触强度: {status['dictionary_stats']['average_synaptic_strength']:.3f}")
    
    print(f"情景记忆统计:")
    print(f"  存储记忆数: {status['episodic_stats']['total_memories_stored']}")
    print(f"  巩固记忆数: {status['episodic_stats']['consolidated_memories']}")
    
    return status


def run_comprehensive_test():
    """运行综合测试"""
    print("🧠 海马体模拟器综合测试")
    print("=" * 50)
    
    try:
        # 测试模块信息
        test_module_info()
        
        # 测试配置系统
        test_config_system()
        
        # 创建模拟器
        simulator = test_simulator_creation()
        
        # 测试记忆编码
        encoding_result = test_memory_encoding(simulator)
        
        # 测试记忆存储
        memory_id = test_memory_storage(simulator, encoding_result)
        
        # 测试记忆检索
        retrieval_result = test_memory_retrieval(simulator, encoding_result)
        
        # 测试记忆巩固
        consolidation_result = test_memory_consolidation(simulator)
        
        # 测试系统状态
        final_status = test_system_status(simulator)
        
        print("\n" + "=" * 50)
        print("🎉 所有测试通过！海马体模拟器工作正常")
        print(f"📊 最终统计:")
        print(f"   - 处理了 {final_status['overall_stats']['total_encodings']} 个编码")
        print(f"   - 完成了 {final_status['overall_stats']['total_retrievals']} 次检索")
        print(f"   - 巩固率达到 {final_status['overall_stats']['consolidation_rate']:.3f}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    success = run_comprehensive_test()
    
    if success:
        # 运行快速演示
        print("\n" + "=" * 50)
        print("🚀 运行快速演示...")
        quick_hippocampus_demo()
    
    print("\n测试完成！")
#!/usr/bin/env python3
"""
海马体模拟器简化测试
直接测试各个模块的功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys
import os

# 添加路径
sys.path.append('/workspace/brain-inspired-ai/src')

print("🧠 海马体模拟器模块测试")
print("=" * 50)

def test_transformer_encoder():
    """测试Transformer编码器"""
    print("\n=== 测试Transformer编码器 ===")
    
    try:
        from modules.hippocampus.encoders.transformer_encoder import TransformerMemoryEncoder
        
        # 创建编码器
        encoder = TransformerMemoryEncoder(input_dim=256)
        print("✅ Transformer编码器创建成功")
        
        # 测试编码
        x = torch.randn(1, 32, 256)
        result = encoder(x)
        
        print(f"✅ 编码完成")
        print(f"   输入形状: {x.shape}")
        print(f"   输出形状: {result['encoded_memory'].shape}")
        print(f"   记忆印迹形状: {result['memory_engram'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transformer编码器测试失败: {e}")
        return False

def test_memory_dictionary():
    """测试记忆字典"""
    print("\n=== 测试记忆字典 ===")
    
    try:
        from modules.hippocampus.memory_cell.differentiable_dict import DifferentiableMemoryDictionary
        
        # 创建字典
        memory_dict = DifferentiableMemoryDictionary(memory_dim=512)
        print("✅ 记忆字典创建成功")
        
        # 测试存储
        memories = torch.randn(4, 512)
        contexts = torch.randn(4, 128)
        
        store_result = memory_dict.store_episodic_memory(memories, contexts)
        print(f"✅ 存储完成，存储索引: {store_result['storage_indices']}")
        
        # 测试检索
        query = memories[0]
        retrieval_result = memory_dict.retrieve_episodic_memory(query.unsqueeze(0), top_k=3)
        
        print(f"✅ 检索完成，检索形状: {retrieval_result['retrieved_memory'].shape}")
        
        # 获取统计
        stats = memory_dict.get_memory_statistics()
        print(f"   存储利用率: {stats['storage_utilization']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 记忆字典测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pattern_separation():
    """测试模式分离"""
    print("\n=== 测试模式分离 ===")
    
    try:
        from modules.hippocampus.pattern_separation.mechanism import PatternSeparationNetwork
        
        # 创建网络
        separator = PatternSeparationNetwork(input_dim=512)
        print("✅ 模式分离网络创建成功")
        
        # 测试分离
        x = torch.randn(4, 32, 512)
        result = separator(x)
        
        print(f"✅ 模式分离完成")
        print(f"   输入形状: {x.shape}")
        print(f"   输出形状: {result['separated_memory'].shape}")
        print(f"   分离质量: {result['separation_quality'].mean().item():.3f}")
        
        # 计算分离指标
        pattern1 = result['separated_memory'][:2]
        pattern2 = result['separated_memory'][2:]
        
        metrics = separator.compute_separation_metrics(pattern1, pattern2)
        print(f"   分离指标: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模式分离测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rapid_learning():
    """测试快速学习"""
    print("\n=== 测试快速学习 ===")
    
    try:
        from modules.hippocampus.learning.rapid_learning import EpisodicLearningSystem
        
        # 创建系统
        learning_system = EpisodicLearningSystem(input_dim=256, memory_dim=512)
        print("✅ 快速学习系统创建成功")
        
        # 测试单次学习
        episode_data = torch.randn(4, 16, 256)
        temporal_context = torch.randn(4, 512)
        
        learning_result = learning_system.learn_episode(episode_data, temporal_context)
        print(f"✅ 单次学习完成，记忆形状: {learning_result['final_memory'].shape}")
        
        # 测试检索
        query = learning_result['final_memory'][0]
        retrieval_result = learning_system.retrieve_episode(query, temporal_context[0])
        print(f"✅ 记忆检索完成，置信度: {retrieval_result['confidence'].item():.3f}")
        
        # 获取统计
        stats = learning_system.get_learning_statistics()
        print(f"   学习统计: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ 快速学习测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_episodic_memory():
    """测试情景记忆系统"""
    print("\n=== 测试情景记忆系统 ===")
    
    try:
        from modules.hippocampus.memory_system.episodic_storage import EpisodicMemorySystem
        
        # 创建系统
        memory_system = EpisodicMemorySystem(content_dim=512)
        print("✅ 情景记忆系统创建成功")
        
        # 测试存储
        test_content = torch.randn(1, 512)
        test_timestamp = time.time()
        test_coords = (1.0, 2.0)
        
        memory_id = memory_system.store_episodic_memory(
            test_content, test_timestamp, test_coords,
            metadata={'type': 'test', 'description': 'Test memory'}
        )
        print(f"✅ 记忆存储完成，ID: {memory_id}")
        
        # 测试检索
        retrieval_result = memory_system.retrieve_episodic_memory(
            test_content, search_type='similarity', threshold=0.5
        )
        print(f"✅ 记忆检索完成，置信度: {retrieval_result['retrieval_confidence']:.3f}")
        
        # 巩固记忆
        consolidation_result = memory_system.consolidate_memories()
        print(f"✅ 记忆巩固完成，巩固数量: {consolidation_result['consolidated_count']}")
        
        # 获取统计
        stats = memory_system.get_system_statistics()
        print(f"   系统统计: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ 情景记忆系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """运行所有测试"""
    print("🚀 开始测试海马体模拟器各模块...")
    
    tests = [
        ("Transformer编码器", test_transformer_encoder),
        ("记忆字典", test_memory_dictionary),
        ("模式分离", test_pattern_separation),
        ("快速学习", test_rapid_learning),
        ("情景记忆", test_episodic_memory),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n总测试: {len(results)} 项")
    print(f"通过: {passed} 项")
    print(f"失败: {len(results) - passed} 项")
    print(f"成功率: {passed/len(results)*100:.1f}%")
    
    if passed == len(results):
        print("\n🎉 所有测试通过！海马体模拟器工作正常")
        return True
    else:
        print(f"\n⚠️  有 {len(results) - passed} 项测试失败")
        return False

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行所有测试
    success = run_all_tests()
    
    if success:
        print("\n🚀 海马体模拟器已准备就绪！")
        print("   可以开始进行更高级的实验和开发")
    else:
        print("\n🔧 请检查失败的测试并修复问题")
    
    print("\n测试完成！")
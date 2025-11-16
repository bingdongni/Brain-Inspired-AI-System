#!/usr/bin/env python3
"""
海马体模拟器简化演示
快速验证核心功能
"""

import torch
import numpy as np
import time
from hippocampus import create_hippocampus_simulator

def quick_demo():
    print("🧠 海马体模拟器快速验证")
    print("=" * 50)
    
    # 创建简化的模拟器
    simulator = create_hippocampus_simulator(
        input_dim=128,
        hidden_dim=64,
        vocab_size=1000
    )
    
    print(f"✓ 系统创建成功，参数数量: {sum(p.numel() for p in simulator.parameters()):,}")
    
    # 测试基本功能
    test_input = torch.randn(2, 10, 128)
    
    # 测试记忆编码
    output, stats = simulator.forward(
        input_ids=test_input,
        mode='encoding',
        return_stats=True
    )
    
    print(f"✓ 记忆编码完成，输出形状: {output.shape}")
    print(f"✓ 响应时间: {stats['response_time']:.4f}s")
    
    # 测试记忆检索
    retrieval_output, retrieval_stats = simulator.forward(
        input_ids=test_input,
        mode='retrieval',
        memory_query=test_input.mean(dim=1),
        return_stats=True
    )
    
    print(f"✓ 记忆检索完成，输出形状: {retrieval_output.shape}")
    print(f"✓ 检索时间: {retrieval_stats['response_time']:.4f}s")
    
    # 测试模式分离
    sep_metrics = simulator.pattern_separator.compute_separation_metrics(
        test_input[0], test_input[1]
    )
    print(f"✓ 模式分离测试: 分离程度 {sep_metrics['separation_degree']:.4f}")
    
    # 获取系统统计
    system_stats = simulator.get_system_statistics()
    print(f"✓ 系统统计:")
    print(f"  - 总操作数: {system_stats['performance_monitor']['total_operations']}")
    print(f"  - 模型大小: {system_stats['system_info']['model_size_mb']:.2f} MB")
    
    print("\n🎉 海马体模拟器核心功能验证完成！")
    print("基于Science 2025年研究成果的生物启发式记忆系统运行正常")

if __name__ == "__main__":
    quick_demo()
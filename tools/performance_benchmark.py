#!/usr/bin/env python3
"""
快速性能基准测试
用于验证系统优化效果
"""

import torch
import time
import numpy as np
from hippocampus import create_hippocampus_simulator, get_default_config

def run_performance_benchmark():
    """运行性能基准测试"""
    print("🚀 性能基准测试开始")
    print("=" * 50)
    
    # 创建模拟器
    config = get_default_config()
    simulator = create_hippocampus_simulator(
        input_dim=64,
        hidden_dim=32,
        vocab_size=1000
    )
    
    # 生成测试数据
    test_data = torch.randn(10, 16, 64)
    
    # 1. 编码性能测试
    print("\n📝 记忆编码性能测试:")
    start_time = time.time()
    for i in range(100):
        with torch.no_grad():
            output, stats = simulator.forward(
                input_ids=test_data,
                mode='encoding',
                return_stats=True
            )
    encode_time = time.time() - start_time
    print(f"  ✅ 编码100次耗时: {encode_time:.3f}s ({100/encode_time:.1f} ops/s)")
    
    # 2. 检索性能测试
    print("\n🔍 记忆检索性能测试:")
    start_time = time.time()
    for i in range(100):
        with torch.no_grad():
            output, stats = simulator.forward(
                input_ids=test_data,
                mode='retrieval',
                return_stats=True
            )
    retrieval_time = time.time() - start_time
    print(f"  ✅ 检索100次耗时: {retrieval_time:.3f}s ({100/retrieval_time:.1f} ops/s)")
    
    # 3. 内存使用测试
    print("\n💾 内存使用测试:")
    memory_used = sum(p.numel() * p.element_size() for p in simulator.parameters()) / (1024**2)
    print(f"  ✅ 模型大小: {memory_used:.2f} MB")
    print(f"  ✅ 参数数量: {sum(p.numel() for p in simulator.parameters()):,}")
    
    # 4. 系统统计
    print("\n📊 系统统计:")
    stats = simulator.get_system_statistics()
    print(f"  ✅ 系统正常运行时间: {stats['performance_monitor']['total_operations']} 操作")
    
    print("\n🎉 性能基准测试完成！")
    return True

if __name__ == "__main__":
    torch.manual_seed(42)  # 固定随机种子
    run_performance_benchmark()
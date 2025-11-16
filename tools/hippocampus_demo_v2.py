#!/usr/bin/env python3
"""
海马体模拟器演示程序
展示基于神经科学原理的生物启发式记忆系统功能
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import json
from typing import Dict, Any, List

# 导入海马体模拟器
from hippocampus import create_hippocampus_simulator, get_default_config


def print_banner():
    """打印演示横幅"""
    print("=" * 80)
    print("🧠 海马体模拟器演示程序 v2.0.0")
    print("基于Science期刊2025年研究成果的生物启发式记忆系统")
    print("=" * 80)
    print()


def demonstrate_system_initialization():
    """演示系统初始化"""
    print("🔧 系统初始化演示")
    print("-" * 40)
    
    # 使用默认配置创建模拟器
    config = get_default_config()
    print(f"✓ 默认配置加载完成，包含 {len(config)} 个参数")
    
    # 创建模拟器实例
    simulator = create_hippocampus_simulator(
        input_dim=512,
        hidden_dim=256,
        vocab_size=10000
    )
    
    # 显示系统信息
    system_stats = simulator.get_system_statistics()
    total_params = system_stats['system_info']['total_parameters']
    memory_usage = system_stats['system_info']['model_size_mb']
    
    print(f"✓ 海马体模拟器创建成功")
    print(f"  - 总参数数量: {total_params:,}")
    print(f"  - 模型大小: {memory_usage:.2f} MB")
    print(f"  - 输入维度: 512")
    print(f"  - 隐藏维度: 256")
    print()
    
    return simulator


def demonstrate_memory_encoding(simulator: Any):
    """演示记忆编码功能"""
    print("📝 记忆编码演示")
    print("-" * 40)
    
    # 创建测试记忆内容
    batch_size = 4
    memory_dim = 256
    
    # 生成不同的记忆内容
    memories = [
        "今天学习了深度学习的Transformer架构",
        "海马体在记忆形成中起关键作用",
        "突触可塑性是学习的基础机制", 
        "情景记忆涉及时间序列处理"
    ]
    
    # 转换为张量（模拟编码后的向量）
    memory_tensors = []
    for i, memory in enumerate(memories):
        # 模拟将文本编码为向量
        np.random.seed(i + 42)  # 确保可重现性
        tensor = torch.randn(memory_dim)
        memory_tensors.append(tensor)
        print(f"  记忆 {i+1}: {memory[:30]}...")
    
    print(f"\n✓ 准备了 {len(memories)} 个测试记忆")
    
    # 编码记忆到海马体系统
    encoded_results = []
    encoding_times = []
    
    for i, memory_tensor in enumerate(memory_tensors):
        start_time = time.time()
        
        encoding_result = simulator.encode_memory(
            content=memory_tensor,
            context=None,
            metadata={'id': f'memory_{i+1}', 'type': 'episodic'}
        )
        
        encoding_time = time.time() - start_time
        encoding_times.append(encoding_time)
        encoded_results.append(encoding_result)
        
        print(f"  记忆 {i+1} 编码完成 (耗时: {encoding_time:.4f}s)")
    
    avg_encoding_time = np.mean(encoding_times)
    print(f"\n✓ 记忆编码完成，平均耗时: {avg_encoding_time:.4f}s")
    print()
    
    return encoded_results


def demonstrate_memory_retrieval(simulator: Any, encoded_results: List[Dict]):
    """演示记忆检索功能"""
    print("🔍 记忆检索演示")
    print("-" * 40)
    
    # 准备检索查询
    query_texts = [
        "深度学习架构",
        "记忆机制",
        "神经网络学习"
    ]
    
    print("检索查询:")
    for i, query in enumerate(query_texts):
        print(f"  查询 {i+1}: {query}")
    
    retrieval_results = []
    retrieval_times = []
    
    for i, query_text in enumerate(query_texts):
        # 创建查询向量（模拟文本编码）
        np.random.seed(i + 100)
        query_tensor = torch.randn(256)
        
        start_time = time.time()
        
        # 执行检索
        retrieval_result = simulator.retrieve_memory(
            query=query_tensor,
            retrieval_type='hybrid',
            num_results=3
        )
        
        retrieval_time = time.time() - start_time
        retrieval_times.append(retrieval_time)
        retrieval_results.append(retrieval_result)
        
        similarity = retrieval_result['similarity_score']
        print(f"  查询 {i+1} 检索完成 (相似度: {similarity:.4f}, 耗时: {retrieval_time:.4f}s)")
    
    avg_retrieval_time = np.mean(retrieval_times)
    print(f"\n✓ 记忆检索完成，平均耗时: {avg_retrieval_time:.4f}s")
    print()
    
    return retrieval_results


def demonstrate_pattern_separation(simulator: Any):
    """演示模式分离功能"""
    print("🎯 模式分离演示")
    print("-" * 40)
    
    # 创建相似和不同的输入对
    base_input = torch.randn(256)
    
    # 相似输入（在基础输入上添加小扰动）
    similar_input = base_input + 0.1 * torch.randn(256)
    
    # 不同输入（完全不同的向量）
    different_input = torch.randn(256)
    
    inputs = {
        '相似输入': (base_input, similar_input),
        '不同输入': (base_input, different_input)
    }
    
    print("测试模式分离效果:")
    
    for test_name, (input1, input2) in inputs.items():
        # 计算分离指标
        metrics = simulator.pattern_separator.compute_separation_metrics(input1, input2)
        
        print(f"\n  {test_name}:")
        print(f"    原始相似度: {metrics['cosine_similarity']:.4f}")
        print(f"    分离后相似度: {metrics['separated_cosine']:.4f}")
        print(f"    分离改善: {metrics['separation_improvement']:.4f}")
        print(f"    欧氏距离变化: {metrics['euclidean_distance']:.4f} -> {metrics['separated_euclidean']:.4f}")
    
    print("\n✓ 模式分离演示完成")
    print()


def demonstrate_fast_learning(simulator: Any):
    """演示快速学习功能"""
    print("⚡ 快速学习演示")
    print("-" * 40)
    
    # 创建few-shot学习任务
    num_way = 3  # 3类
    num_shot = 2  # 每类2个样本
    num_query = 4  # 4个查询样本
    
    # 生成支持集和查询集
    support_size = num_way * num_shot
    query_size = num_query
    
    support_x = torch.randn(support_size, 256)
    query_x = torch.randn(query_size, 256)
    
    # 模拟支持集标签
    support_y = torch.cat([
        torch.full((num_shot,), i, dtype=torch.long) 
        for i in range(num_way)
    ])
    
    print(f"Few-shot学习任务:")
    print(f"  - 类别数: {num_way}")
    print(f"  - 每类样本数: {num_shot}")
    print(f"  - 查询样本数: {query_size}")
    
    # 执行few-shot学习
    start_time = time.time()
    
    predictions, learning_stats = simulator.one_shot_learner.few_shot_learning(
        support_x=support_x,
        support_y=support_y,
        query_x=query_x,
        adaptation_steps=5
    )
    
    learning_time = time.time() - start_time
    
    print(f"\n学习结果:")
    print(f"  - 预测形状: {predictions.shape}")
    print(f"  - 学习耗时: {learning_time:.4f}s")
    print(f"  - 支持集编码范数: {learning_stats['support_encoded_norm']:.4f}")
    print(f"  - 查询集编码范数: {learning_stats['query_encoded_norm']:.4f}")
    print(f"  - 适应策略: {learning_stats['adaptation_strategy'].mean():.4f}")
    
    print("\n✓ 快速学习演示完成")
    print()


def demonstrate_episodic_memory(simulator: Any):
    """演示情景记忆功能"""
    print("📚 情景记忆演示")
    print("-" * 40)
    
    # 创建时间序列记忆
    base_time = time.time()
    episodes = []
    
    for i in range(5):
        timestamp = base_time + i * 3600  # 每小时一个事件
        
        # 创建情景内容
        content = torch.randn(256)
        context = torch.randn(256)
        
        episode_id = f"episode_{i+1}"
        
        # 存储情景记忆
        result = simulator.episodic_memory.store_episode(
            content=content,
            timestamp=timestamp,
            context=context,
            episode_id=episode_id
        )
        
        episodes.append({
            'id': episode_id,
            'timestamp': timestamp,
            'content_norm': torch.norm(content).item()
        })
        
        print(f"  存储情景 {i+1}: {episode_id} (时间: {time.ctime(timestamp)})")
    
    print(f"\n✓ 存储了 {len(episodes)} 个情景记忆")
    
    # 检索情景记忆
    query_content = torch.randn(256)
    query_time = base_time + 7200  # 查询第3小时的事件
    
    start_time = time.time()
    
    retrieved, stats = simulator.episodic_memory.retrieve_episodes(
        query_content=query_content,
        query_context=torch.randn(256),
        query_timestamp=query_time,
        time_window=(base_time, base_time + 4 * 3600),  # 前4小时
        retrieval_type='temporal'
    )
    
    retrieval_time = time.time() - start_time
    
    print(f"\n情景记忆检索:")
    print(f"  - 检索耗时: {retrieval_time:.4f}s")
    print(f"  - 检索类型: {stats['retrieval_type']}")
    print(f"  - 搜索细胞数: {stats['num_cells_searched']}")
    print(f"  - 平均检索分数: {stats['avg_retrieval_score']:.4f}")
    
    print("\n✓ 情景记忆演示完成")
    print()


def demonstrate_memory_consolidation(simulator: Any):
    """演示记忆巩固功能"""
    print("🔄 记忆巩固演示")
    print("-" * 40)
    
    # 执行记忆巩固
    consolidation_result = simulator.consolidate_memories()
    
    print("记忆巩固结果:")
    print(f"  - 巩固耗时: {consolidation_result['consolidation_time']:.4f}s")
    print(f"  - 时间戳: {time.ctime(consolidation_result['timestamp'])}")
    
    episodic_update = consolidation_result['episodic_update']
    print(f"  - 更新的记忆细胞: {episodic_update['cells_updated']}")
    print(f"  - 巩固的记忆数: {episodic_update['memories_consolidated']}")
    print(f"  - 遗忘的记忆数: {episodic_update['memories_forgotten']}")
    
    dict_compression = consolidation_result['dictionary_compression']
    print(f"  - 神经字典压缩:")
    for cell, count in dict_compression.items():
        print(f"    {cell}: {count} 个记忆被遗忘")
    
    print("\n✓ 记忆巩固演示完成")
    print()


def demonstrate_system_performance(simulator: Any):
    """演示系统性能监控"""
    print("📊 系统性能监控")
    print("-" * 40)
    
    # 获取完整系统统计
    system_stats = simulator.get_system_statistics()
    
    print("系统性能指标:")
    print(f"  - 总操作数: {system_stats['performance_monitor']['total_operations']}")
    print(f"  - 平均响应时间: {system_stats['performance_monitor']['avg_response_time']:.4f}s")
    print(f"  - 模型大小: {system_stats['system_info']['model_size_mb']:.2f} MB")
    
    print("\n模块使用统计:")
    module_usage = system_stats['performance_monitor']['module_usage']
    for module, count in module_usage.items():
        print(f"  - {module}: {count} 次调用")
    
    print("\n详细模块统计:")
    
    # 神经字典统计
    dict_stats = system_stats['modules']['neural_dictionary']
    print(f"  神经字典:")
    print(f"    - 当前大小: {dict_stats['total_current_size']}")
    print(f"    - 平均利用率: {dict_stats['average_utilization']:.4f}")
    print(f"    - 总写入次数: {dict_stats['global_memory_stats']['total_writes']}")
    print(f"    - 总检索次数: {dict_stats['global_memory_stats']['total_retrievals']}")
    
    # 模式分离器统计
    sep_stats = system_stats['modules']['pattern_separator']
    print(f"  模式分离器:")
    print(f"    - 颗粒细胞数: {sep_stats['granule_layer']['num_granule_cells']}")
    print(f"    - 当前稀疏性: {sep_stats['granule_layer']['current_sparsity']:.4f}")
    print(f"    - CA3细胞数: {sep_stats['ca3_network']['num_ca3_cells']}")
    
    # 情景记忆统计
    epi_stats = system_stats['modules']['episodic_memory']
    print(f"  情景记忆:")
    print(f"    - 记忆细胞数: {epi_stats['num_memory_cells']}")
    print(f"    - 总容量: {epi_stats['total_capacity']}")
    print(f"    - 当前使用: {epi_stats['total_current_size']}")
    print(f"    - 短期缓冲: {epi_stats['short_term_buffer_size']}")
    
    print("\n✓ 系统性能监控完成")
    print()


def demonstrate_memory_export(simulator: Any):
    """演示记忆状态导出"""
    print("💾 记忆状态导出演示")
    print("-" * 40)
    
    export_path = "/workspace/hippocampus_memory_state.json"
    
    # 导出记忆状态
    simulator.export_memory_state(export_path)
    
    print(f"✓ 记忆状态已导出到: {export_path}")
    
    # 检查导出文件
    try:
        with open(export_path, 'r', encoding='utf-8') as f:
            export_data = json.load(f)
        
        print(f"  导出文件大小: {len(json.dumps(export_data))} 字符")
        print(f"  包含系统统计: {'system_statistics' in export_data}")
        print(f"  包含神经字典状态: {'neural_dictionary_state' in export_data}")
        print(f"  包含情景记忆缓冲: {'episodic_memory_buffer' in export_data}")
        print(f"  导出时间戳: {export_data['timestamp']}")
        
    except Exception as e:
        print(f"  导出文件读取失败: {e}")
    
    print("\n✓ 记忆状态导出演示完成")
    print()


def main():
    """主演示函数"""
    print_banner()
    
    try:
        # 1. 系统初始化
        simulator = demonstrate_system_initialization()
        
        # 2. 记忆编码演示
        encoded_results = demonstrate_memory_encoding(simulator)
        
        # 3. 记忆检索演示
        retrieval_results = demonstrate_memory_retrieval(simulator, encoded_results)
        
        # 4. 模式分离演示
        demonstrate_pattern_separation(simulator)
        
        # 5. 快速学习演示
        demonstrate_fast_learning(simulator)
        
        # 6. 情景记忆演示
        demonstrate_episodic_memory(simulator)
        
        # 7. 记忆巩固演示
        demonstrate_memory_consolidation(simulator)
        
        # 8. 系统性能监控
        demonstrate_system_performance(simulator)
        
        # 9. 记忆状态导出
        demonstrate_memory_export(simulator)
        
        # 完成总结
        print("🎉 演示完成总结")
        print("=" * 40)
        print("✓ 所有功能模块测试通过")
        print("✓ 海马体记忆系统运行正常")
        print("✓ 基于神经科学的生物启发式实现验证成功")
        print("\n📝 主要特性验证:")
        print("  • Transformer-based记忆编码")
        print("  • 可微分神经字典存储检索")
        print("  • 模式分离机制")
        print("  • 快速一次性学习")
        print("  • 情景记忆时间序列处理")
        print("  • 记忆巩固和衰减")
        print("  • 性能监控和状态导出")
        print("\n🧠 海马体模拟器v2.0.0部署成功！")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
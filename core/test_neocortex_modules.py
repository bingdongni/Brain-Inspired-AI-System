"""
新皮层模拟器核心模块测试脚本
============================

测试所有新皮层模拟器的核心组件：
1. 分层抽象机制测试
2. 专业处理模块测试  
3. 知识抽象算法测试
4. 稀疏激活和权重巩固测试
5. 新皮层架构集成测试

运行方式：
python test_neocortex_modules.py
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入新皮层模拟器核心模块
from core import (
    # 分层抽象机制
    HierarchicalLayer, LayerConfig, ProcessingHierarchy, LayerType, ProcessingMode,
    create_visual_hierarchy, create_auditory_hierarchy,
    
    # 专业处理模块
    PredictionModule, AttentionModule, DecisionModule, CrossModalModule,
    PredictionType, AttentionType, DecisionMode, ProcessingConfig,
    create_prediction_module, create_attention_module, 
    create_decision_module, create_crossmodal_module,
    
    # 知识抽象算法
    AbstractionEngine, ConceptUnit, SemanticAbstraction,
    ConceptConfig, ConceptType, AbstractionLevel,
    create_abstraction_engine, create_concept_units,
    
    # 稀疏激活和权重巩固
    ConsolidationEngine, SparseActivation, WeightConsolidation, EngramCell,
    ConsolidationConfig, EngramConfig, CellType, MemoryState,
    create_consolidation_engine,
    
    # 新皮层架构
    NeocortexSimulator, TONN, ModularNeocortex,
    NeocortexConfig, ArchitectureType
)


def test_hierarchical_layers():
    """测试分层抽象机制"""
    print("=== 测试分层抽象机制 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建视觉层次配置
    visual_configs = create_visual_hierarchy(input_channels=3)
    print(f"视觉层次配置: {len(visual_configs)}层")
    for i, config in enumerate(visual_configs):
        print(f"  层{i+1}: {config.layer_type.value} - {config.input_channels}->{config.output_channels}")
    
    # 创建处理层次
    visual_hierarchy = ProcessingHierarchy(visual_configs).to(device)
    print(f"✓ 创建视觉处理层次成功")
    
    # 测试前向传播
    test_input = torch.randn(2, 3, 224, 224).to(device)
    outputs = visual_hierarchy(test_input)
    
    print(f"✓ 视觉层次前向传播成功")
    print(f"  输入形状: {test_input.shape}")
    print(f"  最终输出形状: {outputs['final_output'].shape}")
    print(f"  层次数量: {len(outputs['layer_outputs'])}")
    
    # 创建听觉层次配置
    auditory_configs = create_auditory_hierarchy(input_channels=1)
    print(f"✓ 听觉层次配置: {len(auditory_configs)}层")
    
    return True


def test_processing_modules():
    """测试专业处理模块"""
    print("\n=== 测试专业处理模块 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试预测模块
    prediction_module = create_prediction_module(128).to(device)
    test_input = torch.randn(4, 10, 128).to(device)  # batch, seq_len, dim
    pred_output = prediction_module(test_input)
    
    print(f"✓ 预测模块测试成功")
    print(f"  输入: {test_input.shape}")
    print(f"  预测输出: {pred_output['predictions'].shape}")
    print(f"  预测误差: {pred_output['prediction_errors'].shape}")
    print(f"  平均置信度: {pred_output['confidence'].mean().item():.3f}")
    
    # 测试注意模块
    attention_module = create_attention_module(256).to(device)
    test_features = torch.randn(4, 256, 14, 14).to(device)
    attention_output = attention_module(test_features)
    
    print(f"✓ 注意模块测试成功")
    print(f"  输入特征: {test_features.shape}")
    print(f"  注意特征: {attention_output['attended_features'].shape}")
    print(f"  空间注意: {attention_output['spatial_attention'].shape}")
    
    # 测试决策模块
    decision_module = create_decision_module(512, 2).to(device)
    test_responses = torch.randn(4, 512).to(device)
    decision_output = decision_module(test_responses)
    
    print(f"✓ 决策模块测试成功")
    print(f"  神经元响应: {test_responses.shape}")
    print(f"  决策输出: {decision_output['decision'].shape}")
    print(f"  置信度: {decision_output['confidence'].mean().item():.3f}")
    
    # 测试跨模态模块
    crossmodal_module = create_crossmodal_module(256, 128).to(device)
    visual_input = torch.randn(4, 256).to(device)
    language_input = torch.randn(4, 256).to(device)
    crossmodal_output = crossmodal_module(visual_input, language_input)
    
    print(f"✓ 跨模态模块测试成功")
    print(f"  视觉输入: {visual_input.shape}")
    print(f"  语言输入: {language_input.shape}")
    print(f"  概念表示: {crossmodal_output['concept_representation'].shape}")
    print(f"  抽象表示: {crossmodal_output['abstract_representation'].shape}")
    print(f"  跨模态一致性: {crossmodal_output['cross_modal_consistency'].mean().item():.3f}")
    
    return True


def test_abstraction_algorithms():
    """测试知识抽象算法"""
    print("\n=== 测试知识抽象算法 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建抽象引擎
    abstraction_engine = create_abstraction_engine(256, 20).to(device)
    print(f"✓ 创建抽象引擎成功")
    
    # 测试概念单元
    concept_configs = [
        ConceptConfig(
            concept_id=f"concept_{i}",
            concept_type=ConceptType.OBJECT,
            abstraction_level=AbstractionLevel.CONCEPTUAL
        )
        for i in range(5)
    ]
    
    concept_units = create_concept_units(256, 5)
    print(f"✓ 创建{len(concept_units)}个概念单元")
    
    # 测试前向传播
    test_features = torch.randn(4, 256).to(device)
    results = abstraction_engine(test_features)
    
    print(f"✓ 抽象引擎测试成功")
    print(f"  输入特征: {test_features.shape}")
    print(f"  激活概念数量: {results['abstraction_summary']['num_active_concepts']}")
    print(f"  抽象层次: {results['abstraction_summary']['abstraction_level']:.3f}")
    print(f"  泛化分数: {results['abstraction_summary']['generalization_score']:.3f}")
    print(f"  最终抽象表示: {results['final_abstraction']['integrated_representation'].shape}")
    print(f"  抽象质量: {results['final_abstraction']['abstraction_quality'].mean().item():.3f}")
    
    # 测试语义抽象组件
    semantic_abstraction = SemanticAbstraction(256, 4)
    semantic_results = semantic_abstraction(test_features)
    
    print(f"✓ 语义抽象组件测试成功")
    print(f"  抽象层次: {len(semantic_results['abstraction_levels'])}")
    print(f"  概念聚类数量: {semantic_results['concept_clusters']['num_active_clusters']}")
    print(f"  语义关系数量: {semantic_results['semantic_relations']['num_relations']}")
    
    return True


def test_sparse_activation_consolidation():
    """测试稀疏激活和权重巩固"""
    print("\n=== 测试稀疏激活和权重巩固 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建巩固引擎
    consolidation_engine = create_consolidation_engine(256, 50).to(device)
    print(f"✓ 创建巩固引擎成功")
    
    # 测试输入
    test_features = torch.randn(4, 256).to(device)
    results = consolidation_engine(test_features)
    
    print(f"✓ 巩固引擎测试成功")
    print(f"  输入特征: {test_features.shape}")
    print(f"  稀疏效率: {results['consolidation_summary']['sparse_efficiency']:.3f}")
    print(f"  激活印记细胞: {results['consolidation_summary']['active_engram_cells']}")
    print(f"  形成印记数量: {results['consolidation_summary']['formed_engrams']}")
    print(f"  平均印记强度: {results['consolidation_summary']['avg_engram_strength']:.3f}")
    print(f"  巩固质量: {results['consolidation_summary']['consolidation_quality']:.3f}")
    print(f"  记忆健康度: {results['memory_state']['memory_health']:.3f}")
    
    # 测试稀疏激活组件
    sparse_activation = SparseActivation(ConsolidationConfig(), 256).to(device)
    sparse_results = sparse_activation(test_features)
    
    print(f"✓ 稀疏激活组件测试成功")
    print(f"  实际稀疏性: {sparse_results['actual_sparsity'].mean().item():.3f}")
    print(f"  稀疏效率: {sparse_results['sparse_efficiency'].mean().item():.3f}")
    
    # 测试权重巩固组件
    weight_consolidation = WeightConsolidation(
        ConsolidationConfig(), (256, 256)
    ).to(device)
    
    # 创建测试权重
    current_weights = torch.randn(256, 256) * 0.1
    weight_changes = torch.randn(256, 256) * 0.01
    
    consolidation_results = weight_consolidation(current_weights, weight_changes)
    
    print(f"✓ 权重巩固组件测试成功")
    print(f"  当前权重形状: {current_weights.shape}")
    print(f"  巩固质量: {consolidation_results['consolidation_quality'].item():.3f}")
    print(f"  权重稳定性: {consolidation_results['weight_stability'].item():.3f}")
    print(f"  记忆强度: {consolidation_results['memory_strength'].item():.3f}")
    
    return True


def test_neocortex_architecture():
    """测试新皮层架构"""
    print("\n=== 测试新皮层架构 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建新皮层配置
    config = NeocortexConfig(
        architecture_type=ArchitectureType.TONN,
        input_dim=512,
        hidden_dim=256,
        output_dim=128,
        prediction_enabled=True,
        attention_enabled=True,
        abstraction_enabled=True,
        consolidation_enabled=True,
        decision_enabled=True,
        num_concepts=30,
        num_engram_cells=60
    )
    
    print(f"✓ 创建新皮层配置")
    print(f"  架构类型: {config.architecture_type.value}")
    print(f"  输入维度: {config.input_dim}")
    print(f"  隐藏维度: {config.hidden_dim}")
    
    # 创建新皮层模拟器
    neocortex_simulator = NeocortexSimulator(config).to(device)
    print(f"✓ 创建新皮层模拟器成功")
    
    # 创建测试输入
    test_inputs = {
        'visual': torch.randn(2, 512).to(device),
        'multimodal': torch.randn(2, 512).to(device)
    }
    
    # 测试前向传播
    results = neocortex_simulator(test_inputs)
    
    print(f"✓ 新皮层模拟器测试成功")
    print(f"  输入形状: {test_inputs['visual'].shape}")
    print(f"  总处理阶段: {len(results['stage_outputs'])}")
    print(f"  最终输出形状: {results['final_output'].shape}")
    print(f"  架构类型: {results['summary']['architecture_type']}")
    print(f"  处理效率: {results['summary']['processing_efficiency']:.3f}")
    
    # 显示各阶段信息
    print(f"\n处理阶段详情:")
    for stage, info in results['stage_info'].items():
        print(f"  {stage.value}: {info}")
    
    # 性能指标
    perf_metrics = results['performance_metrics']
    print(f"\n性能指标:")
    print(f"  处理效率: {perf_metrics['processing_efficiency']:.3f}")
    print(f"  输出质量: {perf_metrics['output_quality']:.3f}")
    print(f"  资源使用: {perf_metrics['resource_usage']}")
    print(f"  整体效率: {perf_metrics['overall_efficiency']:.3f}")
    
    # 测试TONN
    tonn = TONN(config).to(device)
    tonn_results = tonn(test_inputs)
    
    print(f"\n✓ TONN测试成功")
    print(f"  TONN输出形状: {tonn_results['final_output'].shape}")
    
    # 测试模块化架构
    modular_neocortex = ModularNeocortex(config).to(device)
    modular_neocortex.configure_modules(['hierarchical', 'attention', 'decision'])
    modular_results = modular_neocortex(test_inputs)
    
    print(f"✓ 模块化架构测试成功")
    print(f"  模块化输出形状: {modular_results['final_output'].shape}")
    print(f"  活跃模块: {list(modular_results['module_results'].keys())}")
    
    return True


def performance_benchmark():
    """性能基准测试"""
    print("\n=== 性能基准测试 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 导入时间模块
    import time
    
    # 创建配置
    config = NeocortexConfig(
        architecture_type=ArchitectureType.TONN,
        input_dim=512,
        hidden_dim=256,
        output_dim=128,
        prediction_enabled=True,
        attention_enabled=True,
        abstraction_enabled=True,
        consolidation_enabled=True,
        decision_enabled=True
    )
    
    # 创建模拟器
    neocortex = NeocortexSimulator(config).to(device)
    
    # 准备测试数据
    batch_sizes = [1, 4, 8, 16]
    test_inputs = {
        'visual': torch.randn(1, 512).to(device),
        'multimodal': torch.randn(1, 512).to(device)
    }
    
    print("批次大小 | 前向传播时间 | 内存使用(MB)")
    print("-" * 45)
    
    for batch_size in batch_sizes:
        # 创建对应批次大小的输入
        batch_inputs = {
            'visual': torch.randn(batch_size, 512).to(device),
            'multimodal': torch.randn(batch_size, 512).to(device)
        }
        
        # 清空GPU缓存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # 测量前向传播时间
        start_time = time.time()
        with torch.no_grad():
            _ = neocortex(batch_inputs)
        forward_time = time.time() - start_time
        
        # 测量内存使用
        if device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated(device) / 1024 / 1024
        else:
            memory_used = 0
        
        print(f"{batch_size:^8} | {forward_time:^12.4f} | {memory_used:^12.1f}")
    
    print(f"\n✓ 性能基准测试完成")


def run_all_tests():
    """运行所有测试"""
    print("🧠 新皮层模拟器核心模块测试")
    print("=" * 50)
    
    try:
        # 基础模块测试
        test_hierarchical_layers()
        test_processing_modules()
        test_abstraction_algorithms()
        test_sparse_activation_consolidation()
        test_neocortex_architecture()
        
        # 性能测试
        performance_benchmark()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试通过！新皮层模拟器核心模块工作正常。")
        print("\n📊 测试总结:")
        print("  ✓ 分层抽象机制 - 正常")
        print("  ✓ 专业处理模块 - 正常")
        print("  ✓ 知识抽象算法 - 正常")
        print("  ✓ 稀疏激活和权重巩固 - 正常")
        print("  ✓ 新皮层架构 - 正常")
        print("\n🚀 新皮层模拟器已准备就绪，可以进行进一步的研究和开发！")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

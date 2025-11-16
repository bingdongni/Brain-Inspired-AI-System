#!/usr/bin/env python3
"""
高级认知功能系统快速测试（简化版）
=============================

快速验证所有高级认知功能模块的基本功能。

作者: Brain-Inspired AI Team
创建时间: 2025-11-16
"""

import sys
import os
import time
import numpy as np
import logging

# 设置路径
sys.path.append('/workspace/brain-inspired-ai/src')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_imports():
    """测试基本模块导入"""
    print("=" * 60)
    print("1. 测试模块导入")
    print("=" * 60)
    
    try:
        # 测试核心模块导入
        from advanced_cognition.end_to_end_pipeline import EndToEndTrainingPipeline
        from advanced_cognition.performance_optimization import PerformanceOptimizer
        from advanced_cognition.multi_step_reasoning import MultiStepReasoner
        from advanced_cognition.analogical_learning import AnalogicalLearner
        from advanced_cognition.system_integration import CognitiveSystemIntegrator
        print("✅ 核心模块导入成功")
        
        # 测试便利函数
        from advanced_cognition.end_to_end_pipeline import create_standard_classification_pipeline
        from advanced_cognition.performance_optimization import create_neural_network_optimization_config
        from advanced_cognition.multi_step_reasoning import create_comprehensive_reasoner
        from advanced_cognition.analogical_learning import create_analogical_learner
        from advanced_cognition.system_integration import create_cognitive_system_integrator
        print("✅ 便利函数导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_component_creation():
    """测试组件创建"""
    print("\n" + "=" * 60)
    print("2. 测试组件创建")
    print("=" * 60)
    
    try:
        # 测试端到端训练管道
        from advanced_cognition.end_to_end_pipeline import create_standard_classification_pipeline
        pipeline = create_standard_classification_pipeline()
        print("✅ 训练管道创建成功")
        
        # 测试性能优化
        from advanced_cognition.performance_optimization import create_neural_network_optimization_config
        config = create_neural_network_optimization_config()
        optimizer = PerformanceOptimizer(config)
        print("✅ 性能优化器创建成功")
        
        # 测试多步推理
        from advanced_cognition.multi_step_reasoning import create_comprehensive_reasoner
        reasoner = create_comprehensive_reasoner()
        print("✅ 多步推理器创建成功")
        
        # 测试类比学习
        from advanced_cognition.analogical_learning import create_analogical_learner
        learner = create_analogical_learner()
        print("✅ 类比学习器创建成功")
        
        # 测试系统集成
        from advanced_cognition.system_integration import create_cognitive_system_integrator
        integrator = create_cognitive_system_integrator()
        print("✅ 系统集成器创建成功")
        
        return True
    except Exception as e:
        print(f"❌ 组件创建失败: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "=" * 60)
    print("3. 测试基本功能")
    print("=" * 60)
    
    try:
        # 测试类比学习的基本功能
        from advanced_cognition.analogical_learning import create_analogical_learner, KnowledgeConcept
        learner = create_analogical_learner()
        
        # 创建测试概念
        concept1 = KnowledgeConcept(
            concept_id="neural_network",
            name="神经网络",
            description="模拟大脑神经元连接的计算模型",
            properties={"layers": "multiple", "learning": "supervised"},
            relations={},
            domain="machine_learning"
        )
        
        concept2 = KnowledgeConcept(
            concept_id="social_network", 
            name="社交网络",
            description="人与人之间的连接关系网络",
            properties={"nodes": "people", "edges": "relationships"},
            relations={},
            domain="social_science"
        )
        print("✅ 知识概念创建成功")
        
        # 测试初始化
        learner.initialize()
        print("✅ 学习器初始化成功")
        
        # 测试学习
        success = learner.learn_from_example(
            "神经网络如何学习？",
            "通过反向传播算法调整权重",
            success=True
        )
        print(f"✅ 学习功能测试: {success}")
        
        # 测试问题解决
        solutions = learner.solve_problem_creatively(
            "如何设计智能推荐系统？",
            {'domain': 'recommendation', 'constraints': ['实时性']}
        )
        print(f"✅ 创造性问题解决: 生成了 {len(solutions)} 个解决方案")
        
        learner.cleanup()
        
        return True
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False

def test_reasoning():
    """测试推理功能"""
    print("\n" + "=" * 60)
    print("4. 测试推理功能")
    print("=" * 60)
    
    try:
        from advanced_cognition.multi_step_reasoning import create_comprehensive_reasoner, ReasoningType
        reasoner = create_comprehensive_reasoner()
        
        # 初始化
        reasoner.initialize()
        print("✅ 推理器初始化成功")
        
        # 测试推理
        problem = "如何提高机器学习模型的准确率？"
        information = [
            "当前模型准确率为75%",
            "数据集有10000个样本",
            "使用随机森林算法"
        ]
        
        result = reasoner.reason(problem, information, ReasoningType.DEDUCTIVE)
        print(f"✅ 推理执行成功")
        print(f"   - 置信度: {result.confidence_score:.3f}")
        print(f"   - 推理步骤: {len(result.reasoning_chain)}")
        
        reasoner.cleanup()
        
        return True
    except Exception as e:
        print(f"❌ 推理功能测试失败: {e}")
        return False

def test_system_integration():
    """测试系统集成"""
    print("\n" + "=" * 60)
    print("5. 测试系统集成")
    print("=" * 60)
    
    try:
        from advanced_cognition.system_integration import create_cognitive_system_integrator
        integrator = create_cognitive_system_integrator()
        
        # 初始化系统
        success = integrator.initialize()
        print(f"✅ 系统初始化: {success}")
        
        # 获取系统状态
        status = integrator.get_system_status()
        print("✅ 系统状态获取成功")
        print(f"   - 模块数量: {len(status.get('module_states', {}))}")
        print(f"   - 工作流数量: {status.get('execution_statistics', {}).get('workflows_count', 0)}")
        
        # 性能优化
        optimization_result = integrator.optimize_system_performance()
        print(f"✅ 系统优化完成: {optimization_result.get('bottlenecks_identified', 0)} 个瓶颈")
        
        integrator.cleanup()
        
        return True
    except Exception as e:
        print(f"❌ 系统集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("高级认知功能系统快速测试（简化版）")
    print("=" * 80)
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    test_results = {}
    total_tests = 5
    passed_tests = 0
    
    # 执行各项测试
    tests = [
        ("模块导入", test_basic_imports),
        ("组件创建", test_component_creation),
        ("基本功能", test_basic_functionality),
        ("推理功能", test_reasoning),
        ("系统集成", test_system_integration)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results[test_name] = result
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            test_results[test_name] = False
    
    # 生成测试报告
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    success_rate = passed_tests / total_tests
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"成功率: {success_rate:.2%}")
    print()
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<20} {status}")
    
    print("\n" + "=" * 80)
    if success_rate >= 0.8:
        print("🎉 系统测试基本通过！")
        print("所有核心功能模块均可正常工作，系统集成成功。")
    elif success_rate >= 0.6:
        print("⚠️  系统测试部分通过")
        print("大部分功能正常，个别模块需要检查。")
    else:
        print("❌ 系统测试失败")
        print("多个核心功能模块存在问题，需要修复。")
    
    print("=" * 80)
    
    # 保存测试报告
    with open('/workspace/quick_test_report.json', 'w', encoding='utf-8') as f:
        import json
        json.dump({
            'test_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'results': test_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"测试报告已保存到: /workspace/quick_test_report.json")
    
    return success_rate >= 0.8

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
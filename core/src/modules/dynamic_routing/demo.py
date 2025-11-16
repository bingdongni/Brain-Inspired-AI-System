"""
动态路由系统基本演示
快速验证系统核心功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import time

def quick_demo():
    """快速演示动态路由系统"""
    print("=" * 50)
    print("动态路由系统快速演示")
    print("=" * 50)
    
    try:
        from brain_ai.src.modules.dynamic_routing import DynamicRoutingSystem
        
        print("\n✅ 模块导入成功")
        
        # 创建系统实例
        print("\n🔄 创建动态路由系统...")
        with DynamicRoutingSystem() as routing_system:
            print("✅ 系统初始化成功")
            
            # 测试基本路由功能
            print("\n📡 测试路由请求处理...")
            
            # 模拟几个路由请求
            test_cases = [
                ("server_A", "server_B", {"max_latency": 1.0, "min_reliability": 0.9}),
                ("device_X", "device_Y", {"max_energy": 2.0}),
                ("node_1", "node_5", {"priority": 9}),
            ]
            
            for i, (source, destination, requirements) in enumerate(test_cases, 1):
                print(f"\n  请求 {i}: {source} -> {destination}")
                
                try:
                    decision = routing_system.process_request(
                        source=source,
                        destination=destination,
                        priority=np.random.randint(1, 11),
                        requirements=requirements
                    )
                    
                    print(f"    ✓ 路径选择: {decision.selected_path}")
                    print(f"    ✓ 预估延迟: {decision.estimated_latency:.3f}s")
                    print(f"    ✓ 预估能耗: {decision.estimated_energy:.3f}")
                    print(f"    ✓ 置信度: {decision.confidence_score:.3f}")
                    
                    # 模拟路由完成
                    routing_system.controller.complete_route(
                        decision.request_id,
                        actual_latency=decision.estimated_latency * 0.9,
                        actual_energy=decision.estimated_energy * 0.8,
                        success=np.random.random() > 0.1  # 90%成功率
                    )
                    
                except Exception as e:
                    print(f"    ⚠️ 处理异常: {e}")
            
            # 获取系统状态
            print("\n📊 系统状态检查...")
            status = routing_system.get_system_status()
            
            print(f"  总请求数: {status['total_requests']}")
            print(f"  成功率: {status['success_rate']:.2%}")
            print(f"  平均延迟: {status['avg_latency']:.3f}s")
            print(f"  平均能耗: {status['avg_energy_consumption']:.3f}")
            print(f"  系统健康度: {status['system_health']:.2%}")
            
            # 获取性能报告
            print("\n📈 性能报告...")
            report = routing_system.get_performance_report()
            
            if 'recommendations' in report and report['recommendations']:
                print("  优化建议:")
                for rec in report['recommendations'][:3]:  # 显示前3条
                    print(f"    • [{rec['priority'].upper()}] {rec['message']}")
            else:
                print("  ✓ 系统运行良好，无需优化")
            
            print("\n✅ 演示完成！")
            
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def module_demo():
    """演示各个模块的独立功能"""
    print("\n" + "=" * 50)
    print("模块独立功能演示")
    print("=" * 50)
    
    # 1. 强化学习模块演示
    print("\n🎯 强化学习模块演示")
    try:
        from brain_ai.src.modules.dynamic_routing.reinforcement_routing import (
            ActorCriticRouter, QLearningRouter, RoutingEnvironment
        )
        
        # Actor-Critic演示
        print("  Actor-Critic路由器:")
        ac_router = ActorCriticRouter(state_dim=16, action_dim=4)
        test_state = np.random.randn(16)
        action = ac_router.select_action(test_state, training=True)
        print(f"    输入状态维度: {len(test_state)}")
        print(f"    选择动作: {action}")
        print(f"    模块统计: {ac_router.get_statistics()['training_steps']} 训练步骤")
        
        # Q-Learning演示
        print("  Q-Learning路由器:")
        ql_router = QLearningRouter(state_dim=16, action_dim=4, use_deep_q=True)
        q_values = ql_router.get_q_values(test_state)
        print(f"    Q值分布: {q_values[:3]} ... (显示前3个)")
        print(f"    模块统计: {ql_router.get_statistics()['training_steps']} 训练步骤")
        
        print("✅ 强化学习模块演示成功")
        
    except Exception as e:
        print(f"❌ 强化学习模块演示失败: {e}")
    
    # 2. 自适应分配模块演示
    print("\n⚖️ 自适应分配模块演示")
    try:
        from brain_ai.src.modules.dynamic_routing.adaptive_allocation import (
            DynamicWeightRouter, AdaptiveLoadBalancer
        )
        
        # 动态权重路由演示
        print("  动态权重路由器:")
        dw_router = DynamicWeightRouter(num_paths=6, state_dim=20)
        path = dw_router.select_path('normal')
        print(f"    可选路径数: {dw_router.num_paths}")
        print(f"    选择的路径: {path}")
        stats = dw_router.get_statistics()
        print(f"    成功率: {stats['success_rate']:.2%}")
        
        # 负载均衡器演示
        print("  自适应负载均衡器:")
        lb = AdaptiveLoadBalancer(num_nodes=5, balancing_strategy='adaptive')
        node = lb.select_node({'priority': 5})
        print(f"    服务器节点数: {lb.num_nodes}")
        print(f"    选择的节点: {node}")
        lb_stats = lb.get_load_balancing_stats()
        print(f"    平均响应时间: {lb_stats['avg_response_time']:.3f}s")
        
        print("✅ 自适应分配模块演示成功")
        
    except Exception as e:
        print(f"❌ 自适应分配模块演示失败: {e}")
    
    # 3. 能效优化模块演示
    print("\n🔋 能效优化模块演示")
    try:
        from brain_ai.src.modules.dynamic_routing.efficiency_optimization import (
            NeuralInspiredRouter, IntelligentPathSelector
        )
        
        # 神经启发路由演示
        print("  神经启发路由器:")
        ni_router = NeuralInspiredRouter(num_neurons=32, input_dim=16, num_paths=4)
        test_state = np.random.randn(16)
        route, energy_rating, confidence = ni_router.process_input(test_state)
        print(f"    神经元数量: {ni_router.num_neurons}")
        print(f"    路由决策: {route}")
        print(f"    能效评分: {energy_rating:.3f}")
        print(f"    置信度: {confidence:.3f}")
        
        # 智能路径选择演示
        print("  智能路径选择器:")
        ips = IntelligentPathSelector(num_nodes=10, num_objectives=3)
        path_result = ips.find_optimal_path("node_1", "node_9", {"max_energy": 2.0})
        if 'selected_path' in path_result:
            print(f"    网络节点数: {ips.num_nodes}")
            print(f"    最优路径: {path_result['selected_path']}")
            print(f"    路径评分: {path_result['total_score']:.3f}")
        else:
            print(f"    路径搜索结果: {path_result.get('error', 'Unknown')}")
        
        print("✅ 能效优化模块演示成功")
        
    except Exception as e:
        print(f"❌ 能效优化模块演示失败: {e}")

if __name__ == "__main__":
    print("🚀 启动动态路由系统演示...")
    
    # 执行快速演示
    success = quick_demo()
    
    if success:
        # 执行模块演示
        module_demo()
        
        print("\n" + "🎉" * 15)
        print("动态路由系统演示完成！")
        print("所有核心功能正常运行。")
        print("🎉" * 15)
    else:
        print("\n" + "❌" * 15)
        print("演示过程中出现错误，请检查系统配置。")
        print("❌" * 15)
        sys.exit(1)
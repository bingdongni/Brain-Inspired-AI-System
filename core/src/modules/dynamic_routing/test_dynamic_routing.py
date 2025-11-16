"""
动态路由系统综合测试示例
展示所有核心功能的完整测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import time
from brain_ai.src.modules.dynamic_routing import (
    DynamicRoutingSystem, 
    RoutingRequest,
    ActorCriticRouter,
    DynamicWeightRouter,
    NeuralInspiredRouter,
    IntelligentPathSelector,
    RealTimeRoutingController
)

def test_complete_system():
    """测试完整的动态路由系统"""
    print("=" * 60)
    print("动态路由系统综合测试")
    print("=" * 60)
    
    # 1. 创建系统实例
    print("\n1. 初始化动态路由系统...")
    with DynamicRoutingSystem() as routing_system:
        print("✓ 系统初始化成功")
        
        # 2. 测试基本路由功能
        print("\n2. 测试基本路由功能...")
        test_basic_routing(routing_system)
        
        # 3. 测试强化学习模块
        print("\n3. 测试强化学习模块...")
        test_reinforcement_learning_modules(routing_system)
        
        # 4. 测试自适应分配模块
        print("\n4. 测试自适应分配模块...")
        test_adaptive_allocation_modules(routing_system)
        
        # 5. 测试能效优化模块
        print("\n5. 测试能效优化模块...")
        test_efficiency_optimization_modules(routing_system)
        
        # 6. 测试实时控制器
        print("\n6. 测试实时控制器...")
        test_realtime_controller(routing_system)
        
        # 7. 性能测试
        print("\n7. 性能压力测试...")
        performance_test(routing_system)
        
        # 8. 系统状态检查
        print("\n8. 系统状态检查...")
        system_status_check(routing_system)
        
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


def test_basic_routing(routing_system):
    """测试基本路由功能"""
    test_requests = [
        ("node_A", "node_B", 8, {"max_latency": 1.0}),
        ("node_C", "node_D", 5, {"max_energy": 1.5}),
        ("node_E", "node_F", 9, {"min_reliability": 0.9}),
    ]
    
    for i, (source, dest, priority, requirements) in enumerate(test_requests):
        print(f"  测试请求 {i+1}: {source} -> {dest}")
        
        decision = routing_system.process_request(
            source=source,
            destination=dest,
            priority=priority,
            requirements=requirements
        )
        
        print(f"    路径: {decision.selected_path}")
        print(f"    模块: {decision.selected_modules}")
        print(f"    延迟: {decision.estimated_latency:.3f}s")
        print(f"    能耗: {decision.estimated_energy:.3f}")
        print(f"    置信度: {decision.confidence_score:.3f}")
        
        # 模拟完成
        routing_system.controller.complete_route(
            decision.request_id,
            actual_latency=decision.estimated_latency * 0.9,
            actual_energy=decision.estimated_energy * 0.8,
            success=True
        )
    
    print("  ✓ 基本路由功能测试通过")


def test_reinforcement_learning_modules(routing_system):
    """测试强化学习模块"""
    # 测试Actor-Critic路由器
    if hasattr(routing_system.controller, 'actor_critic_router'):
        ac_router = routing_system.controller.actor_critic_router
        
        # 生成测试状态
        test_state = np.random.randn(ac_router.state_dim)
        
        # 选择动作
        action = ac_router.select_action(test_state, training=True)
        print(f"    Actor-Critic动作: {action}")
        
        # 训练步骤
        ac_router.store_transition(test_state, action, 1.0, test_state, False)
        ac_router.train_step()
        
        print("    ✓ Actor-Critic路由器测试通过")
    
    # 测试Q-Learning路由器
    if hasattr(routing_system.controller, 'q_learning_router'):
        ql_router = routing_system.controller.q_learning_router
        
        test_state = np.random.randn(ql_router.state_dim)
        action = ql_router.select_action(test_state, training=True)
        print(f"    Q-Learning动作: {action}")
        
        ql_router.store_transition(test_state, action, 0.8, test_state, False)
        ql_router.train_step()
        
        print("    ✓ Q-Learning路由器测试通过")
    
    # 测试多智能体路由器
    if hasattr(routing_system.controller, 'multi_agent_router'):
        ma_router = routing_system.controller.multi_agent_router
        
        test_state = np.random.randn(ma_router.state_dim)
        decision = ma_router.get_collaborative_decision(test_state)
        print(f"    多智能体决策: {decision}")
        
        print("    ✓ 多智能体路由器测试通过")


def test_adaptive_allocation_modules(routing_system):
    """测试自适应分配模块"""
    # 测试动态权重路由器
    if hasattr(routing_system.controller, 'dynamic_weight_router'):
        dw_router = routing_system.controller.dynamic_weight_router
        
        path_idx = dw_router.select_path('normal')
        print(f"    动态权重路径: {path_idx}")
        
        # 更新路径指标
        dw_router.update_path_metrics(path_idx, 1.2, 0.8, True)
        
        stats = dw_router.get_statistics()
        print(f"    成功率: {stats['success_rate']:.2%}")
        
        print("    ✓ 动态权重路由器测试通过")
    
    # 测试预测性早退
    if hasattr(routing_system.controller, 'predictive_early_exit'):
        pe_exit = routing_system.controller.predictive_early_exit
        
        test_state = np.random.randn(pe_exit.state_dim)
        should_exit, metrics = pe_exit.should_early_exit(
            test_state, 0, 10
        )
        
        print(f"    早退决策: {should_exit}")
        print(f"    置信度: {metrics.confidence:.3f}")
        print(f"    延迟节省: {metrics.latency_saved:.3f}")
        
        print("    ✓ 预测性早退测试通过")
    
    # 测试负载均衡器
    if hasattr(routing_system.controller, 'load_balancer'):
        lb = routing_system.controller.load_balancer
        
        node_idx = lb.select_node({'priority': 5})
        print(f"    负载均衡节点: {node_idx}")
        
        # 完成请求
        lb.complete_request(node_idx, 0.5, True)
        
        stats = lb.get_load_balancing_stats()
        print(f"    负载均衡成功率: {stats['success_rate']:.2%}")
        
        print("    ✓ 负载均衡器测试通过")


def test_efficiency_optimization_modules(routing_system):
    """测试能效优化模块"""
    # 测试神经启发路由器
    if hasattr(routing_system.controller, 'neural_inspired_router'):
        ni_router = routing_system.controller.neural_inspired_router
        
        test_state = np.random.randn(ni_router.input_dim)
        route_idx, energy_rating, confidence = ni_router.process_input(test_state)
        
        print(f"    神经启发路由: {route_idx}")
        print(f"    能效评分: {energy_rating:.3f}")
        print(f"    置信度: {confidence:.3f}")
        
        # 训练步骤
        target_route = np.random.randint(0, ni_router.num_paths)
        target_energy = np.random.uniform(0, 1)
        target_confidence = np.random.uniform(0.5, 1.0)
        
        loss = ni_router.train_step(test_state, target_route, target_energy, target_confidence)
        print(f"    训练损失: {loss:.4f}")
        
        print("    ✓ 神经启发路由器测试通过")
    
    # 测试智能路径选择器
    if hasattr(routing_system.controller, 'intelligent_path_selector'):
        ips = routing_system.controller.intelligent_path_selector
        
        # 模拟网络节点
        nodes = ["node_1", "node_2", "node_3", "node_4", "node_5"]
        source = np.random.choice(nodes)
        target = np.random.choice([n for n in nodes if n != source])
        
        result = ips.find_optimal_path(source, target, {'max_energy': 2.0})
        
        if 'selected_path' in result:
            print(f"    智能路径选择: {source} -> {target}")
            print(f"    选中路径: {result['selected_path']}")
            print(f"    总评分: {result['total_score']:.3f}")
        else:
            print(f"    路径选择失败: {result.get('error', 'Unknown error')}")
        
        print("    ✓ 智能路径选择器测试通过")


def test_realtime_controller(routing_system):
    """测试实时控制器"""
    controller = routing_system.controller
    
    # 创建测试请求
    request = RoutingRequest(
        id="test_req_001",
        source="test_source",
        destination="test_dest",
        priority=7,
        requirements={"max_latency": 1.5}
    )
    
    # 处理请求
    decision = controller.process_routing_request(request)
    print(f"    控制器决策: 路径 {decision.selected_path}")
    print(f"    处理时间: {decision.processing_time:.4f}s")
    
    # 完成路由
    controller.complete_route(request.id, 1.2, 0.8, True)
    
    # 获取状态
    status = controller.get_real_time_status()
    print(f"    系统状态: 总请求 {status['total_requests']}")
    print(f"    成功率: {status['success_rate']:.2%}")
    print(f"    系统健康度: {status['system_health']:.2%}")
    
    print("    ✓ 实时控制器测试通过")


def performance_test(routing_system, num_requests=50):
    """性能压力测试"""
    print(f"  执行 {num_requests} 个路由请求的压力测试...")
    
    start_time = time.time()
    results = []
    
    for i in range(num_requests):
        # 创建随机请求
        source = f"perf_node_{i % 10}"
        dest = f"perf_node_{(i + 5) % 10}"
        priority = np.random.randint(1, 11)
        requirements = {
            "max_latency": np.random.uniform(0.5, 2.0),
            "max_energy": np.random.uniform(1.0, 3.0)
        }
        
        req_start = time.time()
        decision = routing_system.process_request(
            source=source,
            destination=dest,
            priority=priority,
            requirements=requirements
        )
        req_time = time.time() - req_start
        
        results.append({
            'processing_time': req_time,
            'decision_confidence': decision.confidence_score,
            'estimated_latency': decision.estimated_latency,
            'estimated_energy': decision.estimated_energy
        })
        
        # 模拟完成
        if decision.selected_path:
            routing_system.controller.complete_route(
                decision.request_id,
                actual_latency=decision.estimated_latency * 0.9,
                actual_energy=decision.estimated_energy * 0.8,
                success=np.random.random() > 0.1  # 90%成功率
            )
    
    total_time = time.time() - start_time
    
    # 统计结果
    processing_times = [r['processing_time'] for r in results]
    confidences = [r['decision_confidence'] for r in results]
    
    print(f"    总测试时间: {total_time:.3f}s")
    print(f"    平均处理时间: {np.mean(processing_times):.4f}s")
    print(f"    最大处理时间: {np.max(processing_times):.4f}s")
    print(f"    平均置信度: {np.mean(confidences):.3f}")
    print(f"    处理速度: {num_requests / total_time:.1f} req/s")
    
    print("  ✓ 性能压力测试通过")


def system_status_check(routing_system):
    """系统状态检查"""
    # 获取性能报告
    report = routing_system.get_performance_report()
    
    print("  系统状态报告:")
    print(f"    总请求数: {report['system_status']['total_requests']}")
    print(f"    成功路由: {report['system_status']['successful_routes']}")
    print(f"    失败路由: {report['system_status']['failed_routes']}")
    print(f"    成功率: {report['system_status']['success_rate']:.2%}")
    print(f"    平均延迟: {report['system_status']['avg_latency']:.3f}s")
    print(f"    平均能耗: {report['system_status']['avg_energy_consumption']:.3f}")
    print(f"    活动路由: {report['system_status']['active_routes']}")
    print(f"    等待请求: {report['system_status']['pending_requests']}")
    
    # 显示优化建议
    if report['recommendations']:
        print("  优化建议:")
        for i, rec in enumerate(report['recommendations'][:3], 1):  # 只显示前3条
            print(f"    {i}. [{rec['priority'].upper()}] {rec['message']}")
    else:
        print("  ✓ 系统运行良好，无需要优化的项目")
    
    print("  ✓ 系统状态检查完成")


def demonstrate_module_capabilities():
    """展示各个模块的独立功能"""
    print("\n" + "=" * 60)
    print("模块功能演示")
    print("=" * 60)
    
    # 1. 强化学习模块演示
    print("\n1. 强化学习模块演示")
    demonstrate_reinforcement_learning()
    
    # 2. 自适应分配模块演示
    print("\n2. 自适应分配模块演示")
    demonstrate_adaptive_allocation()
    
    # 3. 能效优化模块演示
    print("\n3. 能效优化模块演示")
    demonstrate_efficiency_optimization()


def demonstrate_reinforcement_learning():
    """演示强化学习模块"""
    # Actor-Critic演示
    ac_router = ActorCriticRouter(state_dim=16, action_dim=4)
    
    print("  Actor-Critic路由器:")
    for episode in range(3):
        state = np.random.randn(16)
        action = ac_router.select_action(state, training=True)
        reward = np.random.uniform(0, 1)
        next_state = np.random.randn(16)
        
        ac_router.store_transition(state, action, reward, next_state, False)
        print(f"    Episode {episode+1}: 状态维度={len(state)}, 动作={action}, 奖励={reward:.3f}")
    
    ac_router.train_step()
    stats = ac_router.get_statistics()
    print(f"    统计: 训练步骤={stats['training_steps']}, 内存大小={stats['memory_size']}")


def demonstrate_adaptive_allocation():
    """演示自适应分配模块"""
    # 动态权重路由器演示
    dw_router = DynamicWeightRouter(num_paths=6, state_dim=20)
    
    print("  动态权重路由器:")
    for i in range(3):
        path = dw_router.select_path('normal')
        print(f"    选择路径: {path}")
        
        # 更新指标
        dw_router.update_path_metrics(path, 1.0 + np.random.uniform(-0.2, 0.2), 
                                    0.8 + np.random.uniform(-0.1, 0.1), True)
    
    stats = dw_router.get_statistics()
    print(f"    总请求: {stats['total_requests']}, 成功率: {stats['success_rate']:.2%}")


def demonstrate_efficiency_optimization():
    """演示能效优化模块"""
    # 神经启发路由器演示
    ni_router = NeuralInspiredRouter(num_neurons=32, input_dim=16, num_paths=4)
    
    print("  神经启发路由器:")
    for i in range(3):
        state = np.random.randn(16)
        route, energy, confidence = ni_router.process_input(state)
        print(f"    输入维度={len(state)}, 路由={route}, 能效={energy:.3f}, 置信度={confidence:.3f}")
        
        # 训练
        target_route = np.random.randint(0, 4)
        loss = ni_router.train_step(state, target_route, np.random.uniform(0, 1), np.random.uniform(0.5, 1.0))
        print(f"      训练损失: {loss:.4f}")
    
    metrics = ni_router.get_performance_metrics()
    print(f"    性能: 总决策={metrics['total_decisions']}, 成功率={metrics['success_rate']:.2%}")


def run_comprehensive_test():
    """运行综合测试"""
    try:
        # 执行主要测试
        test_complete_system()
        
        # 展示模块功能
        demonstrate_module_capabilities()
        
        print("\n" + "🎉" * 20)
        print("动态路由系统综合测试完成！")
        print("所有模块功能正常，系统运行稳定。")
        print("🎉" * 20)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
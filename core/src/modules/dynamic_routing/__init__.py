"""
动态路由系统
实现基于人工智能的智能路由决策系统
"""

import time

from .reinforcement_routing import (
    ActorCriticRouter, QLearningRouter, MultiAgentRouter, RoutingEnvironment
)

from .adaptive_allocation import (
    DynamicWeightRouter, PredictiveEarlyExit, AdaptiveLoadBalancer, AllocationController
)

from .efficiency_optimization import (
    NeuralInspiredRouter, IntelligentPathSelector
)

from .realtime_routing_controller import (
    RealTimeRoutingController, RoutingRequest, RoutingDecision
)

__version__ = "1.0.0"
__author__ = "Dynamic Routing System"

__all__ = [
    # 强化学习模块
    'ActorCriticRouter',
    'QLearningRouter', 
    'MultiAgentRouter',
    'RoutingEnvironment',
    
    # 自适应分配模块
    'DynamicWeightRouter',
    'PredictiveEarlyExit',
    'AdaptiveLoadBalancer',
    'AllocationController',
    
    # 能效优化模块
    'NeuralInspiredRouter',
    'IntelligentPathSelector',
    
    # 实时控制器
    'RealTimeRoutingController',
    'RoutingRequest',
    'RoutingDecision'
]


class DynamicRoutingSystem:
    """动态路由系统主接口"""
    
    def __init__(self, 
                 config: dict = None,
                 enable_reinforcement_learning: bool = True,
                 enable_adaptive_allocation: bool = True,
                 enable_efficiency_optimization: bool = True,
                 device: str = 'cpu'):
        
        """
        初始化动态路由系统
        
        Args:
            config: 配置字典
            enable_reinforcement_learning: 是否启用强化学习模块
            enable_adaptive_allocation: 是否启用自适应分配模块
            enable_efficiency_optimization: 是否启用能效优化模块
            device: 计算设备
        """
        
        self.controller = RealTimeRoutingController(
            config=config,
            enable_reinforcement_learning=enable_reinforcement_learning,
            enable_adaptive_allocation=enable_adaptive_allocation,
            enable_efficiency_optimization=enable_efficiency_optimization,
            device=device
        )
        
        # 系统状态
        self.is_running = False
        self.system_stats = {}
    
    def start(self):
        """启动路由系统"""
        if not self.is_running:
            self.controller.start_monitoring()
            self.is_running = True
            print("动态路由系统已启动")
    
    def stop(self):
        """停止路由系统"""
        if self.is_running:
            self.controller.stop_monitoring()
            self.is_running = False
            print("动态路由系统已停止")
    
    def process_request(self, source: str, destination: str, priority: int = 5, 
                       requirements: dict = None, constraints: dict = None) -> RoutingDecision:
        """
        处理路由请求
        
        Args:
            source: 源节点
            destination: 目标节点
            priority: 优先级（1-10）
            requirements: 需求参数
            constraints: 约束条件
            
        Returns:
            RoutingDecision: 路由决策结果
        """
        
        request = RoutingRequest(
            id=f"req_{int(time.time() * 1000000)}",
            source=source,
            destination=destination,
            priority=priority,
            requirements=requirements or {},
            constraints=constraints or {}
        )
        
        return self.controller.process_routing_request(request)
    
    def get_system_status(self) -> dict:
        """获取系统状态"""
        return self.controller.get_real_time_status()
    
    def get_performance_report(self) -> dict:
        """获取性能报告"""
        status = self.get_system_status()
        
        return {
            'system_status': status,
            'module_statistics': self._get_module_statistics(),
            'performance_metrics': self._get_performance_metrics(),
            'recommendations': self._get_optimization_recommendations()
        }
    
    def _get_module_statistics(self) -> dict:
        """获取模块统计信息"""
        stats = {}
        
        # 强化学习模块统计
        if hasattr(self.controller, 'actor_critic_router'):
            stats['reinforcement_learning'] = {
                'actor_critic': self.controller.actor_critic_router.get_statistics(),
                'q_learning': self.controller.q_learning_router.get_statistics() if hasattr(self.controller, 'q_learning_router') else {},
                'multi_agent': self.controller.multi_agent_router.get_statistics() if hasattr(self.controller, 'multi_agent_router') else {}
            }
        
        # 自适应分配模块统计
        if hasattr(self.controller, 'dynamic_weight_router'):
            stats['adaptive_allocation'] = {
                'dynamic_weight_routing': self.controller.dynamic_weight_router.get_statistics(),
                'predictive_early_exit': self.controller.predictive_early_exit.get_early_exit_statistics(),
                'load_balancer': self.controller.load_balancer.get_load_balancing_stats(),
                'allocation_controller': self.controller.allocation_controller.get_statistics()
            }
        
        # 能效优化模块统计
        if hasattr(self.controller, 'neural_inspired_router'):
            stats['efficiency_optimization'] = {
                'neural_inspired_routing': self.controller.neural_inspired_router.get_performance_metrics(),
                'intelligent_path_selector': self.controller.intelligent_path_selector.get_performance_metrics()
            }
        
        return stats
    
    def _get_performance_metrics(self) -> dict:
        """获取性能指标"""
        return {
            'throughput': self.controller.successful_routes / max(self.controller.total_requests, 1),
            'latency': self.controller.avg_latency,
            'energy_efficiency': 1.0 / (1.0 + self.controller.avg_energy_consumption),
            'reliability': self.controller.successful_routes / max(self.controller.total_requests, 1),
            'resource_utilization': len(self.controller.active_routes) / self.controller.max_concurrent_routes
        }
    
    def _get_optimization_recommendations(self) -> list:
        """获取优化建议"""
        recommendations = []
        
        # 基于系统状态的建议
        status = self.controller.get_real_time_status()
        
        if status['success_rate'] < 0.9:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'message': f"成功率较低 ({status['success_rate']:.2%})，建议检查路由算法参数"
            })
        
        if status['avg_latency'] > 2.0:
            recommendations.append({
                'type': 'latency',
                'priority': 'medium',
                'message': f"平均延迟较高 ({status['avg_latency']:.2f}s)，建议优化路径选择算法"
            })
        
        if status['avg_energy_consumption'] > 1.5:
            recommendations.append({
                'type': 'energy',
                'priority': 'medium',
                'message': f"能耗较高 ({status['avg_energy_consumption']:.2f})，建议启用能效优化模块"
            })
        
        if len(self.controller.active_routes) / self.controller.max_concurrent_routes > 0.8:
            recommendations.append({
                'type': 'capacity',
                'priority': 'high',
                'message': "系统负载较高，建议扩容或优化负载均衡策略"
            })
        
        return recommendations
    
    def save_state(self, filepath: str = "dynamic_routing_state.pt"):
        """保存系统状态"""
        self.controller.save_controller_state(filepath)
        print(f"系统状态已保存到 {filepath}")
    
    def load_state(self, filepath: str = "dynamic_routing_state.pt"):
        """加载系统状态"""
        self.controller.load_controller_state(filepath)
        print(f"系统状态已从 {filepath} 加载")
    
    def configure(self, config: dict):
        """配置系统参数"""
        self.controller.config.update(config)
        print("系统配置已更新")
    
    def shutdown(self):
        """关闭系统"""
        self.stop()
        self.controller.shutdown()
        print("动态路由系统已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()


# 示例使用代码
if __name__ == "__main__":
    import time
    
    # 创建动态路由系统
    with DynamicRoutingSystem() as routing_system:
        print("开始动态路由系统演示...")
        
        # 模拟多个路由请求
        requests = [
            ("node_A", "node_B", 8, {"max_latency": 1.0, "min_reliability": 0.9}),
            ("node_C", "node_D", 5, {"max_energy": 1.5}),
            ("node_E", "node_F", 9, {"min_bandwidth": 100}),
        ]
        
        results = []
        for source, destination, priority, requirements in requests:
            result = routing_system.process_request(
                source=source,
                destination=destination,
                priority=priority,
                requirements=requirements
            )
            results.append(result)
            print(f"请求 {source} -> {destination}: 路径 {result.selected_path}, "
                  f"延迟 {result.estimated_latency:.2f}s, "
                  f"能耗 {result.estimated_energy:.2f}")
            
            # 模拟路由完成
            time.sleep(0.1)
            routing_system.controller.complete_route(
                result.request_id,
                actual_latency=result.estimated_latency * 0.9,
                actual_energy=result.estimated_energy * 0.8,
                success=True
            )
        
        # 获取性能报告
        report = routing_system.get_performance_report()
        print(f"\n系统性能报告:")
        print(f"总请求数: {report['system_status']['total_requests']}")
        print(f"成功率: {report['system_status']['success_rate']:.2%}")
        print(f"平均延迟: {report['system_status']['avg_latency']:.3f}s")
        print(f"平均能耗: {report['system_status']['avg_energy_consumption']:.3f}")
        
        # 显示优化建议
        if report['recommendations']:
            print(f"\n优化建议:")
            for rec in report['recommendations']:
                print(f"- [{rec['priority'].upper()}] {rec['message']}")
        
        # 保存系统状态
        routing_system.save_state()
        print("\n系统演示完成!")
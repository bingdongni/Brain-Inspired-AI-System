"""
实时动态路由控制器
整合所有路由模块的统一控制器
"""

import numpy as np
import torch
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading
import json


# 导入所有路由模块
try:
    from .reinforcement_routing import (
        ActorCriticRouter, QLearningRouter, MultiAgentRouter, RoutingEnvironment
    )
    from .adaptive_allocation import (
        DynamicWeightRouter, PredictiveEarlyExit, AdaptiveLoadBalancer, AllocationController
    )
    from .efficiency_optimization import (
        NeuralInspiredRouter, IntelligentPathSelector
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    IMPORTS_AVAILABLE = False
    
    # 创建占位符类
    class ActorCriticRouter:
        def __init__(self, *args, **kwargs): pass
        def select_action(self, *args, **kwargs): return 0
        def get_statistics(self, *args, **kwargs): return {}
    
    class QLearningRouter:
        def __init__(self, *args, **kwargs): pass
        def select_action(self, *args, **kwargs): return 0
        def get_statistics(self, *args, **kwargs): return {}
    
    class MultiAgentRouter:
        def __init__(self, *args, **kwargs): pass
        def get_collaborative_decision(self, *args, **kwargs): return 0
        def get_statistics(self, *args, **kwargs): return {}
    
    class RoutingEnvironment:
        def __init__(self, *args, **kwargs): pass
    
    class DynamicWeightRouter:
        def __init__(self, *args, **kwargs): pass
        def select_path(self, *args, **kwargs): return 0
        def update_path_metrics(self, *args, **kwargs): pass
        def get_statistics(self, *args, **kwargs): return {}
    
    class PredictiveEarlyExit:
        def __init__(self, *args, **kwargs): pass
        def should_early_exit(self, *args, **kwargs): return (False, None)
        def get_early_exit_statistics(self, *args, **kwargs): return {}
    
    class AdaptiveLoadBalancer:
        def __init__(self, *args, **kwargs): pass
        def select_node(self, *args, **kwargs): return 0
        def get_load_balancing_stats(self, *args, **kwargs): return {}
    
    class AllocationController:
        def __init__(self, *args, **kwargs): pass
        def get_statistics(self, *args, **kwargs): return {}
    
    class NeuralInspiredRouter:
        def __init__(self, *args, **kwargs): pass
        def process_input(self, *args, **kwargs): return (0, 0.5, 0.5)
        def get_performance_metrics(self, *args, **kwargs): return {}
    
    class IntelligentPathSelector:
        def __init__(self, *args, **kwargs): pass
        def find_optimal_path(self, *args, **kwargs): return {'error': 'Not implemented'}


@dataclass
class RoutingRequest:
    """路由请求"""
    id: str
    source: str
    destination: str
    priority: int = 5  # 1-10, 10为最高优先级
    requirements: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    callback: Optional[Callable] = None


@dataclass
class RoutingDecision:
    """路由决策"""
    request_id: str
    selected_path: List[str]
    selected_modules: List[str]
    estimated_latency: float
    estimated_energy: float
    confidence_score: float
    processing_time: float
    alternative_paths: List[List[str]] = field(default_factory=list)
    decision_metadata: Dict = field(default_factory=dict)


class RealTimeRoutingController:
    """实时动态路由控制器"""
    
    def __init__(self, 
                 config: Dict = None,
                 enable_reinforcement_learning: bool = True,
                 enable_adaptive_allocation: bool = True,
                 enable_efficiency_optimization: bool = True,
                 device: str = 'cpu'):
        
        self.device = device
        self.enable_rl = enable_reinforcement_learning
        self.enable_adaptive = enable_adaptive_allocation
        self.enable_efficiency = enable_efficiency_optimization
        
        # 配置
        self.config = config or self._get_default_config()
        
        # 初始化各个路由模块
        self._initialize_routing_modules()
        
        # 路由决策引擎
        self.decision_engine = self._create_decision_engine()
        
        # 请求管理
        self.pending_requests = []
        self.active_routes = {}
        self.completed_routes = []
        
        # 统计信息
        self.total_requests = 0
        self.successful_routes = 0
        self.failed_routes = 0
        self.total_processing_time = 0.0
        self.avg_latency = 0.0
        self.avg_energy_consumption = 0.0
        
        # 性能监控
        self.performance_history = deque(maxlen=10000)
        self.decision_history = deque(maxlen=5000)
        
        # 实时监控
        self.monitoring_active = False
        self.monitor_thread = None
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # 模块权重（决策时的重要性）
        self.module_weights = {
            'reinforcement_learning': 0.3,
            'adaptive_allocation': 0.3,
            'efficiency_optimization': 0.4
        }
        
        # 负载均衡
        self.load_balancing_enabled = True
        self.max_concurrent_routes = 100
        
        # 异常处理
        self.fallback_enabled = True
        self.circuit_breaker_threshold = 0.8
        
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'reinforcement_learning': {
                'actor_critic': {
                    'state_dim': 32,
                    'action_dim': 8,
                    'learning_rate': 1e-3,
                    'gamma': 0.99
                },
                'q_learning': {
                    'state_dim': 32,
                    'action_dim': 8,
                    'learning_rate': 1e-3,
                    'gamma': 0.99,
                    'use_deep_q': True
                },
                'multi_agent': {
                    'num_agents': 4,
                    'state_dim': 32,
                    'action_dim': 8,
                    'learning_rate': 1e-3
                },
                'routing_environment': {
                    'num_modules': 8,
                    'state_dim': 32,
                    'max_steps': 1000
                }
            },
            'adaptive_allocation': {
                'dynamic_weight_routing': {
                    'num_paths': 8,
                    'state_dim': 32,
                    'learning_rate': 1e-3
                },
                'predictive_early_exit': {
                    'num_modules': 8,
                    'state_dim': 32,
                    'confidence_threshold': 0.85
                },
                'load_balancer': {
                    'num_nodes': 8,
                    'state_dim': 64,
                    'balancing_strategy': 'adaptive'
                },
                'allocation_controller': {
                    'resource_dim': 32,
                    'max_concurrent_allocations': 100
                }
            },
            'efficiency_optimization': {
                'neural_inspired_routing': {
                    'num_neurons': 64,
                    'input_dim': 32,
                    'num_paths': 8,
                    'learning_rate': 1e-3
                },
                'intelligent_path_selector': {
                    'num_nodes': 20,
                    'num_objectives': 5,
                    'input_dim': 32,
                    'learning_rate': 1e-3
                }
            }
        }
    
    def _initialize_routing_modules(self):
        """初始化路由模块"""
        # 强化学习模块
        if self.enable_rl:
            rl_config = self.config['reinforcement_learning']
            
            self.actor_critic_router = ActorCriticRouter(
                **rl_config['actor_critic'],
                device=self.device
            )
            
            self.q_learning_router = QLearningRouter(
                **rl_config['q_learning'],
                device=self.device
            )
            
            self.multi_agent_router = MultiAgentRouter(
                **rl_config['multi_agent'],
                device=self.device
            )
            
            self.routing_environment = RoutingEnvironment(
                **rl_config['routing_environment']
            )
        
        # 自适应分配模块
        if self.enable_adaptive:
            adaptive_config = self.config['adaptive_allocation']
            
            self.dynamic_weight_router = DynamicWeightRouter(
                **adaptive_config['dynamic_weight_routing'],
                device=self.device
            )
            
            self.predictive_early_exit = PredictiveEarlyExit(
                **adaptive_config['predictive_early_exit'],
                device=self.device
            )
            
            self.load_balancer = AdaptiveLoadBalancer(
                **adaptive_config['load_balancer'],
                device=self.device
            )
            
            self.allocation_controller = AllocationController(
                **adaptive_config['allocation_controller']
            )
        
        # 能效优化模块
        if self.enable_efficiency:
            efficiency_config = self.config['efficiency_optimization']
            
            self.neural_inspired_router = NeuralInspiredRouter(
                **efficiency_config['neural_inspired_routing'],
                device=self.device
            )
            
            self.intelligent_path_selector = IntelligentPathSelector(
                **efficiency_config['intelligent_path_selector'],
                device=self.device
            )
    
    def _create_decision_engine(self) -> 'DecisionEngine':
        """创建决策引擎"""
        return DecisionEngine(self)
    
    def process_routing_request(self, request: RoutingRequest) -> RoutingDecision:
        """处理路由请求"""
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # 检查并发限制
            if len(self.active_routes) >= self.max_concurrent_routes:
                return self._create_queue_result(request)
            
            # 生成路由决策
            decision = self.decision_engine.make_decision(request)
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            # 记录决策
            self._record_decision(request, decision, processing_time, True)
            
            # 添加到活动路由
            self.active_routes[request.id] = {
                'request': request,
                'decision': decision,
                'start_time': start_time
            }
            
            self.successful_routes += 1
            return decision
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # 记录失败决策
            self._record_decision(request, None, processing_time, False, str(e))
            
            # 异常处理
            if self.fallback_enabled:
                return self._create_fallback_decision(request, str(e))
            else:
                self.failed_routes += 1
                raise e
    
    def _create_queue_result(self, request: RoutingRequest) -> RoutingDecision:
        """创建排队结果"""
        return RoutingDecision(
            request_id=request.id,
            selected_path=[],
            selected_modules=[],
            estimated_latency=-1,
            estimated_energy=-1,
            confidence_score=0.0,
            processing_time=0.0,
            decision_metadata={'status': 'queued'}
        )
    
    def _create_fallback_decision(self, request: RoutingRequest, error: str) -> RoutingDecision:
        """创建备用决策"""
        # 简单的备用策略：基于最短路径或负载均衡
        simple_path = ['node_1', 'node_2', 'node_3']  # 简化的备用路径
        simple_modules = ['module_1', 'module_2']
        
        return RoutingDecision(
            request_id=request.id,
            selected_path=simple_path,
            selected_modules=simple_modules,
            estimated_latency=1.0,
            estimated_energy=1.0,
            confidence_score=0.5,
            processing_time=0.1,
            decision_metadata={'status': 'fallback', 'error': error}
        )
    
    def _record_decision(self, 
                        request: RoutingRequest, 
                        decision: Optional[RoutingDecision], 
                        processing_time: float, 
                        success: bool, 
                        error: str = None):
        """记录路由决策"""
        decision_entry = {
            'request_id': request.id,
            'timestamp': time.time(),
            'processing_time': processing_time,
            'success': success,
            'priority': request.priority,
            'source': request.source,
            'destination': request.destination,
            'error': error
        }
        
        if decision:
            decision_entry.update({
                'selected_path': decision.selected_path,
                'selected_modules': decision.selected_modules,
                'estimated_latency': decision.estimated_latency,
                'estimated_energy': decision.estimated_energy,
                'confidence_score': decision.confidence_score
            })
        
        self.decision_history.append(decision_entry)
        
        # 更新性能统计
        if decision and success:
            # 更新平均延迟和能耗
            n = self.successful_routes
            self.avg_latency = (self.avg_latency * (n - 1) + decision.estimated_latency) / n
            self.avg_energy_consumption = (self.avg_energy_consumption * (n - 1) + 
                                         decision.estimated_energy) / n
    
    def complete_route(self, request_id: str, actual_latency: float, 
                      actual_energy: float, success: bool):
        """完成路由"""
        if request_id in self.active_routes:
            route_info = self.active_routes[request_id]
            decision = route_info['decision']
            
            # 更新所有相关模块
            self._update_modules_with_feedback(decision, actual_latency, actual_energy, success)
            
            # 移动到完成记录
            route_info['completion_time'] = time.time()
            route_info['actual_latency'] = actual_latency
            route_info['actual_energy'] = actual_energy
            route_info['success'] = success
            
            self.completed_routes.append(route_info)
            del self.active_routes[request_id]
            
            # 处理等待队列
            self._process_pending_queue()
    
    def _update_modules_with_feedback(self, 
                                    decision: RoutingDecision, 
                                    actual_latency: float, 
                                    actual_energy: float, 
                                    success: bool):
        """向所有模块提供反馈"""
        # 更新强化学习模块
        if self.enable_rl:
            # 这里简化处理，实际中需要更复杂的反馈机制
            pass
        
        # 更新自适应分配模块
        if self.enable_adaptive:
            # 更新路径指标
            if hasattr(self, 'dynamic_weight_router') and decision.selected_path:
                for i, module_id in enumerate(decision.selected_modules):
                    if 'path_' in module_id:
                        path_idx = int(module_id.split('_')[1])
                        self.dynamic_weight_router.update_path_metrics(
                            path_idx, actual_energy, actual_latency, success
                        )
        
        # 更新能效优化模块
        if self.enable_efficiency:
            if hasattr(self, 'neural_inspired_router') and decision.selected_path:
                # 简化更新逻辑
                pass
    
    def _process_pending_queue(self):
        """处理等待队列"""
        if not self.pending_requests:
            return
        
        # 按优先级排序
        self.pending_requests.sort(key=lambda x: -x.priority)
        
        # 处理等待的请求
        while self.pending_requests and len(self.active_routes) < self.max_concurrent_routes:
            request = self.pending_requests.pop(0)
            self.process_routing_request(request)
    
    def _start_monitoring(self):
        """启动实时监控"""
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self._update_monitoring_data()
                    time.sleep(1.0)  # 每秒更新一次
                except Exception as e:
                    print(f"Monitoring error: {e}")
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _update_monitoring_data(self):
        """更新监控数据"""
        self.performance_history.append({
            'timestamp': time.time(),
            'total_requests': self.total_requests,
            'active_routes': len(self.active_routes),
            'pending_requests': len(self.pending_requests),
            'success_rate': self.successful_routes / max(self.total_requests, 1),
            'avg_latency': self.avg_latency,
            'avg_energy': self.avg_energy_consumption,
            'total_processing_time': self.total_processing_time
        })
    
    def get_real_time_status(self) -> Dict:
        """获取实时状态"""
        return {
            'timestamp': time.time(),
            'total_requests': self.total_requests,
            'successful_routes': self.successful_routes,
            'failed_routes': self.failed_routes,
            'active_routes': len(self.active_routes),
            'pending_requests': len(self.pending_requests),
            'success_rate': self.successful_routes / max(self.total_requests, 1),
            'avg_latency': self.avg_latency,
            'avg_energy_consumption': self.avg_energy_consumption,
            'avg_processing_time': self.total_processing_time / max(self.total_requests, 1),
            'module_weights': self.module_weights.copy(),
            'system_health': self._calculate_system_health()
        }
    
    def _calculate_system_health(self) -> float:
        """计算系统健康度"""
        # 基于多个指标计算系统健康度
        success_rate = self.successful_routes / max(self.total_requests, 1)
        load_factor = len(self.active_routes) / self.max_concurrent_routes
        
        # 健康度评分
        health_score = success_rate * (1.0 - min(1.0, load_factor))
        
        return max(0.0, min(1.0, health_score))
    
    def adjust_module_weights(self, performance_metrics: Dict[str, float]):
        """调整模块权重"""
        # 基于性能指标调整权重
        rl_performance = performance_metrics.get('reinforcement_learning', 0.5)
        adaptive_performance = performance_metrics.get('adaptive_allocation', 0.5)
        efficiency_performance = performance_metrics.get('efficiency_optimization', 0.5)
        
        # 归一化性能并更新权重
        total_performance = rl_performance + adaptive_performance + efficiency_performance
        
        if total_performance > 0:
            self.module_weights['reinforcement_learning'] = rl_performance / total_performance
            self.module_weights['adaptive_allocation'] = adaptive_performance / total_performance
            self.module_weights['efficiency_optimization'] = efficiency_performance / total_performance
    
    def save_controller_state(self, filepath: str):
        """保存控制器状态"""
        state = {
            'config': self.config,
            'module_weights': self.module_weights,
            'statistics': self.get_real_time_status(),
            'performance_history': list(self.performance_history)[-1000:],
            'decision_history': list(self.decision_history)[-1000:]
        }
        
        # 保存各个模块
        if self.enable_rl:
            rl_state = {}
            if hasattr(self, 'actor_critic_router'):
                rl_state['actor_critic'] = 'saved_separately'
            state['reinforcement_learning_state'] = rl_state
        
        if self.enable_adaptive:
            adaptive_state = {}
            state['adaptive_allocation_state'] = adaptive_state
        
        if self.enable_efficiency:
            efficiency_state = {}
            state['efficiency_optimization_state'] = efficiency_state
        
        torch.save(state, filepath)
    
    def load_controller_state(self, filepath: str):
        """加载控制器状态"""
        state = torch.load(filepath)
        
        self.config = state['config']
        self.module_weights = state['module_weights']
        
        # 加载各模块状态
        if 'reinforcement_learning_state' in state:
            # 加载强化学习模块
            pass
        
        if 'adaptive_allocation_state' in state:
            # 加载自适应分配模块
            pass
        
        if 'efficiency_optimization_state' in state:
            # 加载能效优化模块
            pass
    
    def start_monitoring(self):
        """启动监控系统"""
        if not self.monitoring_active:
            self._start_monitoring()
    
    def stop_monitoring(self):
        """停止监控系统"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join()
    
    def shutdown(self):
        """关闭控制器"""
        self.stop_monitoring()
        self.executor.shutdown(wait=True)
        
        # 保存状态
        self.save_controller_state('controller_state.pt')


class DecisionEngine:
    """路由决策引擎"""
    
    def __init__(self, controller: RealTimeRoutingController):
        self.controller = controller
    
    def make_decision(self, request: RoutingRequest) -> RoutingDecision:
        """制定路由决策"""
        start_time = time.time()
        
        # 收集各个模块的建议
        module_suggestions = self._collect_module_suggestions(request)
        
        # 综合决策
        final_decision = self._integrate_decisions(request, module_suggestions)
        
        decision_time = time.time() - start_time
        
        return RoutingDecision(
            request_id=request.id,
            selected_path=final_decision['path'],
            selected_modules=final_decision['modules'],
            estimated_latency=final_decision['latency'],
            estimated_energy=final_decision['energy'],
            confidence_score=final_decision['confidence'],
            alternative_paths=final_decision.get('alternatives', []),
            processing_time=decision_time,
            decision_metadata=final_decision.get('metadata', {})
        )
    
    def _collect_module_suggestions(self, request: RoutingRequest) -> Dict:
        """收集各模块建议"""
        suggestions = {}
        
        # 强化学习建议
        if self.controller.enable_rl:
            suggestions['reinforcement_learning'] = self._get_rl_suggestions(request)
        
        # 自适应分配建议
        if self.controller.enable_adaptive:
            suggestions['adaptive_allocation'] = self._get_adaptive_suggestions(request)
        
        # 能效优化建议
        if self.controller.enable_efficiency:
            suggestions['efficiency_optimization'] = self._get_efficiency_suggestions(request)
        
        return suggestions
    
    def _get_rl_suggestions(self, request: RoutingRequest) -> Dict:
        """获取强化学习模块建议"""
        suggestions = []
        
        # Actor-Critic建议
        if hasattr(self.controller, 'actor_critic_router'):
            state = self._request_to_state(request)
            action = self.controller.actor_critic_router.select_action(state)
            suggestions.append({
                'module': 'actor_critic',
                'action': action,
                'confidence': 0.8  # 简化
            })
        
        # Q-Learning建议
        if hasattr(self.controller, 'q_learning_router'):
            state = self._request_to_state(request)
            action = self.controller.q_learning_router.select_action(state)
            suggestions.append({
                'module': 'q_learning',
                'action': action,
                'confidence': 0.75
            })
        
        # Multi-Agent建议
        if hasattr(self.controller, 'multi_agent_router'):
            state = self._request_to_state(request)
            action = self.controller.multi_agent_router.get_collaborative_decision(state)
            suggestions.append({
                'module': 'multi_agent',
                'action': action,
                'confidence': 0.85
            })
        
        return {'suggestions': suggestions}
    
    def _get_adaptive_suggestions(self, request: RoutingRequest) -> Dict:
        """获取自适应分配模块建议"""
        suggestions = []
        
        # Dynamic Weight Routing建议
        if hasattr(self.controller, 'dynamic_weight_router'):
            path_idx = self.controller.dynamic_weight_router.select_path()
            suggestions.append({
                'module': 'dynamic_weight',
                'path': path_idx,
                'confidence': 0.7
            })
        
        # Load Balancer建议
        if hasattr(self.controller, 'load_balancer'):
            node_idx = self.controller.load_balancer.select_node()
            suggestions.append({
                'module': 'load_balancer',
                'node': node_idx,
                'confidence': 0.8
            })
        
        return {'suggestions': suggestions}
    
    def _get_efficiency_suggestions(self, request: RoutingRequest) -> Dict:
        """获取能效优化模块建议"""
        suggestions = []
        
        # Neural Inspired Routing建议
        if hasattr(self.controller, 'neural_inspired_router'):
            state = self._request_to_state(request)
            route, energy_score, confidence = self.controller.neural_inspired_router.process_input(state)
            suggestions.append({
                'module': 'neural_inspired',
                'route': route,
                'energy_score': energy_score,
                'confidence': confidence
            })
        
        # Intelligent Path Selector建议
        if hasattr(self.controller, 'intelligent_path_selector'):
            path_result = self.controller.intelligent_path_selector.find_optimal_path(
                request.source, request.destination, request.requirements
            )
            if 'selected_path' in path_result:
                suggestions.append({
                    'module': 'intelligent_path',
                    'path': path_result['selected_path'],
                    'score': path_result['total_score'],
                    'confidence': path_result['total_score']
                })
        
        return {'suggestions': suggestions}
    
    def _request_to_state(self, request: RoutingRequest) -> np.ndarray:
        """将请求转换为状态向量"""
        # 简化的状态表示
        state = np.zeros(32)
        
        # 源和目标节点编码（简化）
        source_hash = hash(request.source) % 1000 / 1000.0
        target_hash = hash(request.destination) % 1000 / 1000.0
        state[0] = source_hash
        state[1] = target_hash
        state[2] = request.priority / 10.0
        
        # 其他维度填充随机值
        for i in range(3, min(32, len(state))):
            state[i] = np.random.random()
        
        return state
    
    def _integrate_decisions(self, request: RoutingRequest, suggestions: Dict) -> Dict:
        """整合决策"""
        # 简化的决策整合逻辑
        best_path = []
        best_modules = []
        best_latency = 1.0
        best_energy = 1.0
        best_confidence = 0.0
        alternatives = []
        
        # 从各个模块收集决策
        module_decisions = []
        
        for module_name, module_suggestions in suggestions.items():
            weight = self.controller.module_weights.get(module_name, 0.33)
            
            for suggestion in module_suggestions.get('suggestions', []):
                module_decisions.append({
                    'module': suggestion['module'],
                    'weight': weight,
                    'suggestion': suggestion,
                    'confidence': suggestion.get('confidence', 0.5)
                })
        
        # 选择最佳决策（简化：选择置信度最高的）
        if module_decisions:
            best_decision = max(module_decisions, key=lambda x: x['confidence'])
            
            # 构建路径
            if 'path' in best_decision['suggestion']:
                best_path = best_decision['suggestion']['path']
                best_modules = [f'module_{i}' for i in range(len(best_path))]
            elif 'action' in best_decision['suggestion']:
                action = best_decision['suggestion']['action']
                best_path = [f'node_{i}' for i in range(3)]
                best_modules = [f'module_{action}']
            else:
                best_path = ['node_1', 'node_2', 'node_3']
                best_modules = ['module_1']
            
            best_confidence = best_decision['confidence']
            best_latency = np.random.uniform(0.5, 2.0)  # 简化的延迟估计
            best_energy = np.random.uniform(0.8, 2.0)   # 简化的能耗估计
        
        return {
            'path': best_path,
            'modules': best_modules,
            'latency': best_latency,
            'energy': best_energy,
            'confidence': best_confidence,
            'alternatives': alternatives,
            'metadata': {'primary_module': best_decision['module'] if module_decisions else 'fallback'}
        }
"""
自适应计算分配控制器
统一管理所有分配策略的协调控制器
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class AllocationRequest:
    """分配请求"""
    id: str
    request_type: str  # 'compute', 'storage', 'network', 'ai_inference'
    priority: int  # 0-10, 10为最高优先级
    requirements: Dict[str, float]
    estimated_duration: float
    submission_time: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    callback: Optional[Callable] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class AllocationResult:
    """分配结果"""
    request_id: str
    allocated_resources: List[str]  # 分配的模块ID列表
    start_time: float
    estimated_completion: float
    confidence_score: float
    alternative_allocations: List[Dict] = field(default_factory=list)


class AllocationController:
    """自适应计算分配控制器"""
    
    def __init__(self,
                 resource_dim: int = 32,
                 allocation_strategies: List[str] = None,
                 enable_monitoring: bool = True,
                 max_concurrent_allocations: int = 100):
        
        self.resource_dim = resource_dim
        self.max_concurrent_allocations = max_concurrent_allocations
        self.enable_monitoring = enable_monitoring
        
        # 默认分配策略
        if allocation_strategies is None:
            allocation_strategies = ['priority_based', 'load_aware', 'predictive', 'collaborative']
        
        self.allocation_strategies = allocation_strategies
        self.current_strategy_index = 0
        
        # 资源状态
        self.resources = self._initialize_resources()
        self.resource_status = {res_id: 'available' for res_id in self.resources.keys()}
        
        # 分配队列
        self.pending_requests = []
        self.active_allocations = {}
        self.completed_allocations = []
        
        # 统计信息
        self.total_requests = 0
        self.successful_allocations = 0
        self.failed_allocations = 0
        self.total_waiting_time = 0.0
        self.total_processing_time = 0.0
        
        # 性能监控
        self.performance_history = deque(maxlen=10000)
        self.strategy_performance = defaultdict(list)
        
        # 策略决策网络（用于学习最佳策略选择）
        self.strategy_network = self._create_strategy_network()
        
        # 监控系统
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 1.0  # 秒
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # 策略权重
        self.strategy_weights = {strategy: 1.0 / len(allocation_strategies) 
                               for strategy in allocation_strategies}
        
        if enable_monitoring:
            self._start_monitoring()
    
    def _initialize_resources(self) -> Dict[str, Dict]:
        """初始化资源"""
        resources = {}
        
        # 计算资源
        for i in range(4):
            resources[f'compute_node_{i}'] = {
                'type': 'compute',
                'capacity': np.random.uniform(80, 120),
                'current_load': 0.0,
                'efficiency': np.random.uniform(0.8, 0.95),
                'cost': np.random.uniform(0.1, 0.3),
                'latency': np.random.uniform(0.1, 0.5)
            }
        
        # 存储资源
        for i in range(3):
            resources[f'storage_node_{i}'] = {
                'type': 'storage',
                'capacity': np.random.uniform(500, 1000),  # GB
                'current_load': 0.0,
                'throughput': np.random.uniform(100, 500),  # MB/s
                'cost': np.random.uniform(0.05, 0.15),
                'latency': np.random.uniform(0.01, 0.1)
            }
        
        # 网络资源
        for i in range(2):
            resources[f'network_node_{i}'] = {
                'type': 'network',
                'capacity': np.random.uniform(1000, 2000),  # Mbps
                'current_load': 0.0,
                'reliability': np.random.uniform(0.95, 0.99),
                'cost': np.random.uniform(0.02, 0.08),
                'latency': np.random.uniform(0.001, 0.01)
            }
        
        # AI推理资源
        for i in range(3):
            resources[f'ai_node_{i}'] = {
                'type': 'ai_inference',
                'capacity': np.random.uniform(50, 100),  # FLOPS
                'current_load': 0.0,
                'accuracy': np.random.uniform(0.85, 0.95),
                'cost': np.random.uniform(0.2, 0.5),
                'latency': np.random.uniform(0.5, 2.0)
            }
        
        return resources
    
    def _create_strategy_network(self) -> nn.Module:
        """创建策略选择网络"""
        class StrategyNetwork(nn.Module):
            def __init__(self, input_dim: int, num_strategies: int):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, num_strategies)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = torch.relu(self.fc2(x))
                x = self.dropout(x)
                strategy_scores = torch.softmax(self.fc3(x), dim=-1)
                return strategy_scores
        
        return StrategyNetwork(self.resource_dim, len(self.allocation_strategies))
    
    def allocate_resources(self, request: AllocationRequest) -> AllocationResult:
        """分配资源"""
        self.total_requests += 1
        
        # 检查并发限制
        if len(self.active_allocations) >= self.max_concurrent_allocations:
            # 将请求加入等待队列
            self.pending_requests.append(request)
            return self._create_waiting_result(request)
        
        # 选择最佳分配策略
        strategy = self._select_strategy(request)
        
        # 执行分配
        start_time = time.time()
        try:
            allocated_resources = self._execute_allocation(request, strategy)
            allocation_time = time.time() - start_time
            
            if allocated_resources:
                result = AllocationResult(
                    request_id=request.id,
                    allocated_resources=allocated_resources,
                    start_time=start_time,
                    estimated_completion=start_time + request.estimated_duration,
                    confidence_score=self._calculate_confidence(allocated_resources, request),
                    alternative_allocations=self._generate_alternatives(request, allocated_resources)
                )
                
                # 记录成功分配
                self._record_successful_allocation(request, result, allocation_time)
                self.active_allocations[request.id] = {
                    'request': request,
                    'result': result,
                    'start_time': start_time
                }
                
                self.successful_allocations += 1
                return result
            else:
                raise Exception("No suitable resources available")
                
        except Exception as e:
            # 记录失败分配
            self._record_failed_allocation(request, str(e))
            self.failed_allocations += 1
            
            # 返回失败结果
            return AllocationResult(
                request_id=request.id,
                allocated_resources=[],
                start_time=start_time,
                estimated_completion=-1,
                confidence_score=0.0
            )
    
    def _select_strategy(self, request: AllocationRequest) -> str:
        """选择分配策略"""
        # 使用策略网络预测最佳策略
        state = self._get_resource_state_vector(request)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            strategy_scores = self.strategy_network(state_tensor).numpy().squeeze()
        
        # 结合策略权重和预测分数
        combined_scores = {}
        for i, strategy in enumerate(self.allocation_strategies):
            combined_score = strategy_scores[i] * self.strategy_weights[strategy]
            combined_scores[strategy] = combined_score
        
        # 选择评分最高的策略
        best_strategy = max(combined_scores.keys(), key=lambda k: combined_scores[k])
        return best_strategy
    
    def _execute_allocation(self, request: AllocationRequest, strategy: str) -> List[str]:
        """执行具体分配策略"""
        if strategy == 'priority_based':
            return self._priority_based_allocation(request)
        elif strategy == 'load_aware':
            return self._load_aware_allocation(request)
        elif strategy == 'predictive':
            return self._predictive_allocation(request)
        elif strategy == 'collaborative':
            return self._collaborative_allocation(request)
        else:
            # 默认使用负载感知分配
            return self._load_aware_allocation(request)
    
    def _priority_based_allocation(self, request: AllocationRequest) -> List[str]:
        """基于优先级的分配"""
        # 按优先级排序可用资源
        available_resources = [res_id for res_id, status in self.resource_status.items() 
                             if status == 'available']
        
        # 根据请求类型筛选合适的资源
        suitable_resources = []
        for res_id in available_resources:
            res_info = self.resources[res_id]
            if self._is_resource_suitable(res_info, request):
                suitable_resources.append(res_id)
        
        # 按效率/成本比排序（优先级高的请求优先选择）
        suitable_resources.sort(key=lambda res_id: (
            -request.priority,  # 优先级高的在前
            self.resources[res_id]['cost'] / self.resources[res_id].get('efficiency', 1.0)
        ))
        
        # 选择足够的资源
        allocated = []
        total_capacity = 0
        target_capacity = self._calculate_required_capacity(request)
        
        for res_id in suitable_resources:
            if total_capacity >= target_capacity:
                break
            
            res_info = self.resources[res_id]
            allocated.append(res_id)
            total_capacity += res_info.get('capacity', 0)
            
            # 更新资源状态
            self.resource_status[res_id] = 'allocated'
            self._update_resource_load(res_id, request)
        
        return allocated
    
    def _load_aware_allocation(self, request: AllocationRequest) -> List[str]:
        """负载感知分配"""
        # 计算各资源的负载评分
        resource_scores = {}
        for res_id, res_info in self.resources.items():
            if self.resource_status[res_id] == 'available':
                if self._is_resource_suitable(res_info, request):
                    # 负载越低，评分越高
                    load_score = 1.0 - (res_info['current_load'] / max(res_info['capacity'], 1))
                    efficiency_score = res_info.get('efficiency', 1.0)
                    cost_score = 1.0 / (1.0 + res_info['cost'])
                    
                    # 综合评分
                    total_score = (load_score * 0.5 + efficiency_score * 0.3 + cost_score * 0.2)
                    resource_scores[res_id] = total_score
        
        # 选择评分最高的资源
        sorted_resources = sorted(resource_scores.keys(), 
                                 key=lambda x: resource_scores[x], reverse=True)
        
        allocated = []
        total_capacity = 0
        target_capacity = self._calculate_required_capacity(request)
        
        for res_id in sorted_resources:
            if total_capacity >= target_capacity:
                break
            
            allocated.append(res_id)
            total_capacity += self.resources[res_id].get('capacity', 0)
            
            # 更新资源状态
            self.resource_status[res_id] = 'allocated'
            self._update_resource_load(res_id, request)
        
        return allocated
    
    def _predictive_allocation(self, request: AllocationRequest) -> List[str]:
        """预测性分配"""
        # 预测未来的负载情况
        predicted_loads = self._predict_future_loads()
        
        # 选择在预测时间窗口内负载最低的资源
        resource_predictions = {}
        for res_id, res_info in self.resources.items():
            if self.resource_status[res_id] == 'available':
                if self._is_resource_suitable(res_info, request):
                    # 预测负载越低，评分越高
                    predicted_load = predicted_loads.get(res_id, res_info['current_load'])
                    load_factor = predicted_load / max(res_info['capacity'], 1)
                    prediction_score = 1.0 - min(1.0, load_factor)
                    
                    efficiency_score = res_info.get('efficiency', 1.0)
                    reliability_score = res_info.get('reliability', 1.0)
                    
                    total_score = prediction_score * 0.5 + efficiency_score * 0.3 + reliability_score * 0.2
                    resource_predictions[res_id] = total_score
        
        # 选择预测评分最高的资源
        sorted_resources = sorted(resource_predictions.keys(), 
                                 key=lambda x: resource_predictions[x], reverse=True)
        
        allocated = []
        total_capacity = 0
        target_capacity = self._calculate_required_capacity(request)
        
        for res_id in sorted_resources:
            if total_capacity >= target_capacity:
                break
            
            allocated.append(res_id)
            total_capacity += self.resources[res_id].get('capacity', 0)
            
            # 更新资源状态
            self.resource_status[res_id] = 'allocated'
            self._update_resource_load(res_id, request)
        
        return allocated
    
    def _collaborative_allocation(self, request: AllocationRequest) -> List[str]:
        """协作式分配"""
        # 简化实现：结合多种策略的结果
        priority_result = self._priority_based_allocation(request)
        load_aware_result = self._load_aware_allocation(request)
        predictive_result = self._predictive_allocation(request)
        
        # 投票选择
        resource_votes = defaultdict(int)
        for result in [priority_result, load_aware_result, predictive_result]:
            for res_id in result:
                resource_votes[res_id] += 1
        
        # 选择得票最多的资源
        sorted_by_votes = sorted(resource_votes.keys(), 
                                key=lambda x: resource_votes[x], reverse=True)
        
        allocated = []
        total_capacity = 0
        target_capacity = self._calculate_required_capacity(request)
        
        for res_id in sorted_by_votes:
            if total_capacity >= target_capacity:
                break
            
            allocated.append(res_id)
            total_capacity += self.resources[res_id].get('capacity', 0)
            
            # 更新资源状态
            self.resource_status[res_id] = 'allocated'
            self._update_resource_load(res_id, request)
        
        return allocated
    
    def _is_resource_suitable(self, resource_info: Dict, request: AllocationRequest) -> bool:
        """判断资源是否适合请求"""
        res_type = resource_info['type']
        req_type = request.request_type
        
        # 类型匹配
        if res_type != req_type:
            return False
        
        # 容量检查
        required_capacity = self._calculate_required_capacity(request)
        if resource_info['capacity'] < required_capacity * 0.1:  # 至少需要10%的容量
            return False
        
        # 性能要求检查
        if 'min_efficiency' in request.requirements:
            if resource_info.get('efficiency', 1.0) < request.requirements['min_efficiency']:
                return False
        
        if 'max_latency' in request.requirements:
            if resource_info.get('latency', 0) > request.requirements['max_latency']:
                return False
        
        if 'min_accuracy' in request.requirements:
            if resource_info.get('accuracy', 1.0) < request.requirements['min_accuracy']:
                return False
        
        return True
    
    def _calculate_required_capacity(self, request: AllocationRequest) -> float:
        """计算所需容量"""
        base_capacity = request.requirements.get('capacity', 100.0)
        priority_multiplier = 1.0 + (request.priority / 10.0) * 0.5  # 高优先级需求更多资源
        return base_capacity * priority_multiplier
    
    def _calculate_confidence(self, allocated_resources: List[str], request: AllocationRequest) -> float:
        """计算分配置信度"""
        if not allocated_resources:
            return 0.0
        
        total_capacity = sum(self.resources[res_id].get('capacity', 0) 
                           for res_id in allocated_resources)
        required_capacity = self._calculate_required_capacity(request)
        
        capacity_ratio = min(1.0, total_capacity / required_capacity)
        
        # 计算资源质量分数
        quality_scores = []
        for res_id in allocated_resources:
            res_info = self.resources[res_id]
            efficiency = res_info.get('efficiency', 1.0)
            reliability = res_info.get('reliability', 1.0)
            quality_scores.append((efficiency + reliability) / 2)
        
        avg_quality = np.mean(quality_scores)
        
        # 综合置信度
        confidence = (capacity_ratio * 0.6 + avg_quality * 0.4)
        return min(1.0, confidence)
    
    def _generate_alternatives(self, request: AllocationRequest, 
                              allocated_resources: List[str]) -> List[Dict]:
        """生成替代分配方案"""
        alternatives = []
        
        # 生成不同的资源组合
        available_resources = [res_id for res_id, status in self.resource_status.items() 
                             if status == 'available']
        
        for i in range(min(3, len(available_resources))):
            if i < len(allocated_resources):
                # 替换一个资源
                alt_resources = allocated_resources.copy()
                alt_resources[i] = available_resources[i % len(available_resources)]
                
                alt_result = {
                    'resources': alt_resources,
                    'confidence': self._calculate_confidence(alt_resources, request) * 0.9,
                    'reason': f'Alternative configuration {i+1}'
                }
                alternatives.append(alt_result)
        
        return alternatives
    
    def _predict_future_loads(self, prediction_window: int = 10) -> Dict[str, float]:
        """预测未来负载"""
        predictions = {}
        
        # 简化的负载预测：基于历史趋势
        for res_id, res_info in self.resources.items():
            current_load = res_info['current_load']
            
            # 基于历史数据的简单预测
            if res_id in self.performance_history:
                recent_loads = [entry.get('load', current_load) 
                              for entry in list(self.performance_history)[-5:]]
                if recent_loads:
                    # 线性趋势预测
                    trend = np.polyfit(range(len(recent_loads)), recent_loads, 1)[0]
                    predicted_load = current_load + trend * prediction_window
                else:
                    predicted_load = current_load
            else:
                predicted_load = current_load
            
            predictions[res_id] = max(0, min(predicted_load, res_info['capacity'] * 1.2))
        
        return predictions
    
    def _update_resource_load(self, resource_id: str, request: AllocationRequest):
        """更新资源负载"""
        if resource_id in self.resources:
            load_increase = self._calculate_required_capacity(request) * 0.1
            self.resources[resource_id]['current_load'] = min(
                self.resources[resource_id]['current_load'] + load_increase,
                self.resources[resource_id]['capacity'] * 1.2
            )
    
    def _get_resource_state_vector(self, request: AllocationRequest) -> np.ndarray:
        """获取资源状态向量"""
        state = np.zeros(self.resource_dim)
        
        # 资源统计信息
        resource_counts = defaultdict(int)
        total_capacity = 0
        total_load = 0
        avg_efficiency = 0
        efficiency_sum = 0
        efficiency_count = 0
        
        for res_id, res_info in self.resources.items():
            resource_counts[res_info['type']] += 1
            total_capacity += res_info['capacity']
            total_load += res_info['current_load']
            
            if 'efficiency' in res_info:
                efficiency_sum += res_info['efficiency']
                efficiency_count += 1
        
        # 填充状态向量
        state[0] = len(self.resources)  # 总资源数
        state[1] = total_capacity
        state[2] = total_load
        state[3] = total_load / max(total_capacity, 1)  # 总体利用率
        state[4] = efficiency_sum / max(efficiency_count, 1)  # 平均效率
        state[5] = request.priority / 10.0  # 归一化优先级
        
        # 不同类型资源的数量
        type_start = 6
        for i, res_type in enumerate(['compute', 'storage', 'network', 'ai_inference']):
            if type_start + i < self.resource_dim:
                state[type_start + i] = resource_counts[res_type] / 10.0  # 归一化
        
        return state
    
    def _create_waiting_result(self, request: AllocationRequest) -> AllocationResult:
        """创建等待结果"""
        return AllocationResult(
            request_id=request.id,
            allocated_resources=[],
            start_time=-1,
            estimated_completion=-1,
            confidence_score=0.0,
            alternative_allocations=[]
        )
    
    def _record_successful_allocation(self, request: AllocationRequest, 
                                    result: AllocationResult, allocation_time: float):
        """记录成功分配"""
        self.performance_history.append({
            'request_id': request.id,
            'strategy': self._select_strategy(request),  # 简化：直接选择
            'success': True,
            'allocation_time': allocation_time,
            'confidence': result.confidence_score,
            'load': sum(self.resources[res_id]['current_load'] 
                       for res_id in result.allocated_resources)
        })
        
        self.total_processing_time += allocation_time
    
    def _record_failed_allocation(self, request: AllocationRequest, error: str):
        """记录失败分配"""
        self.performance_history.append({
            'request_id': request.id,
            'strategy': self._select_strategy(request),
            'success': False,
            'error': error
        })
    
    def release_resources(self, request_id: str):
        """释放资源"""
        if request_id in self.active_allocations:
            allocation_info = self.active_allocations[request_id]
            result = allocation_info['result']
            
            # 释放资源
            for res_id in result.allocated_resources:
                if res_id in self.resources:
                    # 减少负载
                    load_decrease = self.resources[res_id]['capacity'] * 0.1
                    self.resources[res_id]['current_load'] = max(
                        0, self.resources[res_id]['current_load'] - load_decrease
                    )
                
                # 更新状态
                if res_id in self.resource_status:
                    self.resource_status[res_id] = 'available'
            
            # 移动到已完成记录
            allocation_info['completion_time'] = time.time()
            self.completed_allocations.append(allocation_info)
            del self.active_allocations[request_id]
            
            # 处理等待队列
            self._process_pending_requests()
    
    def _process_pending_requests(self):
        """处理等待队列"""
        if not self.pending_requests:
            return
        
        # 按优先级排序
        self.pending_requests.sort(key=lambda x: -x.priority)
        
        # 处理等待的请求
        while self.pending_requests and len(self.active_allocations) < self.max_concurrent_allocations:
            request = self.pending_requests.pop(0)
            self.allocate_resources(request)
    
    def _start_monitoring(self):
        """启动监控系统"""
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self._update_monitoring_data()
                    time.sleep(self.monitor_interval)
                except Exception as e:
                    print(f"Monitoring error: {e}")
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _update_monitoring_data(self):
        """更新监控数据"""
        # 记录当前系统状态
        self.performance_history.append({
            'timestamp': time.time(),
            'total_resources': len(self.resources),
            'active_allocations': len(self.active_allocations),
            'pending_requests': len(self.pending_requests),
            'total_load': sum(res['current_load'] for res in self.resources.values()),
            'total_capacity': sum(res['capacity'] for res in self.resources.values())
        })
        
        # 清理过期数据
        if len(self.performance_history) > 10000:
            # 保留最新的8000条记录
            recent_history = list(self.performance_history)[-8000:]
            self.performance_history.clear()
            self.performance_history.extend(recent_history)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_requests': self.total_requests,
            'successful_allocations': self.successful_allocations,
            'failed_allocations': self.failed_allocations,
            'success_rate': self.successful_allocations / max(self.total_requests, 1),
            'active_allocations': len(self.active_allocations),
            'pending_requests': len(self.pending_requests),
            'avg_processing_time': self.total_processing_time / max(self.successful_allocations, 1),
            'resource_utilization': self._calculate_resource_utilization(),
            'strategy_weights': self.strategy_weights.copy(),
            'performance_history_size': len(self.performance_history)
        }
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """计算资源利用率"""
        utilization = {}
        for res_type in ['compute', 'storage', 'network', 'ai_inference']:
            type_resources = [(res_id, res_info) for res_id, res_info in self.resources.items() 
                            if res_info['type'] == res_type]
            
            if type_resources:
                total_capacity = sum(info['capacity'] for _, info in type_resources)
                total_load = sum(info['current_load'] for _, info in type_resources)
                utilization[res_type] = total_load / max(total_capacity, 1)
            else:
                utilization[res_type] = 0.0
        
        return utilization
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join()
    
    def shutdown(self):
        """关闭控制器"""
        self.stop_monitoring()
        self.executor.shutdown(wait=True)
    
    def save_state(self, filepath: str):
        """保存状态"""
        state = {
            'resources': self.resources,
            'resource_status': self.resource_status,
            'allocation_strategies': self.allocation_strategies,
            'strategy_weights': self.strategy_weights,
            'statistics': self.get_statistics(),
            'performance_history': list(self.performance_history)[-1000:]  # 保存最近1000条
        }
        
        torch.save(state, filepath)
    
    def load_state(self, filepath: str):
        """加载状态"""
        state = torch.load(filepath)
        self.resources = state['resources']
        self.resource_status = state['resource_status']
        self.allocation_strategies = state['allocation_strategies']
        self.strategy_weights = state['strategy_weights']
        self.performance_history = deque(state['performance_history'], maxlen=10000)
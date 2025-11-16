"""
负载均衡器
在多个处理单元间分配负载，优化系统性能
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque
import time


class LoadBalancer:
    """
    负载均衡器
    智能分配负载到不同的处理单元
    """
    
    def __init__(self, processing_units: List[str]):
        self.processing_units = processing_units
        self.unit_loads = {unit: 0.0 for unit in processing_units}
        self.unit_capacities = {unit: 1.0 for unit in processing_units}
        self.load_history = deque(maxlen=100)
        
        # 负载均衡策略
        self.strategies = {
            'round_robin': self._round_robin_select,
            'least_loaded': self._least_loaded_select,
            'weighted': self._weighted_select,
            'adaptive': self._adaptive_select
        }
        
        self.current_strategy = 'adaptive'
        self.round_robin_index = 0
        
        # 统计信息
        self.stats = {
            'total_assignments': 0,
            'strategy_usage': {strategy: 0 for strategy in self.strategies.keys()},
            'average_assignment_time': 0.0,
            'load_distribution_variance': 0.0
        }
    
    def select_processing_unit(
        self,
        operation: Any,
        strategy: str = None,
        operation_priority: int = 2,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> str:
        """
        选择处理单元
        
        Args:
            operation: 操作对象
            strategy: 选择策略
            operation_priority: 操作优先级
            custom_weights: 自定义权重
            
        Returns:
            选中的处理单元ID
        """
        start_time = time.time()
        strategy = strategy or self.current_strategy
        
        if strategy not in self.strategies:
            strategy = 'adaptive'
        
        # 应用自定义权重
        if custom_weights:
            for unit, weight in custom_weights.items():
                if unit in self.unit_capacities:
                    self.unit_capacities[unit] *= weight
        
        # 执行选择策略
        selected_unit = self.strategies[strategy](operation, operation_priority)
        
        # 更新负载
        load_increment = self._calculate_load_increment(operation_priority)
        self.unit_loads[selected_unit] += load_increment
        
        # 更新统计
        assignment_time = time.time() - start_time
        self._update_assignment_stats(strategy, assignment_time)
        
        # 记录选择历史
        self.load_history.append({
            'timestamp': time.time(),
            'selected_unit': selected_unit,
            'strategy': strategy,
            'priority': operation_priority,
            'assignment_time': assignment_time,
            'load_increment': load_increment
        })
        
        return selected_unit
    
    def update_unit_capacity(self, unit_id: str, new_capacity: float, 
                           adjustment_type: str = 'absolute'):
        """更新处理单元容量"""
        if unit_id not in self.unit_capacities:
            return
        
        if adjustment_type == 'absolute':
            self.unit_capacities[unit_id] = max(0.1, new_capacity)
        elif adjustment_type == 'relative':
            self.unit_capacities[unit_id] *= new_capacity
            self.unit_capacities[unit_id] = max(0.1, self.unit_capacities[unit_id])
    
    def get_load_balance_status(self) -> Dict:
        """获取负载均衡状态"""
        total_load = sum(self.unit_loads.values())
        total_capacity = sum(self.unit_capacities.values())
        
        utilization = {}
        for unit in self.processing_units:
            capacity = self.unit_capacities[unit]
            utilization[unit] = (self.unit_loads[unit] / capacity) if capacity > 0 else 0
        
        # 计算均衡分数
        if utilization:
            utilization_values = list(utilization.values())
            balance_score = self._calculate_balance_score(utilization_values)
        else:
            balance_score = 0.0
        
        return {
            'unit_loads': self.unit_loads.copy(),
            'unit_capacities': self.unit_capacities.copy(),
            'unit_utilization': utilization,
            'total_load': total_load,
            'total_capacity': total_capacity,
            'overall_utilization': total_load / total_capacity if total_capacity > 0 else 0,
            'balance_score': balance_score,
            'current_strategy': self.current_strategy,
            'statistics': self.stats.copy()
        }
    
    def optimize_load_distribution(self) -> Dict:
        """优化负载分布"""
        current_status = self.get_load_balance_status()
        
        optimization_actions = []
        utilization = current_status['unit_utilization']
        mean_utilization = current_status['overall_utilization']
        
        # 检测和生成优化建议
        for unit, unit_utilization in utilization.items():
            if unit_utilization > mean_utilization * 1.5:  # 负载过高
                optimization_actions.append({
                    'action': 'reduce_load',
                    'unit': unit,
                    'current_utilization': unit_utilization,
                    'target_utilization': mean_utilization,
                    'suggestion': 'redirect_operations_to_other_units',
                    'priority': 'high'
                })
            elif unit_utilization < mean_utilization * 0.5:  # 负载过低
                optimization_actions.append({
                    'action': 'increase_load',
                    'unit': unit,
                    'current_utilization': unit_utilization,
                    'target_utilization': mean_utilization,
                    'suggestion': 'assign_more_operations_to_this_unit',
                    'priority': 'low'
                })
        
        # 动态策略调整建议
        balance_score = current_status['balance_score']
        recommended_strategy = self._recommend_strategy(utilization, balance_score)
        
        return {
            'current_balance_score': balance_score,
            'optimization_actions': optimization_actions,
            'mean_utilization': mean_utilization,
            'recommended_strategy': recommended_strategy,
            'load_variance': current_status['statistics']['load_distribution_variance'],
            'optimization_timestamp': time.time()
        }
    
    def apply_load_optimization(self, optimization_actions: List[Dict]) -> Dict:
        """应用负载优化"""
        optimization_results = []
        
        for action in optimization_actions:
            unit = action['unit']
            current_action = action['action']
            
            if current_action == 'reduce_load':
                # 通过调整容量来减少负载分配
                reduction_factor = 0.8
                self.update_unit_capacity(unit, reduction_factor, 'relative')
                optimization_results.append({
                    'unit': unit,
                    'action': 'capacity_reduced',
                    'factor': reduction_factor
                })
                
            elif current_action == 'increase_load':
                # 通过增加容量来增加负载分配
                increase_factor = 1.2
                self.update_unit_capacity(unit, increase_factor, 'relative')
                optimization_results.append({
                    'unit': unit,
                    'action': 'capacity_increased',
                    'factor': increase_factor
                })
        
        return {
            'optimization_applied': len(optimization_results),
            'results': optimization_results,
            'new_balance_status': self.get_load_balance_status()
        }
    
    def _round_robin_select(self, operation: Any, priority: int) -> str:
        """轮询选择策略"""
        selected = self.processing_units[self.round_robin_index]
        self.round_robin_index = (self.round_robin_index + 1) % len(self.processing_units)
        return selected
    
    def _least_loaded_select(self, operation: Any, priority: int) -> str:
        """最少负载选择策略"""
        return min(self.processing_units, key=lambda unit: self.unit_loads[unit])
    
    def _weighted_select(self, operation: Any, priority: int) -> str:
        """加权选择策略"""
        weights = [self.unit_capacities[unit] for unit in self.processing_units]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return self.processing_units[0]
        
        # 按权重随机选择
        normalized_weights = [w / total_weight for w in weights]
        rand = np.random.random()
        
        cumulative = 0
        for i, unit in enumerate(self.processing_units):
            cumulative += normalized_weights[i]
            if rand <= cumulative:
                return unit
        
        return self.processing_units[-1]
    
    def _adaptive_select(self, operation: Any, priority: int) -> str:
        """自适应选择策略"""
        # 综合多个因素的智能选择
        
        scores = {}
        for unit in self.processing_units:
            # 负载分数 (负载越低分数越高)
            load_score = 1.0 - min(1.0, self.unit_loads[unit] / self.unit_capacities[unit])
            
            # 容量分数 (容量越大分数越高)
            max_capacity = max(self.unit_capacities.values())
            capacity_score = self.unit_capacities[unit] / max_capacity
            
            # 优先级权重
            priority_weight = self._get_priority_weight(priority)
            
            # 综合分数
            scores[unit] = (
                0.4 * load_score + 
                0.3 * capacity_score + 
                0.3 * priority_weight
            )
        
        return max(scores, key=scores.get)
    
    def _calculate_load_increment(self, priority: int) -> float:
        """计算负载增量"""
        # 基于优先级的负载增量
        base_increment = 1.0
        
        priority_multipliers = {
            0: 3.0,  # 最高优先级
            1: 2.0,  # 高优先级
            2: 1.0,  # 正常优先级
            3: 0.6,  # 低优先级
            4: 0.3   # 最低优先级
        }
        
        multiplier = priority_multipliers.get(priority, 1.0)
        return base_increment * multiplier
    
    def _get_priority_weight(self, priority: int) -> float:
        """获取优先级权重"""
        # 高优先级操作倾向于分配到更强大的单元
        if priority <= 1:
            return 1.0
        elif priority <= 2:
            return 0.8
        else:
            return 0.6
    
    def _calculate_balance_score(self, utilization_values: List[float]) -> float:
        """计算负载均衡分数"""
        if not utilization_values:
            return 0.0
        
        if len(utilization_values) <= 1:
            return 1.0
        
        # 计算标准差 (越小越均衡)
        mean_util = np.mean(utilization_values)
        std_util = np.std(utilization_values)
        
        # 转换为均衡分数 (1.0 = 完全均衡)
        # 归一化标准差
        normalized_std = std_util / max(mean_util, 0.1)
        balance_score = max(0.0, 1.0 - normalized_std)
        
        # 更新统计信息中的方差
        self.stats['load_distribution_variance'] = std_util
        
        return balance_score
    
    def _recommend_strategy(self, utilization: Dict[str, float], balance_score: float) -> str:
        """推荐负载均衡策略"""
        utilizations = list(utilization.values())
        max_utilization = max(utilizations) if utilizations else 0
        utilization_variance = np.var(utilizations) if len(utilizations) > 1 else 0
        
        # 基于当前状态推荐策略
        if balance_score > 0.8:
            # 已经很均衡，使用轮询保持
            return 'round_robin'
        elif max_utilization > 0.9:
            # 有单元过载，使用最少负载
            return 'least_loaded'
        elif utilization_variance < 0.1:
            # 差异很小，加权分配
            return 'weighted'
        else:
            # 一般情况使用自适应
            return 'adaptive'
    
    def _update_assignment_stats(self, strategy: str, assignment_time: float):
        """更新分配统计信息"""
        self.stats['total_assignments'] += 1
        self.stats['strategy_usage'][strategy] += 1
        
        # 更新平均分配时间
        current_avg = self.stats['average_assignment_time']
        total_assignments = self.stats['total_assignments']
        
        if total_assignments > 1:
            self.stats['average_assignment_time'] = (
                (current_avg * (total_assignments - 1) + assignment_time) / total_assignments
            )
        else:
            self.stats['average_assignment_time'] = assignment_time
    
    def reset_loads(self):
        """重置所有负载"""
        for unit in self.processing_units:
            self.unit_loads[unit] = 0.0
    
    def get_unit_performance(self, unit_id: str) -> Dict:
        """获取特定单元的性能指标"""
        if unit_id not in self.unit_loads:
            return {'error': f'Unit {unit_id} not found'}
        
        unit_load = self.unit_loads[unit_id]
        unit_capacity = self.unit_capacities[unit_id]
        
        # 计算该单元的历史利用率
        recent_assignments = [
            entry for entry in self.load_history
            if entry['timestamp'] > time.time() - 60  # 最近1分钟
        ]
        
        unit_assignments = [
            entry for entry in recent_assignments
            if entry['selected_unit'] == unit_id
        ]
        
        return {
            'unit_id': unit_id,
            'current_load': unit_load,
            'capacity': unit_capacity,
            'utilization': unit_load / unit_capacity if unit_capacity > 0 else 0,
            'recent_assignments': len(unit_assignments),
            'average_load_increment': np.mean([entry['load_increment'] for entry in unit_assignments]) if unit_assignments else 0,
            'last_assignment': max([entry['timestamp'] for entry in unit_assignments]) if unit_assignments else None
        }


if __name__ == "__main__":
    # 测试负载均衡器
    units = ['cpu_unit_1', 'cpu_unit_2', 'gpu_unit_1', 'memory_unit_1']
    load_balancer = LoadBalancer(units)
    
    # 调整单元容量
    load_balancer.update_unit_capacity('gpu_unit_1', 2.0)  # GPU容量更大
    load_balancer.update_unit_capacity('memory_unit_1', 0.5)  # 内存单元容量较小
    
    print(f"初始容量: {load_balancer.unit_capacities}")
    
    # 测试任务分配
    import random
    
    for i in range(20):
        priority = random.randint(0, 4)  # 随机优先级
        selected_unit = load_balancer.select_processing_unit(
            operation=f"task_{i}",
            operation_priority=priority
        )
        print(f"任务_{i} (优先级 {priority}) -> {selected_unit}")
    
    # 获取状态
    status = load_balancer.get_load_balance_status()
    print(f"\n负载状态:")
    print(f"单元负载: {status['unit_loads']}")
    print(f"单元利用率: {status['unit_utilization']}")
    print(f"均衡分数: {status['balance_score']:.3f}")
    
    # 优化建议
    optimization = load_balancer.optimize_load_distribution()
    print(f"\n优化建议: {optimization}")
    
    # 应用优化
    if optimization['optimization_actions']:
        print("\n应用负载优化...")
        optimization_result = load_balancer.apply_load_optimization(
            optimization['optimization_actions']
        )
        print(f"优化结果: {optimization_result}")
    
    # 获取单个单元性能
    gpu_performance = load_balancer.get_unit_performance('gpu_unit_1')
    print(f"\nGPU单元性能: {gpu_performance}")
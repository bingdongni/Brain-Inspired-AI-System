"""
优先级管理器
管理记忆操作和消息的优先级分配
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class PriorityConfig:
    """优先级配置"""
    max_priority_levels: int = 5
    importance_weights: Dict[str, float] = None
    urgency_weights: Dict[str, float] = None
    adaptive_priority: bool = True
    priority_decay: float = 0.95


class PriorityManager:
    """
    优先级管理器
    智能分配和管理操作优先级
    """
    
    def __init__(self, config: PriorityConfig = None):
        self.config = config or PriorityConfig()
        
        # 默认权重
        self.default_importance_weights = self.config.importance_weights or {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4,
            'background': 0.2
        }
        
        self.default_urgency_weights = self.config.urgency_weights or {
            'immediate': 1.0,
            'urgent': 0.9,
            'normal': 0.7,
            'low': 0.5,
            'deferred': 0.3
        }
        
        # 优先级队列
        self.priority_queues = [
            deque() for _ in range(self.config.max_priority_levels)
        ]
        
        # 优先级历史
        self.priority_history = deque(maxlen=1000)
        
        # 统计信息
        self.stats = {
            'total_assignments': 0,
            'priority_distribution': {i: 0 for i in range(self.config.max_priority_levels)},
            'average_wait_time': 0.0,
            'priority_adjustments': 0
        }
    
    def assign_priority(
        self,
        operation: Any,
        importance: str = 'medium',
        urgency: str = 'normal',
        custom_factors: Optional[Dict[str, float]] = None
    ) -> int:
        """
        分配优先级
        
        Args:
            operation: 操作对象
            importance: 重要性级别
            urgency: 紧急程度
            custom_factors: 自定义权重因子
            
        Returns:
            分配的优先级 (0=最高优先级)
        """
        # 计算基础优先级分数
        importance_score = self.default_importance_weights.get(importance, 0.6)
        urgency_score = self.default_urgency_weights.get(urgency, 0.7)
        
        # 应用自定义因子
        custom_score = 1.0
        if custom_factors:
            for factor, weight in custom_factors.items():
                custom_score *= weight
        
        # 综合分数
        total_score = importance_score * urgency_score * custom_score
        
        # 自适应优先级调整
        if self.config.adaptive_priority:
            total_score = self._apply_adaptive_adjustment(total_score, operation)
        
        # 转换为优先级级别 (0=最高)
        priority_level = self._score_to_priority_level(total_score)
        
        # 记录到统计
        self.stats['total_assignments'] += 1
        self.stats['priority_distribution'][priority_level] += 1
        
        # 记录优先级历史
        self.priority_history.append({
            'timestamp': time.time(),
            'priority_level': priority_level,
            'total_score': total_score,
            'importance': importance,
            'urgency': urgency
        })
        
        return priority_level
    
    def get_queue_status(self) -> Dict:
        """获取队列状态"""
        return {
            'queue_sizes': [len(queue) for queue in self.priority_queues],
            'total_items': sum(len(queue) for queue in self.priority_queues),
            'priority_distribution': self.stats['priority_distribution'].copy(),
            'max_priority_levels': self.config.max_priority_levels
        }
    
    def optimize_priority_distribution(self) -> Dict:
        """优化优先级分布"""
        current_distribution = self.stats['priority_distribution'].copy()
        total_items = sum(current_distribution.values())
        
        if total_items == 0:
            return {'status': 'no_items_to_optimize'}
        
        # 计算理想分布 (均衡分布)
        ideal_per_level = total_items / self.config.max_priority_levels
        
        optimization_suggestions = {}
        
        for level, count in current_distribution.items():
            deviation = count - ideal_per_level
            if abs(deviation) > ideal_per_level * 0.3:  # 超过30%偏差
                if deviation > 0:
                    optimization_suggestions[f'level_{level}'] = {
                        'current': count,
                        'suggested': 'reduce',
                        'reason': 'overloaded'
                    }
                else:
                    optimization_suggestions[f'level_{level}'] = {
                        'current': count,
                        'suggested': 'increase',
                        'reason': 'underloaded'
                    }
        
        return {
            'current_distribution': current_distribution,
            'ideal_distribution': {i: ideal_per_level for i in range(self.config.max_priority_levels)},
            'optimization_suggestions': optimization_suggestions
        }
    
    def _apply_adaptive_adjustment(self, base_score: float, operation: Any) -> float:
        """应用自适应调整"""
        # 基于当前队列负载调整
        current_load = sum(len(queue) for queue in self.priority_queues)
        
        if current_load > 100:  # 高负载时，提升高优先级操作的比例
            if base_score > 0.7:
                base_score *= 1.1  # 提升高重要性操作的优先级
            elif base_score < 0.4:
                base_score *= 0.9  # 降低低重要性操作的优先级
        
        # 基于时间因素调整
        current_time = time.time()
        time_factor = 1.0
        
        # 如果操作长时间未处理，逐渐提升优先级
        if hasattr(operation, 'creation_time'):
            age = current_time - operation.creation_time
            if age > 60:  # 超过1分钟
                time_factor = min(1.5, 1.0 + age / 300)  # 最大1.5倍提升
                base_score *= time_factor
                self.stats['priority_adjustments'] += 1
        
        return min(1.0, base_score)
    
    def _score_to_priority_level(self, score: float) -> int:
        """将分数转换为优先级级别"""
        # 分数越高，优先级越高 (数字越小)
        if score >= 0.9:
            return 0  # 最高优先级
        elif score >= 0.75:
            return 1
        elif score >= 0.6:
            return 2
        elif score >= 0.45:
            return 3
        else:
            return 4  # 最低优先级


class LoadBalancer:
    """
    负载均衡器
    在多个处理单元间分配负载
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
    
    def select_processing_unit(
        self,
        operation: Any,
        strategy: str = None,
        operation_priority: int = 2
    ) -> str:
        """
        选择处理单元
        
        Args:
            operation: 操作对象
            strategy: 选择策略
            operation_priority: 操作优先级
            
        Returns:
            选中的处理单元ID
        """
        strategy = strategy or self.current_strategy
        
        if strategy not in self.strategies:
            strategy = 'adaptive'
        
        selected_unit = self.strategies[strategy](operation, operation_priority)
        
        # 更新负载
        self.unit_loads[selected_unit] += self._calculate_load_increment(operation_priority)
        
        # 记录选择历史
        self.load_history.append({
            'timestamp': time.time(),
            'selected_unit': selected_unit,
            'strategy': strategy,
            'priority': operation_priority
        })
        
        return selected_unit
    
    def update_unit_capacity(self, unit_id: str, new_capacity: float):
        """更新处理单元容量"""
        self.unit_capacities[unit_id] = max(0.1, new_capacity)
    
    def get_load_balance_status(self) -> Dict:
        """获取负载均衡状态"""
        total_load = sum(self.unit_loads.values())
        total_capacity = sum(self.unit_capacities.values())
        
        utilization = {
            unit: (self.unit_loads[unit] / self.unit_capacacies[unit]) if self.unit_capacities[unit] > 0 else 0
            for unit in self.processing_units
        }
        
        return {
            'unit_loads': self.unit_loads.copy(),
            'unit_capacities': self.unit_capacities.copy(),
            'unit_utilization': utilization,
            'total_load': total_load,
            'total_capacity': total_capacity,
            'overall_utilization': total_load / total_capacity if total_capacity > 0 else 0,
            'balance_score': self._calculate_balance_score(utilization),
            'current_strategy': self.current_strategy
        }
    
    def _round_robin_select(self, operation: Any, priority: int) -> str:
        """轮询选择"""
        selected = self.processing_units[self.round_robin_index]
        self.round_robin_index = (self.round_robin_index + 1) % len(self.processing_units)
        return selected
    
    def _least_loaded_select(self, operation: Any, priority: int) -> str:
        """最少负载选择"""
        return min(self.processing_units, key=lambda unit: self.unit_loads[unit])
    
    def _weighted_select(self, operation: Any, priority: int) -> str:
        """加权选择"""
        # 基于容量的加权随机选择
        weights = [self.unit_capacities[unit] for unit in self.processing_units]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return self.processing_units[0]
        
        # 归一化权重
        normalized_weights = [w / total_weight for w in weights]
        
        # 按权重随机选择
        rand = np.random.random()
        cumulative = 0
        
        for i, unit in enumerate(self.processing_units):
            cumulative += normalized_weights[i]
            if rand <= cumulative:
                return unit
        
        return self.processing_units[-1]
    
    def _adaptive_select(self, operation: Any, priority: int) -> str:
        """自适应选择"""
        # 结合多种策略的自适应选择
        
        # 1. 优先考虑低负载单元
        least_loaded = min(self.processing_units, key=lambda unit: self.unit_loads[unit])
        
        # 2. 考虑容量因素
        best_utilization = min(
            self.processing_units,
            key=lambda unit: self.unit_loads[unit] / self.unit_capacities[unit]
        )
        
        # 3. 优先级考虑
        if priority <= 1:  # 高优先级操作
            # 倾向于选择容量大的单元
            largest_capacity = max(self.processing_units, key=lambda unit: self.unit_capacities[unit])
            return largest_capacity
        
        # 4. 综合评估
        scores = {}
        for unit in self.processing_units:
            load_score = 1.0 - (self.unit_loads[unit] / self.unit_capacities[unit])
            capacity_score = self.unit_capacities[unit]
            
            # 综合分数
            scores[unit] = 0.6 * load_score + 0.4 * (capacity_score / max(self.unit_capacities.values()))
        
        return max(scores, key=scores.get)
    
    def _calculate_load_increment(self, priority: int) -> float:
        """计算负载增量"""
        # 优先级越高，负载增量越大
        base_increment = 1.0
        
        if priority == 0:
            return base_increment * 2.0
        elif priority == 1:
            return base_increment * 1.5
        elif priority == 2:
            return base_increment * 1.0
        elif priority == 3:
            return base_increment * 0.7
        else:
            return base_increment * 0.5
    
    def _calculate_balance_score(self, utilization: Dict[str, float]) -> float:
        """计算负载均衡分数"""
        if not utilization:
            return 0.0
        
        utilizations = list(utilization.values())
        
        if len(utilizations) <= 1:
            return 1.0
        
        # 计算标准差 (越小越均衡)
        mean_util = np.mean(utilizations)
        std_util = np.std(utilizations)
        
        # 转换为均衡分数 (1.0 = 完全均衡)
        balance_score = max(0.0, 1.0 - std_util)
        
        return balance_score
    
    def optimize_load_distribution(self) -> Dict:
        """优化负载分布"""
        current_status = self.get_load_balance_status()
        
        optimization_actions = []
        
        # 检测负载不均衡
        utilizations = current_status['unit_utilization']
        mean_utilization = np.mean(list(utilizations.values()))
        
        for unit, utilization in utilizations.items():
            if utilization > mean_utilization * 1.5:  # 负载过高
                optimization_actions.append({
                    'action': 'reduce_load',
                    'unit': unit,
                    'current_utilization': utilization,
                    'target_utilization': mean_utilization,
                    'suggestion': 'redirect_operations_to_other_units'
                })
            elif utilization < mean_utilization * 0.5:  # 负载过低
                optimization_actions.append({
                    'action': 'increase_load',
                    'unit': unit,
                    'current_utilization': utilization,
                    'target_utilization': mean_utilization,
                    'suggestion': 'assign_more_operations_to_this_unit'
                })
        
        return {
            'current_balance_score': current_status['balance_score'],
            'optimization_actions': optimization_actions,
            'mean_utilization': mean_utilization,
            'recommended_strategy': self._recommend_strategy(utilizations)
        }
    
    def _recommend_strategy(self, utilization: Dict[str, float]) -> str:
        """推荐负载均衡策略"""
        utilizations = list(utilization.values())
        utilization_variance = np.var(utilizations)
        
        if utilization_variance < 0.1:
            return 'round_robin'  # 负载均衡，使用轮询
        elif max(utilizations) > 0.8:
            return 'least_loaded'  # 有高负载单元，使用最少负载
        else:
            return 'adaptive'  # 一般情况使用自适应策略


if __name__ == "__main__":
    # 测试优先级管理器
    priority_manager = PriorityManager()
    
    # 测试优先级分配
    priorities = []
    for i in range(10):
        priority = priority_manager.assign_priority(
            operation=f"操作_{i}",
            importance=['high', 'medium', 'low'][i % 3],
            urgency=['urgent', 'normal', 'low'][i % 3]
        )
        priorities.append(priority)
        print(f"操作_{i}: 优先级 {priority}")
    
    print(f"优先级分布: {priority_manager.get_queue_status()}")
    
    # 测试负载均衡器
    units = ['unit_1', 'unit_2', 'unit_3']
    load_balancer = LoadBalancer(units)
    
    # 测试负载分配
    for i in range(15):
        selected_unit = load_balancer.select_processing_unit(
            operation=f"任务_{i}",
            operation_priority=i % 5
        )
        print(f"任务_{i} -> {selected_unit}")
    
    print(f"负载状态: {load_balancer.get_load_balance_status()}")
    print(f"优化建议: {load_balancer.optimize_load_distribution()}")
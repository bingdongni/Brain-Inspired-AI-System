"""
信息流控制系统
管理和优化记忆系统中信息流动的效率和准确性
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import asyncio
import time
from collections import deque, defaultdict
from dataclasses import dataclass


@dataclass
class FlowControlConfig:
    """流控制配置"""
    max_buffer_size: int = 1000
    max_throughput: float = 100.0  # 每秒最大处理量
    priority_levels: int = 5
    batching_enabled: bool = True
    batch_size: int = 32
    adaptive_throttling: bool = True
    latency_threshold: float = 0.1  # 延迟阈值（秒）
    quality_threshold: float = 0.7  # 质量阈值


class InformationFlowController:
    """
    高效的信息流控制系统
    优化记忆处理和传输的效率
    """
    
    def __init__(self, config: FlowControlConfig):
        self.config = config
        
        # 输入缓冲区
        self.input_buffer = deque(maxlen=config.max_buffer_size)
        
        # 处理队列
        self.processing_queues = [
            deque() for _ in range(config.priority_levels)
        ]
        
        # 输出缓冲区
        self.output_buffer = deque(maxlen=config.max_buffer_size)
        
        # 流控制指标
        self.flow_metrics = {
            'current_throughput': 0.0,
            'average_latency': 0.0,
            'quality_score': 0.0,
            'buffer_occupancy': 0.0,
            'queue_efficiency': 0.0
        }
        
        # 历史指标
        self.metrics_history = deque(maxlen=100)
        
        # 质量评估器
        self.quality_evaluator = QualityEvaluator()
        
        # 自适应节流器
        if config.adaptive_throttling:
            self.throttle_controller = AdaptiveThrottleController(config)
        
        # 批处理器
        if config.batching_enabled:
            self.batch_processor = BatchProcessor(config.batch_size)
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'total_dropped': 0,
            'total_batched': 0,
            'quality_improvements': 0,
            'throttling_events': 0,
            'average_processing_time': 0.0
        }
        
        # 性能监控
        self.performance_monitor = FlowPerformanceMonitor()
    
    async def submit_information(
        self,
        information: Any,
        priority: int = 2,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        提交信息到流控制系统
        
        Args:
            information: 要处理的信息
            priority: 优先级 (0=最高, 优先级数-1=最低)
            metadata: 元数据
            
        Returns:
            是否成功提交
        """
        try:
            # 优先级范围检查
            priority = max(0, min(priority, self.config.priority_levels - 1))
            
            # 评估信息质量
            quality_score = self.quality_evaluator.evaluate(information)
            
            # 自适应节流检查
            if self.config.adaptive_throttling:
                if not self.throttle_controller.should_accept_information():
                    self.stats['throttling_events'] += 1
                    return False
            
            # 质量过滤
            if quality_score < self.config.quality_threshold:
                self.stats['total_dropped'] += 1
                return False
            
            # 添加到输入缓冲区
            info_item = {
                'data': information,
                'priority': priority,
                'quality': quality_score,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
            
            self.input_buffer.append(info_item)
            
            # 更新流指标
            self._update_flow_metrics()
            
            return True
            
        except Exception as e:
            print(f"提交信息失败: {e}")
            return False
    
    async def process_next_batch(self) -> List[Any]:
        """
        处理下一批信息
        
        Returns:
            处理结果列表
        """
        try:
            if not self.input_buffer:
                return []
            
            # 按优先级从队列中提取信息
            batch_items = []
            
            for priority_level in range(self.config.priority_levels):
                while (len(batch_items) < self.config.batch_size and 
                       self.processing_queues[priority_level]):
                    item = self.processing_queues[priority_level].popleft()
                    batch_items.append(item)
                
                if len(batch_items) >= self.config.batch_size:
                    break
            
            # 如果没有足够的高优先级项目，从输入缓冲区补充
            while len(batch_items) < self.config.batch_size and self.input_buffer:
                batch_items.append(self.input_buffer.popleft())
            
            if not batch_items:
                return []
            
            # 批量处理
            if self.config.batching_enabled:
                processed_batch = await self.batch_processor.process_batch(batch_items)
                self.stats['total_batched'] += len(processed_batch)
            else:
                processed_batch = []
                for item in batch_items:
                    processed = await self._process_single_item(item)
                    processed_batch.append(processed)
            
            # 更新统计
            self.stats['total_processed'] += len(processed_batch)
            
            # 更新质量评估
            if processed_batch:
                quality_improvement = self._assess_quality_improvement(processed_batch)
                if quality_improvement > 0:
                    self.stats['quality_improvements'] += 1
            
            return processed_batch
            
        except Exception as e:
            print(f"批处理失败: {e}")
            return []
    
    async def get_flow_status(self) -> Dict:
        """获取流控制状态"""
        return {
            'buffer_status': {
                'input_buffer_size': len(self.input_buffer),
                'output_buffer_size': len(self.output_buffer),
                'queue_sizes': [len(queue) for queue in self.processing_queues]
            },
            'performance_metrics': {
                **self.flow_metrics,
                'throughput_limit': self.config.max_throughput,
                'latency_threshold': self.config.latency_threshold,
                'quality_threshold': self.config.quality_threshold
            },
            'statistics': self.stats,
            'health_status': self._assess_system_health(),
            'configuration': {
                'batching_enabled': self.config.batching_enabled,
                'adaptive_throttling': self.config.adaptive_throttling,
                'priority_levels': self.config.priority_levels
            }
        }
    
    async def optimize_flow(self) -> Dict:
        """优化流控制"""
        try:
            optimization_results = {}
            
            # 缓冲区优化
            if len(self.input_buffer) > self.config.max_buffer_size * 0.8:
                # 执行缓冲区清理
                removed_items = await self._clean_input_buffer()
                optimization_results['buffer_cleanup'] = {
                    'removed_items': removed_items,
                    'current_size': len(self.input_buffer)
                }
            
            # 优先级队列平衡
            queue_balance = self._balance_priority_queues()
            optimization_results['queue_balance'] = queue_balance
            
            # 自适应参数调整
            if self.config.adaptive_throttling:
                throttle_adjustment = self.throttle_controller.adjust_parameters(
                    self.flow_metrics
                )
                optimization_results['throttle_adjustment'] = throttle_adjustment
            
            # 批处理参数优化
            if self.config.batching_enabled:
                batch_optimization = self.batch_processor.optimize_parameters(
                    self.flow_metrics
                )
                optimization_results['batch_optimization'] = batch_optimization
            
            return {
                'optimization_results': optimization_results,
                'optimization_timestamp': time.time(),
                'flow_metrics_after': self.flow_metrics
            }
            
        except Exception as e:
            print(f"流优化失败: {e}")
            return {'error': str(e)}
    
    def _update_flow_metrics(self):
        """更新流控制指标"""
        current_time = time.time()
        
        # 计算当前吞吐量（每秒处理项目数）
        recent_processing = [
            item for item in self.metrics_history
            if current_time - item['timestamp'] < 1.0
        ]
        self.flow_metrics['current_throughput'] = len(recent_processing)
        
        # 计算缓冲区占用率
        total_buffer_size = (
            len(self.input_buffer) + len(self.output_buffer) +
            sum(len(queue) for queue in self.processing_queues)
        )
        max_possible_size = self.config.max_buffer_size + len(self.processing_queues) * self.config.max_buffer_size
        self.flow_metrics['buffer_occupancy'] = total_buffer_size / max_possible_size
        
        # 计算队列效率
        active_queues = sum(1 for queue in self.processing_queues if len(queue) > 0)
        self.flow_metrics['queue_efficiency'] = active_queues / len(self.processing_queues)
        
        # 添加到历史记录
        self.metrics_history.append({
            'timestamp': current_time,
            'throughput': self.flow_metrics['current_throughput'],
            'buffer_occupancy': self.flow_metrics['buffer_occupancy']
        })
    
    def _balance_priority_queues(self) -> Dict:
        """平衡优先级队列"""
        queue_sizes = [len(queue) for queue in self.processing_queues]
        
        # 计算平衡度
        total_items = sum(queue_sizes)
        if total_items == 0:
            return {'status': 'no_items_to_balance'}
        
        # 计算分布方差
        expected_per_queue = total_items / len(self.processing_queues)
        variance = sum((size - expected_per_queue) ** 2 for size in queue_sizes) / len(queue_sizes)
        
        # 如果方差过大，重新分配
        if variance > expected_per_queue * 0.5:
            self._redistribute_items()
            return {'status': 'redistributed', 'variance_before': variance}
        else:
            return {'status': 'balanced', 'variance': variance}
    
    def _redistribute_items(self):
        """重新分配队列项目"""
        # 收集所有项目
        all_items = []
        for queue in self.processing_queues:
            while queue:
                all_items.append(queue.popleft())
        
        # 按质量分数排序
        all_items.sort(key=lambda x: x['quality'], reverse=True)
        
        # 重新分配到优先级队列
        for i, item in enumerate(all_items):
            target_queue = min(i // (len(all_items) // len(self.processing_queues) + 1),
                             len(self.processing_queues) - 1)
            self.processing_queues[target_queue].append(item)
    
    async def _process_single_item(self, item: Dict) -> Any:
        """处理单个项目"""
        start_time = time.time()
        
        # 模拟处理过程
        processed_data = await self._simulate_processing(item['data'])
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 更新平均处理时间统计
        current_avg = self.stats['average_processing_time']
        total_processed = self.stats['total_processed']
        
        if total_processed > 0:
            self.stats['average_processing_time'] = (
                (current_avg * total_processed + processing_time) / (total_processed + 1)
            )
        else:
            self.stats['average_processing_time'] = processing_time
        
        return {
            'original_data': item['data'],
            'processed_data': processed_data,
            'quality': item['quality'],
            'processing_time': processing_time,
            'priority': item['priority']
        }
    
    async def _simulate_processing(self, data: Any) -> Any:
        """模拟信息处理"""
        # 这里实现实际的处理逻辑
        await asyncio.sleep(0.001)  # 模拟处理延迟
        return data
    
    def _assess_quality_improvement(self, processed_batch: List[Dict]) -> float:
        """评估质量改进"""
        if not processed_batch:
            return 0.0
        
        # 计算处理后质量的平均值
        avg_quality = np.mean([item['quality'] for item in processed_batch])
        
        # 质量阈值检查
        if avg_quality > self.config.quality_threshold:
            return avg_quality - self.config.quality_threshold
        
        return 0.0
    
    def _assess_system_health(self) -> Dict:
        """评估系统健康状态"""
        health_score = 1.0
        
        # 检查吞吐量
        if self.flow_metrics['current_throughput'] > self.config.max_throughput:
            health_score -= 0.2
        
        # 检查延迟
        if self.flow_metrics['average_latency'] > self.config.latency_threshold:
            health_score -= 0.2
        
        # 检查缓冲区占用
        if self.flow_metrics['buffer_occupancy'] > 0.8:
            health_score -= 0.3
        
        # 检查队列效率
        if self.flow_metrics['queue_efficiency'] < 0.5:
            health_score -= 0.2
        
        health_status = {
            'score': max(0.0, health_score),
            'status': 'healthy' if health_score > 0.7 else 'degraded' if health_score > 0.4 else 'critical'
        }
        
        return health_status
    
    async def _clean_input_buffer(self) -> int:
        """清理输入缓冲区"""
        current_time = time.time()
        items_to_remove = []
        
        # 移除过期项目
        for item in self.input_buffer:
            if current_time - item['timestamp'] > 300:  # 5分钟过期
                items_to_remove.append(item)
        
        # 移除低质量项目
        for item in self.input_buffer:
            if item['quality'] < self.config.quality_threshold * 0.5:
                items_to_remove.append(item)
        
        # 执行清理
        for item in items_to_remove:
            try:
                self.input_buffer.remove(item)
                self.stats['total_dropped'] += 1
            except ValueError:
                pass
        
        return len(items_to_remove)


class QualityEvaluator:
    """质量评估器"""
    
    def evaluate(self, information: Any) -> float:
        """评估信息质量"""
        # 基于信息特征计算质量分数
        if isinstance(information, torch.Tensor):
            # 张量信息的质量评估
            activation = torch.norm(information).item()
            consistency = 1.0 / (1.0 + torch.var(information).item())
            quality = 0.6 * min(1.0, activation / 10) + 0.4 * consistency
        else:
            # 默认质量分数
            quality = 0.7
        
        return min(1.0, max(0.0, quality))


class AdaptiveThrottleController:
    """自适应节流控制器"""
    
    def __init__(self, config: FlowControlConfig):
        self.config = config
        self.throttle_factor = 1.0
        self.adjustment_history = deque(maxlen=50)
    
    def should_accept_information(self) -> bool:
        """判断是否应该接受新信息"""
        # 基于当前负载动态调整接受概率
        current_load = self._calculate_current_load()
        
        if current_load > 0.8:
            # 高负载时降低接受概率
            acceptance_probability = max(0.1, 1.0 - current_load)
        else:
            acceptance_probability = 1.0
        
        return np.random.random() < acceptance_probability
    
    def adjust_parameters(self, flow_metrics: Dict) -> Dict:
        """调整节流参数"""
        current_load = flow_metrics['buffer_occupancy']
        throughput = flow_metrics['current_throughput']
        
        # 基于负载调整节流因子
        if current_load > 0.8:
            self.throttle_factor *= 0.9  # 增加节流
        elif current_load < 0.5:
            self.throttle_factor *= 1.1  # 减少节流
        
        # 基于吞吐量调整
        if throughput > self.config.max_throughput * 0.9:
            self.throttle_factor *= 0.95
        elif throughput < self.config.max_throughput * 0.5:
            self.throttle_factor *= 1.05
        
        # 限制节流因子范围
        self.throttle_factor = max(0.1, min(2.0, self.throttle_factor))
        
        adjustment_info = {
            'new_throttle_factor': self.throttle_factor,
            'current_load': current_load,
            'current_throughput': throughput,
            'adjustment_reason': self._get_adjustment_reason(current_load, throughput)
        }
        
        self.adjustment_history.append(adjustment_info)
        
        return adjustment_info
    
    def _calculate_current_load(self) -> float:
        """计算当前系统负载"""
        # 简化的负载计算
        return 0.5  # 占位符
    
    def _get_adjustment_reason(self, load: float, throughput: float) -> str:
        """获取调整原因"""
        if load > 0.8:
            return "high_buffer_occupancy"
        elif throughput > self.config.max_throughput * 0.9:
            return "high_throughput"
        elif load < 0.5 and throughput < self.config.max_throughput * 0.5:
            return "low_utilization"
        else:
            return "normal_operation"


class BatchProcessor:
    """批处理器"""
    
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.processing_history = deque(maxlen=100)
    
    async def process_batch(self, batch_items: List[Dict]) -> List[Dict]:
        """处理一批信息"""
        if not batch_items:
            return []
        
        start_time = time.time()
        
        # 并行处理批次
        tasks = []
        for item in batch_items:
            task = self._process_batch_item(item)
            tasks.append(task)
        
        processed_results = await asyncio.gather(*tasks)
        
        processing_time = time.time() - start_time
        
        # 记录处理历史
        self.processing_history.append({
            'batch_size': len(batch_items),
            'processing_time': processing_time,
            'throughput': len(batch_items) / processing_time
        })
        
        return processed_results
    
    async def _process_batch_item(self, item: Dict) -> Dict:
        """处理批次中的单个项目"""
        # 模拟批处理
        await asyncio.sleep(0.001 * len(item['data']) if hasattr(item['data'], '__len__') else 0.001)
        
        return {
            'original_data': item['data'],
            'processed_data': item['data'],  # 简化处理
            'quality': item['quality'],
            'priority': item['priority'],
            'batch_processed': True
        }
    
    def optimize_parameters(self, flow_metrics: Dict) -> Dict:
        """优化批处理参数"""
        if not self.processing_history:
            return {'status': 'no_history'}
        
        # 分析处理历史
        recent_entries = list(self.processing_history)[-10:]
        avg_throughput = np.mean([entry['throughput'] for entry in recent_entries])
        avg_processing_time = np.mean([entry['processing_time'] for entry in recent_entries])
        
        optimization_result = {
            'current_batch_size': self.batch_size,
            'average_throughput': avg_throughput,
            'average_processing_time': avg_processing_time,
            'optimization_suggested': False
        }
        
        # 如果平均处理时间过长，建议减小批次大小
        if avg_processing_time > 0.1:
            new_batch_size = max(8, int(self.batch_size * 0.8))
            optimization_result['suggested_batch_size'] = new_batch_size
            optimization_result['optimization_suggested'] = True
        
        # 如果吞吐量较低，建议增大批次大小
        elif avg_throughput < 50:
            new_batch_size = min(64, int(self.batch_size * 1.2))
            optimization_result['suggested_batch_size'] = new_batch_size
            optimization_result['optimization_suggested'] = True
        
        return optimization_result


class FlowPerformanceMonitor:
    """流性能监控器"""
    
    def __init__(self):
        self.monitoring_data = deque(maxlen=1000)
        self.alert_thresholds = {
            'latency': 0.2,
            'throughput_drop': 0.3,
            'buffer_overflow': 0.9
        }
    
    def record_performance(self, metrics: Dict):
        """记录性能数据"""
        self.monitoring_data.append({
            'timestamp': time.time(),
            **metrics
        })
    
    def check_alerts(self) -> List[Dict]:
        """检查性能告警"""
        alerts = []
        
        if not self.monitoring_data:
            return alerts
        
        recent_data = list(self.monitoring_data)[-10:]
        
        # 检查延迟告警
        avg_latency = np.mean([data.get('average_latency', 0) for data in recent_data])
        if avg_latency > self.alert_thresholds['latency']:
            alerts.append({
                'type': 'high_latency',
                'value': avg_latency,
                'threshold': self.alert_thresholds['latency']
            })
        
        # 检查吞吐量下降告警
        throughputs = [data.get('current_throughput', 0) for data in recent_data]
        if len(throughputs) >= 2:
            recent_throughput = np.mean(throughputs[-3:])
            earlier_throughput = np.mean(throughputs[:-3]) if len(throughputs) > 3 else recent_throughput
            
            if earlier_throughput > 0 and (earlier_throughput - recent_throughput) / earlier_throughput > self.alert_thresholds['throughput_drop']:
                alerts.append({
                    'type': 'throughput_drop',
                    'current': recent_throughput,
                    'previous': earlier_throughput,
                    'drop_rate': (earlier_throughput - recent_throughput) / earlier_throughput
                })
        
        return alerts


# 使用示例和测试
if __name__ == "__main__":
    import asyncio
    
    async def test_flow_control():
        # 创建配置
        config = FlowControlConfig(
            max_buffer_size=500,
            max_throughput=50.0,
            priority_levels=3,
            batching_enabled=True,
            adaptive_throttling=True
        )
        
        # 创建流控制器
        controller = InformationFlowController(config)
        
        # 测试提交信息
        for i in range(10):
            success = await controller.submit_information(
                information=f"测试信息_{i}",
                priority=i % 3,
                metadata={'source': 'test', 'id': i}
            )
            print(f"提交信息 {i}: {'成功' if success else '失败'}")
        
        # 处理信息
        processed_batch = await controller.process_next_batch()
        print(f"处理批次大小: {len(processed_batch)}")
        
        # 获取状态
        status = await controller.get_flow_status()
        print(f"流控制状态: {status}")
        
        # 优化流
        optimization = await controller.optimize_flow()
        print(f"优化结果: {optimization}")
        
        # 性能监控
        monitor = FlowPerformanceMonitor()
        monitor.record_performance(status['performance_metrics'])
        alerts = monitor.check_alerts()
        print(f"性能告警: {alerts}")
    
    # 运行测试
    asyncio.run(test_flow_control())
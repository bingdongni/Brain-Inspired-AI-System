"""
通信控制器
统一管理海马体与新皮层之间的双向通信
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from collections import deque, defaultdict
import json

from .protocol_handler import CommunicationProtocol, NetworkConfig
from .message_types import (
    MessageType, MemoryType, Priority, MessageBuilder, Message
)
from .hippocampus_interface import HippocampusInterface, HippocampusConfig
from .neocortex_interface import NeocortexInterface, NeocortexConfig


@dataclass
class CommunicationControllerConfig:
    """通信控制器配置"""
    enable_bidirectional_sync: bool = True
    message_queue_size: int = 10000
    sync_interval: float = 5.0
    consolidation_trigger_threshold: float = 0.8
    load_balancing: bool = True
    error_recovery: bool = True
    performance_monitoring: bool = True


class CommunicationController:
    """
    通信控制器
    协调海马体与新皮层之间的双向通信
    """
    
    def __init__(self, config: CommunicationControllerConfig = None):
        self.config = config or CommunicationControllerConfig()
        
        # 初始化网络协议
        network_config = NetworkConfig(max_queue_size=self.config.message_queue_size)
        self.protocol = CommunicationProtocol(network_config)
        
        # 初始化脑区接口
        hippocampus_config = HippocampusConfig(memory_capacity=50000)
        neocortex_config = NeocortexConfig(
            cortical_columns=32,
            hierarchical_levels=3,
            attention_mechanism=True
        )
        
        self.hippocampus = HippocampusInterface(hippocampus_config, self.protocol)
        self.neocortex = NeocortexInterface(neocortex_config, self.protocol)
        
        # 通信状态
        self.is_initialized = False
        self.is_running = False
        self.active_connections = {}
        
        # 消息路由
        self.message_routes = {
            'hippocampus_to_neocortex': [],
            'neocortex_to_hippocampus': [],
            'bidirectional': []
        }
        
        # 负载均衡
        if self.config.load_balancing:
            self.load_balancer = LoadBalancer()
        
        # 性能监控
        if self.config.performance_monitoring:
            self.performance_monitor = PerformanceMonitor()
        
        # 错误恢复
        if self.config.error_recovery:
            self.error_recovery = ErrorRecovery()
        
        # 统计信息
        self.stats = {
            'messages_routed': 0,
            'hippocampus_requests': 0,
            'neocortex_requests': 0,
            'sync_events': 0,
            'consolidation_events': 0,
            'error_count': 0,
            'average_response_time': 0.0,
            'throughput': 0.0
        }
        
        # 任务队列
        self.task_queue = deque()
        self.response_cache = {}
        
        # 回调函数
        self.callbacks = {
            'on_message_routed': [],
            'on_sync_completed': [],
            'on_consolidation_triggered': [],
            'on_error_occurred': [],
            'on_performance_update': []
        }
    
    async def initialize(self) -> bool:
        """初始化通信控制器"""
        try:
            # 启动协议
            await self.protocol.start()
            
            # 初始化接口
            hippocampus_ok = await self.hippocampus.initialize()
            neocortex_ok = await self.neocortex.initialize()
            
            if not hippocampus_ok or not neocortex_ok:
                print("接口初始化失败")
                return False
            
            # 设置消息路由
            self._setup_message_routes()
            
            # 启动后台任务
            await self._start_background_tasks()
            
            self.is_initialized = True
            self.is_running = True
            
            print("通信控制器初始化成功")
            return True
            
        except Exception as e:
            print(f"通信控制器初始化失败: {e}")
            return False
    
    async def shutdown(self):
        """关闭通信控制器"""
        self.is_running = False
        
        # 关闭接口
        await self.hippocampus.protocol.stop()
        await self.neocortex.protocol.stop()
        
        # 关闭协议
        await self.protocol.stop()
        
        print("通信控制器已关闭")
    
    async def route_memory_request(
        self,
        request_type: str,
        data: Dict,
        source: str,
        target: str
    ) -> Dict:
        """
        路由记忆请求
        
        Args:
            request_type: 请求类型
            data: 请求数据
            source: 源脑区
            target: 目标脑区
            
        Returns:
            路由结果
        """
        try:
            start_time = time.time()
            
            # 负载均衡（如果启用）
            if self.config.load_balancing:
                target = self.load_balancer.select_optimal_target(target, data)
            
            # 构建消息
            if request_type == "store_memory":
                message = await self._create_store_message(data, source, target)
            elif request_type == "retrieve_memory":
                message = await self._create_retrieve_message(data, source, target)
            elif request_type == "query_memory":
                message = await self._create_query_message(data, source, target)
            else:
                return {'error': f'未知请求类型: {request_type}'}
            
            # 发送消息
            success = await self.protocol.send_message(message)
            
            if not success:
                return {'error': '消息发送失败'}
            
            # 等待响应
            response = await self._wait_for_response(message.message_id, timeout=30.0)
            
            # 记录性能指标
            response_time = time.time() - start_time
            self._update_performance_metrics(response_time, request_type)
            
            # 触发回调
            self._trigger_callback('on_message_routed', request_type, source, target, response)
            
            # 更新统计
            self.stats['messages_routed'] += 1
            if source == 'hippocampus':
                self.stats['hippocampus_requests'] += 1
            else:
                self.stats['neocortex_requests'] += 1
            
            return {
                'success': True,
                'response': response,
                'response_time': response_time,
                'routed_from': source,
                'routed_to': target
            }
            
        except Exception as e:
            self.stats['error_count'] += 1
            self._trigger_callback('on_error_occurred', f'路由请求失败: {e}')
            return {'error': str(e)}
    
    async def synchronize_brain_regions(self, sync_type: str = "full") -> Dict:
        """
        同步脑区间信息
        
        Args:
            sync_type: 同步类型 (full, partial, incremental)
            
        Returns:
            同步结果
        """
        try:
            start_time = time.time()
            
            sync_results = {}
            
            if sync_type in ["full", "partial"]:
                # 海马体到新皮层的同步
                hippocampus_state = await self._get_hippocampus_state()
                neocortex_sync_result = await self._sync_to_neocortex(hippocampus_state)
                sync_results['hippocampus_to_neocortex'] = neocortex_sync_result
            
            if sync_type in ["full", "incremental"]:
                # 新皮层到海马体的同步
                neocortex_state = await self._get_neocortex_state()
                hippocampus_sync_result = await self._sync_to_hippocampus(neocortex_state)
                sync_results['neocortex_to_hippocampus'] = hippocampus_sync_result
            
            sync_time = time.time() - start_time
            self.stats['sync_events'] += 1
            
            # 触发同步完成回调
            self._trigger_callback('on_sync_completed', sync_type, sync_results)
            
            return {
                'sync_type': sync_type,
                'sync_results': sync_results,
                'sync_time': sync_time,
                'success': True
            }
            
        except Exception as e:
            self.stats['error_count'] += 1
            self._trigger_callback('on_error_occurred', f'脑区同步失败: {e}')
            return {'error': str(e)}
    
    async def trigger_consolidation(self, trigger_type: str = "automatic") -> Dict:
        """
        触发记忆巩固
        
        Args:
            trigger_type: 触发类型 (automatic, manual, threshold)
            
        Returns:
            巩固结果
        """
        try:
            # 获取海马体统计信息
            hippocampus_stats = self.hippocampus.get_statistics()
            neocortex_stats = self.neocortex.get_statistics()
            
            # 检查是否满足巩固条件
            if trigger_type == "automatic":
                memory_load = hippocampus_stats['episodic_memories'] / hippocampus_stats.get('episodic_memory_size', 1000)
                if memory_load < self.config.consolidation_trigger_threshold:
                    return {'status': 'no_consolidation_needed', 'memory_load': memory_load}
            
            # 执行巩固
            consolidation_result = await self.hippocampus.consolidate_memories()
            semantic_consolidation_result = await self.neocortex.consolidate_to_semantic_memory(
                []  # 这里应该传递实际的记忆数据
            )
            
            combined_result = {
                'episodic_consolidation': consolidation_result,
                'semantic_consolidation': semantic_consolidation_result,
                'trigger_type': trigger_type,
                'timestamp': time.time()
            }
            
            self.stats['consolidation_events'] += 1
            
            # 触发巩固完成回调
            self._trigger_callback('on_consolidation_triggered', combined_result)
            
            return {
                'success': True,
                'consolidation_result': combined_result
            }
            
        except Exception as e:
            self.stats['error_count'] += 1
            self._trigger_callback('on_error_occurred', f'记忆巩固失败: {e}')
            return {'error': str(e)}
    
    def register_callback(self, event_type: str, callback: Callable):
        """注册回调函数"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def get_communication_status(self) -> Dict:
        """获取通信状态"""
        hippocampus_state = self.hippocampus.get_statistics()
        neocortex_state = self.neocortex.get_statistics()
        protocol_stats = self.protocol.get_statistics()
        
        return {
            'is_running': self.is_running,
            'protocol_status': protocol_stats,
            'hippocampus_status': hippocampus_state,
            'neocortex_status': neocortex_state,
            'communication_stats': self.stats,
            'active_routes': len(self.message_routes['bidirectional']),
            'config': {
                'load_balancing': self.config.load_balancing,
                'error_recovery': self.config.error_recovery,
                'performance_monitoring': self.config.performance_monitoring
            }
        }
    
    async def _start_background_tasks(self):
        """启动后台任务"""
        # 定期同步任务
        if self.config.enable_bidirectional_sync:
            asyncio.create_task(self._periodic_sync_task())
        
        # 性能监控任务
        if self.config.performance_monitoring:
            asyncio.create_task(self._performance_monitoring_task())
        
        # 负载监控任务
        if self.config.load_balancing:
            asyncio.create_task(self._load_monitoring_task())
    
    async def _periodic_sync_task(self):
        """定期同步任务"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.sync_interval)
                await self.synchronize_brain_regions(sync_type="incremental")
            except Exception as e:
                print(f"定期同步任务错误: {e}")
    
    async def _performance_monitoring_task(self):
        """性能监控任务"""
        while self.is_running:
            try:
                await asyncio.sleep(10.0)  # 每10秒监控一次
                
                # 收集性能指标
                performance_data = {
                    'timestamp': time.time(),
                    'throughput': self.stats['messages_routed'] / 10.0,  # 每秒消息数
                    'average_response_time': self.stats['average_response_time'],
                    'error_rate': self.stats['error_count'] / max(self.stats['messages_routed'], 1)
                }
                
                # 触发性能更新回调
                self._trigger_callback('on_performance_update', performance_data)
                
            except Exception as e:
                print(f"性能监控任务错误: {e}")
    
    async def _load_monitoring_task(self):
        """负载监控任务"""
        while self.is_running:
            try:
                await asyncio.sleep(5.0)  # 每5秒监控一次负载
                
                # 更新负载均衡器
                self.load_balancer.update_load_metrics({
                    'hippocampus': self.hippocampus.get_statistics(),
                    'neocortex': self.neocortex.get_statistics()
                })
                
            except Exception as e:
                print(f"负载监控任务错误: {e}")
    
    def _setup_message_routes(self):
        """设置消息路由"""
        # 海马体到新皮层的路由
        self.message_routes['hippocampus_to_neocortex'] = [
            MessageType.MEMORY_STORE,
            MessageType.QUERY_PROCESS,
            MessageType.SYNC_REQUEST
        ]
        
        # 新皮层到海马体的路由
        self.message_routes['neocortex_to_hippocampus'] = [
            MessageType.MEMORY_RETRIEVE,
            MessageType.CONTROL_COMMAND,
            MessageType.LEARNING_UPDATE
        ]
        
        # 双向路由
        self.message_routes['bidirectional'] = [
            MessageType.HEARTBEAT,
            MessageType.STATUS_UPDATE,
            MessageType.ERROR_REPORT
        ]
    
    async def _create_store_message(self, data: Dict, source: str, target: str):
        """创建存储消息"""
        return MessageBuilder.create_memory_store(
            content=data['content'],
            memory_type=data.get('memory_type', MemoryType.EPISODIC),
            importance=data.get('importance', 0.5),
            context=data.get('context'),
            sender=source,
            receiver=target,
            requires_response=True
        )
    
    async def _create_retrieve_message(self, data: Dict, source: str, target: str):
        """创建检索消息"""
        return MessageBuilder.create_memory_retrieve(
            query_vector=data['query_vector'],
            memory_type=data.get('memory_type'),
            top_k=data.get('top_k', 10),
            threshold=data.get('threshold', 0.5),
            sender=source,
            receiver=target,
            requires_response=True
        )
    
    async def _create_query_message(self, data: Dict, source: str, target: str):
        """创建查询消息"""
        return MessageBuilder.create_query_message(
            query_vector=data['query_vector'],
            search_space=data.get('search_space', []),
            filters=data.get('filters', {}),
            sender=source,
            receiver=target,
            requires_response=True
        )
    
    async def _wait_for_response(self, message_id: str, timeout: float = 30.0):
        """等待响应"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = self.protocol.receive_message(timeout=1.0)
            if response and hasattr(response, 'metadata') and \
               response.metadata.get('original_message_id') == message_id:
                return response
            
            await asyncio.sleep(0.1)
        
        return None
    
    def _update_performance_metrics(self, response_time: float, request_type: str):
        """更新性能指标"""
        # 更新平均响应时间
        current_avg = self.stats['average_response_time']
        total_requests = self.stats['messages_routed']
        
        self.stats['average_response_time'] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
        
        # 更新吞吐量
        self.stats['throughput'] = self.stats['messages_routed'] / time.time()
    
    def _trigger_callback(self, event_type: str, *args):
        """触发回调函数"""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(*args)
            except Exception as e:
                print(f"回调函数错误 ({event_type}): {e}")
    
    async def _get_hippocampus_state(self) -> Dict:
        """获取海马体状态"""
        return {
            'statistics': self.hippocampus.get_statistics(),
            'working_memory': self.hippocampus.get_working_memory_state(),
            'episodic_memory_size': len(self.hippocampus.episodic_memory),
            'timestamp': time.time()
        }
    
    async def _get_neocortex_state(self) -> Dict:
        """获取新皮层状态"""
        return {
            'statistics': self.neocortex.get_statistics(),
            'associative_memory_sizes': {
                memory_type: len(memory_bank)
                for memory_type, memory_bank in self.neocortex.associative_memory.items()
            },
            'knowledge_graph_size': len(self.neocortex.knowledge_graph),
            'timestamp': time.time()
        }
    
    async def _sync_to_neocortex(self, hippocampus_state: Dict) -> Dict:
        """同步到新皮层"""
        try:
            # 创建同步消息
            sync_message = MessageBuilder.create_sync_message(
                sync_data=hippocampus_state,
                sync_type="hippocampus_update",
                sender="hippocampus",
                receiver="neocortex"
            )
            
            success = await self.protocol.send_message(sync_message)
            
            return {
                'success': success,
                'data_size': len(str(hippocampus_state)),
                'sync_timestamp': time.time()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _sync_to_hippocampus(self, neocortex_state: Dict) -> Dict:
        """同步到海马体"""
        try:
            # 创建同步消息
            sync_message = MessageBuilder.create_sync_message(
                sync_data=neocortex_state,
                sync_type="neocortex_update",
                sender="neocortex",
                receiver="hippocampus"
            )
            
            success = await self.protocol.send_message(sync_message)
            
            return {
                'success': success,
                'data_size': len(str(neocortex_state)),
                'sync_timestamp': time.time()
            }
            
        except Exception as e:
            return {'error': str(e)}


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self):
        self.load_metrics = {}
        self.target_capabilities = {}
    
    def update_load_metrics(self, metrics: Dict):
        """更新负载指标"""
        self.load_metrics = metrics
    
    def select_optimal_target(self, target: str, data: Dict) -> str:
        """选择最优目标"""
        # 简单的负载均衡策略
        if target in self.load_metrics:
            current_load = self.load_metrics[target].get('memory_utilization', 0.5)
            
            if current_load > 0.8:  # 高负载，尝试备用目标
                if target == 'hippocampus':
                    return 'neocortex'
                else:
                    return 'hippocampus'
        
        return target


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=100)
        self.alert_thresholds = {
            'response_time': 1.0,
            'error_rate': 0.1,
            'throughput_drop': 0.5
        }
    
    def record_metrics(self, metrics: Dict):
        """记录指标"""
        self.metrics_history.append({
            'timestamp': time.time(),
            **metrics
        })
    
    def check_alerts(self) -> List[str]:
        """检查告警"""
        alerts = []
        
        if not self.metrics_history:
            return alerts
        
        recent_metrics = list(self.metrics_history)[-10:]  # 最近10个指标
        
        # 检查响应时间
        avg_response_time = sum(m.get('response_time', 0) for m in recent_metrics) / len(recent_metrics)
        if avg_response_time > self.alert_thresholds['response_time']:
            alerts.append(f"高响应时间: {avg_response_time:.2f}s")
        
        # 检查错误率
        error_count = sum(1 for m in recent_metrics if m.get('error', False))
        error_rate = error_count / len(recent_metrics)
        if error_rate > self.alert_thresholds['error_rate']:
            alerts.append(f"高错误率: {error_rate:.2%}")
        
        return alerts


class ErrorRecovery:
    """错误恢复机制"""
    
    def __init__(self):
        self.retry_count = 0
        self.max_retries = 3
        self.recovery_strategies = {
            'connection_error': self._recover_connection,
            'timeout_error': self._recover_timeout,
            'data_error': self._recover_data
        }
    
    async def handle_error(self, error_type: str, context: Dict) -> bool:
        """处理错误"""
        strategy = self.recovery_strategies.get(error_type)
        
        if strategy and self.retry_count < self.max_retries:
            try:
                self.retry_count += 1
                return await strategy(context)
            except Exception as e:
                print(f"错误恢复失败: {e}")
                return False
        
        return False
    
    async def _recover_connection(self, context: Dict) -> bool:
        """恢复连接"""
        # 重新建立连接
        await asyncio.sleep(1)  # 等待一下
        return True
    
    async def _recover_timeout(self, context: Dict) -> bool:
        """恢复超时"""
        # 增加超时时间
        context['timeout'] = context.get('timeout', 30) * 2
        return True
    
    async def _recover_data(self, context: Dict) -> bool:
        """恢复数据"""
        # 验证和修复数据
        if 'data' in context:
            context['data'] = self._validate_data(context['data'])
        return True
    
    def _validate_data(self, data: Any) -> Any:
        """验证数据"""
        # 简单的数据验证
        return data if data is not None else {}


if __name__ == "__main__":
    import asyncio
    
    # 创建控制器
    controller = CommunicationController()
    
    # 初始化
    async def main():
        if await controller.initialize():
            print("通信控制器初始化成功")
            
            # 测试消息路由
            test_data = {
                'content': torch.randn(512),
                'memory_type': MemoryType.EPISODIC,
                'importance': 0.8
            }
            
            result = await controller.route_memory_request(
                request_type="store_memory",
                data=test_data,
                source="neocortex",
                target="hippocampus"
            )
            
            print(f"消息路由结果: {result}")
            
            # 测试同步
            sync_result = await controller.synchronize_brain_regions()
            print(f"同步结果: {sync_result}")
            
            # 测试巩固
            consolidation_result = await controller.trigger_consolidation()
            print(f"巩固结果: {consolidation_result}")
            
            # 获取状态
            status = controller.get_communication_status()
            print(f"通信状态: {status}")
            
            # 关闭控制器
            await controller.shutdown()
        else:
            print("通信控制器初始化失败")
    
    asyncio.run(main())
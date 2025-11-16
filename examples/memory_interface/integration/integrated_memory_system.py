"""
集成记忆系统
统一管理所有记忆-计算接口组件
"""

import torch
import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..attention_mechanism import AttentionController, AttentionConfig
from ..communication import CommunicationController
from .memory_integrator import MemoryIntegrator, InformationFlowController


@dataclass
class SystemConfig:
    """系统配置"""
    # 注意力机制配置
    attention_config: AttentionConfig = None
    
    # 记忆整合配置
    integration_threshold: float = 0.7
    buffer_size: int = 1000
    flow_control_enabled: bool = True
    
    # 系统参数
    auto_consolidation: bool = True
    auto_optimization: bool = True
    performance_monitoring: bool = True


class IntegratedMemorySystem:
    """
    集成记忆系统
    统一协调所有记忆组件的工作
    """
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        
        # 初始化注意力系统
        attention_config = self.config.attention_config or AttentionConfig()
        self.attention_system = AttentionController(attention_config)
        
        # 初始化通信系统
        self.communication_system = CommunicationController()
        
        # 初始化记忆整合器
        self.memory_integrator = MemoryIntegrator(
            integration_threshold=self.config.integration_threshold
        )
        
        # 初始化信息流控制器
        self.flow_controller = InformationFlowController(
            buffer_size=self.config.buffer_size,
            flow_control_enabled=self.config.flow_control_enabled
        )
        
        # 系统状态
        self.is_initialized = False
        self.is_running = False
        
        # 性能统计
        self.system_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'average_response_time': 0.0,
            'system_uptime': 0.0,
            'component_health': {}
        }
        
        # 启动时间
        self.start_time = None
    
    async def initialize(self) -> bool:
        """初始化集成记忆系统"""
        try:
            print("正在初始化集成记忆系统...")
            
            # 初始化注意力系统
            attention_initialized = self.attention_system.initialize()
            if not attention_initialized:
                print("注意力系统初始化失败")
                return False
            
            # 初始化通信系统
            communication_initialized = await self.communication_system.initialize()
            if not communication_initialized:
                print("通信系统初始化失败")
                return False
            
            self.is_initialized = True
            self.start_time = time.time()
            
            print("集成记忆系统初始化完成")
            return True
            
        except Exception as e:
            print(f"集成记忆系统初始化失败: {e}")
            return False
    
    async def start(self):
        """启动系统"""
        if not self.is_initialized:
            raise RuntimeError("系统未初始化")
        
        self.is_running = True
        print("集成记忆系统已启动")
        
        # 启动后台任务
        if self.config.auto_consolidation:
            asyncio.create_task(self._auto_consolidation_task())
        
        if self.config.auto_optimization:
            asyncio.create_task(self._auto_optimization_task())
        
        if self.config.performance_monitoring:
            asyncio.create_task(self._performance_monitoring_task())
    
    async def stop(self):
        """停止系统"""
        self.is_running = False
        
        # 关闭各个系统
        await self.communication_system.shutdown()
        self.attention_system.shutdown()
        
        print("集成记忆系统已停止")
    
    async def store_memory(
        self,
        content: torch.Tensor,
        memory_type: str = 'episodic',
        importance: float = 0.5,
        context: Optional[torch.Tensor] = None
    ) -> str:
        """
        存储记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            importance: 重要性
            context: 上下文
            
        Returns:
            记忆ID
        """
        start_time = time.time()
        
        try:
            # 1. 存储到注意力系统
            memory_id = self.attention_system.store_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                context=context
            )
            
            # 2. 通过通信系统发送到海马体
            await self.communication_system.route_memory_request(
                request_type="store_memory",
                data={
                    'content': content,
                    'memory_type': memory_type,
                    'importance': importance,
                    'context': context
                },
                source="system",
                target="hippocampus"
            )
            
            # 3. 记录操作
            self._record_operation(start_time, True)
            
            return memory_id
            
        except Exception as e:
            self._record_operation(start_time, False)
            print(f"存储记忆失败: {e}")
            return ""
    
    async def retrieve_memory(
        self,
        query: torch.Tensor,
        memory_type: Optional[str] = None,
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        检索记忆
        
        Args:
            query: 查询向量
            memory_type: 记忆类型
            top_k: 返回数量
            threshold: 阈值
            
        Returns:
            检索结果
        """
        start_time = time.time()
        
        try:
            # 1. 注意力系统检索
            attention_results = self.attention_system.retrieve_memory(
                query=query,
                memory_types=[memory_type] if memory_type else None,
                top_k=top_k,
                threshold=threshold
            )
            
            # 2. 通信系统检索
            comm_result = await self.communication_system.route_memory_request(
                request_type="retrieve_memory",
                data={
                    'query_vector': query,
                    'memory_type': memory_type,
                    'top_k': top_k,
                    'threshold': threshold
                },
                source="system",
                target="hippocampus"
            )
            
            # 3. 整合结果
            if comm_result.get('success') and comm_result.get('response'):
                comm_results = comm_result['response'].results
                integrated_results = self._integrate_retrieval_results(
                    attention_results, comm_results
                )
            else:
                integrated_results = attention_results
            
            # 4. 记录操作
            self._record_operation(start_time, True)
            
            return integrated_results
            
        except Exception as e:
            self._record_operation(start_time, False)
            print(f"检索记忆失败: {e}")
            return []
    
    async def process_information(
        self,
        information: torch.Tensor,
        processing_type: str = "general"
    ) -> Dict:
        """
        处理信息
        
        Args:
            information: 信息内容
            processing_type: 处理类型
            
        Returns:
            处理结果
        """
        start_time = time.time()
        
        try:
            # 添加到流控制器
            flow_success = self.flow_controller.add_information(
                information, priority="normal"
            )
            
            if not flow_success:
                return {'error': '流控制失败'}
            
            # 处理信息
            processed_info = self.flow_controller.process_next()
            
            # 记录操作
            self._record_operation(start_time, True)
            
            return {
                'processed_information': processed_info,
                'processing_type': processing_type,
                'flow_control_status': self.flow_controller.get_flow_status()
            }
            
        except Exception as e:
            self._record_operation(start_time, False)
            print(f"信息处理失败: {e}")
            return {'error': str(e)}
    
    async def consolidate_memories(self) -> Dict:
        """
        巩固记忆
        
        Returns:
            巩固结果
        """
        start_time = time.time()
        
        try:
            # 通信系统巩固
            comm_consolidation = await self.communication_system.trigger_consolidation(
                trigger_type="automatic"
            )
            
            # 记录操作
            self._record_operation(start_time, True)
            
            return {
                'communication_consolidation': comm_consolidation,
                'consolidation_timestamp': time.time()
            }
            
        except Exception as e:
            self._record_operation(start_time, False)
            print(f"记忆巩固失败: {e}")
            return {'error': str(e)}
    
    async def optimize_system(self) -> Dict:
        """
        系统优化
        
        Returns:
            优化结果
        """
        start_time = time.time()
        
        try:
            # 注意力系统优化
            attention_optimization = self.attention_system.optimize_system()
            
            # 流控制器优化
            flow_status = self.flow_controller.get_flow_status()
            
            # 记录操作
            self._record_operation(start_time, True)
            
            return {
                'attention_optimization': attention_optimization,
                'flow_control_status': flow_status,
                'optimization_timestamp': time.time()
            }
            
        except Exception as e:
            self._record_operation(start_time, False)
            print(f"系统优化失败: {e}")
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict:
        """
        获取系统状态
        
        Returns:
            系统状态信息
        """
        try:
            # 各组件状态
            attention_status = self.attention_system.get_memory_state()
            communication_status = self.communication_system.get_communication_status()
            flow_status = self.flow_controller.get_flow_status()
            
            # 系统统计
            uptime = time.time() - self.start_time if self.start_time else 0
            
            return {
                'system_status': {
                    'is_running': self.is_running,
                    'uptime_seconds': uptime,
                    'initialization_time': self.start_time
                },
                'component_status': {
                    'attention_system': attention_status,
                    'communication_system': communication_status,
                    'flow_controller': flow_status
                },
                'performance_metrics': {
                    'total_operations': self.system_stats['total_operations'],
                    'success_rate': (
                        self.system_stats['successful_operations'] / 
                        max(self.system_stats['total_operations'], 1)
                    ),
                    'average_response_time': self.system_stats['average_response_time']
                },
                'configuration': {
                    'auto_consolidation': self.config.auto_consolidation,
                    'auto_optimization': self.config.auto_optimization,
                    'performance_monitoring': self.config.performance_monitoring
                }
            }
            
        except Exception as e:
            print(f"获取系统状态失败: {e}")
            return {'error': str(e)}
    
    def _integrate_retrieval_results(
        self,
        attention_results: List[Dict],
        communication_results: List[Dict]
    ) -> List[Dict]:
        """整合检索结果"""
        # 简单的结果合并和去重
        integrated = attention_results.copy()
        
        # 合并通信结果
        for comm_result in communication_results:
            # 检查是否已存在
            exists = any(
                att_result.get('memory_id') == comm_result.get('memory_id')
                for att_result in integrated
            )
            
            if not exists:
                integrated.append(comm_result)
        
        # 按相关性排序
        integrated.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return integrated
    
    def _record_operation(self, start_time: float, success: bool):
        """记录操作统计"""
        operation_time = time.time() - start_time
        
        self.system_stats['total_operations'] += 1
        
        if success:
            self.system_stats['successful_operations'] += 1
            
            # 更新平均响应时间
            current_avg = self.system_stats['average_response_time']
            total_ops = self.system_stats['successful_operations']
            
            if total_ops > 1:
                self.system_stats['average_response_time'] = (
                    (current_avg * (total_ops - 1) + operation_time) / total_ops
                )
            else:
                self.system_stats['average_response_time'] = operation_time
        else:
            self.system_stats['failed_operations'] += 1
    
    async def _auto_consolidation_task(self):
        """自动巩固任务"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # 每30秒执行一次
                await self.consolidate_memories()
            except Exception as e:
                print(f"自动巩固任务错误: {e}")
    
    async def _auto_optimization_task(self):
        """自动优化任务"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # 每60秒执行一次
                await self.optimize_system()
            except Exception as e:
                print(f"自动优化任务错误: {e}")
    
    async def _performance_monitoring_task(self):
        """性能监控任务"""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # 每10秒监控一次
                status = self.get_system_status()
                
                # 检查性能告警
                success_rate = status['performance_metrics']['success_rate']
                if success_rate < 0.9:
                    print(f"警告: 系统成功率低于90%: {success_rate:.2%}")
                
                avg_response_time = status['performance_metrics']['average_response_time']
                if avg_response_time > 1.0:
                    print(f"警告: 平均响应时间过长: {avg_response_time:.2f}秒")
                    
            except Exception as e:
                print(f"性能监控任务错误: {e}")


# 使用示例和测试
if __name__ == "__main__":
    async def main():
        # 创建系统配置
        config = SystemConfig(
            auto_consolidation=True,
            auto_optimization=True,
            performance_monitoring=True
        )
        
        # 创建集成记忆系统
        system = IntegratedMemorySystem(config)
        
        # 初始化系统
        if await system.initialize():
            print("系统初始化成功")
            
            # 启动系统
            await system.start()
            
            # 测试存储记忆
            test_content = torch.randn(512)
            memory_id = await system.store_memory(
                content=test_content,
                memory_type='episodic',
                importance=0.8
            )
            print(f"存储记忆 ID: {memory_id}")
            
            # 测试检索记忆
            test_query = torch.randn(512)
            results = await system.retrieve_memory(
                query=test_query,
                top_k=5
            )
            print(f"检索到 {len(results)} 个记忆")
            
            # 测试信息处理
            test_info = torch.randn(256)
            processing_result = await system.process_information(
                test_info,
                processing_type="visual"
            )
            print(f"信息处理结果: {processing_result}")
            
            # 手动巩固和优化
            consolidation_result = await system.consolidate_memories()
            print(f"记忆巩固结果: {consolidation_result}")
            
            optimization_result = await system.optimize_system()
            print(f"系统优化结果: {optimization_result}")
            
            # 获取系统状态
            system_status = system.get_system_status()
            print(f"系统状态: {system_status['system_status']}")
            print(f"性能指标: {system_status['performance_metrics']}")
            
            # 停止系统
            await system.stop()
            print("系统已停止")
            
        else:
            print("系统初始化失败")
    
    # 运行测试
    asyncio.run(main())
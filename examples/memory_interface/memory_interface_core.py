"""
记忆-计算接口核心
统一管理所有记忆接口组件的主控制器
"""

import torch
import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging

from .attention_mechanism import AttentionController, AttentionConfig
from .communication import CommunicationController
from .consolidation import ConsolidationEngine, ConsolidationConfig
from .integration import IntegratedMemorySystem, SystemConfig
from .control_flow import InformationFlowController, FlowControlConfig


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemoryInterfaceConfig:
    """记忆接口配置"""
    # 注意力机制配置
    memory_dim: int = 512
    hidden_dim: int = 512
    num_heads: int = 8
    max_memories: int = 10000
    
    # 通信配置
    enable_bidirectional_sync: bool = True
    message_queue_size: int = 10000
    
    # 巩固配置
    consolidation_threshold: float = 0.7
    strengthening_rate: float = 0.05
    
    # 流控制配置
    max_buffer_size: int = 1000
    max_throughput: float = 100.0
    
    # 系统配置
    auto_consolidation: bool = True
    auto_optimization: bool = True
    performance_monitoring: bool = True
    debug_mode: bool = False


class MemoryInterfaceCore:
    """
    记忆-计算接口核心
    统一管理所有记忆接口组件
    """
    
    def __init__(self, config: MemoryInterfaceConfig = None):
        self.config = config or MemoryInterfaceConfig()
        
        # 初始化各个组件
        self._initialize_components()
        
        # 系统状态
        self.is_initialized = False
        self.is_running = False
        self.start_time = None
        
        # 性能统计
        self.performance_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'average_response_time': 0.0,
            'memory_utilization': 0.0,
            'system_uptime': 0.0,
            'component_health': {}
        }
        
        # 操作计数器
        self.operation_counter = 0
        
        # 设置日志
        if self.config.debug_mode:
            logger.setLevel(logging.DEBUG)
    
    def _initialize_components(self):
        """初始化所有组件"""
        try:
            # 注意力机制配置
            attention_config = AttentionConfig(
                memory_dim=self.config.memory_dim,
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.num_heads,
                max_memories=self.config.max_memories
            )
            
            # 通信配置
            communication_config = type('Config', (), {
                'enable_bidirectional_sync': self.config.enable_bidirectional_sync,
                'message_queue_size': self.config.message_queue_size
            })()
            
            # 巩固配置
            consolidation_config = ConsolidationConfig(
                consolidation_threshold=self.config.consolidation_threshold,
                strengthening_rate=self.config.strengthening_rate
            )
            
            # 流控制配置
            flow_config = FlowControlConfig(
                max_buffer_size=self.config.max_buffer_size,
                max_throughput=self.config.max_throughput
            )
            
            # 系统配置
            system_config = SystemConfig(
                attention_config=attention_config,
                integration_threshold=0.7,
                buffer_size=self.config.max_buffer_size,
                flow_control_enabled=True,
                auto_consolidation=self.config.auto_consolidation,
                auto_optimization=self.config.auto_optimization,
                performance_monitoring=self.config.performance_monitoring
            )
            
            # 创建组件实例
            self.attention_controller = AttentionController(attention_config)
            self.communication_controller = CommunicationController(communication_config)
            self.consolidation_engine = ConsolidationEngine(consolidation_config)
            self.flow_controller = InformationFlowController(flow_config)
            self.integrated_system = IntegratedMemorySystem(system_config)
            
            logger.info("所有组件初始化完成")
            
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise
    
    async def initialize(self) -> bool:
        """初始化整个记忆接口系统"""
        try:
            logger.info("正在初始化记忆-计算接口系统...")
            
            # 按顺序初始化各个组件
            if not self.attention_controller.initialize():
                logger.error("注意力控制器初始化失败")
                return False
            
            if not await self.communication_controller.initialize():
                logger.error("通信控制器初始化失败")
                return False
            
            await self.consolidation_engine.initialize()
            
            self.is_initialized = True
            self.start_time = time.time()
            
            logger.info("记忆-计算接口系统初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            return False
    
    async def start(self):
        """启动记忆接口系统"""
        if not self.is_initialized:
            raise RuntimeError("系统未初始化，请先调用initialize()方法")
        
        self.is_running = True
        logger.info("记忆-计算接口系统已启动")
        
        # 启动集成系统
        await self.integrated_system.start()
        
        # 启动后台任务
        if self.config.auto_consolidation:
            asyncio.create_task(self._auto_consolidation_loop())
        
        if self.config.auto_optimization:
            asyncio.create_task(self._auto_optimization_loop())
        
        if self.config.performance_monitoring:
            asyncio.create_task(self._performance_monitoring_loop())
    
    async def stop(self):
        """停止记忆接口系统"""
        self.is_running = False
        
        # 停止集成系统
        await self.integrated_system.stop()
        
        # 关闭各组件
        await self.communication_controller.shutdown()
        self.attention_controller.shutdown()
        
        logger.info("记忆-计算接口系统已停止")
    
    # 主要接口方法
    
    async def store_memory(
        self,
        content: torch.Tensor,
        memory_type: str = 'episodic',
        importance: float = 0.5,
        context: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        存储记忆到系统中
        
        Args:
            content: 记忆内容张量
            memory_type: 记忆类型 ('episodic', 'semantic', 'procedural', 'declarative')
            importance: 重要性分数 (0-1)
            context: 上下文信息
            metadata: 元数据
            
        Returns:
            记忆ID
        """
        start_time = time.time()
        
        try:
            # 使用集成系统存储记忆
            memory_id = await self.integrated_system.store_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                context=context
            )
            
            # 通过流控制器记录操作
            await self.flow_controller.submit_information({
                'memory_id': memory_id,
                'operation': 'store',
                'memory_type': memory_type,
                'importance': importance
            })
            
            self._record_operation(start_time, True)
            logger.debug(f"记忆存储成功: {memory_id}")
            
            return memory_id
            
        except Exception as e:
            self._record_operation(start_time, False)
            logger.error(f"记忆存储失败: {e}")
            return ""
    
    async def retrieve_memory(
        self,
        query: torch.Tensor,
        memory_type: Optional[str] = None,
        top_k: int = 10,
        threshold: float = 0.5,
        context: Optional[torch.Tensor] = None
    ) -> List[Dict]:
        """
        从系统中检索记忆
        
        Args:
            query: 查询向量
            memory_type: 记忆类型过滤
            top_k: 返回结果数量
            threshold: 相关性阈值
            context: 上下文信息
            
        Returns:
            检索结果列表
        """
        start_time = time.time()
        
        try:
            # 使用集成系统检索记忆
            results = await self.integrated_system.retrieve_memory(
                query=query,
                memory_type=memory_type,
                top_k=top_k,
                threshold=threshold
            )
            
            # 通过通信系统进行海马体检索
            comm_results = await self.communication_controller.route_memory_request(
                request_type="retrieve_memory",
                data={
                    'query_vector': query,
                    'memory_type': memory_type,
                    'top_k': top_k,
                    'threshold': threshold,
                    'context': context
                },
                source="memory_interface",
                target="hippocampus"
            )
            
            # 整合结果
            if comm_results.get('success') and comm_results.get('response'):
                combined_results = self._merge_retrieval_results(results, comm_results['response'].results)
            else:
                combined_results = results
            
            self._record_operation(start_time, True)
            logger.debug(f"记忆检索成功，返回 {len(combined_results)} 个结果")
            
            return combined_results
            
        except Exception as e:
            self._record_operation(start_time, False)
            logger.error(f"记忆检索失败: {e}")
            return []
    
    async def update_memory(
        self,
        memory_id: str,
        new_content: torch.Tensor,
        update_type: str = 'additive'
    ) -> bool:
        """
        更新现有记忆
        
        Args:
            memory_id: 记忆ID
            new_content: 新内容
            update_type: 更新类型 ('additive', 'replace', 'weighted_average')
            
        Returns:
            更新是否成功
        """
        start_time = time.time()
        
        try:
            # 通过注意力系统更新记忆
            success = await self.integrated_system.attention_system.update_memory(
                memory_id=memory_id,
                new_content=new_content,
                update_type=update_type
            )
            
            # 通过通信系统同步更新
            if success:
                await self.communication_controller.route_memory_request(
                    request_type="update_memory",
                    data={
                        'memory_id': memory_id,
                        'new_content': new_content,
                        'update_type': update_type
                    },
                    source="memory_interface",
                    target="hippocampus"
                )
            
            self._record_operation(start_time, success)
            logger.debug(f"记忆更新 {memory_id}: {'成功' if success else '失败'}")
            
            return success
            
        except Exception as e:
            self._record_operation(start_time, False)
            logger.error(f"记忆更新失败: {e}")
            return False
    
    async def consolidate_memories(
        self,
        consolidation_type: str = "automatic"
    ) -> Dict:
        """
        执行记忆巩固
        
        Args:
            consolidation_type: 巩固类型 ("automatic", "manual", "threshold")
            
        Returns:
            巩固结果
        """
        start_time = time.time()
        
        try:
            # 通过通信系统执行巩固
            comm_consolidation = await self.communication_controller.trigger_consolidation(
                trigger_type=consolidation_type
            )
            
            # 通过巩固引擎执行巩固
            engine_consolidation = await self.consolidation_engine.automatic_consolidation()
            
            consolidation_result = {
                'communication_consolidation': comm_consolidation,
                'engine_consolidation': engine_consolidation,
                'consolidation_type': consolidation_type,
                'timestamp': time.time()
            }
            
            self._record_operation(start_time, True)
            logger.info(f"记忆巩固完成: {consolidation_type}")
            
            return consolidation_result
            
        except Exception as e:
            self._record_operation(start_time, False)
            logger.error(f"记忆巩固失败: {e}")
            return {'error': str(e)}
    
    async def optimize_system(self) -> Dict:
        """
        优化整个记忆系统
        
        Returns:
            优化结果
        """
        start_time = time.time()
        
        try:
            # 各组件优化
            attention_optimization = self.attention_controller.optimize_system()
            flow_optimization = await self.flow_controller.optimize_flow()
            system_optimization = await self.integrated_system.optimize_system()
            
            optimization_result = {
                'attention_optimization': attention_optimization,
                'flow_optimization': flow_optimization,
                'system_optimization': system_optimization,
                'optimization_timestamp': time.time()
            }
            
            self._record_operation(start_time, True)
            logger.info("系统优化完成")
            
            return optimization_result
            
        except Exception as e:
            self._record_operation(start_time, False)
            logger.error(f"系统优化失败: {e}")
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict:
        """
        获取整个系统的状态
        
        Returns:
            系统状态信息
        """
        try:
            uptime = time.time() - self.start_time if self.start_time else 0
            
            # 各组件状态
            attention_status = self.attention_controller.get_memory_state()
            communication_status = self.communication_system.get_communication_status()
            flow_status = asyncio.run(self.flow_controller.get_flow_status())
            consolidation_status = self.consolidation_engine.get_consolidation_status()
            integrated_status = self.integrated_system.get_system_status()
            
            # 计算系统健康度
            health_score = self._calculate_system_health()
            
            return {
                'system_info': {
                    'is_initialized': self.is_initialized,
                    'is_running': self.is_running,
                    'uptime_seconds': uptime,
                    'health_score': health_score
                },
                'performance_stats': self.performance_stats.copy(),
                'component_status': {
                    'attention_system': attention_status,
                    'communication_system': communication_status,
                    'flow_controller': flow_status,
                    'consolidation_engine': consolidation_status,
                    'integrated_system': integrated_status
                },
                'configuration': {
                    'memory_dim': self.config.memory_dim,
                    'max_memories': self.config.max_memories,
                    'auto_consolidation': self.config.auto_consolidation,
                    'auto_optimization': self.config.auto_optimization,
                    'performance_monitoring': self.config.performance_monitoring
                }
            }
            
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {'error': str(e)}
    
    # 后台任务
    
    async def _auto_consolidation_loop(self):
        """自动巩固循环"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # 每30秒执行一次
                await self.consolidate_memories("automatic")
            except Exception as e:
                logger.error(f"自动巩固任务错误: {e}")
    
    async def _auto_optimization_loop(self):
        """自动优化循环"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # 每分钟执行一次
                await self.optimize_system()
            except Exception as e:
                logger.error(f"自动优化任务错误: {e}")
    
    async def _performance_monitoring_loop(self):
        """性能监控循环"""
        while self.is_running:
            try:
                await asyncio.sleep(10)  # 每10秒监控一次
                await self._update_performance_metrics()
                
                # 检查性能告警
                self._check_performance_alerts()
                
            except Exception as e:
                logger.error(f"性能监控任务错误: {e}")
    
    # 辅助方法
    
    def _record_operation(self, start_time: float, success: bool):
        """记录操作统计"""
        operation_time = time.time() - start_time
        self.operation_counter += 1
        
        self.performance_stats['total_operations'] += 1
        
        if success:
            self.performance_stats['successful_operations'] += 1
            
            # 更新平均响应时间
            current_avg = self.performance_stats['average_response_time']
            successful_ops = self.performance_stats['successful_operations']
            
            if successful_ops > 1:
                self.performance_stats['average_response_time'] = (
                    (current_avg * (successful_ops - 1) + operation_time) / successful_ops
                )
            else:
                self.performance_stats['average_response_time'] = operation_time
        else:
            self.performance_stats['failed_operations'] += 1
        
        # 更新系统正常运行时间
        if self.start_time:
            self.performance_stats['system_uptime'] = time.time() - self.start_time
    
    async def _update_performance_metrics(self):
        """更新性能指标"""
        # 更新内存利用率
        attention_stats = self.attention_controller.get_memory_state()
        memory_utilization = attention_stats.get('system_stats', {}).get('memory_utilization', 0)
        self.performance_stats['memory_utilization'] = memory_utilization
    
    def _check_performance_alerts(self):
        """检查性能告警"""
        success_rate = (
            self.performance_stats['successful_operations'] / 
            max(self.performance_stats['total_operations'], 1)
        )
        
        avg_response_time = self.performance_stats['average_response_time']
        memory_utilization = self.performance_stats['memory_utilization']
        
        # 成功率告警
        if success_rate < 0.9:
            logger.warning(f"系统成功率低于90%: {success_rate:.2%}")
        
        # 响应时间告警
        if avg_response_time > 1.0:
            logger.warning(f"平均响应时间过长: {avg_response_time:.2f}秒")
        
        # 内存使用率告警
        if memory_utilization > 0.8:
            logger.warning(f"内存使用率过高: {memory_utilization:.2%}")
    
    def _calculate_system_health(self) -> float:
        """计算系统健康度"""
        health_score = 1.0
        
        # 成功率权重
        success_rate = (
            self.performance_stats['successful_operations'] / 
            max(self.performance_stats['total_operations'], 1)
        )
        health_score *= success_rate
        
        # 响应时间权重
        avg_response_time = self.performance_stats['average_response_time']
        if avg_response_time > 0:
            response_score = max(0, 1 - avg_response_time / 2.0)  # 2秒为基准
            health_score *= response_score
        
        # 内存使用率权重
        memory_utilization = self.performance_stats['memory_utilization']
        memory_score = max(0, 1 - memory_utilization)
        health_score *= memory_score
        
        return max(0.0, min(1.0, health_score))
    
    def _merge_retrieval_results(
        self,
        attention_results: List[Dict],
        communication_results: List[Dict]
    ) -> List[Dict]:
        """合并检索结果"""
        # 简单合并并去重
        combined = attention_results.copy()
        
        for comm_result in communication_results:
            if not any(
                att_result.get('memory_id') == comm_result.get('memory_id')
                for att_result in combined
            ):
                combined.append(comm_result)
        
        # 按相关性排序
        combined.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return combined


# 使用示例和测试
if __name__ == "__main__":
    async def main():
        # 创建配置
        config = MemoryInterfaceConfig(
            memory_dim=512,
            max_memories=5000,
            auto_consolidation=True,
            auto_optimization=True,
            debug_mode=True
        )
        
        # 创建记忆接口
        memory_interface = MemoryInterfaceCore(config)
        
        # 初始化系统
        if await memory_interface.initialize():
            print("记忆-计算接口初始化成功")
            
            # 启动系统
            await memory_interface.start()
            
            # 测试存储记忆
            test_content = torch.randn(512)
            memory_id = await memory_interface.store_memory(
                content=test_content,
                memory_type='episodic',
                importance=0.8,
                metadata={'source': 'test', 'id': '001'}
            )
            print(f"存储记忆 ID: {memory_id}")
            
            # 测试检索记忆
            test_query = torch.randn(512)
            results = await memory_interface.retrieve_memory(
                query=test_query,
                top_k=5,
                threshold=0.5
            )
            print(f"检索到 {len(results)} 个记忆")
            
            for result in results[:3]:  # 显示前3个结果
                print(f"  记忆ID: {result.get('memory_id', 'N/A')}, "
                      f"相关性: {result.get('relevance_score', 0):.3f}")
            
            # 测试更新记忆
            if memory_id:
                new_content = torch.randn(512)
                update_success = await memory_interface.update_memory(
                    memory_id=memory_id,
                    new_content=new_content,
                    update_type='additive'
                )
                print(f"记忆更新: {'成功' if update_success else '失败'}")
            
            # 手动巩固和优化
            consolidation_result = await memory_interface.consolidate_memories()
            print(f"记忆巩固: {consolidation_result.get('success', False)}")
            
            optimization_result = await memory_interface.optimize_system()
            print(f"系统优化: {optimization_result.get('attention_optimization', {}).get('cache_cleared', 0)} 缓存项已清理")
            
            # 获取系统状态
            system_status = memory_interface.get_system_status()
            print(f"\n=== 系统状态 ===")
            print(f"健康度: {system_status['system_info']['health_score']:.3f}")
            print(f"成功率: {system_status['performance_stats']['successful_operations'] / max(system_status['performance_stats']['total_operations'], 1):.2%}")
            print(f"平均响应时间: {system_status['performance_stats']['average_response_time']:.3f}秒")
            print(f"内存使用率: {system_status['performance_stats']['memory_utilization']:.2%}")
            print(f"系统运行时间: {system_status['system_info']['uptime_seconds']:.1f}秒")
            
            # 等待一段时间观察自动任务
            print("\n等待自动任务执行...")
            await asyncio.sleep(10)
            
            # 获取最终状态
            final_status = memory_interface.get_system_status()
            print(f"\n最终系统状态: {final_status['system_info']['health_score']:.3f}")
            
            # 停止系统
            await memory_interface.stop()
            print("记忆-计算接口已停止")
            
        else:
            print("记忆-计算接口初始化失败")
    
    # 运行示例
    asyncio.run(main())
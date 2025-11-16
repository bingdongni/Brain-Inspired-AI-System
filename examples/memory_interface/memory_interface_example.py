"""
记忆-计算接口系统使用示例
演示系统的主要功能和使用方法
"""

import torch
import asyncio
import time
import logging
from typing import List, Dict
import random

from memory_interface import MemoryInterfaceCore, MemoryInterfaceConfig

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MemoryInterfaceDemo:
    """记忆接口演示类"""
    
    def __init__(self):
        self.memory_interface = None
        self.demo_data = []
        
    async def setup(self):
        """设置演示环境"""
        logger.info("正在初始化记忆-计算接口系统...")
        
        # 创建配置
        config = MemoryInterfaceConfig(
            memory_dim=512,
            max_memories=10000,
            auto_consolidation=True,
            auto_optimization=True,
            performance_monitoring=True,
            debug_mode=True
        )
        
        # 创建接口实例
        self.memory_interface = MemoryInterfaceCore(config)
        
        # 初始化系统
        if not await self.memory_interface.initialize():
            raise RuntimeError("系统初始化失败")
        
        # 启动系统
        await self.memory_interface.start()
        
        logger.info("记忆-计算接口系统启动成功")
    
    async def generate_demo_data(self, num_items: int = 20):
        """生成演示数据"""
        logger.info(f"生成 {num_items} 个演示记忆...")
        
        memory_types = ['episodic', 'semantic', 'procedural', 'declarative']
        
        for i in range(num_items):
            # 生成随机内容
            content = torch.randn(512)
            
            # 随机选择记忆类型
            memory_type = random.choice(memory_types)
            
            # 随机重要性
            importance = random.uniform(0.3, 1.0)
            
            # 生成元数据
            metadata = {
                'id': f'demo_{i:03d}',
                'category': f'category_{i % 5}',
                'tags': [f'tag_{j}' for j in range(random.randint(1, 4))],
                'demo_data': True
            }
            
            self.demo_data.append({
                'content': content,
                'memory_type': memory_type,
                'importance': importance,
                'metadata': metadata
            })
        
        logger.info(f"生成了 {len(self.demo_data)} 个演示记忆")
    
    async def demonstrate_storage(self):
        """演示记忆存储功能"""
        logger.info("=== 记忆存储演示 ===")
        
        stored_ids = []
        start_time = time.time()
        
        for i, data in enumerate(self.demo_data):
            try:
                # 存储记忆
                memory_id = await self.memory_interface.store_memory(
                    content=data['content'],
                    memory_type=data['memory_type'],
                    importance=data['importance'],
                    metadata=data['metadata']
                )
                
                stored_ids.append(memory_id)
                
                if (i + 1) % 5 == 0:
                    logger.info(f"已存储 {i + 1}/{len(self.demo_data)} 个记忆")
                    
            except Exception as e:
                logger.error(f"存储记忆失败 (id={i}): {e}")
        
        storage_time = time.time() - start_time
        success_rate = len(stored_ids) / len(self.demo_data)
        
        logger.info(f"存储完成: {len(stored_ids)}/{len(self.demo_data)} 个记忆")
        logger.info(f"存储时间: {storage_time:.2f}秒")
        logger.info(f"成功率: {success_rate:.2%}")
        
        return stored_ids
    
    async def demonstrate_retrieval(self, memory_ids: List[str]):
        """演示记忆检索功能"""
        logger.info("=== 记忆检索演示 ===")
        
        # 使用多个不同的查询进行检索
        test_queries = [
            torch.randn(512),  # 随机查询1
            torch.randn(512),  # 随机查询2
            self.demo_data[0]['content'],  # 使用已知内容的变体
            torch.randn(512)   # 随机查询3
        ]
        
        retrieval_results = []
        
        for i, query in enumerate(test_queries):
            try:
                start_time = time.time()
                
                results = await self.memory_interface.retrieve_memory(
                    query=query,
                    memory_type=None,  # 不限制类型
                    top_k=5,
                    threshold=0.3
                )
                
                retrieval_time = time.time() - start_time
                
                retrieval_results.append({
                    'query_id': i,
                    'results_count': len(results),
                    'retrieval_time': retrieval_time,
                    'top_result_score': results[0].get('relevance_score', 0) if results else 0
                })
                
                logger.info(f"查询 {i+1}: 找到 {len(results)} 个结果, 耗时 {retrieval_time:.3f}秒")
                
                # 显示前3个结果
                for j, result in enumerate(results[:3]):
                    memory_id = result.get('memory_id', 'N/A')
                    score = result.get('relevance_score', 0)
                    logger.info(f"  结果 {j+1}: ID={memory_id}, 相关性={score:.3f}")
                
            except Exception as e:
                logger.error(f"检索失败 (查询 {i+1}): {e}")
        
        return retrieval_results
    
    async def demonstrate_update(self, memory_ids: List[str]):
        """演示记忆更新功能"""
        logger.info("=== 记忆更新演示 ===")
        
        if not memory_ids:
            logger.warning("没有可更新的记忆")
            return []
        
        # 更新前几个记忆
        update_count = min(5, len(memory_ids))
        update_results = []
        
        for i in range(update_count):
            try:
                memory_id = memory_ids[i]
                new_content = torch.randn(512) * 2  # 生成新的内容
                
                success = await self.memory_interface.update_memory(
                    memory_id=memory_id,
                    new_content=new_content,
                    update_type='additive'
                )
                
                update_results.append({
                    'memory_id': memory_id,
                    'success': success,
                    'operation': 'additive'
                })
                
                logger.info(f"更新记忆 {memory_id}: {'成功' if success else '失败'}")
                
            except Exception as e:
                logger.error(f"更新记忆失败 ({memory_id}): {e}")
        
        return update_results
    
    async def demonstrate_consolidation(self):
        """演示记忆巩固功能"""
        logger.info("=== 记忆巩固演示 ===")
        
        try:
            start_time = time.time()
            
            # 执行巩固
            consolidation_result = await self.memory_interface.consolidate_memories(
                consolidation_type="manual"
            )
            
            consolidation_time = time.time() - start_time
            
            logger.info(f"记忆巩固完成, 耗时 {consolidation_time:.2f}秒")
            logger.info(f"巩固结果: {consolidation_result}")
            
            return consolidation_result
            
        except Exception as e:
            logger.error(f"记忆巩固失败: {e}")
            return {}
    
    async def demonstrate_optimization(self):
        """演示系统优化功能"""
        logger.info("=== 系统优化演示 ===")
        
        try:
            start_time = time.time()
            
            # 执行优化
            optimization_result = await self.memory_interface.optimize_system()
            
            optimization_time = time.time() - start_time
            
            logger.info(f"系统优化完成, 耗时 {optimization_time:.2f}秒")
            
            # 解析优化结果
            if optimization_result:
                attention_opt = optimization_result.get('attention_optimization', {})
                cache_cleared = attention_opt.get('cache_cleared', 0)
                logger.info(f"清理了 {cache_cleared} 个缓存项")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"系统优化失败: {e}")
            return {}
    
    async def monitor_system_performance(self, duration: int = 30):
        """监控系统性能"""
        logger.info(f"=== 系统性能监控 ({duration}秒) ===")
        
        performance_data = []
        check_interval = 5  # 每5秒检查一次
        
        for elapsed in range(0, duration, check_interval):
            try:
                # 获取系统状态
                status = self.memory_interface.get_system_status()
                
                # 提取关键指标
                health_score = status['system_info']['health_score']
                success_rate = (
                    status['performance_stats']['successful_operations'] / 
                    max(status['performance_stats']['total_operations'], 1)
                )
                avg_response_time = status['performance_stats']['average_response_time']
                memory_utilization = status['performance_stats']['memory_utilization']
                
                performance_data.append({
                    'elapsed_time': elapsed + check_interval,
                    'health_score': health_score,
                    'success_rate': success_rate,
                    'avg_response_time': avg_response_time,
                    'memory_utilization': memory_utilization
                })
                
                logger.info(f"时间 {elapsed + check_interval}s: "
                          f"健康度={health_score:.3f}, "
                          f"成功率={success_rate:.2%}, "
                          f"响应时间={avg_response_time:.3f}s, "
                          f"内存使用={memory_utilization:.2%}")
                
                # 等待下次检查
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"性能监控错误: {e}")
        
        return performance_data
    
    async def run_full_demo(self):
        """运行完整演示"""
        logger.info("开始记忆-计算接口系统完整演示")
        
        try:
            # 1. 设置系统
            await self.setup()
            
            # 2. 生成演示数据
            await self.generate_demo_data(30)
            
            # 3. 演示存储功能
            stored_ids = await self.demonstrate_storage()
            
            # 4. 演示检索功能
            retrieval_results = await self.demonstrate_retrieval(stored_ids)
            
            # 5. 演示更新功能
            await asyncio.sleep(2)  # 等待系统稳定
            update_results = await self.demonstrate_update(stored_ids)
            
            # 6. 演示巩固功能
            await asyncio.sleep(2)
            consolidation_result = await self.demonstrate_consolidation()
            
            # 7. 演示优化功能
            await asyncio.sleep(2)
            optimization_result = await self.demonstrate_optimization()
            
            # 8. 性能监控
            await asyncio.sleep(5)
            performance_data = await self.monitor_system_performance(20)
            
            # 9. 获取最终系统状态
            final_status = self.memory_interface.get_system_status()
            
            # 10. 输出演示总结
            self.print_demo_summary(
                stored_ids=stored_ids,
                retrieval_results=retrieval_results,
                update_results=update_results,
                consolidation_result=consolidation_result,
                optimization_result=optimization_result,
                performance_data=performance_data,
                final_status=final_status
            )
            
        except Exception as e:
            logger.error(f"演示过程出错: {e}")
            raise
        finally:
            # 清理资源
            if self.memory_interface:
                await self.memory_interface.stop()
                logger.info("系统已停止")
    
    def print_demo_summary(self, **kwargs):
        """打印演示总结"""
        logger.info("=" * 60)
        logger.info("记忆-计算接口系统演示总结")
        logger.info("=" * 60)
        
        # 存储统计
        stored_ids = kwargs.get('stored_ids', [])
        logger.info(f"存储统计: {len(stored_ids)} 个记忆成功存储")
        
        # 检索统计
        retrieval_results = kwargs.get('retrieval_results', [])
        if retrieval_results:
            total_results = sum(r['results_count'] for r in retrieval_results)
            avg_time = sum(r['retrieval_time'] for r in retrieval_results) / len(retrieval_results)
            logger.info(f"检索统计: {total_results} 个结果检索, 平均耗时 {avg_time:.3f}秒")
        
        # 更新统计
        update_results = kwargs.get('update_results', [])
        successful_updates = sum(1 for r in update_results if r['success'])
        logger.info(f"更新统计: {successful_updates}/{len(update_results)} 个记忆更新成功")
        
        # 巩固和优化
        consolidation_result = kwargs.get('consolidation_result', {})
        optimization_result = kwargs.get('optimization_result', {})
        logger.info(f"巩固状态: {'成功' if consolidation_result.get('success') else '失败'}")
        logger.info(f"优化状态: {'成功' if optimization_result else '失败'}")
        
        # 性能统计
        performance_data = kwargs.get('performance_data', [])
        if performance_data:
            avg_health = sum(d['health_score'] for d in performance_data) / len(performance_data)
            avg_success_rate = sum(d['success_rate'] for d in performance_data) / len(performance_data)
            avg_memory_util = sum(d['memory_utilization'] for d in performance_data) / len(performance_data)
            
            logger.info(f"性能统计:")
            logger.info(f"  平均健康度: {avg_health:.3f}")
            logger.info(f"  平均成功率: {avg_success_rate:.2%}")
            logger.info(f"  平均内存使用率: {avg_memory_util:.2%}")
        
        # 最终系统状态
        final_status = kwargs.get('final_status', {})
        if final_status:
            uptime = final_status['system_info']['uptime_seconds']
            total_ops = final_status['performance_stats']['total_operations']
            logger.info(f"系统总运行时间: {uptime:.1f}秒")
            logger.info(f"总操作数: {total_ops}")
        
        logger.info("=" * 60)
        logger.info("演示完成!")
        logger.info("=" * 60)


async def quick_demo():
    """快速演示"""
    logger.info("开始快速演示...")
    
    # 创建配置
    config = MemoryInterfaceConfig(
        memory_dim=256,
        max_memories=1000,
        auto_consolidation=False,  # 关闭自动巩固以便快速演示
        auto_optimization=False,   # 关闭自动优化以便快速演示
        debug_mode=False
    )
    
    # 创建接口
    memory_interface = MemoryInterfaceCore(config)
    
    try:
        # 初始化
        await memory_interface.initialize()
        await memory_interface.start()
        
        # 存储几个记忆
        memory_ids = []
        for i in range(5):
            content = torch.randn(256)
            memory_id = await memory_interface.store_memory(
                content=content,
                memory_type=['episodic', 'semantic', 'procedural'][i % 3],
                importance=0.7 + 0.1 * i
            )
            memory_ids.append(memory_id)
        
        logger.info(f"存储了 {len(memory_ids)} 个记忆")
        
        # 检索记忆
        query = torch.randn(256)
        results = await memory_interface.retrieve_memory(
            query=query,
            top_k=3
        )
        
        logger.info(f"检索到 {len(results)} 个结果")
        
        # 获取状态
        status = memory_interface.get_system_status()
        health = status['system_info']['health_score']
        logger.info(f"系统健康度: {health:.3f}")
        
    finally:
        await memory_interface.stop()


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='记忆-计算接口系统演示')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='演示模式: quick=快速演示, full=完整演示')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        await quick_demo()
    else:
        demo = MemoryInterfaceDemo()
        await demo.run_full_demo()


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())
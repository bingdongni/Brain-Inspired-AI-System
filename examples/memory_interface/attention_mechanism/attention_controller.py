"""
注意力控制器 - 统一管理注意力机制组件
协调读写操作，提供外部接口
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import time
import threading
from dataclasses import dataclass
from collections import deque

from .attention_reader import AttentionReader
from .attention_writer import AttentionWriter
from .attention_memory import AttentionMemory, MemoryEntry


@dataclass
class AttentionConfig:
    """注意力系统配置"""
    memory_dim: int = 512
    hidden_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    max_memories: int = 10000
    importance_threshold: float = 0.7
    decay_rate: float = 0.95
    backup_size: int = 100


class AttentionController:
    """
    注意力控制器
    统一管理注意力机制的所有组件
    """
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        
        # 初始化核心组件
        self.attention_memory = AttentionMemory(
            memory_dim=config.memory_dim,
            max_memories=config.max_memories,
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            importance_threshold=config.importance_threshold,
            decay_rate=config.decay_rate,
            backup_size=config.backup_size
        )
        
        # 独立组件（可选使用）
        self.reader = AttentionReader(
            memory_dim=config.memory_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_memories=config.max_memories
        )
        
        self.writer = AttentionWriter(
            memory_dim=config.memory_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_memories=config.max_memories,
            importance_threshold=config.importance_threshold,
            decay_rate=config.decay_rate
        )
        
        # 系统状态
        self.is_initialized = False
        self.is_running = False
        
        # 任务队列
        self.task_queue = deque()
        self.result_cache = {}
        
        # 性能监控
        self.performance_metrics = {
            'total_queries': 0,
            'total_stores': 0,
            'average_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'memory_efficiency': 0.0
        }
        
        # 回调函数
        self.callbacks = {
            'on_store': [],
            'on_retrieve': [],
            'on_consolidate': [],
            'on_compress': []
        }
        
        # 线程锁
        self.lock = threading.RLock()
        
    def initialize(self) -> bool:
        """初始化注意力系统"""
        try:
            # 验证配置
            if not self._validate_config():
                return False
            
            # 重置状态
            self.is_initialized = True
            self.is_running = True
            
            # 启动后台任务
            self._start_background_tasks()
            
            return True
            
        except Exception as e:
            print(f"注意力系统初始化失败: {e}")
            return False
    
    def shutdown(self):
        """关闭注意力系统"""
        self.is_running = False
        
        # 清理缓存
        self.result_cache.clear()
        
        # 等待任务完成
        while self.task_queue:
            time.sleep(0.1)
    
    def store_memory(
        self,
        content: torch.Tensor,
        memory_type: str = 'episodic',
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        blocking: bool = True
    ) -> str:
        """
        存储记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            importance: 重要性分数
            metadata: 元数据
            blocking: 是否阻塞等待结果
            
        Returns:
            记忆ID
        """
        if not self.is_initialized:
            raise RuntimeError("注意力系统未初始化")
        
        def _store_task():
            memory_id = self.attention_memory.store(
                content, memory_type, importance, metadata
            )
            
            # 触发回调
            self._trigger_callbacks('on_store', memory_id, content, memory_type)
            
            return memory_id
        
        if blocking:
            with self.lock:
                return _store_task()
        else:
            task_id = f"task_{int(time.time() * 1000)}"
            self.task_queue.append((task_id, _store_task))
            return task_id
    
    def retrieve_memory(
        self,
        query: torch.Tensor,
        memory_types: Optional[List[str]] = None,
        top_k: int = 10,
        threshold: float = 0.5,
        use_cache: bool = True,
        blocking: bool = True
    ) -> List[Dict]:
        """
        检索记忆
        
        Args:
            query: 查询向量
            memory_types: 记忆类型过滤
            top_k: 返回数量
            threshold: 相关性阈值
            use_cache: 是否使用缓存
            blocking: 是否阻塞等待结果
            
        Returns:
            检索结果列表
        """
        if not self.is_initialized:
            raise RuntimeError("注意力系统未初始化")
        
        start_time = time.time()
        
        # 检查缓存
        cache_key = self._generate_cache_key(query, memory_types, top_k, threshold)
        if use_cache and cache_key in self.result_cache:
            self._update_cache_hit_rate()
            return self.result_cache[cache_key]
        
        def _retrieve_task():
            results = self.attention_memory.retrieve(
                query, memory_types, top_k, threshold
            )
            
            # 触发回调
            self._trigger_callbacks('on_retrieve', results, query)
            
            # 更新缓存
            if use_cache:
                with self.lock:
                    self.result_cache[cache_key] = results
                    # 清理过期缓存
                    if len(self.result_cache) > 100:
                        oldest_key = next(iter(self.result_cache))
                        del self.result_cache[oldest_key]
            
            return results
        
        if blocking:
            with self.lock:
                results = _retrieve_task()
        else:
            task_id = f"task_{int(time.time() * 1000)}"
            self.task_queue.append((task_id, _retrieve_task))
            return task_id
        
        # 更新性能指标
        response_time = time.time() - start_time
        self._update_response_time(response_time)
        self.performance_metrics['total_queries'] += 1
        
        return results
    
    def update_memory(
        self,
        memory_id: str,
        new_content: torch.Tensor,
        update_type: str = 'additive',
        blocking: bool = True
    ) -> bool:
        """更新记忆"""
        if not self.is_initialized:
            raise RuntimeError("注意力系统未初始化")
        
        def _update_task():
            success = self.attention_memory.update(memory_id, new_content, update_type)
            
            # 清理相关缓存
            if success:
                with self.lock:
                    keys_to_remove = []
                    for key in self.result_cache.keys():
                        if memory_id in key:
                            keys_to_remove.append(key)
                    for key in keys_to_remove:
                        del self.result_cache[key]
            
            return success
        
        if blocking:
            with self.lock:
                return _update_task()
        else:
            task_id = f"task_{int(time.time() * 1000)}"
            self.task_queue.append((task_id, _update_task))
            return task_id
    
    def consolidate_memories(
        self,
        consolidation_ratio: float = 0.3,
        auto_trigger: bool = False
    ) -> Dict:
        """
        巩固记忆
        
        Args:
            consolidation_ratio: 巩固比例
            auto_trigger: 是否自动触发
            
        Returns:
            巩固结果
        """
        if not self.is_initialized:
            raise RuntimeError("注意力系统未初始化")
        
        # 检查是否需要自动触发
        if auto_trigger and not self._should_consolidate():
            return {'status': 'consolidation_not_needed'}
        
        result = self.attention_memory.consolidate(consolidation_ratio)
        
        # 触发回调
        self._trigger_callbacks('on_consolidate', result)
        
        return result
    
    def compress_memories(
        self,
        compression_ratio: float = 0.5,
        force_compression: bool = False
    ) -> Dict:
        """
        压缩记忆
        
        Args:
            compression_ratio: 压缩比例
            force_compression: 是否强制压缩
            
        Returns:
            压缩结果
        """
        if not self.is_initialized:
            raise RuntimeError("注意力系统未初始化")
        
        # 检查是否需要压缩
        if not force_compression and not self._should_compress():
            return {'status': 'compression_not_needed'}
        
        result = self.attention_memory.compress(compression_ratio)
        
        # 触发回调
        self._trigger_callbacks('on_compress', result)
        
        return result
    
    def get_memory_state(self) -> Dict:
        """获取记忆系统状态"""
        if not self.is_initialized:
            return {'status': 'not_initialized'}
        
        with self.lock:
            stats = self.attention_memory.get_statistics()
            analysis = self.attention_memory.get_memory_analysis()
            
            return {
                'system_stats': stats,
                'memory_analysis': analysis,
                'performance_metrics': self.performance_metrics,
                'cache_info': {
                    'cache_size': len(self.result_cache),
                    'task_queue_size': len(self.task_queue)
                },
                'is_running': self.is_running
            }
    
    def register_callback(self, event_type: str, callback: Callable):
        """注册回调函数"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def batch_store(
        self,
        contents: List[torch.Tensor],
        memory_types: Optional[List[str]] = None,
        importances: Optional[List[float]] = None,
        metadata_list: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        批量存储记忆
        
        Args:
            contents: 内容列表
            memory_types: 记忆类型列表
            importances: 重要性列表
            metadata_list: 元数据列表
            
        Returns:
            记忆ID列表
        """
        if not self.is_initialized:
            raise RuntimeError("注意力系统未初始化")
        
        memory_ids = []
        
        for i, content in enumerate(contents):
            memory_type = memory_types[i] if memory_types else 'episodic'
            importance = importances[i] if importances else None
            metadata = metadata_list[i] if metadata_list else None
            
            memory_id = self.store_memory(content, memory_type, importance, metadata)
            memory_ids.append(memory_id)
        
        return memory_ids
    
    def batch_retrieve(
        self,
        queries: List[torch.Tensor],
        memory_types: Optional[List[str]] = None,
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List[List[Dict]]:
        """
        批量检索记忆
        
        Args:
            queries: 查询列表
            memory_types: 记忆类型过滤
            top_k: 返回数量
            threshold: 相关性阈值
            
        Returns:
            检索结果列表
        """
        results = []
        
        for query in queries:
            result = self.retrieve_memory(query, memory_types, top_k, threshold)
            results.append(result)
        
        return results
    
    def clear_cache(self):
        """清理缓存"""
        with self.lock:
            self.result_cache.clear()
    
    def optimize_system(self) -> Dict:
        """系统优化"""
        if not self.is_initialized:
            raise RuntimeError("注意力系统未初始化")
        
        optimization_results = {}
        
        # 记忆巩固
        consolidation_result = self.consolidate_memories(consolidation_ratio=0.3)
        optimization_results['consolidation'] = consolidation_result
        
        # 记忆压缩（如果需要）
        if self._should_compress():
            compression_result = self.compress_memories(force_compression=False)
            optimization_results['compression'] = compression_result
        
        # 缓存清理
        cache_size_before = len(self.result_cache)
        self.clear_cache()
        optimization_results['cache_cleared'] = cache_size_before
        
        # 统计更新
        memory_stats = self.attention_memory.get_statistics()
        optimization_results['memory_utilization'] = memory_stats.get('memory_utilization', 0)
        
        return optimization_results
    
    def _validate_config(self) -> bool:
        """验证配置"""
        required_fields = ['memory_dim', 'hidden_dim', 'num_heads', 'max_memories']
        return all(hasattr(self.config, field) for field in required_fields)
    
    def _start_background_tasks(self):
        """启动后台任务"""
        def task_processor():
            while self.is_running:
                if self.task_queue:
                    try:
                        task_id, task_func = self.task_queue.popleft()
                        result = task_func()
                        # 可以将结果存储到某个地方
                    except Exception as e:
                        print(f"任务处理错误: {e}")
                else:
                    time.sleep(0.01)
        
        # 启动后台线程
        background_thread = threading.Thread(target=task_processor, daemon=True)
        background_thread.start()
    
    def _generate_cache_key(self, query: torch.Tensor, memory_types: Optional[List[str]], 
                           top_k: int, threshold: float) -> str:
        """生成缓存键"""
        query_hash = str(hash(query.cpu().numpy().tobytes()))
        types_str = str(sorted(memory_types)) if memory_types else "all"
        return f"{query_hash}_{types_str}_{top_k}_{threshold}"
    
    def _update_response_time(self, response_time: float):
        """更新响应时间统计"""
        current_avg = self.performance_metrics['average_response_time']
        total_queries = self.performance_metrics['total_queries']
        
        # 移动平均
        self.performance_metrics['average_response_time'] = (
            (current_avg * (total_queries - 1) + response_time) / total_queries
        )
    
    def _update_cache_hit_rate(self):
        """更新缓存命中率"""
        cache_hits = self.performance_metrics.get('cache_hits', 0)
        cache_hits += 1
        self.performance_metrics['cache_hits'] = cache_hits
        
        total_requests = self.performance_metrics['total_queries']
        if total_requests > 0:
            self.performance_metrics['cache_hit_rate'] = cache_hits / total_requests
    
    def _should_consolidate(self) -> bool:
        """判断是否需要巩固"""
        stats = self.attention_memory.get_statistics()
        utilization = stats.get('memory_utilization', 0)
        return utilization > 0.8
    
    def _should_compress(self) -> bool:
        """判断是否需要压缩"""
        stats = self.attention_memory.get_statistics()
        utilization = stats.get('memory_utilization', 0)
        return utilization > 0.9
    
    def _trigger_callbacks(self, event_type: str, *args):
        """触发回调函数"""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(*args)
            except Exception as e:
                print(f"回调函数错误 ({event_type}): {e}")
    
    def save_state(self, filepath: str):
        """保存系统状态"""
        import pickle
        
        state = {
            'config': self.config,
            'memory_bank': self.attention_memory.memory_bank,
            'memory_entries': self.attention_memory.memory_entries,
            'current_size': self.attention_memory.current_size,
            'performance_metrics': self.performance_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath: str):
        """加载系统状态"""
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        self.attention_memory.memory_bank = state['memory_bank']
        self.attention_memory.memory_entries = state['memory_entries']
        self.attention_memory.current_size = state['current_size']
        self.performance_metrics = state['performance_metrics']


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建配置
    config = AttentionConfig(
        memory_dim=512,
        hidden_dim=512,
        num_heads=8,
        max_memories=5000
    )
    
    # 创建控制器
    controller = AttentionController(config)
    
    # 初始化
    if controller.initialize():
        print("注意力系统初始化成功")
        
        # 创建测试数据
        test_content = torch.randn(512)
        
        # 存储记忆
        memory_id = controller.store_memory(
            test_content, 
            memory_type='episodic', 
            importance=0.8
        )
        print(f"存储记忆 ID: {memory_id}")
        
        # 检索记忆
        query = torch.randn(512)
        results = controller.retrieve_memory(query, top_k=5)
        print(f"检索到 {len(results)} 个记忆")
        
        # 获取系统状态
        state = controller.get_memory_state()
        print(f"系统状态: {state['system_stats']}")
        
        # 优化系统
        optimization = controller.optimize_system()
        print(f"优化结果: {optimization}")
        
        # 关闭系统
        controller.shutdown()
        print("注意力系统已关闭")
    else:
        print("注意力系统初始化失败")
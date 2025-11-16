#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能优化工具模块
===============

提供系统性的性能优化工具，包括：
- 智能缓存机制
- 内存管理优化
- 并发处理工具
- 性能监控
- 自动优化建议

作者: Brain-Inspired AI Team
创建时间: 2025-11-16
"""

import time
import threading
import psutil
import gc
import functools
import weakref
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import logging
import cProfile
import pstats
from io import StringIO


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0  # MB
    execution_time: float = 0.0  # seconds
    throughput: float = 0.0  # operations per second
    timestamp: float = field(default_factory=time.time)
    function_name: str = ""
    operation_count: int = 0


class SmartLRUCache:
    """线程安全的LRU缓存实现"""
    
    def __init__(self, maxsize: int = 128, ttl: int = 3600):
        """
        初始化智能LRU缓存
        
        Args:
            maxsize: 最大缓存项数
            ttl: 缓存过期时间（秒）
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = {}  # key -> (value, timestamp)
        self.order = deque()  # 记录访问顺序
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                
                # 检查是否过期
                if time.time() - timestamp < self.ttl:
                    # 移动到末尾（最近使用）
                    self.order.remove(key)
                    self.order.append(key)
                    self.hits += 1
                    return value
                else:
                    # 已过期，删除
                    del self.cache[key]
                    self.order.remove(key)
            
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存值"""
        with self.lock:
            ttl = ttl or self.ttl
            
            if key in self.cache:
                # 更新现有项
                self.order.remove(key)
            elif len(self.cache) >= self.maxsize:
                # 移除最旧的项
                oldest_key = self.order.popleft()
                del self.cache[oldest_key]
            
            self.cache[key] = (value, time.time() + ttl)
            self.order.append(key)
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.order.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'size': len(self.cache),
                'maxsize': self.maxsize
            }


class MemoryPool:
    """内存池实现，优化频繁的小对象分配"""
    
    def __init__(self, block_size: int = 1024, max_blocks: int = 100):
        """
        初始化内存池
        
        Args:
            block_size: 每个内存块的大小（字节）
            max_blocks: 最大块数
        """
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.free_blocks = []
        self.allocated_blocks = set()
        self.lock = threading.RLock()
    
    def allocate(self) -> bytearray:
        """分配内存块"""
        with self.lock:
            # 尝试从空闲块中获取
            if self.free_blocks:
                block = self.free_blocks.pop()
            elif len(self.allocated_blocks) < self.maxblocks:
                # 创建新块
                block = bytearray(self.block_size)
                self.allocated_blocks.add(id(block))
            else:
                raise MemoryError("内存池已满")
            
            return block
    
    def deallocate(self, block: bytearray):
        """释放内存块"""
        with self.lock:
            if id(block) in self.allocated_blocks:
                self.allocated_blocks.remove(id(block))
                # 清零块内容
                block[:] = b'\x00' * len(block)
                self.free_blocks.append(block)
    
    def get_stats(self) -> Dict[str, int]:
        """获取内存池统计信息"""
        with self.lock:
            return {
                'allocated_blocks': len(self.allocated_blocks),
                'free_blocks': len(self.free_blocks),
                'total_capacity': self.max_blocks,
                'utilization': len(self.allocated_blocks) / self.max_blocks
            }


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.active_monitors: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
    
    def start_monitor(self, name: str, interval: float = 1.0):
        """启动性能监控"""
        with self.lock:
            if name in self.active_monitors:
                logger.warning(f"监控器 {name} 已存在")
                return
            
            monitor_data = {
                'start_time': time.time(),
                'interval': interval,
                'last_measure': 0,
                'metrics': []
            }
            self.active_monitors[name] = monitor_data
    
    def stop_monitor(self, name: str) -> Optional[List[PerformanceMetrics]]:
        """停止性能监控"""
        with self.lock:
            if name not in self.active_monitors:
                return None
            
            monitor_data = self.active_monitors.pop(name)
            return monitor_data['metrics']
    
    def measure_function(self, func: Callable, *args, **kwargs) -> Any:
        """测量函数执行性能"""
        # 获取系统状态
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        start_memory = psutil.virtual_memory()
        
        try:
            # 执行函数
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            logger.error(f"函数 {func.__name__} 执行失败: {e}")
            raise
        finally:
            # 计算性能指标
            end_time = time.time()
            end_cpu = psutil.cpu_percent()
            end_memory = psutil.virtual_memory()
            
            metrics = PerformanceMetrics(
                cpu_usage=max(0, end_cpu - start_cpu),
                memory_usage=(end_memory.used - start_memory.used) / (1024 * 1024),  # MB
                execution_time=end_time - start_time,
                throughput=(1 if success else 0),
                function_name=func.__name__
            )
            
            with self.lock:
                self.metrics_history.append(metrics)
        
        return result
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """使用cProfile分析函数性能"""
        profiler = cProfile.Profile()
        
        try:
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            
            # 收集分析结果
            stats_stream = StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.strip_dirs()
            stats.sort_stats('cumulative')
            stats.print_stats(10)  # 显示前10个最耗时的函数
            
            profile_output = stats_stream.getvalue()
            
            return {
                'result': result,
                'profile_output': profile_output,
                'stats': stats
            }
        except Exception as e:
            logger.error(f"性能分析失败: {e}")
            raise
    
    def get_average_metrics(self, function_name: Optional[str] = None) -> Optional[PerformanceMetrics]:
        """获取平均性能指标"""
        with self.lock:
            if not self.metrics_history:
                return None
            
            # 筛选指定函数的指标
            filtered_metrics = self.metrics_history
            if function_name:
                filtered_metrics = [m for m in self.metrics_history if m.function_name == function_name]
            
            if not filtered_metrics:
                return None
            
            # 计算平均值
            avg_metrics = PerformanceMetrics(
                cpu_usage=sum(m.cpu_usage for m in filtered_metrics) / len(filtered_metrics),
                memory_usage=sum(m.memory_usage for m in filtered_metrics) / len(filtered_metrics),
                execution_time=sum(m.execution_time for m in filtered_metrics) / len(filtered_metrics),
                throughput=sum(m.throughput for m in filtered_metrics) / len(filtered_metrics),
                function_name=function_name or "all_functions"
            )
            
            return avg_metrics
    
    def get_system_metrics(self) -> Dict[str, float]:
        """获取当前系统性能指标"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_mb': memory_info.rss / (1024 * 1024),
                'memory_available_mb': psutil.virtual_memory().available / (1024 * 1024),
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'network_io_count': psutil.net_io_counters().packets_recv if psutil.net_io_counters() else 0
            }
        except Exception as e:
            logger.error(f"获取系统指标失败: {e}")
            return {}


class AsyncBatchProcessor:
    """异步批处理器"""
    
    def __init__(self, max_workers: int = 4, batch_size: int = 100):
        """
        初始化异步批处理器
        
        Args:
            max_workers: 最大工作线程数
            batch_size: 批处理大小
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_batch(self, items: List[Any], processor_func: Callable, 
                          progress_callback: Optional[Callable] = None) -> List[Any]:
        """批量异步处理项目"""
        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(items))
            batch = items[start_idx:end_idx]
            
            # 创建任务
            tasks = []
            for item in batch:
                task = asyncio.create_task(
                    self._process_single_item(processor_func, item)
                )
                tasks.append(task)
            
            # 等待批次完成
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for result in batch_results:
                if not isinstance(result, Exception):
                    results.append(result)
            
            # 进度回调
            if progress_callback:
                progress = (batch_idx + 1) / total_batches * 100
                progress_callback(progress)
        
        return results
    
    async def _process_single_item(self, processor_func: Callable, item: Any) -> Any:
        """异步处理单个项目"""
        loop = asyncio.get_event_loop()
        
        def run_in_thread():
            return processor_func(item)
        
        return await loop.run_in_executor(self.executor, run_in_thread)
    
    def shutdown(self):
        """关闭处理器"""
        self.executor.shutdown(wait=True)


def smart_cache(cache_name: str, maxsize: int = 128, ttl: int = 3600):
    """智能缓存装饰器"""
    cache = SmartLRUCache(maxsize=maxsize, ttl=ttl)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 创建缓存键
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # 尝试从缓存获取
            result = cache.get(cache_key)
            if result is not None:
                logger.debug(f"缓存命中: {cache_key}")
                return result
            
            # 执行函数并缓存结果
            logger.debug(f"缓存未命中，执行函数: {cache_key}")
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            
            return result
        
        # 添加缓存管理方法
        wrapper.cache = cache
        wrapper.clear_cache = cache.clear
        wrapper.get_cache_stats = cache.get_stats
        
        return wrapper
    
    return decorator


def measure_performance(include_memory: bool = True, include_cpu: bool = True):
    """性能测量装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = PerformanceMonitor()
            return monitor.measure_function(func, *args, **kwargs)
        return wrapper
    return decorator


class OptimizedListProcessor:
    """优化的列表处理器"""
    
    @staticmethod
    def vectorized_operation(data: np.ndarray, operation: str = 'multiply', factor: float = 2.0) -> np.ndarray:
        """向量化操作，比循环快10-100倍"""
        if operation == 'multiply':
            return data * factor
        elif operation == 'add':
            return data + factor
        elif operation == 'square':
            return data ** 2
        elif operation == 'sqrt':
            return np.sqrt(data)
        else:
            raise ValueError(f"不支持的操作: {operation}")
    
    @staticmethod
    def efficient_iteration(data: list, func: Callable) -> list:
        """使用列表推导式替代循环"""
        return [func(item) for item in data]
    
    @staticmethod
    def parallel_processing(data: list, func: Callable, num_workers: int = 4) -> list:
        """并行处理列表"""
        if len(data) < 100:  # 小数据集不并行处理
            return [func(item) for item in data]
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(func, data))
        return results
    
    @staticmethod
    def chunked_processing(data: list, chunk_size: int = 1000) -> List[list]:
        """将大列表分块处理"""
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


# 全局性能监控实例
_global_monitor = PerformanceMonitor()


def get_global_monitor() -> PerformanceMonitor:
    """获取全局性能监控器实例"""
    return _global_monitor


# 示例使用
if __name__ == "__main__":
    # 性能测试
    import random
    
    # 测试缓存
    @smart_cache(maxsize=10, ttl=60)
    def expensive_computation(x):
        """模拟耗时计算"""
        time.sleep(0.1)  # 模拟计算时间
        return x ** 2
    
    # 测试数据
    test_data = list(range(1000))
    
    # 性能测试
    monitor = get_global_monitor()
    
    # 测试函数性能
    result = monitor.measure_function(expensive_computation, 42)
    print(f"计算结果: {result}")
    print(f"性能指标: {monitor.get_average_metrics()}")
    
    # 测试缓存效果
    print("\n=== 缓存测试 ===")
    for i in range(5):
        start_time = time.time()
        result = expensive_computation(i)
        end_time = time.time()
        print(f"第{i+1}次调用，耗时: {end_time - start_time:.4f}秒")
    
    print(f"缓存统计: {expensive_computation.get_cache_stats()}")
    
    # 测试内存池
    print("\n=== 内存池测试 ===")
    pool = MemoryPool(block_size=1024, max_blocks=10)
    
    # 分配和释放内存块
    blocks = []
    for i in range(5):
        block = pool.allocate()
        blocks.append(block)
        print(f"分配块 {i+1}, 统计: {pool.get_stats()}")
    
    # 释放部分块
    for i in range(3):
        pool.deallocate(blocks[i])
        print(f"释放块 {i+1}, 统计: {pool.get_stats()}")
    
    # 测试向量化操作
    print("\n=== 向量化操作测试 ===")
    large_array = np.random.random(100000)
    
    start_time = time.time()
    result = OptimizedListProcessor.vectorized_operation(large_array, 'multiply', 2.0)
    vectorized_time = time.time() - start_time
    print(f"向量化操作耗时: {vectorized_time:.4f}秒")
    
    start_time = time.time()
    manual_result = [x * 2.0 for x in large_array]
    manual_time = time.time() - start_time
    print(f"手动循环耗时: {manual_time:.4f}秒")
    print(f"向量化加速比: {manual_time / vectorized_time:.2f}x")
    
    print("\n=== 系统性能指标 ===")
    system_metrics = monitor.get_system_metrics()
    for key, value in system_metrics.items():
        print(f"{key}: {value:.2f}")

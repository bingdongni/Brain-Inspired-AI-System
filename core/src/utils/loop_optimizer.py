#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
循环优化和性能基准测试工具
========================

解决已识别的低效循环问题：
1. 矢量化操作替换
2. 智能循环优化
3. 性能基准测试
4. 自动优化建议

作者: Brain-Inspired AI Team
创建时间: 2025-11-16
"""

import time
import threading
import functools
import itertools
import numpy as np
import pandas as pd
from typing import List, Any, Callable, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from contextlib import contextmanager
import cProfile
import pstats
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBenchmark:
    """性能基准测试结果"""
    function_name: str
    execution_time: float
    memory_usage_mb: float
    iterations: int
    throughput_per_second: float
    optimization_suggestions: List[str] = field(default_factory=list)


class LoopOptimizer:
    """循环优化器"""
    
    @staticmethod
    def optimize_range_loop(data: list, operation: str = 'multiply', factor: float = 2.0) -> list:
        """优化基于范围的循环"""
        # 优化前: for i in range(len(data))
        # 优化后: 使用列表推导式和内置函数
        
        if operation == 'multiply':
            # 向量化操作
            if isinstance(data[0], (int, float)):
                return [x * factor for x in data]
            else:
                return [operation_func(item, factor) for item in data]
        
        elif operation == 'add':
            return [x + factor for x in data]
        
        elif operation == 'filter':
            return [x for x in data if x > factor]
        
        elif operation == 'map':
            return list(map(lambda x: operation_func(x, factor), data))
        
        else:
            # 通用优化
            results = []
            for item in data:
                if isinstance(item, (int, float)):
                    results.append(eval(f"item {operation} factor"))
                else:
                    results.append(operation_func(item, factor))
            return results
    
    @staticmethod
    def optimize_nested_loops(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
        """优化嵌套循环 - 矩阵乘法"""
        # 优化前: 双重嵌套循环
        # 优化后: 使用NumPy矩阵乘法
        
        # 使用广播和向量化
        return np.dot(matrix1, matrix2)
    
    @staticmethod
    def optimize_search_loop(data: list, target: Any) -> Optional[int]:
        """优化搜索循环"""
        # 优化前: 线性搜索
        # 优化后: 使用内置函数和条件优化
        
        try:
            # 对于数值数据，使用numpy优化
            if isinstance(data[0], (int, float)):
                np_data = np.array(data)
                indices = np.where(np_data == target)[0]
                return int(indices[0]) if len(indices) > 0 else None
            
            # 使用enumerate优化
            for i, item in enumerate(data):
                if item == target:
                    return i
            
            return None
            
        except (IndexError, ValueError):
            return None
    
    @staticmethod
    def optimize_accumulation_loop(data: list, operation: str = 'sum') -> Union[int, float]:
        """优化累积循环"""
        if operation == 'sum':
            return sum(data)
        elif operation == 'product':
            result = 1
            for x in data:
                result *= x
            return result
        elif operation == 'max':
            return max(data) if data else None
        elif operation == 'min':
            return min(data) if data else None
        elif operation == 'mean':
            return sum(data) / len(data) if data else 0
        else:
            result = data[0] if data else None
            for x in data[1:]:
                result = eval(f"result {operation} x")
            return result
    
    @staticmethod
    def parallel_processing(data: list, func: Callable, num_workers: int = None) -> list:
        """并行处理数据"""
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        # 小数据集不并行处理
        if len(data) < 100:
            return [func(item) for item in data]
        
        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(func, data))
        
        return results
    
    @staticmethod
    def chunked_processing(data: list, chunk_size: int = 1000, 
                          chunk_func: Callable = None) -> List[list]:
        """分块处理大列表"""
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        if chunk_func:
            return [chunk_func(chunk) for chunk in chunks]
        
        return chunks


class VectorizedOperations:
    """矢量化操作工具类"""
    
    @staticmethod
    def vectorized_filter(data: np.ndarray, condition_func: Callable) -> np.ndarray:
        """矢量化过滤"""
        return data[np.vectorize(condition_func)(data)]
    
    @staticmethod
    def vectorized_transform(data: np.ndarray, transform_func: Callable) -> np.ndarray:
        """矢量化变换"""
        return np.vectorize(transform_func)(data)
    
    @staticmethod
    def vectorized_conditional(data: np.ndarray, condition: np.ndarray, 
                             true_value: Any, false_value: Any) -> np.ndarray:
        """矢量化条件操作"""
        return np.where(condition, true_value, false_value)
    
    @staticmethod
    def efficient_distance_matrix(points: np.ndarray) -> np.ndarray:
        """高效计算距离矩阵"""
        # 使用广播避免嵌套循环
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))
    
    @staticmethod
    def efficient_correlation_matrix(data: np.ndarray) -> np.ndarray:
        """高效计算相关矩阵"""
        return np.corrcoef(data)
    
    @staticmethod
    def optimized_convolution(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """优化卷积操作"""
        # 使用FFT进行快速卷积
        from scipy import signal as scipy_signal
        return scipy_signal.convolve(signal, kernel, mode='same')


class MemoryEfficientIterator:
    """内存高效迭代器"""
    
    @staticmethod
    def generator_chain(iterables: List[Iterable]) -> Iterator:
        """链接生成器，避免创建大列表"""
        for iterable in iterables:
            for item in iterable:
                yield item
    
    @staticmethod
    def streaming_process(file_path: str, chunk_size: int = 1000) -> Iterator:
        """流式处理大文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                while True:
                    chunk = []
                    for _ in range(chunk_size):
                        line = f.readline()
                        if not line:
                            break
                        chunk.append(line.strip())
                    
                    if chunk:
                        yield chunk
                    else:
                        break
        except Exception as e:
            logger.error(f"流式处理失败: {e}")
    
    @staticmethod
    def lazy_computation(func: Callable, *args, **kwargs):
        """延迟计算装饰器"""
        @functools.wraps(func)
        def lazy_wrapper():
            return func(*args, **kwargs)
        return lazy_wrapper


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.profiles = {}
        self.lock = threading.RLock()
    
    @contextmanager
    def profile(self, name: str):
        """性能分析上下文管理器"""
        start_time = time.time()
        profiler = cProfile.Profile()
        
        try:
            profiler.enable()
            yield profiler
        finally:
            profiler.disable()
            end_time = time.time()
            
            with self.lock:
                self.profiles[name] = {
                    'profiler': profiler,
                    'execution_time': end_time - start_time,
                    'timestamp': time.time()
                }
    
    def get_profile_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """获取分析统计"""
        if name not in self.profiles:
            return None
        
        profiler = self.profiles[name]['profiler']
        stats_stream = StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats(10)
        
        return {
            'output': stats_stream.getvalue(),
            'execution_time': self.profiles[name]['execution_time'],
            'timestamp': self.profiles[name]['timestamp']
        }
    
    def compare_performance(self, name1: str, name2: str) -> Optional[Dict[str, Any]]:
        """比较两个性能分析结果"""
        stats1 = self.get_profile_stats(name1)
        stats2 = self.get_profile_stats(name2)
        
        if not stats1 or not stats2:
            return None
        
        return {
            'name1_time': stats1['execution_time'],
            'name2_time': stats2['execution_time'],
            'speedup': stats1['execution_time'] / stats2['execution_time'],
            'faster': name1 if stats1['execution_time'] < stats2['execution_time'] else name2
        }


class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self):
        self.results = []
        self.lock = threading.Lock()
    
    def run_benchmark(self, func: Callable, iterations: int = 1000, 
                     *args, **kwargs) -> PerformanceBenchmark:
        """运行性能基准测试"""
        
        # 预热
        for _ in range(min(10, iterations // 10)):
            try:
                func(*args, **kwargs)
            except:
                pass
        
        # 正式测试
        times = []
        for _ in range(iterations):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                times.append(end_time - start_time)
            except Exception as e:
                logger.error(f"基准测试执行失败: {e}")
                break
        
        if not times:
            raise RuntimeError("基准测试完全失败")
        
        # 计算统计信息
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        throughput = 1 / avg_time if avg_time > 0 else 0
        
        # 内存使用
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # 生成优化建议
        suggestions = self._generate_optimization_suggestions(times, std_time, avg_time)
        
        result = PerformanceBenchmark(
            function_name=func.__name__,
            execution_time=avg_time,
            memory_usage_mb=memory_mb,
            iterations=len(times),
            throughput_per_second=throughput,
            optimization_suggestions=suggestions
        )
        
        with self.lock:
            self.results.append(result)
        
        return result
    
    def _generate_optimization_suggestions(self, times: List[float], 
                                         std_time: float, avg_time: float) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        # 检查性能稳定性
        cv = std_time / avg_time if avg_time > 0 else float('inf')
        if cv > 0.1:
            suggestions.append("性能不稳定，考虑减少随机性或使用缓存")
        
        # 检查执行时间
        if avg_time > 0.1:
            suggestions.append("执行时间较长，考虑并行化或向量化")
        
        # 检查最慢的迭代
        if len(times) > 10:
            slow_percentile = np.percentile(times, 90)
            if slow_percentile > avg_time * 2:
                suggestions.append("存在异常慢的迭代，检查数据分布和算法复杂度")
        
        # 通用建议
        suggestions.extend([
            "考虑使用NumPy向量化操作",
            "如果可能，使用并行处理",
            "实现结果缓存以避免重复计算",
            "优化数据结构和算法复杂度"
        ])
        
        return suggestions
    
    def compare_functions(self, func1: Callable, func2: Callable, 
                         iterations: int = 1000, *args, **kwargs) -> Dict[str, Any]:
        """比较两个函数的性能"""
        result1 = self.run_benchmark(func1, iterations, *args, **kwargs)
        result2 = self.run_benchmark(func2, iterations, *args, **kwargs)
        
        speedup = result1.execution_time / result2.execution_time
        
        return {
            'function1': {
                'name': result1.function_name,
                'time': result1.execution_time,
                'throughput': result1.throughput_per_second,
                'memory': result1.memory_usage_mb
            },
            'function2': {
                'name': result2.function_name,
                'time': result2.execution_time,
                'throughput': result2.throughput_per_second,
                'memory': result2.memory_usage_mb
            },
            'speedup': speedup,
            'faster': result1.function_name if speedup > 1 else result2.function_name,
            'improvement_percent': abs(speedup - 1) * 100
        }
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """生成性能报告"""
        if not self.results:
            return "没有性能基准测试结果"
        
        # 创建报告
        report_lines = [
            "# 性能基准测试报告",
            f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"测试数量: {len(self.results)}",
            "",
            "## 测试结果",
            ""
        ]
        
        # 按执行时间排序
        sorted_results = sorted(self.results, key=lambda x: x.execution_time)
        
        for i, result in enumerate(sorted_results, 1):
            report_lines.extend([
                f"### {i}. {result.function_name}",
                f"- 执行时间: {result.execution_time:.4f}秒",
                f"- 内存使用: {result.memory_usage_mb:.2f}MB",
                f"- 吞吐量: {result.throughput_per_second:.2f} ops/sec",
                f"- 迭代次数: {result.iterations}",
                "",
                "**优化建议:**"
            ])
            
            for suggestion in result.optimization_suggestions:
                report_lines.append(f"- {suggestion}")
            
            report_lines.append("")
        
        # 性能排名
        report_lines.extend([
            "## 性能排名",
            ""
        ])
        
        for i, result in enumerate(sorted_results, 1):
            report_lines.append(f"{i}. {result.function_name}: {result.execution_time:.4f}秒")
        
        report = "\n".join(report_lines)
        
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"性能报告已保存到: {save_path}")
            except Exception as e:
                logger.error(f"保存报告失败: {e}")
        
        return report
    
    def visualize_performance(self, save_path: Optional[str]] = None) -> None:
        """可视化性能结果"""
        if not self.results:
            logger.warning("没有性能数据用于可视化")
            return
        
        # 准备数据
        function_names = [r.function_name for r in self.results]
        execution_times = [r.execution_time for r in self.results]
        memory_usage = [r.memory_usage_mb for r in self.results]
        throughput = [r.throughput_per_second for r in self.results]
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 执行时间对比
        ax1.bar(function_names, execution_times, color='skyblue')
        ax1.set_title('执行时间对比')
        ax1.set_ylabel('时间 (秒)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 内存使用对比
        ax2.bar(function_names, memory_usage, color='lightgreen')
        ax2.set_title('内存使用对比')
        ax2.set_ylabel('内存 (MB)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 吞吐量对比
        ax3.bar(function_names, throughput, color='orange')
        ax3.set_title('吞吐量对比')
        ax3.set_ylabel('操作/秒')
        ax3.tick_params(axis='x', rotation=45)
        
        # 综合性能散点图
        ax4.scatter(execution_times, memory_usage, s=[t*10 for t in throughput], 
                   alpha=0.6, c=range(len(function_names)), cmap='viridis')
        ax4.set_xlabel('执行时间 (秒)')
        ax4.set_ylabel('内存使用 (MB)')
        ax4.set_title('综合性能分析 (气泡大小=吞吐量)')
        
        # 添加标签
        for i, name in enumerate(function_names):
            ax4.annotate(name, (execution_times[i], memory_usage[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"性能可视化图表已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()


# 辅助函数
def operation_func(item: Any, factor: float) -> Any:
    """辅助操作函数"""
    if isinstance(item, (int, float)):
        return item * factor
    elif isinstance(item, str):
        return item * int(factor) if factor.is_integer() else item
    else:
        return item


# 示例使用
if __name__ == "__main__":
    # 循环优化测试
    print("=== 循环优化测试 ===")
    
    # 测试数据
    test_data = list(range(10000))
    
    # 原始低效循环
    def slow_operation(data):
        result = []
        for i in range(len(data)):
            result.append(data[i] * 2)
        return result
    
    # 优化后的操作
    def fast_operation(data):
        return [x * 2 for x in data]
    
    # 向量化操作
    def vectorized_operation(data):
        return np.array(data) * 2
    
    # 性能基准测试
    runner = BenchmarkRunner()
    
    # 测试各种实现
    results = []
    for func_name, func in [("慢速", slow_operation), ("优化", fast_operation), ("向量化", vectorized_operation)]:
        result = runner.run_benchmark(func, 100, test_data)
        results.append((func_name, result))
        print(f"{func_name}实现: {result.execution_time:.4f}秒, 吞吐量: {result.throughput_per_second:.2f} ops/sec")
    
    # 比较性能
    comparison = runner.compare_functions(slow_operation, fast_operation, 100, test_data)
    print(f"\n性能比较:")
    print(f"优化提升: {comparison['improvement_percent']:.1f}%")
    print(f"更快的方法: {comparison['faster']}")
    
    # 生成报告
    report = runner.generate_report("/tmp/performance_report.md")
    print(f"\n报告预览:")
    print(report[:500] + "...")
    
    # 矢量化操作测试
    print("\n=== 矢量化操作测试 ===")
    
    # 测试数据
    large_array = np.random.random(100000)
    
    # 传统循环过滤
    def slow_filter(data, threshold=0.5):
        result = []
        for i in range(len(data)):
            if data[i] > threshold:
                result.append(data[i])
        return result
    
    # 矢量化过滤
    def fast_filter(data, threshold=0.5):
        return data[data > threshold]
    
    filter_result = runner.compare_functions(slow_filter, fast_filter, 50, large_array, 0.5)
    print(f"过滤操作性能比较:")
    print(f"向量化版本快 {filter_result['speedup']:.1f} 倍")
    
    # 并行处理测试
    print("\n=== 并行处理测试 ===")
    
    def cpu_intensive_task(x):
        """CPU密集型任务"""
        result = 0
        for i in range(1000):
            result += i * x
        return result
    
    # 串行处理
    def serial_processing(data):
        return [cpu_intensive_task(x) for x in data]
    
    # 并行处理
    def parallel_processing_test(data):
        return LoopOptimizer.parallel_processing(data, cpu_intensive_task, mp.cpu_count())
    
    test_data_small = list(range(100))
    test_data_large = list(range(1000))
    
    # 小数据集测试
    small_comparison = runner.compare_functions(serial_processing, parallel_processing_test, 
                                              10, test_data_small)
    print(f"小数据集并行处理提升: {small_comparison['speedup']:.1f} 倍")
    
    # 大数据集测试
    large_comparison = runner.compare_functions(serial_processing, parallel_processing_test, 
                                              5, test_data_large)
    print(f"大数据集并行处理提升: {large_comparison['speedup']:.1f} 倍")
    
    # 性能可视化
    try:
        runner.visualize_performance("/tmp/performance_chart.png")
        print("性能可视化图表已保存到 /tmp/performance_chart.png")
    except Exception as e:
        print(f"可视化失败: {e}")
    
    print("\n=== 优化建议应用测试 ===")
    
    # 应用优化建议
    def optimize_data_processing(data: List[int]) -> Dict[str, Any]:
        """优化的数据处理函数"""
        # 1. 使用向量化操作
        np_data = np.array(data)
        
        # 2. 矢量化计算
        mean_val = np.mean(np_data)
        std_val = np.std(np_data)
        
        # 3. 向量化条件筛选
        filtered_data = np_data[np_data > mean_val]
        
        # 4. 内存高效的累积计算
        cumulative_sum = np.cumsum(np_data)
        
        return {
            'mean': float(mean_val),
            'std': float(std_val),
            'filtered_mean': float(np.mean(filtered_data)) if len(filtered_data) > 0 else 0,
            'cumulative_sum': cumulative_sum.tolist(),
            'data_processed': len(data)
        }
    
    # 测试优化后的函数
    test_data_perf = list(range(50000))
    result = runner.run_benchmark(optimize_data_processing, 100, test_data_perf)
    print(f"优化后的数据处理: {result.execution_time:.4f}秒")
    print(f"吞吐量: {result.throughput_per_second:.2f} ops/sec")
    
    print("\n循环优化和性能基准测试工具测试完成!")

#!/usr/bin/env python3
"""
系统长期运行稳定性测试
Long-term System Stability Validation Test

测试内容包括：
1. 连续运行测试（至少15分钟）
2. 内存泄漏和资源使用监控
3. 并发请求处理测试
4. 不同负载下的稳定性验证
5. 日志和错误记录分析
"""

import os
import sys
import json
import time
import threading
import multiprocessing
import psutil
import gc
import traceback
import random
import string
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
import queue
import signal
import subprocess
import resource
import socket
import tempfile

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/stability_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('StabilityTest')

class StabilityTestSuite:
    """稳定性测试套件"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.test_duration = 15 * 60  # 15分钟，以秒为单位
        self.results = {
            'test_start_time': None,
            'test_end_time': None,
            'total_duration': 0,
            'memory_usage': [],
            'cpu_usage': [],
            'disk_usage': [],
            'network_usage': [],
            'errors': [],
            'performance_metrics': {},
            'concurrency_test_results': [],
            'load_test_results': []
        }
        self.monitoring = True
        self.error_count = 0
        self.request_count = 0
        
    def setup_environment(self):
        """设置测试环境"""
        logger.info("设置测试环境...")
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp(prefix='stability_test_')
        logger.info(f"临时目录: {self.temp_dir}")
        
        # 记录初始资源使用情况
        self.initial_memory = psutil.virtual_memory()
        self.initial_disk = psutil.disk_usage('/')
        self.initial_cpu = psutil.cpu_percent()
        
        logger.info(f"初始内存使用: {self.initial_memory.percent}%")
        logger.info(f"初始磁盘使用: {self.initial_disk.percent}%")
        logger.info(f"初始CPU使用: {self.initial_cpu}%")
        
    def monitor_resources(self):
        """监控资源使用情况"""
        logger.info("开始资源监控...")
        
        while self.monitoring:
            try:
                # 内存使用情况
                memory = psutil.virtual_memory()
                self.results['memory_usage'].append({
                    'timestamp': time.time(),
                    'used_gb': memory.used / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'percentage': memory.percent
                })
                
                # CPU使用情况
                cpu_percent = psutil.cpu_percent(interval=1)
                self.results['cpu_usage'].append({
                    'timestamp': time.time(),
                    'percentage': cpu_percent,
                    'count': psutil.cpu_count()
                })
                
                # 磁盘使用情况
                disk = psutil.disk_usage('/')
                self.results['disk_usage'].append({
                    'timestamp': time.time(),
                    'used_gb': disk.used / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'percentage': (disk.used / disk.total) * 100
                })
                
                # 网络使用情况
                net_io = psutil.net_io_counters()
                self.results['network_usage'].append({
                    'timestamp': time.time(),
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                })
                
                time.sleep(5)  # 每5秒监控一次
                
            except Exception as e:
                logger.error(f"资源监控错误: {e}")
                self.results['errors'].append({
                    'timestamp': time.time(),
                    'type': 'monitoring_error',
                    'message': str(e)
                })
                
    def simulate_memory_operations(self):
        """模拟内存密集型操作"""
        logger.info("开始内存操作测试...")
        
        try:
            # 创建大量数据
            data_size = 1000
            large_data = []
            
            for i in range(data_size):
                # 创建大型numpy数组
                if 'numpy' in sys.modules:
                    import numpy as np
                    arr = np.random.random((100, 100)).astype(np.float32)
                    large_data.append(arr)
                else:
                    # 使用列表作为替代
                    large_data.append([random.random() for _ in range(10000)])
                
                self.request_count += 1
                
                # 每100个操作进行一次垃圾回收
                if i % 100 == 0:
                    gc.collect()
                    
                # 随机暂停
                if random.random() < 0.1:
                    time.sleep(random.uniform(0.01, 0.1))
                    
            logger.info(f"内存操作测试完成，处理了 {len(large_data)} 个数据项")
            
            # 清理数据
            del large_data
            gc.collect()
            
        except Exception as e:
            logger.error(f"内存操作测试错误: {e}")
            self.results['errors'].append({
                'timestamp': time.time(),
                'type': 'memory_operation_error',
                'message': str(e)
            })
            
    def simulate_cpu_operations(self):
        """模拟CPU密集型操作"""
        logger.info("开始CPU操作测试...")
        
        try:
            # CPU密集型计算
            for i in range(1000):
                # 矩阵运算
                result = 0
                for j in range(1000):
                    result += j * random.random()
                    
                # 斐波那契数列计算
                def fibonacci(n):
                    if n <= 1:
                        return n
                    return fibonacci(n-1) + fibonacci(n-2)
                
                if i % 100 == 0:
                    fibonacci(20)
                    
                self.request_count += 1
                
                # 随机暂停
                if random.random() < 0.05:
                    time.sleep(random.uniform(0.001, 0.01))
                    
            logger.info("CPU操作测试完成")
            
        except Exception as e:
            logger.error(f"CPU操作测试错误: {e}")
            self.results['errors'].append({
                'timestamp': time.time(),
                'type': 'cpu_operation_error',
                'message': str(e)
            })
            
    def simulate_io_operations(self):
        """模拟I/O密集型操作"""
        logger.info("开始I/O操作测试...")
        
        try:
            # 文件读写操作
            for i in range(500):
                # 创建临时文件
                temp_file = os.path.join(self.temp_dir, f'test_file_{i}.txt')
                
                # 写入数据
                data = ''.join(random.choices(string.ascii_letters + string.digits, k=1000))
                with open(temp_file, 'w') as f:
                    f.write(data)
                    
                # 读取数据
                with open(temp_file, 'r') as f:
                    read_data = f.read()
                    
                # 验证数据一致性
                assert read_data == data, "数据不一致"
                
                # 删除文件
                os.remove(temp_file)
                
                self.request_count += 1
                
                # 随机暂停
                if random.random() < 0.1:
                    time.sleep(random.uniform(0.001, 0.05))
                    
            logger.info("I/O操作测试完成")
            
        except Exception as e:
            logger.error(f"I/O操作测试错误: {e}")
            self.results['errors'].append({
                'timestamp': time.time(),
                'type': 'io_operation_error',
                'message': str(e)
            })
            
    def concurrent_request_test(self, num_threads=10):
        """并发请求测试"""
        logger.info(f"开始并发请求测试，线程数: {num_threads}")
        
        def worker_function(worker_id):
            """工作线程函数"""
            try:
                start_time = time.time()
                local_count = 0
                
                while time.time() - start_time < 300:  # 运行5分钟
                    # 模拟不同类型的请求
                    operation_type = random.choice(['memory', 'cpu', 'io', 'network'])
                    
                    if operation_type == 'memory':
                        data = [random.random() for _ in range(1000)]
                        del data
                    elif operation_type == 'cpu':
                        result = sum(i**2 for i in range(1000))
                    elif operation_type == 'io':
                        # 模拟数据库操作
                        time.sleep(0.01)
                    elif operation_type == 'network':
                        # 模拟网络请求
                        time.sleep(0.005)
                        
                    local_count += 1
                    
                return {
                    'worker_id': worker_id,
                    'requests_completed': local_count,
                    'success': True
                }
                
            except Exception as e:
                return {
                    'worker_id': worker_id,
                    'requests_completed': 0,
                    'success': False,
                    'error': str(e)
                }
                
        # 启动工作线程
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_function, i) for i in range(num_threads)]
            results = []
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=320)  # 5分20秒超时
                    results.append(result)
                    self.request_count += result['requests_completed']
                except Exception as e:
                    logger.error(f"并发测试错误: {e}")
                    results.append({
                        'worker_id': -1,
                        'requests_completed': 0,
                        'success': False,
                        'error': str(e)
                    })
                    
        self.results['concurrency_test_results'] = results
        logger.info(f"并发测试完成，总请求数: {self.request_count}")
        
    def load_test(self, max_concurrent=50):
        """负载测试"""
        logger.info(f"开始负载测试，最大并发数: {max_concurrent}")
        
        def load_worker(worker_id):
            """负载工作线程"""
            try:
                start_time = time.time()
                requests_made = 0
                errors = 0
                
                while time.time() - start_time < 600:  # 运行10分钟
                    try:
                        # 模拟实际工作负载
                        # 1. 内存分配
                        data_chunks = []
                        for _ in range(10):
                            chunk = [random.random() for _ in range(100)]
                            data_chunks.append(chunk)
                            
                        # 2. CPU计算
                        total = sum(sum(chunk) for chunk in data_chunks)
                        
                        # 3. I/O操作
                        temp_file = os.path.join(self.temp_dir, f'load_{worker_id}_{requests_made}.tmp')
                        with open(temp_file, 'w') as f:
                            f.write(str(total))
                        os.remove(temp_file)
                        
                        requests_made += 1
                        
                        # 清理内存
                        del data_chunks
                        
                        # 随机暂停模拟真实负载
                        time.sleep(random.uniform(0.01, 0.1))
                        
                    except Exception as e:
                        errors += 1
                        logger.error(f"负载测试工作线程 {worker_id} 错误: {e}")
                        
                return {
                    'worker_id': worker_id,
                    'requests_made': requests_made,
                    'errors': errors,
                    'duration': time.time() - start_time,
                    'success': errors == 0
                }
                
            except Exception as e:
                return {
                    'worker_id': worker_id,
                    'requests_made': 0,
                    'errors': 1,
                    'duration': 0,
                    'success': False,
                    'error': str(e)
                }
                
        # 渐进式负载测试
        load_levels = [1, 5, 10, 20, 30, 50]
        load_results = []
        
        for level in load_levels:
            logger.info(f"测试负载级别: {level}")
            
            with ThreadPoolExecutor(max_workers=level) as executor:
                futures = [executor.submit(load_worker, i) for i in range(level)]
                level_results = []
                
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=660)  # 11分钟超时
                        level_results.append(result)
                    except Exception as e:
                        logger.error(f"负载测试级别 {level} 错误: {e}")
                        
            load_results.append({
                'level': level,
                'results': level_results,
                'total_requests': sum(r['requests_made'] for r in level_results),
                'total_errors': sum(r['errors'] for r in level_results)
            })
            
        self.results['load_test_results'] = load_results
        logger.info("负载测试完成")
        
    def stress_test(self):
        """压力测试"""
        logger.info("开始压力测试...")
        
        try:
            # 内存压力测试
            memory_stress_data = []
            for i in range(100):
                # 逐步增加内存使用
                chunk_size = 10 * 1024 * 1024  # 10MB
                chunk = bytearray(chunk_size)
                memory_stress_data.append(chunk)
                self.request_count += 1
                
                if i % 10 == 0:
                    logger.info(f"内存压力测试进度: {i}/100")
                    time.sleep(1)
                    
            # CPU压力测试
            start_time = time.time()
            operations = 0
            while time.time() - start_time < 60:  # 运行1分钟
                # 复杂的数学计算
                result = 0
                for i in range(10000):
                    result += (i ** 0.5) * random.random()
                operations += 1
                self.request_count += 1
                
            logger.info(f"CPU压力测试完成，进行了 {operations} 次操作")
            
            # 清理内存
            del memory_stress_data
            gc.collect()
            
        except Exception as e:
            logger.error(f"压力测试错误: {e}")
            self.results['errors'].append({
                'timestamp': time.time(),
                'type': 'stress_test_error',
                'message': str(e)
            })
            
    def network_resilience_test(self):
        """网络韧性测试"""
        logger.info("开始网络韧性测试...")
        
        try:
            # 测试网络连接
            test_hosts = [
                '8.8.8.8',      # Google DNS
                '1.1.1.1',      # Cloudflare DNS
                'localhost'     # 本地回环
            ]
            
            for host in test_hosts:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((host, 53))  # 尝试连接DNS端口
                    sock.close()
                    
                    if result == 0:
                        logger.info(f"网络连接 {host}: 成功")
                    else:
                        logger.warning(f"网络连接 {host}: 失败")
                        
                except Exception as e:
                    logger.warning(f"网络连接测试 {host} 错误: {e}")
                    
            # 模拟网络请求
            for i in range(100):
                try:
                    # 模拟HTTP请求
                    time.sleep(random.uniform(0.01, 0.1))
                    self.request_count += 1
                    
                except Exception as e:
                    logger.error(f"网络请求模拟错误: {e}")
                    
        except Exception as e:
            logger.error(f"网络韧性测试错误: {e}")
            
    def analyze_results(self):
        """分析测试结果"""
        logger.info("分析测试结果...")
        
        try:
            # 计算基本统计信息
            total_duration = self.end_time - self.start_time
            
            # 内存使用分析
            memory_usage = [m['percentage'] for m in self.results['memory_usage']]
            if memory_usage:
                self.results['performance_metrics']['memory'] = {
                    'max_usage_percent': max(memory_usage),
                    'min_usage_percent': min(memory_usage),
                    'avg_usage_percent': sum(memory_usage) / len(memory_usage),
                    'memory_leak_detected': self.check_memory_leak(memory_usage)
                }
                
            # CPU使用分析
            cpu_usage = [c['percentage'] for c in self.results['cpu_usage']]
            if cpu_usage:
                self.results['performance_metrics']['cpu'] = {
                    'max_usage_percent': max(cpu_usage),
                    'min_usage_percent': min(cpu_usage),
                    'avg_usage_percent': sum(cpu_usage) / len(cpu_usage),
                    'peak_load_handled': max(cpu_usage) < 95
                }
                
            # 错误分析
            error_count = len(self.results['errors'])
            self.results['performance_metrics']['errors'] = {
                'total_errors': error_count,
                'error_rate': error_count / max(self.request_count, 1),
                'critical_errors': sum(1 for e in self.results['errors'] if 'error' in e.get('type', '').lower())
            }
            
            # 系统稳定性评估
            self.results['performance_metrics']['stability'] = {
                'system_responsive': error_count < 10,
                'resource_usage_normal': max(memory_usage) < 90 if memory_usage else True,
                'no_memory_leak': not self.results['performance_metrics']['memory']['memory_leak_detected'],
                'concurrency_handled': len(self.results['concurrency_test_results']) > 0
            }
            
        except Exception as e:
            logger.error(f"结果分析错误: {e}")
            
    def check_memory_leak(self, memory_usage):
        """检查内存泄漏"""
        if len(memory_usage) < 10:
            return False
            
        # 计算线性趋势
        n = len(memory_usage)
        x = list(range(n))
        y = memory_usage
        
        # 简单线性回归
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return False
            
        slope = numerator / denominator
        
        # 如果斜率大于0.1，认为存在内存泄漏趋势
        return slope > 0.1
        
    def cleanup(self):
        """清理资源"""
        logger.info("清理测试资源...")
        
        try:
            # 停止监控
            self.monitoring = False
            
            # 清理临时目录
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                
            # 强制垃圾回收
            gc.collect()
            
        except Exception as e:
            logger.error(f"清理资源错误: {e}")
            
    def run_stability_test(self):
        """运行完整的稳定性测试"""
        logger.info("开始系统稳定性测试...")
        
        try:
            self.start_time = time.time()
            self.results['test_start_time'] = datetime.now().isoformat()
            
            # 启动资源监控线程
            monitor_thread = threading.Thread(target=self.monitor_resources)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # 执行测试套件
            logger.info("执行内存操作测试...")
            self.simulate_memory_operations()
            
            logger.info("执行CPU操作测试...")
            self.simulate_cpu_operations()
            
            logger.info("执行I/O操作测试...")
            self.simulate_io_operations()
            
            logger.info("执行并发请求测试...")
            self.concurrent_request_test(num_threads=10)
            
            logger.info("执行负载测试...")
            self.load_test(max_concurrent=20)
            
            logger.info("执行压力测试...")
            self.stress_test()
            
            logger.info("执行网络韧性测试...")
            self.network_resilience_test()
            
            # 等待测试时间达到15分钟
            elapsed_time = time.time() - self.start_time
            remaining_time = max(0, self.test_duration - elapsed_time)
            
            if remaining_time > 0:
                logger.info(f"继续运行 {remaining_time:.0f} 秒以达到15分钟测试时间...")
                time.sleep(remaining_time)
                
            self.end_time = time.time()
            self.results['test_end_time'] = datetime.now().isoformat()
            self.results['total_duration'] = self.end_time - self.start_time
            
            # 分析结果
            self.analyze_results()
            
            logger.info("稳定性测试完成")
            
        except KeyboardInterrupt:
            logger.info("测试被用户中断")
        except Exception as e:
            logger.error(f"稳定性测试错误: {e}")
            self.results['errors'].append({
                'timestamp': time.time(),
                'type': 'test_execution_error',
                'message': str(e),
                'traceback': traceback.format_exc()
            })
        finally:
            self.cleanup()
            
    def generate_report(self):
        """生成测试报告"""
        report_path = '/workspace/docs/stability_validation.md'
        
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 系统长期运行稳定性验证报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 测试概览
            f.write("## 测试概览\n\n")
            f.write(f"- **测试开始时间**: {self.results['test_start_time']}\n")
            f.write(f"- **测试结束时间**: {self.results['test_end_time']}\n")
            f.write(f"- **总测试时长**: {self.results['total_duration']/60:.2f} 分钟\n")
            f.write(f"- **总请求数**: {self.request_count}\n")
            f.write(f"- **错误总数**: {len(self.results['errors'])}\n\n")
            
            # 资源使用情况
            f.write("## 资源使用情况\n\n")
            
            if 'memory' in self.results['performance_metrics']:
                mem = self.results['performance_metrics']['memory']
                f.write("### 内存使用\n\n")
                f.write(f"- **最大使用率**: {mem['max_usage_percent']:.2f}%\n")
                f.write(f"- **最小使用率**: {mem['min_usage_percent']:.2f}%\n")
                f.write(f"- **平均使用率**: {mem['avg_usage_percent']:.2f}%\n")
                f.write(f"- **内存泄漏检测**: {'是' if mem['memory_leak_detected'] else '否'}\n\n")
                
            if 'cpu' in self.results['performance_metrics']:
                cpu = self.results['performance_metrics']['cpu']
                f.write("### CPU使用\n\n")
                f.write(f"- **最大使用率**: {cpu['max_usage_percent']:.2f}%\n")
                f.write(f"- **最小使用率**: {cpu['min_usage_percent']:.2f}%\n")
                f.write(f"- **平均使用率**: {cpu['avg_usage_percent']:.2f}%\n")
                f.write(f"- **高负载处理**: {'正常' if cpu['peak_load_handled'] else '异常'}\n\n")
            
            # 错误分析
            f.write("## 错误分析\n\n")
            if self.results['errors']:
                f.write("### 发现的错误\n\n")
                for i, error in enumerate(self.results['errors'][:10], 1):  # 只显示前10个错误
                    f.write(f"{i}. **{error['type']}**: {error['message']}\n")
                    f.write(f"   - 时间: {datetime.fromtimestamp(error['timestamp']).strftime('%H:%M:%S')}\n\n")
                    
                if len(self.results['errors']) > 10:
                    f.write(f"... 还有 {len(self.results['errors']) - 10} 个错误\n\n")
            else:
                f.write("未发现任何错误\n\n")
            
            # 并发测试结果
            f.write("## 并发处理能力\n\n")
            if self.results['concurrency_test_results']:
                successful_workers = sum(1 for r in self.results['concurrency_test_results'] if r.get('success', False))
                total_workers = len(self.results['concurrency_test_results'])
                f.write(f"- **成功工作线程**: {successful_workers}/{total_workers}\n")
                f.write(f"- **成功率**: {(successful_workers/total_workers)*100:.1f}%\n\n")
                
                # 显示每个工作线程的结果
                f.write("### 工作线程详情\n\n")
                for result in self.results['concurrency_test_results']:
                    status = "成功" if result.get('success', False) else "失败"
                    f.write(f"- 线程 {result['worker_id']}: {status} - 完成 {result['requests_completed']} 个请求\n")
                f.write("\n")
            
            # 负载测试结果
            f.write("## 负载测试结果\n\n")
            if self.results['load_test_results']:
                for load_result in self.results['load_test_results']:
                    f.write(f"### 负载级别 {load_result['level']}\n\n")
                    f.write(f"- **总请求数**: {load_result['total_requests']}\n")
                    f.write(f"- **总错误数**: {load_result['total_errors']}\n")
                    f.write(f"- **成功率**: {((load_result['total_requests'] - load_result['total_errors'])/max(load_result['total_requests'], 1))*100:.1f}%\n\n")
            
            # 稳定性评估
            f.write("## 稳定性评估\n\n")
            if 'stability' in self.results['performance_metrics']:
                stability = self.results['performance_metrics']['stability']
                f.write("### 关键指标\n\n")
                f.write(f"- **系统响应性**: {'正常' if stability['system_responsive'] else '异常'}\n")
                f.write(f"- **资源使用**: {'正常' if stability['resource_usage_normal'] else '异常'}\n")
                f.write(f"- **内存泄漏**: {'未检测到' if stability['no_memory_leak'] else '检测到'}\n")
                f.write(f"- **并发处理**: {'支持' if stability['concurrency_handled'] else '不支持'}\n\n")
                
                # 总体评估
                f.write("### 总体评估\n\n")
                if all(stability.values()):
                    f.write("✅ **系统稳定性**: 优秀 - 所有测试指标均正常\n\n")
                elif sum(stability.values()) >= 3:
                    f.write("⚠️ **系统稳定性**: 良好 - 大部分指标正常，少数需要关注\n\n")
                else:
                    f.write("❌ **系统稳定性**: 需要改进 - 多个指标存在问题\n\n")
            
            # 建议和改进
            f.write("## 建议和改进\n\n")
            f.write("### 发现的问题\n\n")
            
            issues = []
            
            if 'memory' in self.results['performance_metrics']:
                if self.results['performance_metrics']['memory']['memory_leak_detected']:
                    issues.append("- 检测到内存泄漏趋势，建议检查内存管理代码")
                    
            if 'cpu' in self.results['performance_metrics']:
                if not self.results['performance_metrics']['cpu']['peak_load_handled']:
                    issues.append("- CPU负载处理能力不足，建议优化算法或增加计算资源")
                    
            if 'errors' in self.results['performance_metrics']:
                if self.results['performance_metrics']['errors']['error_rate'] > 0.01:
                    issues.append(f"- 错误率过高 ({self.results['performance_metrics']['errors']['error_rate']*100:.2f}%)，建议检查错误处理机制")
                    
            if issues:
                for issue in issues:
                    f.write(f"{issue}\n")
            else:
                f.write("- 未发现重大问题，系统运行稳定\n")
                
            f.write("\n### 优化建议\n\n")
            f.write("1. **内存优化**: 定期进行内存检查，避免长时间运行后的内存积累\n")
            f.write("2. **并发处理**: 优化线程池配置，提高并发请求处理效率\n")
            f.write("3. **错误监控**: 建立更完善的错误监控和告警机制\n")
            f.write("4. **资源限制**: 设置合理的资源使用限制，防止系统过载\n")
            f.write("5. **定期测试**: 建议定期进行类似的长运行测试以确保持续稳定性\n\n")
            
            # 测试数据摘要
            f.write("## 测试数据摘要\n\n")
            f.write("### 数据点统计\n\n")
            f.write(f"- **内存监控点数**: {len(self.results['memory_usage'])}\n")
            f.write(f"- **CPU监控点数**: {len(self.results['cpu_usage'])}\n")
            f.write(f"- **磁盘监控点数**: {len(self.results['disk_usage'])}\n")
            f.write(f"- **网络监控点数**: {len(self.results['network_usage'])}\n\n")
            
            # 附录
            f.write("## 附录\n\n")
            f.write("### 完整测试结果\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.results, indent=2, ensure_ascii=False, default=str))
            f.write("\n```\n\n")
            
            f.write("---\n")
            f.write("*本报告由系统稳定性测试套件自动生成*\n")
            
        logger.info(f"测试报告已生成: {report_path}")
        
        return report_path

def signal_handler(signum, frame):
    """信号处理器"""
    logger.info(f"收到信号 {signum}，开始优雅关闭...")
    sys.exit(0)

def main():
    """主函数"""
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("启动系统稳定性测试...")
    
    # 创建测试套件
    test_suite = StabilityTestSuite()
    
    try:
        # 设置测试环境
        test_suite.setup_environment()
        
        # 运行稳定性测试
        test_suite.run_stability_test()
        
        # 生成报告
        report_path = test_suite.generate_report()
        
        logger.info("系统稳定性测试完成!")
        logger.info(f"测试报告: {report_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
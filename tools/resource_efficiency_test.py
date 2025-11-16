#!/usr/bin/env python3
"""
资源使用和效率优化测试脚本
监控CPU、内存、GPU、IO和并发性能
"""

import psutil
import time
import threading
import multiprocessing
import os
import json
import statistics
from datetime import datetime
import concurrent.futures
import subprocess
import tempfile
import random
import gc
import sys

class ResourceMonitor:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'cpu_tests': {},
            'memory_tests': {},
            'gpu_tests': {},
            'io_tests': {},
            'concurrency_tests': {},
            'system_info': self.get_system_info()
        }
    
    def get_system_info(self):
        """获取系统基本信息"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total': psutil.virtual_memory().total,
            'platform': sys.platform,
            'python_version': sys.version
        }
        
        # GPU信息检查
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            info['gpu_count'] = len(gpus)
            if gpus:
                info['gpus'] = []
                for gpu in gpus:
                    info['gpus'].append({
                        'name': gpu.name,
                        'memory_total': gpu.memoryTotal,
                        'memory_used': gpu.memoryUsed,
                        'memory_free': gpu.memoryFree
                    })
        except ImportError:
            info['gpu_available'] = False
        
        return info
    
    def monitor_cpu_usage(self, duration=30):
        """监控CPU使用率"""
        print("开始CPU使用率监控...")
        
        cpu_samples = []
        cpu_per_core_samples = []
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # 获取整体CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_samples.append(cpu_percent)
            
            # 获取每个核心的使用率
            per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            cpu_per_core_samples.append(per_core)
        
        # 计算统计数据
        self.results['cpu_tests'] = {
            'duration_seconds': duration,
            'average_cpu_usage': statistics.mean(cpu_samples),
            'max_cpu_usage': max(cpu_samples),
            'min_cpu_usage': min(cpu_samples),
            'std_dev_cpu_usage': statistics.stdev(cpu_samples) if len(cpu_samples) > 1 else 0,
            'average_per_core': {
                f'core_{i}': statistics.mean([samples[i] for samples in cpu_per_core_samples])
                for i in range(len(cpu_per_core_samples[0]))
            } if cpu_per_core_samples else {},
            'peak_per_core': {
                f'core_{i}': max([samples[i] for samples in cpu_per_core_samples])
                for i in range(len(cpu_per_core_samples[0]))
            } if cpu_per_core_samples else {}
        }
        
        print(f"CPU监控完成 - 平均使用率: {self.results['cpu_tests']['average_cpu_usage']:.2f}%")
    
    def stress_test_cpu(self):
        """CPU压力测试"""
        print("开始CPU压力测试...")
        
        # 单线程性能测试
        start_time = time.time()
        single_result = self.cpu_intensive_task()
        single_duration = time.time() - start_time
        
        # 多线程性能测试
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.cpu_intensive_task) for _ in range(4)]
            multi_results = [future.result() for future in futures]
        multi_duration = time.time() - start_time
        
        # 多进程性能测试（使用进程间通信）
        start_time = time.time()
        with multiprocessing.Pool(processes=4) as pool:
            process_results = pool.map(ResourceMonitor.cpu_intensive_task_for_process, range(4))
        process_duration = time.time() - start_time
        
        self.results['cpu_tests']['stress_test'] = {
            'single_thread_duration': single_duration,
            'multi_thread_duration': multi_duration,
            'multi_process_duration': process_duration,
            'thread_efficiency': single_duration / multi_duration if multi_duration > 0 else 0,
            'process_efficiency': single_duration / process_duration if process_duration > 0 else 0
        }
        
        print(f"CPU压力测试完成 - 线程效率: {self.results['cpu_tests']['stress_test']['thread_efficiency']:.2f}")
    
    @staticmethod
    def cpu_intensive_task():
        """CPU密集型任务"""
        result = 0
        for i in range(10**6):
            result += i * i
        return result
    
    @staticmethod
    def cpu_intensive_task_for_process(n):
        """用于多进程的CPU密集型任务"""
        result = 0
        for i in range(10**6):
            result += i * i
        return result
    
    def test_memory_usage(self):
        """测试内存使用模式"""
        print("开始内存使用测试...")
        
        # 获取初始内存状态
        initial_memory = psutil.virtual_memory()
        
        # 内存分配测试
        large_objects = []
        allocation_sizes = []
        
        for size_mb in [10, 50, 100, 200]:
            start_time = time.time()
            
            # 分配内存
            obj = [0] * (size_mb * 1024 * 1024 // 8)  # 假设每个元素8字节
            large_objects.append(obj)
            allocation_sizes.append(size_mb)
            
            # 检查内存使用
            current_memory = psutil.virtual_memory()
            memory_used = current_memory.used - initial_memory.used
            
            allocation_time = time.time() - start_time
            
            self.results['memory_tests'][f'allocation_{size_mb}MB'] = {
                'allocation_time': allocation_time,
                'memory_used_bytes': memory_used,
                'theoretical_size': size_mb * 1024 * 1024,
                'efficiency': memory_used / (size_mb * 1024 * 1024) if size_mb > 0 else 0
            }
            
            time.sleep(1)  # 等待内存稳定
        
        # 内存释放测试
        start_time = time.time()
        large_objects.clear()
        gc.collect()
        release_time = time.time() - start_time
        
        # 内存碎片测试
        def memory_fragmentation_test():
            objects = []
            for i in range(100):
                obj = bytearray(random.randint(1000, 10000))
                objects.append(obj)
            
            # 释放一半对象
            for i in range(0, len(objects), 2):
                objects[i] = None
            
            gc.collect()
            return len(objects)
        
        fragmentation_time = time.time()
        objects_remaining = memory_fragmentation_test()
        fragmentation_duration = time.time() - fragmentation_time
        
        final_memory = psutil.virtual_memory()
        
        self.results['memory_tests'].update({
            'memory_release_time': release_time,
            'fragmentation_test': {
                'duration': fragmentation_duration,
                'objects_remaining': objects_remaining
            },
            'memory_before': {
                'used': initial_memory.used,
                'available': initial_memory.available
            },
            'memory_after': {
                'used': final_memory.used,
                'available': final_memory.available
            }
        })
        
        print("内存使用测试完成")
    
    def test_gpu_usage(self):
        """测试GPU利用率（如果可用）"""
        print("开始GPU使用率测试...")
        
        try:
            import GPUtil
            
            gpus = GPUtil.getGPUs()
            if not gpus:
                self.results['gpu_tests']['available'] = False
                return
            
            gpu_samples = []
            
            # GPU监控
            for _ in range(10):
                for gpu in gpus:
                    gpu_samples.append({
                        'gpu_id': gpu.id,
                        'gpu_name': gpu.name,
                        'load': gpu.load * 100,
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'memory_util': gpu.memoryUtil * 100,
                        'temperature': gpu.temperature
                    })
                time.sleep(1)
            
            self.results['gpu_tests']['available'] = True
            self.results['gpu_tests']['gpu_count'] = len(gpus)
            
            # 计算每个GPU的统计数据
            for gpu_id in range(len(gpus)):
                gpu_data = [sample for sample in gpu_samples if sample['gpu_id'] == gpu_id]
                
                self.results['gpu_tests'][f'gpu_{gpu_id}'] = {
                    'name': gpu_data[0]['gpu_name'],
                    'average_load': statistics.mean([d['load'] for d in gpu_data]),
                    'max_load': max([d['load'] for d in gpu_data]),
                    'average_memory_util': statistics.mean([d['memory_util'] for d in gpu_data]),
                    'max_memory_util': max([d['memory_util'] for d in gpu_data]),
                    'average_temperature': statistics.mean([d['temperature'] for d in gpu_data])
                }
        
        except ImportError:
            self.results['gpu_tests']['available'] = False
            self.results['gpu_tests']['error'] = "GPUtil not available"
        
        print("GPU使用率测试完成")
    
    def test_io_efficiency(self):
        """测试IO操作效率"""
        print("开始IO效率测试...")
        
        # 磁盘写入测试
        temp_dir = tempfile.mkdtemp()
        test_file = os.path.join(temp_dir, 'test_io.tmp')
        
        test_sizes = [1024, 10240, 102400, 1048576]  # 1KB, 10KB, 100KB, 1MB
        
        for size in test_sizes:
            # 写入测试
            start_time = time.time()
            with open(test_file, 'wb') as f:
                data = os.urandom(size)
                f.write(data)
            write_duration = time.time() - start_time
            
            # 读取测试
            start_time = time.time()
            with open(test_file, 'rb') as f:
                read_data = f.read()
            read_duration = time.time() - start_time
            
            # 随机读取测试
            start_time = time.time()
            with open(test_file, 'rb') as f:
                for _ in range(100):
                    f.seek(random.randint(0, size - 1024))
                    f.read(1024)
            random_read_duration = time.time() - start_time
            
            self.results['io_tests'][f'size_{size}_bytes'] = {
                'write_time': write_duration,
                'read_time': read_duration,
                'random_read_time': random_read_duration,
                'write_speed_mb_s': size / (write_duration * 1024 * 1024),
                'read_speed_mb_s': size / (read_duration * 1024 * 1024),
                'data_integrity': read_data == data
            }
        
        # 清理
        try:
            os.remove(test_file)
            os.rmdir(temp_dir)
        except:
            pass
        
        # 网络IO测试（如果可能）
        try:
            import requests
            test_url = "https://httpbin.org/get"
            start_time = time.time()
            response = requests.get(test_url, timeout=10)
            network_duration = time.time() - start_time
            
            self.results['io_tests']['network_io'] = {
                'url': test_url,
                'response_time': network_duration,
                'status_code': response.status_code,
                'content_length': len(response.content)
            }
        except Exception as e:
            self.results['io_tests']['network_io'] = {
                'error': str(e)
            }
        
        print("IO效率测试完成")
    
    def test_concurrency_performance(self):
        """测试并发性能和资源竞争"""
        print("开始并发性能测试...")
        
        # 不同线程数量的测试
        thread_counts = [1, 2, 4, 8]
        concurrency_results = {}
        
        for thread_count in thread_counts:
            # CPU密集型任务
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(self.cpu_task, 10000) for _ in range(10)]
                cpu_results = [future.result() for future in futures]
            cpu_duration = time.time() - start_time
            
            # IO密集型任务
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(self.io_task) for _ in range(20)]
                io_results = [future.result() for future in futures]
            io_duration = time.time() - start_time
            
            concurrency_results[f'threads_{thread_count}'] = {
                'cpu_task_duration': cpu_duration,
                'io_task_duration': io_duration,
                'cpu_results_count': len(cpu_results),
                'io_results_count': len(io_results)
            }
        
        # 资源竞争测试
        shared_counter = {'value': 0}
        lock = threading.Lock()
        num_threads = 10
        
        def increment_with_lock(counter, lock_obj):
            with lock_obj:
                counter['value'] += 1
                return counter['value']
        
        def increment_without_lock(counter):
            counter['value'] += 1
            return counter['value']
        
        # 有锁测试
        shared_counter['value'] = 0
        start_time = time.time()
        threads = [threading.Thread(target=increment_with_lock, args=(shared_counter, lock)) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        with_lock_duration = time.time() - start_time
        
        # 无锁测试（可能有竞态条件）
        shared_counter['value'] = 0
        start_time = time.time()
        threads = [threading.Thread(target=increment_without_lock, args=(shared_counter,)) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        without_lock_duration = time.time() - start_time
        final_value = shared_counter['value']
        
        self.results['concurrency_tests'] = {
            'threading_performance': concurrency_results,
            'resource_contention': {
                'with_lock_duration': with_lock_duration,
                'without_lock_duration': without_lock_duration,
                'lock_overhead': with_lock_duration - without_lock_duration,
                'final_counter_value': final_value,
                'expected_value': num_threads,
                'contention_detected': final_value != num_threads
            }
        }
        
        print("并发性能测试完成")
    
    @staticmethod
    def cpu_task(n):
        """CPU密集型任务"""
        result = 0
        for i in range(n):
            result += i ** 2
        return result
    
    @staticmethod
    def io_task():
        """IO密集型任务"""
        time.sleep(0.1)
        return "IO task completed"
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始资源使用和效率优化测试...")
        print("=" * 50)
        
        # CPU测试
        self.monitor_cpu_usage(duration=10)
        self.stress_test_cpu()
        
        # 内存测试
        self.test_memory_usage()
        
        # GPU测试
        self.test_gpu_usage()
        
        # IO测试
        self.test_io_efficiency()
        
        # 并发测试
        self.test_concurrency_performance()
        
        print("=" * 50)
        print("所有测试完成！")
        
        return self.results

def generate_report(results, output_file):
    """生成测试报告"""
    report = f"""# 资源使用和效率优化测试报告

**测试时间**: {results['timestamp']}

## 系统信息

- **CPU核心数**: {results['system_info']['cpu_count']} (逻辑核心: {results['system_info']['cpu_count_logical']})
- **内存总量**: {results['system_info']['memory_total'] / (1024**3):.2f} GB
- **平台**: {results['system_info']['platform']}
- **Python版本**: {results['system_info']['python_version']}

## 1. CPU使用率分析

### 监控结果
- **平均CPU使用率**: {results['cpu_tests'].get('average_cpu_usage', 0):.2f}%
- **最大CPU使用率**: {results['cpu_tests'].get('max_cpu_usage', 0):.2f}%
- **最小CPU使用率**: {results['cpu_tests'].get('min_cpu_usage', 0):.2f}%
- **标准差**: {results['cpu_tests'].get('std_dev_cpu_usage', 0):.2f}

### 压力测试结果
"""

    if 'stress_test' in results['cpu_tests']:
        stress = results['cpu_tests']['stress_test']
        report += f"""
- **单线程执行时间**: {stress['single_thread_duration']:.4f}秒
- **多线程执行时间**: {stress['multi_thread_duration']:.4f}秒
- **多进程执行时间**: {stress['multi_process_duration']:.4f}秒
- **线程效率**: {stress['thread_efficiency']:.2f}x
- **进程效率**: {stress['process_efficiency']:.2f}x
"""

    report += f"""
## 2. 内存使用模式分析

### 内存分配测试
"""

    for key, value in results['memory_tests'].items():
        if key.startswith('allocation_'):
            report += f"""
**{key}**:
- 分配时间: {value['allocation_time']:.4f}秒
- 实际使用内存: {value['memory_used_bytes'] / (1024**2):.2f} MB
- 理论大小: {value['theoretical_size'] / (1024**2):.2f} MB
- 内存效率: {value['efficiency']:.2f}
"""

    if 'memory_release_time' in results['memory_tests']:
        report += f"""
- **内存释放时间**: {results['memory_tests']['memory_release_time']:.4f}秒
- **内存碎片测试**: {results['memory_tests']['fragmentation_test']['duration']:.4f}秒
"""

    report += """
### 内存使用对比
"""

    if 'memory_before' in results['memory_tests']:
        before = results['memory_tests']['memory_before']
        after = results['memory_tests']['memory_after']
        report += f"""
- **测试前内存使用**: {before['used'] / (1024**3):.2f} GB
- **测试后内存使用**: {after['used'] / (1024**3):.2f} GB
- **内存变化**: {(after['used'] - before['used']) / (1024**2):.2f} MB
"""

    # GPU测试结果
    report += "\n## 3. GPU利用率分析\n"
    
    if results['gpu_tests'].get('available', False):
        report += f"- **可用GPU数量**: {results['gpu_tests']['gpu_count']}\n"
        
        for gpu_key, gpu_data in results['gpu_tests'].items():
            if gpu_key.startswith('gpu_'):
                report += f"""
### GPU {gpu_key}
- **名称**: {gpu_data['name']}
- **平均负载**: {gpu_data['average_load']:.2f}%
- **最大负载**: {gpu_data['max_load']:.2f}%
- **平均内存使用**: {gpu_data['average_memory_util']:.2f}%
- **最大内存使用**: {gpu_data['max_memory_util']:.2f}%
- **平均温度**: {gpu_data['average_temperature']:.1f}°C
"""
    else:
        report += "- **GPU不可用或未检测到**\n"
        if 'error' in results['gpu_tests']:
            report += f"- **错误信息**: {results['gpu_tests']['error']}\n"

    # IO测试结果
    report += "\n## 4. IO操作效率分析\n"
    
    for key, value in results['io_tests'].items():
        if key.startswith('size_'):
            report += f"""
### {key}
- **写入时间**: {value['write_time']:.4f}秒
- **读取时间**: {value['read_time']:.4f}秒
- **随机读取时间**: {value['random_read_time']:.4f}秒
- **写入速度**: {value['write_speed_mb_s']:.2f} MB/s
- **读取速度**: {value['read_speed_mb_s']:.2f} MB/s
- **数据完整性**: {'✓' if value['data_integrity'] else '✗'}
"""

    if 'network_io' in results['io_tests']:
        net_io = results['io_tests']['network_io']
        if 'error' not in net_io:
            report += f"""
### 网络IO测试
- **响应时间**: {net_io['response_time']:.4f}秒
- **状态码**: {net_io['status_code']}
- **内容大小**: {net_io['content_length']} 字节
"""
        else:
            report += f"- **网络IO测试失败**: {net_io['error']}\n"

    # 并发性能测试结果
    report += "\n## 5. 并发性能和资源竞争分析\n"
    
    if 'threading_performance' in results['concurrency_tests']:
        report += "### 线程池性能测试\n"
        for key, value in results['concurrency_tests']['threading_performance'].items():
            report += f"""
**{key}**:
- CPU任务执行时间: {value['cpu_task_duration']:.4f}秒
- IO任务执行时间: {value['io_task_duration']:.4f}秒
"""

    if 'resource_contention' in results['concurrency_tests']:
        contention = results['concurrency_tests']['resource_contention']
        report += f"""
### 资源竞争测试
- **有锁执行时间**: {contention['with_lock_duration']:.4f}秒
- **无锁执行时间**: {contention['without_lock_duration']:.4f}秒
- **锁开销**: {contention['lock_overhead']:.4f}秒
- **最终计数器值**: {contention['final_counter_value']}
- **期望值**: {contention['expected_value']}
- **检测到竞态条件**: {'是' if contention['contention_detected'] else '否'}
"""

    # 总结和建议
    report += """
## 总结和建议

### 性能优化建议
1. **CPU优化**: 
   - 根据CPU核心数合理设置线程池大小
   - 考虑使用多进程而非多线程来处理CPU密集型任务

2. **内存优化**:
   - 监控内存使用情况，避免内存泄漏
   - 适当进行垃圾回收
   - 考虑使用内存池技术

3. **IO优化**:
   - 使用异步IO提高并发性能
   - 考虑使用缓冲区减少磁盘访问
   - 优化网络请求的并发数量

4. **并发优化**:
   - 合理使用锁机制避免竞态条件
   - 根据任务类型选择合适的并发模式
   - 监控系统资源使用情况

### 监控建议
- 定期监控CPU、内存和IO使用率
- 设置告警阈值，及时发现性能问题
- 记录性能基准，便于后续优化对比

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report

def main():
    """主函数"""
    monitor = ResourceMonitor()
    results = monitor.run_all_tests()
    
    # 生成报告
    output_file = '/workspace/docs/resource_efficiency_test.md'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    report = generate_report(results, output_file)
    
    print(f"\n报告已生成: {output_file}")
    
    # 保存详细数据
    json_file = '/workspace/docs/resource_efficiency_data.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"详细数据已保存: {json_file}")
    
    return results, report

if __name__ == "__main__":
    main()
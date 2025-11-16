#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件操作和内存管理优化工具
=======================

修复已识别的文件操作和内存泄漏问题，包括：
1. 安全的文件操作
2. 智能序列化/反序列化
3. 内存泄漏检测
4. 资源自动清理

作者: Brain-Inspired AI Team
创建时间: 2025-11-16
"""

import os
import sys
import pickle
import json
import gzip
import tempfile
import contextlib
import threading
import time
import gc
import weakref
from typing import Any, Dict, List, Optional, Union, Generator, Iterator
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
import psutil
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class FileOperationStats:
    """文件操作统计"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_size_mb: float = 0.0
    total_time_seconds: float = 0.0
    compression_ratio: float = 0.0


class SafeFileManager:
    """安全的文件管理器"""
    
    def __init__(self, max_file_size_mb: int = 100, compression_enabled: bool = True):
        """
        初始化安全文件管理器
        
        Args:
            max_file_size_mb: 最大文件大小（MB）
            compression_enabled: 是否启用压缩
        """
        self.max_file_size_mb = max_file_size_mb
        self.compression_enabled = compression_enabled
        self.open_files = {}  # track open files
        self.stats = FileOperationStats()
        self.lock = threading.RLock()
        
        # 内存泄漏检测
        self._memory_snapshots = deque(maxlen=100)
        self._large_object_threshold = 10 * 1024 * 1024  # 10MB
    
    def _check_file_size(self, filepath: Union[str, Path]) -> bool:
        """检查文件大小是否超出限制"""
        try:
            size = Path(filepath).stat().st_size
            size_mb = size / (1024 * 1024)
            return size_mb <= self.max_file_size_mb
        except (OSError, FileNotFoundError):
            return True  # 新文件，认为可以创建
    
    def _compress_data(self, data: bytes) -> bytes:
        """压缩数据"""
        if not self.compression_enabled:
            return data
        
        try:
            return gzip.compress(data)
        except Exception as e:
            logger.warning(f"数据压缩失败: {e}")
            return data
    
    def _decompress_data(self, data: bytes) -> bytes:
        """解压缩数据"""
        if not self.compression_enabled:
            return data
        
        try:
            return gzip.decompress(data)
        except Exception:
            return data  # 假设数据未压缩
    
    @contextlib.contextmanager
    def safe_open(self, filepath: Union[str, Path], mode: str = 'r', 
                  encoding: str = 'utf-8', **kwargs) -> Generator:
        """安全的文件打开上下文管理器"""
        filepath = Path(filepath)
        
        # 检查文件大小
        if mode.startswith('w') and not self._check_file_size(filepath):
            raise ValueError(f"文件大小超过限制: {self.max_file_size_mb}MB")
        
        # 检查文件句柄管理
        file_handle = None
        try:
            with self.lock:
                self.stats.total_operations += 1
            
            start_time = time.time()
            
            # 自动添加压缩扩展名
            if self.compression_enabled and mode == 'rb':
                filepath = filepath.with_suffix(filepath.suffix + '.gz')
            elif self.compression_enabled and mode == 'wb':
                filepath = filepath.with_suffix(filepath.suffix + '.gz')
            
            # 安全打开文件
            if mode.endswith('b'):
                file_handle = open(filepath, mode, **kwargs)
            else:
                file_handle = open(filepath, mode, encoding=encoding, **kwargs)
            
            # 记录打开的文件
            with self.lock:
                self.open_files[str(filepath)] = {
                    'handle': file_handle,
                    'opened_at': time.time(),
                    'mode': mode
                }
            
            yield file_handle
            
        except Exception as e:
            with self.lock:
                self.stats.failed_operations += 1
            logger.error(f"文件操作失败: {filepath}, 模式: {mode}, 错误: {e}")
            raise
        finally:
            # 确保文件关闭
            if file_handle and not file_handle.closed:
                file_handle.close()
            
            # 从打开文件列表中移除
            with self.lock:
                if str(filepath) in self.open_files:
                    del self.open_files[str(filepath)]
                
                self.stats.successful_operations += 1
                self.stats.total_time_seconds += time.time() - start_time
    
    def safe_save_pickle(self, data: Any, filepath: Union[str, Path]) -> bool:
        """安全的pickle保存"""
        try:
            # 检查数据大小
            data_size = sys.getsizeof(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
            data_size_mb = data_size / (1024 * 1024)
            
            if data_size_mb > self.max_file_size_mb:
                raise ValueError(f"数据大小 {data_size_mb:.2f}MB 超过限制 {self.max_file_size_mb}MB")
            
            with self.safe_open(filepath, 'wb') as f:
                serialized_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                compressed_data = self._compress_data(serialized_data)
                f.write(compressed_data)
            
            with self.lock:
                self.stats.total_size_mb += data_size_mb
                if self.compression_enabled:
                    actual_size_mb = len(compressed_data) / (1024 * 1024)
                    self.stats.compression_ratio = (data_size_mb - actual_size_mb) / data_size_mb
            
            logger.info(f"数据已保存到: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"保存pickle文件失败: {filepath}, 错误: {e}")
            return False
    
    def safe_load_pickle(self, filepath: Union[str, Path]) -> Optional[Any]:
        """安全的pickle加载"""
        try:
            with self.safe_open(filepath, 'rb') as f:
                compressed_data = f.read()
                data = self._decompress_data(compressed_data)
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"加载pickle文件失败: {filepath}, 错误: {e}")
            return None
    
    def safe_save_json(self, data: Any, filepath: Union[str, Path], 
                      ensure_ascii: bool = False, indent: int = 2) -> bool:
        """安全的JSON保存"""
        try:
            # 检查数据大小
            json_str = json.dumps(data, ensure_ascii=ensure_ascii, indent=indent)
            json_bytes = json_str.encode('utf-8')
            data_size_mb = len(json_bytes) / (1024 * 1024)
            
            if data_size_mb > self.max_file_size_mb:
                raise ValueError(f"JSON数据大小 {data_size_mb:.2f}MB 超过限制")
            
            with self.safe_open(filepath, 'w', encoding='utf-8') as f:
                if self.compression_enabled:
                    compressed_data = self._compress_data(json_bytes)
                    f.write(compressed_data.decode('utf-8', errors='ignore'))
                else:
                    f.write(json_str)
            
            with self.lock:
                self.stats.total_size_mb += data_size_mb
            
            return True
            
        except Exception as e:
            logger.error(f"保存JSON文件失败: {filepath}, 错误: {e}")
            return False
    
    def safe_load_json(self, filepath: Union[str, Path]) -> Optional[Any]:
        """安全的JSON加载"""
        try:
            with self.safe_open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
                if self.compression_enabled:
                    try:
                        content_bytes = content.encode('utf-8')
                        decompressed_data = self._decompress_data(content_bytes)
                        json_str = decompressed_data.decode('utf-8')
                    except Exception:
                        json_str = content
                else:
                    json_str = content
                
                return json.loads(json_str)
                
        except Exception as e:
            logger.error(f"加载JSON文件失败: {filepath}, 错误: {e}")
            return None
    
    def batch_save(self, data_dict: Dict[str, Any], directory: Union[str, Path], 
                   file_format: str = 'pickle') -> Dict[str, bool]:
        """批量保存文件"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            for filename, data in data_dict.items():
                filepath = directory / f"{filename}.{file_format}"
                
                if file_format.lower() == 'pickle':
                    future = executor.submit(self.safe_save_pickle, data, filepath)
                elif file_format.lower() == 'json':
                    future = executor.submit(self.safe_save_json, data, filepath)
                else:
                    raise ValueError(f"不支持的文件格式: {file_format}")
                
                futures[filename] = future
            
            for filename, future in futures.items():
                try:
                    results[filename] = future.result()
                except Exception as e:
                    logger.error(f"批量保存失败 {filename}: {e}")
                    results[filename] = False
        
        return results
    
    def batch_load(self, directory: Union[str, Path], 
                   file_format: str = 'pickle') -> Dict[str, Any]:
        """批量加载文件"""
        directory = Path(directory)
        results = {}
        
        if not directory.exists():
            logger.warning(f"目录不存在: {directory}")
            return results
        
        # 查找匹配的文件
        pattern = f"*.{file_format}"
        files = list(directory.glob(pattern))
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            for filepath in files:
                filename = filepath.stem
                
                if file_format.lower() == 'pickle':
                    future = executor.submit(self.safe_load_pickle, filepath)
                elif file_format.lower() == 'json':
                    future = executor.submit(self.safe_load_json, filepath)
                else:
                    continue
                
                futures[filename] = future
            
            for filename, future in futures.items():
                try:
                    results[filename] = future.result()
                except Exception as e:
                    logger.error(f"批量加载失败 {filename}: {e}")
                    results[filename] = None
        
        return results
    
    def cleanup_old_files(self, directory: Union[str, Path], 
                         max_age_days: int = 7) -> int:
        """清理旧文件"""
        directory = Path(directory)
        if not directory.exists():
            return 0
        
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        removed_count = 0
        
        try:
            for filepath in directory.rglob('*'):
                if filepath.is_file():
                    file_age = current_time - filepath.stat().st_mtime
                    if file_age > max_age_seconds:
                        filepath.unlink()
                        removed_count += 1
                        logger.debug(f"已删除旧文件: {filepath}")
        except Exception as e:
            logger.error(f"清理旧文件失败: {e}")
        
        return removed_count
    
    def get_stats(self) -> FileOperationStats:
        """获取操作统计"""
        with self.lock:
            return FileOperationStats(
                total_operations=self.stats.total_operations,
                successful_operations=self.stats.successful_operations,
                failed_operations=self.stats.failed_operations,
                total_size_mb=self.stats.total_size_mb,
                total_time_seconds=self.stats.total_time_seconds,
                compression_ratio=self.stats.compression_ratio
            )
    
    def force_cleanup(self):
        """强制清理所有资源"""
        # 关闭所有打开的文件
        with self.lock:
            for filepath, file_info in self.open_files.items():
                try:
                    file_info['handle'].close()
                except Exception as e:
                    logger.error(f"关闭文件失败 {filepath}: {e}")
            self.open_files.clear()
        
        # 强制垃圾回收
        gc.collect()
        
        logger.info("文件管理器强制清理完成")


class MemoryLeakDetector:
    """内存泄漏检测器"""
    
    def __init__(self, threshold_mb: float = 50.0, check_interval: float = 5.0):
        """
        初始化内存泄漏检测器
        
        Args:
            threshold_mb: 内存增长阈值（MB）
            check_interval: 检查间隔（秒）
        """
        self.threshold_mb = threshold_mb
        self.check_interval = check_interval
        self.memory_history = deque(maxlen=100)
        self.reference_counts = defaultdict(int)
        self.watched_objects = weakref.WeakSet()
        self.lock = threading.RLock()
        self.monitoring = False
        self.monitor_thread = None
        
    def record_memory_snapshot(self):
        """记录内存快照"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            snapshot = {
                'timestamp': time.time(),
                'memory_mb': memory_mb,
                'cpu_percent': psutil.cpu_percent(),
                'system_memory_percent': psutil.virtual_memory().percent
            }
            
            with self.lock:
                self.memory_history.append(snapshot)
            
            # 检查内存泄漏
            self._check_memory_leak(snapshot)
            
        except Exception as e:
            logger.error(f"记录内存快照失败: {e}")
    
    def _check_memory_leak(self, current_snapshot: Dict[str, Any]):
        """检查内存泄漏"""
        with self.lock:
            if len(self.memory_history) < 2:
                return
            
            # 获取早期快照
            early_snapshot = self.memory_history[-2]
            memory_diff = current_snapshot['memory_mb'] - early_snapshot['memory_mb']
            
            # 如果内存增长超过阈值，报告可能的内存泄漏
            if memory_diff > self.threshold_mb:
                logger.warning(f"检测到可能的内存泄漏: 内存增长 {memory_diff:.2f}MB")
                
                # 分析对象引用
                self._analyze_object_references()
    
    def _analyze_object_references(self):
        """分析对象引用"""
        with self.lock:
            for obj_id, count in self.reference_counts.items():
                if count > 10:  # 引用数过多的对象
                    logger.warning(f"对象 {obj_id} 引用计数过高: {count}")
            
            # 检查是否有未释放的大对象
            large_objects = [obj for obj in self.watched_objects 
                           if hasattr(obj, '__sizeof__') and sys.getsizeof(obj) > self._large_object_threshold]
            
            if large_objects:
                logger.info(f"发现 {len(large_objects)} 个大对象未被释放")
    
    def watch_object(self, obj: Any, name: str = None):
        """监控对象"""
        try:
            obj_id = id(obj)
            with self.lock:
                self.reference_counts[obj_id] += 1
                self.watched_objects.add(obj)
            
            name = name or f"Object_{obj_id}"
            logger.debug(f"开始监控对象: {name} (ID: {obj_id})")
            
        except Exception as e:
            logger.error(f"监控对象失败: {e}")
    
    def unwatch_object(self, obj: Any):
        """停止监控对象"""
        try:
            obj_id = id(obj)
            with self.lock:
                if obj_id in self.reference_counts:
                    self.reference_counts[obj_id] -= 1
                    if self.reference_counts[obj_id] <= 0:
                        del self.reference_counts[obj_id]
                        logger.debug(f"停止监控对象 ID: {obj_id}")
            
        except Exception as e:
            logger.error(f"停止监控对象失败: {e}")
    
    def start_monitoring(self):
        """开始监控"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("内存泄漏监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            logger.info("内存泄漏监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                self.record_memory_snapshot()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"内存监控循环错误: {e}")
                time.sleep(self.check_interval)
    
    def get_memory_trend(self) -> Dict[str, Any]:
        """获取内存趋势分析"""
        with self.lock:
            if len(self.memory_history) < 2:
                return {'trend': 'unknown', 'growth_rate_mb_per_hour': 0}
            
            memory_values = [snapshot['memory_mb'] for snapshot in self.memory_history]
            
            # 计算趋势
            if len(memory_values) >= 2:
                recent_memory = memory_values[-1]
                initial_memory = memory_values[0]
                time_diff_hours = (self.memory_history[-1]['timestamp'] - 
                                 self.memory_history[0]['timestamp']) / 3600
                
                growth_rate = (recent_memory - initial_memory) / max(time_diff_hours, 0.001)
                
                if growth_rate > 1:  # MB per hour
                    trend = 'increasing'
                elif growth_rate < -1:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
                
                return {
                    'trend': trend,
                    'growth_rate_mb_per_hour': growth_rate,
                    'memory_range_mb': [min(memory_values), max(memory_values)],
                    'sample_count': len(memory_values)
                }
            
            return {'trend': 'stable', 'growth_rate_mb_per_hour': 0}


class ResourceTracker:
    """资源跟踪器"""
    
    def __init__(self):
        self.tracked_resources = {}
        self.lock = threading.RLock()
    
    def track_file(self, filepath: str, resource_type: str = 'file'):
        """跟踪文件资源"""
        with self.lock:
            self.tracked_resources[filepath] = {
                'type': resource_type,
                'created_at': time.time(),
                'size_mb': 0,
                'access_count': 0
            }
    
    def track_memory_object(self, obj_id: int, obj_type: str, size_bytes: int):
        """跟踪内存对象"""
        with self.lock:
            self.tracked_resources[f"mem_{obj_id}"] = {
                'type': f'memory_{obj_type}',
                'created_at': time.time(),
                'size_mb': size_bytes / (1024 * 1024),
                'access_count': 0
            }
    
    def update_access(self, resource_id: str):
        """更新访问计数"""
        with self.lock:
            if resource_id in self.tracked_resources:
                self.tracked_resources[resource_id]['access_count'] += 1
    
    def get_resource_report(self) -> Dict[str, Any]:
        """获取资源报告"""
        with self.lock:
            total_files = sum(1 for r in self.tracked_resources.values() 
                            if r['type'] == 'file')
            total_memory_mb = sum(r['size_mb'] for r in self.tracked_resources.values() 
                                if r['type'].startswith('memory_'))
            
            high_access_resources = []
            for resource_id, info in self.tracked_resources.items():
                if info['access_count'] > 100:
                    high_access_resources.append({
                        'resource': resource_id,
                        'type': info['type'],
                        'access_count': info['access_count']
                    })
            
            return {
                'total_resources': len(self.tracked_resources),
                'total_files': total_files,
                'total_memory_mb': total_memory_mb,
                'high_access_resources': high_access_resources,
                'oldest_resource_age_hours': max((time.time() - r['created_at']) / 3600 
                                               for r in self.tracked_resources.values()) if self.tracked_resources else 0
            }


# 全局实例
_global_file_manager = SafeFileManager()
_global_memory_detector = MemoryLeakDetector()
_global_resource_tracker = ResourceTracker()


def get_global_file_manager() -> SafeFileManager:
    """获取全局文件管理器"""
    return _global_file_manager


def get_global_memory_detector() -> MemoryLeakDetector:
    """获取全局内存泄漏检测器"""
    return _global_memory_detector


def get_global_resource_tracker() -> ResourceTracker:
    """获取全局资源跟踪器"""
    return _global_resource_tracker


# 使用示例
if __name__ == "__main__":
    # 启动内存监控
    detector = get_global_memory_detector()
    detector.start_monitoring()
    
    # 测试文件操作
    file_manager = get_global_file_manager()
    
    # 创建测试数据
    test_data = {
        'models': {'key1': 'value1', 'key2': [1, 2, 3, 4, 5]},
        'config': {'setting1': True, 'setting2': 42},
        'metrics': {'accuracy': 0.95, 'loss': 0.05}
    }
    
    # 安全的文件保存
    print("测试安全文件操作...")
    
    success = file_manager.safe_save_json(test_data, "/tmp/test_data.json")
    print(f"JSON保存结果: {success}")
    
    # 批量保存
    batch_results = file_manager.batch_save(test_data, "/tmp", file_format='json')
    print(f"批量保存结果: {batch_results}")
    
    # 安全的文件加载
    loaded_data = file_manager.safe_load_json("/tmp/test_data.json")
    print(f"数据加载成功: {loaded_data == test_data}")
    
    # 内存监控测试
    print("\n测试内存监控...")
    
    large_data = [list(range(10000)) for _ in range(100)]
    detector.watch_object(large_data, "large_test_data")
    
    time.sleep(2)  # 等待监控
    
    memory_trend = detector.get_memory_trend()
    print(f"内存趋势: {memory_trend}")
    
    # 资源跟踪测试
    print("\n测试资源跟踪...")
    
    tracker = get_global_resource_tracker()
    tracker.track_file("/tmp/test_data.json")
    tracker.update_access("/tmp/test_data.json")
    
    report = tracker.get_resource_report()
    print(f"资源报告: {report}")
    
    # 文件操作统计
    stats = file_manager.get_stats()
    print(f"\n文件操作统计:")
    print(f"总操作数: {stats.total_operations}")
    print(f"成功操作数: {stats.successful_operations}")
    print(f"失败操作数: {stats.failed_operations}")
    print(f"总大小MB: {stats.total_size_mb:.2f}")
    print(f"压缩比: {stats.compression_ratio:.2%}")
    
    # 清理
    file_manager.cleanup_old_files("/tmp", max_age_days=0)  # 删除所有临时文件
    file_manager.force_cleanup()
    detector.stop_monitoring()
    
    print("\n文件操作和内存管理优化工具测试完成!")

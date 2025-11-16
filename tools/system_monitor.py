#!/usr/bin/env python3
"""
实时系统监控脚本
Real-time System Monitoring Script

监控系统的CPU、内存、磁盘、网络等资源使用情况
"""

import psutil
import time
import json
from datetime import datetime

def monitor_system():
    """监控系统资源"""
    print("开始系统监控...")
    
    # 创建监控数据存储
    monitoring_data = {
        'start_time': datetime.now().isoformat(),
        'cpu_percent': [],
        'memory_percent': [],
        'disk_usage': [],
        'network_io': [],
        'process_count': [],
        'load_average': []
    }
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < 900:  # 监控15分钟
            current_time = datetime.now()
            
            # CPU监控
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # 内存监控
            memory = psutil.virtual_memory()
            
            # 磁盘监控
            disk = psutil.disk_usage('/')
            
            # 网络监控
            net_io = psutil.net_io_counters()
            
            # 进程监控
            process_count = len(psutil.pids())
            
            # 负载平均（仅Linux/macOS）
            try:
                load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            except:
                load_avg = [0, 0, 0]
            
            # 记录数据
            monitoring_data['cpu_percent'].append({
                'time': current_time.isoformat(),
                'percent': cpu_percent,
                'count': cpu_count
            })
            
            monitoring_data['memory_percent'].append({
                'time': current_time.isoformat(),
                'percent': memory.percent,
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3)
            })
            
            monitoring_data['disk_usage'].append({
                'time': current_time.isoformat(),
                'percent': (disk.used / disk.total) * 100,
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3)
            })
            
            monitoring_data['network_io'].append({
                'time': current_time.isoformat(),
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            })
            
            monitoring_data['process_count'].append({
                'time': current_time.isoformat(),
                'count': process_count
            })
            
            monitoring_data['load_average'].append({
                'time': current_time.isoformat(),
                '1min': load_avg[0],
                '5min': load_avg[1],
                '15min': load_avg[2]
            })
            
            # 每30秒输出一次状态
            if len(monitoring_data['cpu_percent']) % 30 == 0:
                print(f"[{current_time.strftime('%H:%M:%S')}] CPU: {cpu_percent}%, 内存: {memory.percent}%, 进程: {process_count}")
            
            time.sleep(1)  # 每秒监控一次
            
    except KeyboardInterrupt:
        print("监控被用户中断")
    except Exception as e:
        print(f"监控错误: {e}")
    
    # 保存监控数据
    monitoring_data['end_time'] = datetime.now().isoformat()
    
    with open('/workspace/system_monitoring_data.json', 'w') as f:
        json.dump(monitoring_data, f, indent=2)
    
    print("系统监控完成，数据已保存到 system_monitoring_data.json")

if __name__ == '__main__':
    import os
    monitor_system()
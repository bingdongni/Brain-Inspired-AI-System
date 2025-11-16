#!/usr/bin/env python3
"""
测试进度检查脚本
Test Progress Check Script
"""

import time
import os
from datetime import datetime

def check_test_progress():
    """检查测试进度"""
    print("=== 稳定性测试进度检查 ===")
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查进程状态
    try:
        # 检查系统监控进程
        if os.path.exists('/workspace/system_monitoring_data.json'):
            stat = os.stat('/workspace/system_monitoring_data.json')
            print(f"系统监控数据文件已创建: {datetime.fromtimestamp(stat.st_mtime).strftime('%H:%M:%S')}")
            
        # 检查日志文件
        if os.path.exists('/workspace/stability_test.log'):
            with open('/workspace/stability_test.log', 'r') as f:
                lines = f.readlines()
                print(f"稳定性测试日志: {len(lines)} 行")
                
                # 显示最后几行
                print("=== 最后几条日志 ===")
                for line in lines[-5:]:
                    print(line.strip())
        else:
            print("稳定性测试日志文件尚未创建")
            
    except Exception as e:
        print(f"检查进度时出错: {e}")
    
    # 手动检查系统资源
    try:
        import psutil
        
        print("\n=== 当前系统状态 ===")
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"CPU使用率: {cpu_percent}%")
        
        # 内存
        memory = psutil.virtual_memory()
        print(f"内存使用率: {memory.percent}%")
        print(f"可用内存: {memory.available / (1024**3):.2f} GB")
        
        # 磁盘
        disk = psutil.disk_usage('/')
        print(f"磁盘使用率: {(disk.used / disk.total) * 100:.1f}%")
        
        # 进程数
        process_count = len(psutil.pids())
        print(f"当前进程数: {process_count}")
        
    except Exception as e:
        print(f"获取系统状态时出错: {e}")

if __name__ == '__main__':
    check_test_progress()
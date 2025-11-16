#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Brain-Inspired AI Framework - 主入口
====================================

这是脑启发AI框架的主入口文件，提供统一的命令行接口。

使用方式:
    python main.py --help
    python main.py train --model-type hippocampus
    python main.py demo --demo-type basic
    python main.py interactive

主要功能:
- 模型训练和评估
- 交互式演示
- 系统信息查看
- 配置管理
- 性能监控
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

# 检查和安装依赖
def check_dependencies():
    """检查依赖包是否已安装"""
    required_packages = [
        'torch', 'numpy', 'scipy', 'pandas', 'scikit-learn',
        'matplotlib', 'seaborn', 'click', 'rich', 'pyyaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("警告: 以下依赖包未安装:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\n请运行以下命令安装依赖:")
        print("pip install -r requirements.txt")
        print("pip install -e .")
        return False
    
    return True

def main():
    """主入口函数"""
    try:
        # 检查依赖
        if not check_dependencies():
            sys.exit(1)
        
        # 导入并运行CLI
        from brain_ai.cli import cli
        cli()
        
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(0)
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已正确安装 brain_ai 包")
        print("运行: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
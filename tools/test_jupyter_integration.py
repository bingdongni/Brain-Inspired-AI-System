#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Jupyter集成测试脚本
==================

测试Jupyter Notebook集成的各个功能组件

作者: Brain-Inspired AI Team
日期: 2025-11-16
"""

import sys
import os
import traceback
import importlib
from datetime import datetime

# 添加路径
sys.path.append('/workspace/brain-inspired-ai/ui')

def test_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")
    try:
        import jupyter_core
        print(f"  ✅ jupyter_core: {jupyter_core.__version__}")
    except ImportError as e:
        print(f"  ❌ jupyter_core: {e}")
        return False
    
    try:
        import ipywidgets
        print(f"  ✅ ipywidgets: {ipywidgets.__version__}")
    except ImportError as e:
        print(f"  ❌ ipywidgets: {e}")
        return False
    
    try:
        import plotly
        print(f"  ✅ plotly: {plotly.__version__}")
    except ImportError as e:
        print(f"  ❌ plotly: {e}")
        return False
    
    try:
        import numpy as np
        print(f"  ✅ numpy: {np.__version__}")
    except ImportError as e:
        print(f"  ❌ numpy: {e}")
        return False
    
    try:
        import matplotlib
        print(f"  ✅ matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ❌ matplotlib: {e}")
        return False
    
    return True

def test_jupyter_integration():
    """测试Jupyter集成模块"""
    print("\n🔍 测试Jupyter集成模块...")
    try:
        from jupyter_integration import (
            notebook_ui,
            show_brain_dashboard,
            show_training_console,
            show_performance_dashboard,
            show_system_diagram,
            create_brain_monitor_widget,
            create_training_widget,
            create_performance_chart
        )
        print("  ✅ 所有函数导入成功")
        return True
    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        traceback.print_exc()
        return False

def test_widgets():
    """测试ipywidgets功能"""
    print("\n🔍 测试ipywidgets功能...")
    try:
        from jupyter_integration import create_brain_monitor_widget, create_training_widget
        
        brain_widget = create_brain_monitor_widget()
        print("  ✅ 大脑监控小部件创建成功")
        
        training_widget = create_training_widget()
        print("  ✅ 训练控制小部件创建成功")
        
        return True
    except Exception as e:
        print(f"  ❌ 小部件创建失败: {e}")
        traceback.print_exc()
        return False

def test_charts():
    """测试图表功能"""
    print("\n🔍 测试图表功能...")
    try:
        from jupyter_integration import create_performance_chart
        
        chart = create_performance_chart()
        print("  ✅ 性能图表创建成功")
        
        # 检查图表属性
        if hasattr(chart, 'data'):
            print(f"  ✅ 图表数据点数量: {len(chart.data)}")
        
        return True
    except Exception as e:
        print(f"  ❌ 图表创建失败: {e}")
        traceback.print_exc()
        return False

def test_ui_components():
    """测试UI组件"""
    print("\n🔍 测试UI组件...")
    try:
        from jupyter_integration import (
            JupyterUIIntegration
        )
        
        ui = JupyterUIIntegration()
        
        # 测试仪表板嵌入
        dashboard_html = ui.embed_ui_dashboard()
        print("  ✅ 主仪表板HTML生成成功")
        
        # 测试训练界面
        training_html = ui.embed_training_interface()
        print("  ✅ 训练界面HTML生成成功")
        
        # 测试性能监控
        performance_html = ui.embed_performance_monitor()
        print("  ✅ 性能监控HTML生成成功")
        
        # 测试系统架构
        architecture_html = ui.embed_system_architecture()
        print("  ✅ 系统架构HTML生成成功")
        
        return True
    except Exception as e:
        print(f"  ❌ UI组件测试失败: {e}")
        traceback.print_exc()
        return False

def test_notebook_file():
    """测试notebook文件"""
    print("\n🔍 测试notebook文件...")
    notebook_path = "/workspace/brain-inspired-ai/ui/界面使用演示.ipynb"
    
    if not os.path.exists(notebook_path):
        print(f"  ❌ Notebook文件不存在: {notebook_path}")
        return False
    
    try:
        import json
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_data = json.load(f)
        
        # 检查notebook结构
        if 'cells' in notebook_data:
            cell_count = len(notebook_data['cells'])
            print(f"  ✅ Notebook文件格式正确，包含 {cell_count} 个单元格")
            
            # 统计单元格类型
            markdown_cells = sum(1 for cell in notebook_data['cells'] if cell['cell_type'] == 'markdown')
            code_cells = sum(1 for cell in notebook_data['cells'] if cell['cell_type'] == 'code')
            
            print(f"  ✅ Markdown单元格: {markdown_cells}")
            print(f"  ✅ 代码单元格: {code_cells}")
            
        else:
            print("  ❌ Notebook文件格式不正确")
            return False
        
        return True
    except Exception as e:
        print(f"  ❌ Notebook文件测试失败: {e}")
        traceback.print_exc()
        return False

def test_kernel_connection():
    """测试Python内核连接"""
    print("\n🔍 测试Python内核连接...")
    try:
        import sys
        print(f"  ✅ Python版本: {sys.version}")
        
        # 测试基本Python功能
        test_code = """
import numpy as np
import matplotlib.pyplot as plt

# 简单计算测试
result = np.sum([1, 2, 3, 4, 5])
print(f"NumPy计算测试: {result}")

# 列表操作测试
test_list = [1, 2, 3, 4, 5]
squared = [x**2 for x in test_list]
print(f"列表推导式测试: {squared}")
"""
        
        exec(test_code)
        print("  ✅ Python内核连接正常")
        return True
    except Exception as e:
        print(f"  ❌ Python内核测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("🧪 Jupyter集成功能测试开始")
    print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    tests = [
        ("模块导入", test_imports),
        ("Jupyter集成模块", test_jupyter_integration),
        ("ipywidgets功能", test_widgets),
        ("图表功能", test_charts),
        ("UI组件", test_ui_components),
        ("Notebook文件", test_notebook_file),
        ("Python内核连接", test_kernel_connection),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 执行测试: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ 测试异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20s} : {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-" * 60)
    print(f"总计: {passed + failed} 项测试")
    print(f"通过: {passed} 项")
    print(f"失败: {failed} 项")
    print(f"成功率: {passed / (passed + failed) * 100:.1f}%")
    
    if failed == 0:
        print("\n🎉 所有测试通过！Jupyter集成功能正常工作。")
    else:
        print(f"\n⚠️  有 {failed} 项测试失败，请检查相关功能。")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

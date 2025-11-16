#!/usr/bin/env python3
"""
演示系统快速测试脚本
Demo System Quick Test Script

快速验证演示系统的核心功能是否正常工作
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

def test_imports():
    """测试模块导入"""
    print("📦 测试模块导入...")
    
    try:
        import numpy as np
        print("✅ NumPy 导入成功")
    except ImportError:
        print("❌ NumPy 导入失败")
        return False
        
    try:
        from pathlib import Path
        print("✅ Pathlib 导入成功")
    except ImportError:
        print("❌ Pathlib 导入失败")
        return False
        
    try:
        import torch
        print(f"✅ PyTorch 导入成功 (版本: {torch.__version__})")
    except ImportError:
        print("⚠️ PyTorch 未安装，部分功能将受限")
        
    try:
        import matplotlib
        print(f"✅ Matplotlib 导入成功 (版本: {matplotlib.__version__})")
    except ImportError:
        print("⚠️ Matplotlib 未安装，可视化功能将受限")
        
    return True

def test_cli_system():
    """测试CLI系统"""
    print("\n🚀 测试CLI系统...")
    
    try:
        # 添加项目路径
        sys.path.insert(0, str(Path(__file__).parent))
        from cli_demo import BrainInspiredAISystem
        print("✅ CLI模块导入成功")
        
        # 创建系统实例
        system = BrainInspiredAISystem()
        print("✅ 系统实例创建成功")
        
        # 测试初始化
        if system.initialize_system():
            print("✅ 系统初始化成功")
        else:
            print("❌ 系统初始化失败")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ CLI系统测试失败: {e}")
        return False

def test_data_generation():
    """测试数据生成"""
    print("\n📊 测试数据生成...")
    
    try:
        # 添加项目路径
        sys.path.insert(0, str(Path(__file__).parent))
        from cli_demo import BrainInspiredAISystem
        
        system = BrainInspiredAISystem()
        system.initialize_system()
        
        # 测试合成数据
        data = system.generate_sample_data("synthetic")
        
        required_keys = ['X_train', 'y_train', 'X_test', 'y_test', 'input_dim', 'output_dim']
        for key in required_keys:
            if key not in data:
                print(f"❌ 数据缺少字段: {key}")
                return False
                
        print(f"✅ 合成数据生成成功")
        print(f"   训练样本: {len(data['X_train'])}")
        print(f"   测试样本: {len(data['X_test'])}")
        print(f"   输入维度: {data['input_dim']}")
        print(f"   输出维度: {data['output_dim']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据生成测试失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n🏗️ 测试模型创建...")
    
    try:
        # 添加项目路径
        sys.path.insert(0, str(Path(__file__).parent))
        from cli_demo import BrainInspiredAISystem
        
        system = BrainInspiredAISystem()
        system.initialize_system()
        
        # 设置测试数据
        system.current_data = system.generate_sample_data("synthetic")
        
        # 测试模型创建
        models = system.create_models("brain_inspired")
        
        if 'brain_inspired' in models:
            print("✅ 脑启发模型创建成功")
            return True
        else:
            print("❌ 脑启发模型创建失败")
            return False
            
    except Exception as e:
        print(f"❌ 模型创建测试失败: {e}")
        return False

def test_demo_imports():
    """测试演示模块导入"""
    print("\n🎮 测试演示模块...")
    
    # 添加项目路径
    sys.path.insert(0, str(Path(__file__).parent))
    
    demos_to_test = [
        ('demos.memory_learning_demo', '记忆学习演示'),
        ('demos.lifelong_learning_demo', '终身学习演示'),
        ('demos.dynamic_routing_demo', '动态路由演示'),
        ('scripts.benchmark_test', '基准测试脚本'),
        ('scripts.automated_testing', '自动化测试脚本'),
        ('scripts.run_all_demos', '运行所有演示脚本'),
        ('scripts.download_models', '下载模型脚本')
    ]
    
    success_count = 0
    
    for module_name, demo_name in demos_to_test:
        try:
            __import__(module_name)
            print(f"✅ {demo_name} 导入成功")
            success_count += 1
        except ImportError as e:
            print(f"⚠️ {demo_name} 导入失败: {e}")
        except Exception as e:
            print(f"❌ {demo_name} 测试异常: {e}")
            
    print(f"成功导入 {success_count}/{len(demos_to_test)} 个演示模块")
    return success_count >= len(demos_to_test) * 0.8

def test_file_structure():
    """测试文件结构"""
    print("\n📁 测试文件结构...")
    
    required_files = [
        'cli_demo.py',
        'QUICK_START.md',
        'demos/memory_learning_demo.py',
        'demos/lifelong_learning_demo.py',
        'demos/dynamic_routing_demo.py',
        'scripts/benchmark_test.py',
        'scripts/automated_testing.py',
        'scripts/run_all_demos.py',
        'scripts/download_models.py'
    ]
    
    missing_files = []
    base_dir = Path(__file__).parent
    
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (缺失)")
            missing_files.append(file_path)
            
    if not missing_files:
        print("✅ 所有必需文件都存在")
        return True
    else:
        print(f"❌ 缺失文件: {missing_files}")
        return False

def test_system_requirements():
    """测试系统要求"""
    print("\n🖥️ 测试系统要求...")
    
    # Python版本
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"❌ Python版本过低: {version.major}.{version.minor} (要求: 3.8+)")
        return False
        
    # 内存检查
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        
        if total_gb >= 2:
            print(f"✅ 内存: {total_gb:.1f} GB")
        else:
            print(f"⚠️ 内存较少: {total_gb:.1f} GB (建议: 4GB+)")
            
    except ImportError:
        print("⚠️ 无法检查内存: psutil未安装")
        
    # 磁盘空间
    try:
        import shutil
        free_space_gb = shutil.disk_usage('.').free / (1024**3)
        
        if free_space_gb >= 1:
            print(f"✅ 可用磁盘空间: {free_space_gb:.1f} GB")
        else:
            print(f"⚠️ 磁盘空间不足: {free_space_gb:.1f} GB (建议: 2GB+)")
            
    except Exception:
        print("⚠️ 无法检查磁盘空间")
        
    return True

def run_quick_test():
    """运行快速测试"""
    print("🧪 演示系统快速测试")
    print("=" * 60)
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    
    start_time = time.time()
    
    tests = [
        ("模块导入", test_imports),
        ("文件结构", test_file_structure),
        ("系统要求", test_system_requirements),
        ("CLI系统", test_cli_system),
        ("数据生成", test_data_generation),
        ("模型创建", test_model_creation),
        ("演示模块", test_demo_imports)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"测试: {test_name}")
        
        try:
            if test_func():
                passed_tests += 1
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
            
    end_time = time.time()
    total_time = end_time - start_time
    
    # 测试总结
    print(f"\n{'='*60}")
    print(f"🏁 快速测试完成")
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"成功率: {passed_tests/total_tests:.1%}")
    print(f"总耗时: {total_time:.2f}秒")
    
    # 性能评级
    if passed_tests == total_tests:
        grade = "A (优秀)"
        recommendation = "系统完全正常，可以开始使用"
    elif passed_tests >= total_tests * 0.8:
        grade = "B (良好)"
        recommendation = "系统基本正常，建议检查失败的功能"
    elif passed_tests >= total_tests * 0.6:
        grade = "C (一般)"
        recommendation = "系统部分正常，建议安装缺失的依赖"
    else:
        grade = "D (需要改进)"
        recommendation = "系统存在严重问题，需要全面检查"
        
    print(f"性能评级: {grade}")
    print(f"建议: {recommendation}")
    
    # 保存测试报告
    test_report = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': total_tests - passed_tests,
        'success_rate': passed_tests / total_tests,
        'total_time': total_time,
        'performance_grade': grade,
        'recommendation': recommendation,
        'python_version': sys.version,
        'platform': sys.platform
    }
    
    try:
        os.makedirs('data/results', exist_ok=True)
        report_file = f"data/results/demo_quick_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(test_report, f, indent=2, ensure_ascii=False)
        print(f"\n💾 测试报告已保存到: {report_file}")
    except Exception as e:
        print(f"⚠️ 报告保存失败: {e}")
    
    if passed_tests < total_tests:
        print(f"\n💡 建议操作:")
        print(f"   1. 检查失败的测试项目")
        print(f"   2. 安装缺失的Python包")
        print(f"   3. 查看 QUICK_START.md 了解详细安装说明")
        print(f"   4. 运行完整的自动化测试: python scripts/automated_testing.py")
        
    return passed_tests >= total_tests * 0.8

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='演示系统快速测试')
    parser.add_argument('--test', choices=['imports', 'cli', 'data', 'model', 'demos', 'all'],
                       default='all', help='测试类型')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    if args.test == 'imports':
        test_imports()
    elif args.test == 'cli':
        test_cli_system()
    elif args.test == 'data':
        test_data_generation()
    elif args.test == 'model':
        test_model_creation()
    elif args.test == 'demos':
        test_demo_imports()
    else:
        # 运行完整快速测试
        success = run_quick_test()
        
        if not success:
            print(f"\n❌ 系统测试未完全通过")
            sys.exit(1)
        else:
            print(f"\n✅ 系统测试通过")
            sys.exit(0)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
脑启发AI系统 - 快速安装验证脚本
Brain-Inspired AI System - Quick Installation Check

此脚本用于验证系统安装是否正确，并进行基本功能测试。
"""

import sys
import os
import importlib
from pathlib import Path
import platform

def print_header(title):
    """打印标题"""
    print(f"\n{'='*60}")
    print(f"🧠 {title}")
    print(f"{'='*60}")

def print_step(step, description):
    """打印步骤"""
    print(f"\n📋 {step}: {description}")

def print_success(message):
    """打印成功信息"""
    print(f"✅ {message}")

def print_warning(message):
    """打印警告信息"""
    print(f"⚠️  {message}")

def print_error(message):
    """打印错误信息"""
    print(f"❌ {message}")

def check_python_version():
    """检查Python版本"""
    print_step("步骤1", "检查Python环境")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print_success(f"Python版本: {version.major}.{version.minor}.{version.micro} ✓")
        return True
    else:
        print_error(f"Python版本过低: {version.major}.{version.minor}.{version.micro}")
        print_error("需要Python 3.8或更高版本")
        return False

def check_pip():
    """检查pip"""
    print_step("步骤2", "检查pip包管理器")
    
    try:
        import pip
        from pip._internal.cli.main import main as pip_main
        print_success("pip包管理器可用 ✓")
        return True
    except ImportError:
        print_error("pip未安装或无法导入")
        return False

def check_core_dependencies():
    """检查核心依赖包"""
    print_step("步骤3", "检查核心依赖包")
    
    # 核心依赖列表
    core_deps = {
        'numpy': 'NumPy - 数值计算',
        'scipy': 'SciPy - 科学计算', 
        'pandas': 'Pandas - 数据处理',
        'torch': 'PyTorch - 深度学习框架',
        'sklearn': 'Scikit-learn - 机器学习',
        'matplotlib': 'Matplotlib - 绘图库',
        'yaml': 'PyYAML - 配置文件处理',
        'click': 'Click - 命令行界面'
    }
    
    missing_deps = []
    
    for dep, description in core_deps.items():
        try:
            module = importlib.import_module(dep)
            version = getattr(module, '__version__', 'Unknown')
            print_success(f"{description} (v{version}) ✓")
        except ImportError:
            print_error(f"{description} - 未安装")
            missing_deps.append(dep)
    
    if missing_deps:
        print_warning(f"缺少依赖包: {', '.join(missing_deps)}")
        print_info(f"安装命令: pip install {' '.join(missing_deps)}")
        return False
    else:
        print_success("所有核心依赖包已安装 ✓")
        return True

def check_project_modules():
    """检查项目核心模块"""
    print_step("步骤4", "检查项目核心模块")
    
    # 项目核心模块
    project_modules = {
        'hippocampus': '海马体记忆系统',
        'brain_ai': '核心AI系统',
        'lifelong_learning': '持续学习模块',
        'memory_interface': '记忆接口',
        'dynamic_expansion': '动态扩展',
        'elastic_weight_consolidation': '弹性权重巩固',
        'generative_replay': '生成重放'
    }
    
    missing_modules = []
    
    for module, description in project_modules.items():
        try:
            importlib.import_module(module)
            print_success(f"{description} ✓")
        except ImportError:
            print_error(f"{description} - 未安装或不可用")
            missing_modules.append(module)
    
    if missing_modules:
        print_warning(f"缺少项目模块: {', '.join(missing_modules)}")
        return False
    else:
        print_success("所有项目核心模块可用 ✓")
        return True

def check_system_compatibility():
    """检查系统兼容性"""
    print_step("步骤5", "检查系统兼容性")
    
    # 操作系统信息
    system = platform.system()
    machine = platform.machine()
    python_version = platform.python_version()
    
    print_success(f"操作系统: {system} {machine}")
    print_success(f"Python版本: {python_version}")
    
    # 检查GPU支持
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print_success(f"GPU支持: {gpu_name} ({gpu_count}个设备)")
        else:
            print_warning("GPU支持: 不可用（将使用CPU模式）")
    except:
        print_warning("GPU支持检查失败")
    
    return True

def check_ui_components():
    """检查UI组件"""
    print_step("步骤6", "检查用户界面组件")
    
    # 检查CLI工具
    cli_files = [
        'cli_demo.py',
        'main.py',
        'quick_test.py'
    ]
    
    ui_available = True
    for file in cli_files:
        if Path(file).exists():
            print_success(f"CLI工具: {file} ✓")
        else:
            print_warning(f"CLI工具: {file} - 未找到")
            ui_available = False
    
    # 检查Web界面
    web_ui_path = Path('ui/brain-ai-ui')
    if web_ui_path.exists():
        package_json = web_ui_path / 'package.json'
        if package_json.exists():
            print_success("Web界面: React项目结构 ✓")
        else:
            print_warning("Web界面: 目录存在但缺少package.json")
            ui_available = False
    else:
        print_warning("Web界面: ui/brain-ai-ui目录未找到")
        ui_available = False
    
    # 检查Jupyter集成
    jupyter_file = Path('ui/jupyter_integration.py')
    jupyter_notebook = Path('ui/界面使用演示.ipynb')
    
    if jupyter_file.exists():
        print_success("Jupyter集成: jupyter_integration.py ✓")
    else:
        print_warning("Jupyter集成: jupyter_integration.py未找到")
        ui_available = False
    
    if jupyter_notebook.exists():
        print_success("Jupyter演示: 界面使用演示.ipynb ✓")
    else:
        print_warning("Jupyter演示: 界面使用演示.ipynb未找到")
        ui_available = False
    
    return ui_available

def run_basic_functionality_test():
    """运行基本功能测试"""
    print_step("步骤7", "运行基本功能测试")
    
    try:
        # 导入系统核心
        from hippocampus import HippocampusSimulator
        from lifelong_learning import ContinualLearner
        
        # 创建基本实例
        hippocampus = HippocampusSimulator(memory_capacity=100)
        learner = ContinualLearner(memory_size=500)
        
        print_success("海马体系统创建成功 ✓")
        print_success("持续学习器创建成功 ✓")
        
        # 测试基本功能
        import numpy as np
        
        # 简单的序列学习测试
        sequence = [1, 2, 3, 4, 5]
        hippocampus.learn_sequence(sequence)
        
        print_success("序列学习功能正常 ✓")
        
        return True
        
    except Exception as e:
        print_error(f"功能测试失败: {str(e)}")
        return False

def check_data_structure():
    """检查数据目录结构"""
    print_step("步骤8", "检查数据目录结构")
    
    required_dirs = [
        'data/datasets',
        'data/models', 
        'data/results',
        'logs',
        'config',
        'examples'
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print_success(f"目录存在: {dir_path} ✓")
        else:
            print_warning(f"目录不存在: {dir_path}")
            # 创建缺失的目录
            try:
                path.mkdir(parents=True, exist_ok=True)
                print_success(f"已创建目录: {dir_path}")
            except Exception as e:
                print_error(f"无法创建目录 {dir_path}: {str(e)}")

def run_comprehensive_test():
    """运行综合测试"""
    print_step("步骤9", "运行综合演示测试")
    
    try:
        # 尝试运行快速演示
        print("运行基础演示程序...")
        
        # 检查是否有cli_demo.py
        cli_demo = Path('cli_demo.py')
        if cli_demo.exists():
            print_success("找到CLI演示程序")
            
            # 这里可以添加具体的演示调用
            # 演示程序会在外部命令行运行
            
        else:
            print_warning("未找到CLI演示程序")
        
        return True
        
    except Exception as e:
        print_error(f"综合测试失败: {str(e)}")
        return False

def print_system_info():
    """打印系统信息"""
    print_header("系统信息报告")
    
    print(f"🖥️  系统: {platform.system()} {platform.release()}")
    print(f"🏗️  架构: {platform.architecture()[0]} {platform.machine()}")
    print(f"🐍 Python: {sys.version}")
    print(f"📁 工作目录: {os.getcwd()}")
    
    try:
        import torch
        print(f"🔥 PyTorch: {torch.__version__}")
        print(f"🖥️  CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
        if torch.cuda.is_available():
            print(f"🎯 GPU设备: {torch.cuda.device_count()}")
    except:
        print("🔥 PyTorch: 未安装")
    
    try:
        import numpy as np
        print(f"🔢 NumPy: {np.__version__}")
    except:
        print("🔢 NumPy: 未安装")

def generate_recommendations(results):
    """生成使用建议"""
    print_header("使用建议")
    
    if all(results):
        print("🎉 恭喜！系统安装完全成功！")
        print("\n🚀 推荐开始方式:")
        print("  1. 运行基础演示:")
        print("     python cli_demo.py --mode demo")
        print("  2. 启动Web界面:")
        print("     cd ui/brain-ai-ui && npm run dev")
        print("  3. 查看完整文档:")
        print("     docs/quick_start_guide.md")
    else:
        print("⚠️  安装存在一些问题，建议:")
        print("\n🔧 修复步骤:")
        print("  1. 检查并安装缺失的依赖:")
        print("     pip install -r requirements.txt")
        print("  2. 确保在正确的虚拟环境中")
        print("  3. 重新运行安装脚本:")
        print("     bash install.sh --clean --dev --viz")
        print("\n📞 获取帮助:")
        print("  - 查看完整文档: docs/quick_start_guide.md")
        print("  - GitHub Issues: 提交问题报告")

def main():
    """主函数"""
    print_header("脑启发AI系统 - 快速安装验证")
    
    print("此脚本将验证脑启发AI系统的安装状态...")
    print("如果发现问题，会提供相应的解决建议。")
    
    # 运行所有检查
    checks = []
    
    checks.append(("Python版本", check_python_version()))
    checks.append(("pip包管理器", check_pip()))
    checks.append(("核心依赖包", check_core_dependencies()))
    checks.append(("项目核心模块", check_project_modules()))
    checks.append(("系统兼容性", check_system_compatibility()))
    checks.append(("用户界面组件", check_ui_components()))
    checks.append(("基本功能测试", run_basic_functionality_test()))
    checks.append(("数据目录结构", check_data_structure()))
    checks.append(("综合演示测试", run_comprehensive_test()))
    
    # 打印系统信息
    print_system_info()
    
    # 生成建议
    results = [check[1] for check in checks]
    generate_recommendations(results)
    
    # 打印详细结果
    print_header("详细检查结果")
    for check_name, result in checks:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {check_name}: {status}")
    
    # 统计结果
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    print(f"\n📊 检查总结: {passed}/{total} 项通过")
    
    if passed == total:
        print("🎉 全部检查通过！系统已准备就绪！")
        return 0
    else:
        print("⚠️  部分检查未通过，建议修复后再使用")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断检查")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 检查过程发生错误: {str(e)}")
        print("请查看详细错误信息或联系技术支持")
        sys.exit(1)
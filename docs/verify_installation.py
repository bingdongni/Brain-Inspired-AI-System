#!/usr/bin/env python3
"""
Brain AI Windows 11 安装验证脚本
快速验证所有组件是否正常工作
"""

import sys
import os
import time
import traceback
from datetime import datetime

# 设置路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class InstallationVerifier:
    def __init__(self):
        self.passed_tests = 0
        self.total_tests = 0
        self.test_results = []
    
    def log_test(self, test_name, passed, message=""):
        """记录测试结果"""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        result = {
            'name': test_name,
            'passed': passed,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        print(f"[{status}] {test_name}")
        if message:
            print(f"    {message}")
    
    def test_basic_imports(self):
        """测试基础包导入"""
        print("\n📦 测试基础科学计算包...")
        
        packages = [
            ('numpy', 'NumPy'),
            ('scipy', 'SciPy'),
            ('pandas', 'Pandas'),
            ('sklearn', 'Scikit-learn'),
            ('matplotlib', 'Matplotlib'),
            ('PIL', 'Pillow')
        ]
        
        for import_name, display_name in packages:
            try:
                __import__(import_name)
                self.log_test(f"{display_name}导入", True)
            except ImportError as e:
                self.log_test(f"{display_name}导入", False, f"未安装: {e}")
    
    def test_pytorch(self):
        """测试PyTorch"""
        print("\n🔥 测试PyTorch...")
        
        try:
            import torch
            self.log_test("PyTorch导入", True, f"版本: {torch.__version__}")
            
            # 测试张量创建
            x = torch.randn(100, 100)
            self.log_test("张量创建", True)
            
            # 测试张量运算
            y = torch.mm(x, x.t())
            self.log_test("张量运算", True)
            
            # GPU测试
            cuda_available = torch.cuda.is_available()
            self.log_test("CUDA可用性", cuda_available, 
                         f"CUDA: {torch.cuda.is_available()}")
            
            if cuda_available:
                # GPU计算测试
                x_gpu = x.to('cuda')
                y_gpu = torch.mm(x_gpu, x_gpu.t())
                self.log_test("GPU计算", True)
                
                # 获取GPU信息
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "None"
                self.log_test("GPU设备信息", True, 
                             f"{gpu_count} GPU(s): {gpu_name}")
            
        except ImportError:
            self.log_test("PyTorch导入", False, "PyTorch未安装")
        except Exception as e:
            self.log_test("PyTorch测试", False, f"错误: {str(e)}")
    
    def test_project_modules(self):
        """测试项目模块"""
        print("\n🧠 测试Brain AI项目模块...")
        
        # 检查项目结构
        required_files = [
            'requirements.txt',
            'cli_demo.py',
            'setup.py'
        ]
        
        for file_path in required_files:
            exists = os.path.exists(file_path)
            self.log_test(f"文件存在: {file_path}", exists)
        
        # 测试核心模块导入
        core_modules = [
            'src.core.brain_system',
            'src.modules.hippocampus.hippocampus_simulator',
            'src.modules.neocortex.neocortex_simulator',
            'src.modules.lifelong_learning.lifelong_learning_system'
        ]
        
        for module_name in core_modules:
            try:
                __import__(module_name)
                self.log_test(f"模块导入: {module_name.split('.')[-1]}", True)
            except ImportError as e:
                self.log_test(f"模块导入: {module_name.split('.')[-1]}", False, str(e))
    
    def test_configuration(self):
        """测试配置系统"""
        print("\n⚙️  测试配置系统...")
        
        try:
            import yaml
            self.log_test("PyYAML导入", True)
            
            # 测试配置读取
            config_files = ['config.yaml', 'config/development.yaml']
            for config_file in config_files:
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    self.log_test(f"配置读取: {config_file}", True, 
                                 f"键数: {len(config) if config else 0}")
                    break
            else:
                self.log_test("配置文件", False, "未找到配置文件")
                
        except ImportError:
            self.log_test("PyYAML导入", False, "PyYAML未安装")
        except Exception as e:
            self.log_test("配置系统", False, str(e))
    
    def test_cli_tools(self):
        """测试CLI工具"""
        print("\n💻 测试CLI工具...")
        
        try:
            import click
            self.log_test("Click导入", True)
            
            # 检查CLI演示脚本
            if os.path.exists('cli_demo.py'):
                self.log_test("CLI演示脚本存在", True)
                
                # 尝试导入CLI模块
                spec = __import__('cli_demo')
                self.log_test("CLI模块导入", True)
            else:
                self.log_test("CLI演示脚本存在", False, "cli_demo.py不存在")
                
        except Exception as e:
            self.log_test("CLI工具", False, str(e))
    
    def test_jupyter_integration(self):
        """测试Jupyter集成"""
        print("\n📓 测试Jupyter集成...")
        
        try:
            import jupyter
            import jupyterlab
            self.log_test("Jupyter导入", True, f"版本: {jupyter.__version__}")
            
            # 测试ipywidgets
            import ipywidgets
            self.log_test("IPywidgets导入", True)
            
        except ImportError:
            self.log_test("Jupyter导入", False, "Jupyter未安装")
        except Exception as e:
            self.log_test("Jupyter集成", False, str(e))
    
    def test_web_interface(self):
        """测试Web界面"""
        print("\n🌐 测试Web界面配置...")
        
        # 检查Web界面目录
        web_dirs = [
            'ui/brain-ai-ui',
            'ui'
        ]
        
        web_ui_exists = False
        for web_dir in web_dirs:
            if os.path.exists(web_dir):
                # 检查package.json
                package_json = os.path.join(web_dir, 'package.json')
                if os.path.exists(package_json):
                    web_ui_exists = True
                    self.log_test("Web界面目录", True, f"找到: {web_dir}")
                    break
        
        if not web_ui_exists:
            self.log_test("Web界面目录", False, "未找到Web界面配置")
        
        # 检查Node.js环境（如果在Windows环境中）
        try:
            import subprocess
            result = subprocess.run(['node', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                node_version = result.stdout.strip()
                self.log_test("Node.js环境", True, f"版本: {node_version}")
            else:
                self.log_test("Node.js环境", False, "Node.js不可用")
        except Exception:
            self.log_test("Node.js环境", False, "Node.js未安装或不可用")
    
    def test_performance(self):
        """测试性能"""
        print("\n⚡ 测试系统性能...")
        
        try:
            import numpy as np
            import time
            
            # NumPy性能测试
            size = 1000
            start_time = time.time()
            x = np.random.randn(size, size)
            y = np.random.randn(size, size)
            result = np.dot(x, y)
            numpy_time = time.time() - start_time
            
            self.log_test("NumPy性能", True, f"{size}x{size}矩阵乘法: {numpy_time:.3f}s")
            
            # PyTorch性能测试
            try:
                import torch
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    start_time = time.time()
                    x_gpu = torch.randn(size, size, device=device)
                    y_gpu = torch.randn(size, size, device=device)
                    result_gpu = torch.mm(x_gpu, y_gpu)
                    gpu_time = time.time() - start_time
                    
                    speedup = numpy_time / gpu_time
                    self.log_test("GPU性能", True, 
                                 f"GPU计算: {gpu_time:.3f}s, 加速比: {speedup:.1f}x")
                else:
                    self.log_test("GPU性能", False, "CUDA不可用")
                    
            except Exception as e:
                self.log_test("GPU性能", False, str(e))
                
        except Exception as e:
            self.log_test("性能测试", False, str(e))
    
    def test_memory_management(self):
        """测试内存管理"""
        print("\n💾 测试内存管理...")
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            self.log_test("内存监控", True, 
                         f"使用率: {memory.percent}%")
            
            # 检查内存使用是否过高
            if memory.percent > 90:
                self.log_test("内存使用状态", False, "内存使用率过高 (>90%)")
            elif memory.percent > 80:
                self.log_test("内存使用状态", True, "内存使用率较高 (80-90%)")
            else:
                self.log_test("内存使用状态", True, "内存使用正常")
                
        except ImportError:
            self.log_test("内存监控", False, "psutil未安装")
        except Exception as e:
            self.log_test("内存管理", False, str(e))
    
    def test_network_connectivity(self):
        """测试网络连接"""
        print("\n🌍 测试网络连接...")
        
        try:
            import socket
            
            # 测试常用端口
            test_hosts = [
                ('pypi.org', 443),
                ('github.com', 443),
                ('localhost', 8888)
            ]
            
            for host, port in test_hosts:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    
                    if result == 0:
                        self.log_test(f"网络连接: {host}:{port}", True)
                    else:
                        self.log_test(f"网络连接: {host}:{port}", False, "连接失败")
                        
                except Exception as e:
                    self.log_test(f"网络连接: {host}:{port}", False, str(e))
                    
        except Exception as e:
            self.log_test("网络测试", False, str(e))
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        print("🧠 Brain AI 安装验证")
        print("=" * 50)
        print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        
        try:
            # 按顺序运行所有测试
            self.test_basic_imports()
            self.test_pytorch()
            self.test_project_modules()
            self.test_configuration()
            self.test_cli_tools()
            self.test_jupyter_integration()
            self.test_web_interface()
            self.test_performance()
            self.test_memory_management()
            self.test_network_connectivity()
            
            self.print_summary()
            self.generate_report()
            
        except KeyboardInterrupt:
            print("\n\n⏹️  验证被用户中断")
        except Exception as e:
            print(f"\n❌ 验证过程中发生错误: {str(e)}")
            traceback.print_exc()
    
    def print_summary(self):
        """打印总结"""
        print("\n" + "=" * 50)
        print("📊 验证总结")
        print("=" * 50)
        
        passed_rate = (self.passed_tests / self.total_tests) * 100
        
        print(f"总测试数: {self.total_tests}")
        print(f"通过测试: {self.passed_tests}")
        print(f"失败测试: {self.total_tests - self.passed_tests}")
        print(f"通过率: {passed_rate:.1f}%")
        
        if passed_rate >= 90:
            print("🎉 安装状态: 优秀！所有主要功能正常。")
        elif passed_rate >= 70:
            print("✅ 安装状态: 良好！大部分功能正常，可能有一些可选组件缺失。")
        elif passed_rate >= 50:
            print("⚠️  安装状态: 一般！核心功能可用，但需要解决一些问题。")
        else:
            print("❌ 安装状态: 需要修复！存在多个问题需要解决。")
        
        print("\n建议下一步:")
        if passed_rate >= 90:
            print("🚀 可以开始使用Brain AI系统了！运行 python cli_demo.py --mode demo")
        elif passed_rate >= 70:
            print("🔧 修复失败的测试，然后开始使用")
        else:
            print("🛠️  请参考安装文档解决所有问题")
    
    def generate_report(self):
        """生成详细报告"""
        print("\n📄 生成详细报告...")
        
        try:
            # 生成JSON报告
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_tests': self.total_tests,
                    'passed_tests': self.passed_tests,
                    'pass_rate': (self.passed_tests / self.total_tests) * 100
                },
                'test_results': self.test_results
            }
            
            import json
            with open('verification_report.json', 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            print("✅ 报告已保存到: verification_report.json")
            
        except Exception as e:
            print(f"❌ 生成报告失败: {str(e)}")

def main():
    """主函数"""
    verifier = InstallationVerifier()
    verifier.run_comprehensive_test()

if __name__ == "__main__":
    main()
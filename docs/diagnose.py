#!/usr/bin/env python3
"""
Windows 11 Brain AI 环境诊断工具
完整版诊断脚本，包含所有检查项目
"""

import sys
import subprocess
import platform
import os
import time
import json
import psutil
from datetime import datetime

# 尝试导入必要的库
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

class BrainAIDiagnosis:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {},
            'python_environment': {},
            'dependencies': {},
            'gpu_status': {},
            'performance': {},
            'issues': [],
            'recommendations': []
        }
    
    def check_system_info(self):
        """检查系统信息"""
        print("🔍 检查系统信息...")
        
        try:
            system_info = {
                'os': platform.system(),
                'os_version': platform.release(),
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'python_version': sys.version,
                'python_executable': sys.executable,
                'ram_total': f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
                'ram_available': f"{psutil.virtual_memory().available / (1024**3):.1f} GB",
                'cpu_count': psutil.cpu_count(),
                'cpu_frequency': f"{psutil.cpu_freq().current:.0f} MHz" if psutil.cpu_freq() else "Unknown"
            }
            
            self.results['system_info'] = system_info
            
            print(f"✅ 操作系统: {system_info['os']} {system_info['os_version']}")
            print(f"✅ 架构: {system_info['architecture']}")
            print(f"✅ 内存: {system_info['ram_total']}")
            print(f"✅ CPU: {system_info['cpu_count']} cores @ {system_info['cpu_frequency']}")
            print(f"✅ Python: {platform.python_version()}")
            
        except Exception as e:
            self.results['issues'].append(f"系统信息检查失败: {str(e)}")
            print(f"❌ 系统信息检查失败: {str(e)}")
    
    def check_python_environment(self):
        """检查Python环境"""
        print("\n🐍 检查Python环境...")
        
        try:
            env_info = {
                'python_path': sys.executable,
                'python_version': sys.version,
                'pip_version': subprocess.run(['pip', '--version'], 
                                            capture_output=True, text=True).stdout.strip(),
                'virtual_env': sys.prefix != sys.base_prefix,
                'path': sys.path
            }
            
            self.results['python_environment'] = env_info
            
            print(f"✅ Python路径: {sys.executable}")
            print(f"✅ 虚拟环境: {'是' if env_info['virtual_env'] else '否'}")
            print(f"✅ pip版本: {env_info['pip_version'].split()[1]}")
            
        except Exception as e:
            self.results['issues'].append(f"Python环境检查失败: {str(e)}")
            print(f"❌ Python环境检查失败: {str(e)}")
    
    def check_dependencies(self):
        """检查依赖包"""
        print("\n📦 检查依赖包...")
        
        required_packages = [
            'torch', 'torchvision', 'numpy', 'scipy', 'pandas',
            'sklearn', 'matplotlib', 'jupyter', 'click', 'pyyaml',
            'tqdm', 'rich', 'psutil'
        ]
        
        dependencies = {}
        
        for package in required_packages:
            try:
                if package == 'sklearn':
                    import sklearn
                    version = sklearn.__version__
                else:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'Unknown')
                
                dependencies[package] = {
                    'installed': True,
                    'version': version,
                    'status': '✅'
                }
                print(f"✅ {package}: {version}")
                
            except ImportError:
                dependencies[package] = {
                    'installed': False,
                    'version': None,
                    'status': '❌'
                }
                print(f"❌ {package}: 未安装")
                self.results['issues'].append(f"缺少依赖包: {package}")
        
        self.results['dependencies'] = dependencies
    
    def check_gpu_status(self):
        """检查GPU状态"""
        print("\n🎮 检查GPU状态...")
        
        gpu_info = {
            'cuda_available': False,
            'gpu_count': 0,
            'gpu_devices': []
        }
        
        if TORCH_AVAILABLE:
            try:
                gpu_info['cuda_available'] = torch.cuda.is_available()
                gpu_info['gpu_count'] = torch.cuda.device_count()
                
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpu_device = {
                        'id': i,
                        'name': props.name,
                        'memory': f"{props.total_memory / (1024**3):.1f} GB",
                        'compute_capability': f"{props.major}.{props.minor}"
                    }
                    gpu_info['gpu_devices'].append(gpu_device)
                    
                print(f"✅ CUDA可用: {gpu_info['cuda_available']}")
                print(f"✅ GPU数量: {gpu_info['gpu_count']}")
                
                for device in gpu_info['gpu_devices']:
                    print(f"  GPU {device['id']}: {device['name']} ({device['memory']})")
                
            except Exception as e:
                print(f"❌ GPU检查失败: {str(e)}")
                self.results['issues'].append(f"GPU检查失败: {str(e)}")
        else:
            print("❌ PyTorch未安装，无法检查GPU")
            self.results['issues'].append("PyTorch未安装")
        
        # 检查NVIDIA驱动
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ NVIDIA驱动正常")
                gpu_info['nvidia_driver'] = True
            else:
                print("❌ NVIDIA驱动问题")
                gpu_info['nvidia_driver'] = False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("❌ NVIDIA驱动未安装或不可用")
            gpu_info['nvidia_driver'] = False
            self.results['issues'].append("NVIDIA驱动未安装")
        
        self.results['gpu_status'] = gpu_info
    
    def check_ports(self):
        """检查端口占用"""
        print("\n🔌 检查端口占用...")
        
        ports_to_check = {
            8888: 'Jupyter Lab',
            5173: 'Web界面 (开发)',
            6006: 'TensorBoard'
        }
        
        port_status = {}
        
        for port, description in ports_to_check.items():
            try:
                result = subprocess.run(
                    ['netstat', '-ano'], 
                    capture_output=True, 
                    text=True,
                    timeout=5
                )
                
                if f':{port}' in result.stdout:
                    port_status[port] = {
                        'status': '占用',
                        'description': description,
                        'available': False
                    }
                    print(f"⚠️  端口 {port} ({description}) 被占用")
                else:
                    port_status[port] = {
                        'status': '可用',
                        'description': description,
                        'available': True
                    }
                    print(f"✅ 端口 {port} ({description}) 可用")
                    
            except Exception as e:
                port_status[port] = {
                    'status': '检查失败',
                    'description': description,
                    'available': None,
                    'error': str(e)
                }
                print(f"❌ 端口 {port} 检查失败: {str(e)}")
        
        self.results['ports'] = port_status
    
    def run_performance_tests(self):
        """运行性能测试"""
        print("\n⚡ 运行性能测试...")
        
        performance = {}
        
        # CPU性能测试
        print("  测试CPU计算性能...")
        start_time = time.time()
        try:
            # NumPy计算测试
            if NUMPY_AVAILABLE:
                x = np.random.randn(1000, 1000)
                y = np.random.randn(1000, 1000)
                _ = np.dot(x, y)
                numpy_time = time.time() - start_time
                performance['numpy_matrix_multiply'] = f"{numpy_time:.4f}s"
                print(f"    NumPy矩阵乘法: {numpy_time:.4f}s")
            
            # Python循环测试
            start_time = time.time()
            result = sum(range(1000000))
            python_loop_time = time.time() - start_time
            performance['python_loop'] = f"{python_loop_time:.4f}s"
            print(f"    Python循环(1M次): {python_loop_time:.4f}s")
            
        except Exception as e:
            performance['cpu_test_error'] = str(e)
            print(f"  ❌ CPU测试失败: {str(e)}")
        
        # GPU性能测试
        if TORCH_AVAILABLE and torch.cuda.is_available():
            print("  测试GPU计算性能...")
            try:
                device = torch.device('cuda')
                x = torch.randn(1000, 1000, device=device)
                y = torch.randn(1000, 1000, device=device)
                
                # 预热
                for _ in range(3):
                    _ = torch.mm(x, y)
                
                # 正式测试
                start_time = time.time()
                result = torch.mm(x, y)
                gpu_time = time.time() - start_time
                
                performance['gpu_matrix_multiply'] = f"{gpu_time:.4f}s"
                print(f"    GPU矩阵乘法: {gpu_time:.4f}s")
                
                # 计算加速比
                if 'numpy_matrix_multiply' in performance:
                    speedup = float(performance['numpy_matrix_multiply'].replace('s', '')) / gpu_time
                    performance['gpu_speedup'] = f"{speedup:.1f}x"
                    print(f"    GPU加速比: {speedup:.1f}x")
                
            except Exception as e:
                performance['gpu_test_error'] = str(e)
                print(f"  ❌ GPU测试失败: {str(e)}")
        else:
            print("  跳过GPU测试 (CUDA不可用)")
        
        # 内存测试
        try:
            memory = psutil.virtual_memory()
            performance['memory_usage'] = {
                'total': f"{memory.total / (1024**3):.1f} GB",
                'used': f"{memory.used / (1024**3):.1f} GB",
                'available': f"{memory.available / (1024**3):.1f} GB",
                'percent': f"{memory.percent}%"
            }
            print(f"  内存使用率: {memory.percent}%")
            
        except Exception as e:
            performance['memory_test_error'] = str(e)
            print(f"  ❌ 内存测试失败: {str(e)}")
        
        self.results['performance'] = performance
    
    def check_project_structure(self):
        """检查项目结构"""
        print("\n📁 检查项目结构...")
        
        project_files = [
            'requirements.txt',
            'setup.py',
            'cli_demo.py',
            'src/',
            'demos/',
            'ui/brain-ai-ui/'
        ]
        
        project_structure = {}
        
        for file_path in project_files:
            exists = os.path.exists(file_path)
            project_structure[file_path] = exists
            status = "✅" if exists else "❌"
            print(f"{status} {file_path}")
        
        self.results['project_structure'] = project_structure
    
    def generate_recommendations(self):
        """生成建议"""
        print("\n💡 生成建议...")
        
        recommendations = []
        
        # 根据检查结果生成建议
        if not TORCH_AVAILABLE:
            recommendations.append("安装PyTorch: pip install torch torchvision torchaudio")
        
        gpu_status = self.results.get('gpu_status', {})
        if not gpu_status.get('cuda_available', False) and gpu_status.get('nvidia_driver', False):
            recommendations.append("安装CUDA版本的PyTorch以启用GPU加速")
        
        if self.results['issues']:
            recommendations.append("解决上述标记的问题以获得最佳性能")
        
        # 性能建议
        performance = self.results.get('performance', {})
        if 'gpu_speedup' in performance:
            speedup = float(performance['gpu_speedup'].replace('x', ''))
            if speedup > 5:
                recommendations.append("GPU加速效果良好，建议充分利用GPU进行训练")
        
        # 内存建议
        memory_info = performance.get('memory_usage', {})
        if memory_info.get('percent'):
            usage_percent = int(memory_info['percent'].replace('%', ''))
            if usage_percent > 80:
                recommendations.append("内存使用率较高，建议关闭不必要的程序或增加内存")
        
        recommendations.extend([
            "定期更新GPU驱动程序以获得最佳性能",
            "使用虚拟环境隔离项目依赖",
            "考虑使用SSD存储以提高I/O性能",
            "定期清理临时文件和缓存"
        ])
        
        self.results['recommendations'] = recommendations
        
        print("建议生成完成")
    
    def save_report(self, filename='diagnosis_report.json'):
        """保存诊断报告"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"\n📄 诊断报告已保存到: {filename}")
        except Exception as e:
            print(f"❌ 保存报告失败: {str(e)}")
    
    def print_summary(self):
        """打印总结"""
        print("\n" + "="*60)
        print("📊 诊断总结")
        print("="*60)
        
        # 系统状态
        print(f"🖥️  系统: {self.results['system_info'].get('os', 'Unknown')} {self.results['system_info'].get('os_version', 'Unknown')}")
        print(f"🧠 Python: {platform.python_version()}")
        print(f"💾 内存: {self.results['system_info'].get('ram_total', 'Unknown')}")
        
        # 依赖状态
        deps = self.results.get('dependencies', {})
        installed = sum(1 for v in deps.values() if v.get('installed', False))
        total = len(deps)
        print(f"📦 依赖: {installed}/{total} 已安装")
        
        # GPU状态
        gpu_info = self.results.get('gpu_status', {})
        print(f"🎮 GPU: {'可用' if gpu_info.get('cuda_available', False) else '不可用'}")
        
        # 问题统计
        issues = self.results.get('issues', [])
        if issues:
            print(f"⚠️  问题: {len(issues)} 个问题需要解决")
        else:
            print("✅ 没有发现问题")
        
        # 性能指标
        performance = self.results.get('performance', {})
        if 'gpu_speedup' in performance:
            print(f"⚡ 性能: GPU加速 {performance['gpu_speedup']}")
        
        print("\n" + "="*60)
    
    def run_full_diagnosis(self):
        """运行完整诊断"""
        print("🧠 Brain AI Windows 11 环境诊断")
        print("="*60)
        print(f"诊断时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        try:
            self.check_system_info()
            self.check_python_environment()
            self.check_dependencies()
            self.check_gpu_status()
            self.check_ports()
            self.run_performance_tests()
            self.check_project_structure()
            self.generate_recommendations()
            
            self.print_summary()
            self.save_report()
            
            # 打印建议
            print("\n💡 建议:")
            for i, rec in enumerate(self.results['recommendations'], 1):
                print(f"{i}. {rec}")
            
            print("\n" + "="*60)
            print("✅ 诊断完成！详细报告已保存。")
            
        except KeyboardInterrupt:
            print("\n\n⏹️  诊断被用户中断")
        except Exception as e:
            print(f"\n❌ 诊断过程中发生错误: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    diagnosis = BrainAIDiagnosis()
    diagnosis.run_full_diagnosis()

if __name__ == "__main__":
    main()
# Windows 11 完整安装指南

## 📋 目录
- [系统要求](#系统要求)
- [安装前准备](#安装前准备)
- [Python环境安装](#python环境安装)
- [依赖包安装](#依赖包安装)
- [Node.js和Web界面设置](#nodejs和web界面设置)
- [Jupyter环境配置](#jupyter环境配置)
- [GPU支持配置](#gpu支持配置)
- [CLI工具使用指南](#cli工具使用指南)
- [性能优化](#性能优化)
- [故障排除](#故障排除)
- [环境验证](#环境验证)

## 🎯 系统要求

### 最低配置
- **操作系统**: Windows 11 (21H2或更高版本)
- **内存**: 8GB RAM (推荐16GB+)
- **存储**: 至少20GB可用空间 (推荐50GB+)
- **Python**: 3.8+ (推荐Python 3.11或3.12)
- **网络**: 宽带连接 (用于下载依赖包)

### 推荐配置
- **处理器**: Intel i5/AMD Ryzen 5或更高
- **内存**: 32GB RAM
- **GPU**: NVIDIA RTX 3060或更高 (支持CUDA)
- **存储**: SSD 100GB+

## 🛠️ 安装前准备

### 1. 启用Windows功能

在开始之前，请确保以下Windows功能已启用：

```powershell
# 以管理员身份运行PowerShell，执行以下命令：
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-Management-PowerShell -All
```

### 2. 设置Windows Terminal（推荐）

```powershell
# 安装Windows Terminal (通过Microsoft Store)
# 或者通过winget
winget install Microsoft.WindowsTerminal
```

### 3. 配置Windows子系统Linux (WSL2) - 可选

如果您想要Linux环境体验：

```powershell
# 启用WSL2
wsl --install

# 重启电脑后，设置Ubuntu作为默认发行版
wsl --setdefault Ubuntu-22.04
```

## 🐍 Python环境安装

### 方案一：官方Python安装（推荐）

#### 1. 下载Python

访问 [python.org](https://www.python.org/downloads/) 下载Python 3.11或3.12版本。

#### 2. 安装Python

```bash
# 下载完成后，运行安装程序
# 重要：勾选 "Add Python to PATH"
# 选择 "Customize Installation"
# 在 "Optional Features" 页面勾选：
# - pip
# - tcl/tk and IDLE
# - Python test suite
# - py launcher
# - for all users

# 在 "Advanced Options" 页面勾选：
# - Install for all users
# - Add Python to environment variables
```

#### 3. 验证安装

```bash
# 打开新的CMD或PowerShell窗口
python --version
pip --version

# 如果显示版本信息，说明安装成功
```

### 方案二：Anaconda/Miniconda安装

#### 1. 下载Anaconda

访问 [anaconda.com](https://www.anaconda.com/products/distribution) 下载最新版本。

#### 2. 安装Anaconda

```bash
# 运行安装程序
# 建议选择 "Just Me" 安装
# 勾选 "Add Anaconda to PATH"
# 勾选 "Register Anaconda as my default Python"
```

#### 3. 创建虚拟环境

```bash
# 创建专门的虚拟环境
conda create -n brain_ai python=3.11
conda activate brain_ai

# 验证环境
python --version
```

### 方案三：Microsoft Store安装

1. 打开Microsoft Store
2. 搜索 "Python 3.11"
3. 点击"安装"
4. 等待安装完成

## 📦 依赖包安装

### 1. 基础依赖安装

#### 使用pip安装

```bash
# 创建项目目录
mkdir brain-inspired-ai
cd brain-inspired-ai

# 创建虚拟环境 (如果未使用conda)
python -m venv brain_ai_env
brain_ai_env\Scripts\activate  # CMD
# 或
.\brain_ai_env\Scripts\Activate.ps1  # PowerShell

# 升级pip
python -m pip install --upgrade pip

# 安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow keras
pip install numpy scipy pandas scikit-learn
pip install matplotlib seaborn pillow opencv-python
pip install jupyter jupyterlab ipywidgets
```

#### 使用conda安装 (推荐速度更快)

```bash
# 如果使用Anaconda
conda create -n brain_ai python=3.11
conda activate brain_ai

# 安装深度学习框架
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install tensorflow
conda install numpy scipy pandas scikit-learn
conda install matplotlib seaborn pillow opencv
conda install jupyter jupyterlab ipywidgets

# 安装额外依赖
conda install -c conda-forge nixio pyefd
conda install wandb tensorboard rich
```

### 2. 完整依赖安装

```bash
# 如果您已获得完整项目包，使用以下命令安装所有依赖：
pip install -r requirements.txt
pip install -e .

# 或者使用conda
conda env create -f environment.yml
```

### 3. 验证PyTorch安装

```python
# 创建测试文件 test_pytorch.py
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU数量: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
```

```bash
# 运行测试
python test_pytorch.py
```

## 🌐 Node.js和Web界面设置

### 1. 安装Node.js

#### 方案一：官方安装包

```bash
# 访问 nodejs.org 下载LTS版本
# 安装时勾选 "Add to PATH"
# 验证安装
node --version
npm --version
```

#### 方案二：winget安装

```powershell
winget install OpenJS.NodeJS
```

#### 方案三：Chocolatey安装

```powershell
choco install nodejs
```

### 2. 安装pnpm（推荐）

```bash
# 安装pnpm（比npm更快）
npm install -g pnpm

# 验证安装
pnpm --version
```

### 3. 配置Web界面

```bash
# 进入Web界面目录
cd ui/brain-ai-ui

# 安装依赖
pnpm install

# 或者使用npm
npm install

# 启动开发服务器
pnpm dev

# 或者
npm run dev

# 访问地址: http://localhost:5173
```

### 4. 构建生产版本

```bash
# 构建生产版本
pnpm build

# 或者
npm run build

# 预览构建结果
pnpm preview
```

## 📓 Jupyter环境配置

### 1. 基础配置

```bash
# 启动Jupyter
jupyter lab

# 或者
jupyter notebook
```

### 2. 配置Jupyter

```bash
# 生成配置文件
jupyter lab --generate-config

# 设置密码
jupyter lab password

# 设置远程访问 (可选)
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

### 3. 安装Jupyter扩展

```bash
# 安装常用扩展
pip install jupyterlab-git
pip install jupyterlab-drawio
pip install @jupyter-widgets/jupyterlab-manager

# 启用扩展
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyterlab-git
```

### 4. 配置内核

```python
# 在Jupyter中注册Python内核
import sys
print(sys.executable)

# 或者使用命令行
python -m ipykernel install --user --name=brain_ai --display-name="Brain AI Environment"
```

## 🎮 GPU支持配置

### NVIDIA GPU配置

#### 1. 检查GPU支持

```bash
# 检查NVIDIA GPU
nvidia-smi

# 检查CUDA安装
nvcc --version
```

#### 2. 安装CUDA工具包

```powershell
# 方法1: 下载安装包
# 访问 developer.nvidia.com/cuda-downloads
# 下载Windows版本CUDA Toolkit

# 方法2: 使用winget
winget install Nvidia.CUDA

# 方法3: 使用conda
conda install cudatoolkit=11.8
```

#### 3. 配置PyTorch GPU支持

```bash
# CPU版本（已安装）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA版本 (选择对应的CUDA版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
```

#### 4. 验证GPU支持

```python
# 创建测试文件 test_gpu.py
import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 简单GPU测试
    device = torch.device('cuda')
    x = torch.rand(1000, 1000).to(device)
    y = torch.mm(x, x.t())
    print("GPU计算测试通过!")
else:
    print("GPU不可用，将使用CPU")
```

```bash
python test_gpu.py
```

### AMD GPU配置 (ROCm)

```bash
# AMD GPU支持有限，推荐使用CPU版本
# 或者使用OpenCL
pip install pyopencl
```

## 💻 CLI工具使用指南

### 1. 项目CLI工具

```bash
# 进入项目目录
cd brain-inspired-ai

# 查看可用命令
python cli_demo.py --help

# 运行演示模式
python cli_demo.py --mode demo

# 交互式模式
python cli_demo.py --mode interactive

# 自定义参数
python cli_demo.py --mode demo --dataset synthetic --model brain_inspired --epochs 5 --batch_size 32
```

### 2. 常用CLI命令

```bash
# 快速测试
python quick_test.py

# 完整测试套件
python comprehensive_test_suite.py

# 性能基准测试
python scripts/benchmark_test.py

# 运行特定演示
python demos/memory_learning_demo.py
python demos/lifelong_learning_demo.py
python demos/dynamic_routing_demo.py
```

### 3. 批处理脚本

创建 `run_demo.bat` 文件：

```batch
@echo off
cd /d "D:\path\to\brain-inspired-ai"
call brain_ai_env\Scripts\activate
echo 启动Brain AI演示...
python cli_demo.py --mode demo
pause
```

### 4. PowerShell脚本

创建 `run_demo.ps1` 文件：

```powershell
# 设置执行策略
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 切换到项目目录
Set-Location "D:\path\to\brain-inspired-ai"

# 激活虚拟环境
& .\brain_ai_env\Scripts\Activate.ps1

# 运行演示
Write-Host "启动Brain AI演示..." -ForegroundColor Green
python cli_demo.py --mode demo

# 保持窗口打开
Read-Host "按任意键退出"
```

## ⚡ 性能优化

### 1. Python性能优化

```python
# 创建优化配置文件 optimize_config.py
import os
import torch

# 设置环境变量
os.environ['PYTHONHASHSEED'] = '0'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# PyTorch优化
torch.set_num_threads(8)  # 设置线程数
torch.backends.cudnn.benchmark = True  # 启用cuDNN基准测试
torch.backends.cudnn.deterministic = False  # 禁用确定性模式以提高性能

# 如果有GPU，设置GPU优化
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # 清空GPU缓存
    torch.backends.cuda.matmul.allow_tf32 = True  # 允许TF32计算
    torch.backends.cudnn.allow_tf32 = True

print("性能优化配置完成")
```

### 2. 内存优化

```python
# 创建内存优化工具 memory_optimizer.py
import gc
import torch
import psutil
import threading
import time

class MemoryOptimizer:
    def __init__(self, threshold=80):
        self.threshold = threshold
        self.monitor = True
        self.start_monitoring()
    
    def get_memory_usage(self):
        """获取内存使用情况"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / 1024**3,  # GB
            'used': memory.used / 1024**3,   # GB
            'percent': memory.percent,
            'available': memory.available / 1024**3  # GB
        }
    
    def optimize_memory(self):
        """内存优化"""
        # Python垃圾回收
        gc.collect()
        
        # GPU内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 强制垃圾回收
        gc.collect()
    
    def monitor_memory(self):
        """内存监控"""
        while self.monitor:
            memory_info = self.get_memory_usage()
            
            if memory_info['percent'] > self.threshold:
                print(f"内存使用率过高 ({memory_info['percent']:.1f}%)，执行优化...")
                self.optimize_memory()
            
            time.sleep(30)  # 每30秒检查一次
    
    def start_monitoring(self):
        """启动内存监控"""
        thread = threading.Thread(target=self.monitor_memory, daemon=True)
        thread.start()
        print("内存监控已启动")
    
    def stop_monitoring(self):
        """停止内存监控"""
        self.monitor = False
        print("内存监控已停止")

# 使用示例
optimizer = MemoryOptimizer(threshold=85)
```

### 3. GPU性能优化

```python
# 创建GPU优化工具 gpu_optimizer.py
import torch
import time

class GPUOptimizer:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.name = torch.cuda.get_device_name(0)
            self.total_memory = torch.cuda.get_device_properties(0).total_memory
            print(f"GPU设备: {self.name}")
            print(f"总显存: {self.total_memory / 1024**3:.1f} GB")
        else:
            self.device = torch.device('cpu')
            self.name = "CPU"
            print("使用CPU进行计算")
    
    def optimize_settings(self):
        """优化GPU设置"""
        if torch.cuda.is_available():
            # 启用TF32 (30%性能提升)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # 启用benchmark模式
            torch.backends.cudnn.benchmark = True
            
            # 设置优化的数据类型
            torch.set_default_dtype(torch.float32)
            
            print("GPU优化设置已应用")
        else:
            print("无可用GPU，跳过优化")
    
    def benchmark_memory_speed(self, size=1000):
        """基准测试内存速度"""
        if not torch.cuda.is_available():
            print("无可用GPU进行内存基准测试")
            return
        
        # 分配测试
        start_time = time.time()
        test_tensor = torch.randn(size, size, device=self.device)
        allocate_time = time.time() - start_time
        
        # 复制测试
        start_time = time.time()
        test_tensor_copy = test_tensor.clone()
        copy_time = time.time() - start_time
        
        # 计算测试
        start_time = time.time()
        result = torch.mm(test_tensor, test_tensor)
        compute_time = time.time() - start_time
        
        print(f"GPU内存基准测试结果:")
        print(f"  分配时间 ({size}x{size}): {allocate_time:.4f}s")
        print(f"  复制时间: {copy_time:.4f}s")
        print(f"  计算时间: {compute_time:.4f}s")

# 使用示例
gpu_opt = GPUOptimizer()
gpu_opt.optimize_settings()
gpu_opt.benchmark_memory_speed()
```

### 4. 系统级优化

#### Windows性能设置

```powershell
# 创建优化脚本 optimize_windows.ps1

# 设置高性能电源计划
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# 禁用Windows索引（可选）
# Set-Service "WSearch" -StartupType Disabled

# 设置环境变量
[Environment]::SetEnvironmentVariable("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512", "Machine")
[Environment]::SetEnvironmentVariable("CUDA_CACHE_MAXSIZE", "2147483648", "Machine")

# 启用开发者模式
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock" /t REG_DWORD /f /v "AllowDevelopmentWithoutDevLicense" /d 1

Write-Host "Windows性能优化完成" -ForegroundColor Green
```

#### 磁盘优化

```bash
# 创建磁盘优化脚本 optimize_disk.bat

@echo off
echo 正在清理临时文件...

# 清理Python缓存
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc
del /s /q *.pyo

# 清理pip缓存
python -m pip cache purge

# 清理系统临时文件
del /q/f/s %TEMP%\*

echo 磁盘清理完成!
pause
```

## 🔧 故障排除

### 常见问题及解决方案

#### 1. Python安装问题

```bash
# 问题: Python命令不可用
# 解决方案: 
# 1. 检查PATH环境变量
echo %PATH%
# 2. 重新安装Python并勾选"Add to PATH"
# 3. 手动添加Python路径到PATH
```

#### 2. pip安装失败

```bash
# 问题: pip install失败
# 解决方案:
pip install --upgrade pip setuptools wheel
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org [package_name]

# 或者使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple [package_name]
```

#### 3. 依赖冲突

```bash
# 问题: 依赖包版本冲突
# 解决方案:
pip check

# 创建新的虚拟环境
python -m venv new_env
new_env\Scripts\activate
pip install -r requirements.txt

# 或者使用conda
conda create --name new_env python=3.11
conda activate new_env
conda install --file requirements.txt
```

#### 4. GPU不可用

```python
# 问题: CUDA不可用
# 解决方案检查:

# 1. 检查CUDA安装
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"PyTorch版本: {torch.__version__}")

# 2. 重新安装CUDA版本PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 检查NVIDIA驱动
# 打开NVIDIA控制面板 → 帮助 → 系统信息 → 组件 → NVCUDA64.DLL
```

#### 5. Jupyter无法启动

```bash
# 问题: Jupyter Lab无法启动
# 解决方案:

# 1. 检查端口占用
netstat -ano | findstr :8888

# 2. 清理Jupyter配置
jupyter lab --generate-config
jupyter lab --reset-config

# 3. 重新安装Jupyter
pip uninstall jupyter jupyterlab
pip install jupyter jupyterlab

# 4. 使用不同端口
jupyter lab --port=9999
```

#### 6. Web界面无法访问

```bash
# 问题: React开发服务器无法访问
# 解决方案:

# 1. 检查防火墙设置
# Windows防火墙可能阻止端口5173

# 2. 配置开发服务器
cd ui/brain-ai-ui
npm run dev -- --host 0.0.0.0

# 3. 检查端口占用
netstat -ano | findstr :5173
```

#### 7. 内存不足错误

```python
# 问题: OOM (Out of Memory) 错误
# 解决方案:

import torch

# 减少批次大小
batch_size = 16  # 原先可能是64或128

# 启用梯度检查点
torch.utils.checkpoint.checkpoint_sequential

# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    output = model(input_data)
    loss = criterion(output, target)

# 清理GPU缓存
torch.cuda.empty_cache()
```

#### 8. 模块导入错误

```bash
# 问题: ModuleNotFoundError
# 解决方案:

# 1. 检查Python路径
import sys
print(sys.path)

# 2. 确保在项目根目录
cd brain-inspired-ai

# 3. 设置PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;.

# 4. 重新安装包
pip install -e .
```

### 诊断工具

创建诊断脚本 `diagnose.py`:

```python
#!/usr/bin/env python3
"""
Windows 11 Brain AI 环境诊断工具
"""
import sys
import subprocess
import platform
import torch
import importlib
import pkg_resources

def check_system_info():
    """检查系统信息"""
    print("=== 系统信息 ===")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"架构: {platform.machine()}")
    print(f"处理器: {platform.processor()}")
    print()

def check_python_packages():
    """检查Python包"""
    print("=== Python包检查 ===")
    required_packages = [
        'torch', 'torchvision', 'numpy', 'scipy', 'pandas',
        'sklearn', 'matplotlib', 'jupyter', 'click', 'pyyaml'
    ]
    
    for package in required_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"✅ {package}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package}: 未安装")
    print()

def check_pytorch():
    """检查PyTorch配置"""
    print("=== PyTorch检查 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}, {props.total_memory/1024**3:.1f}GB")
    else:
        print("⚠️  GPU不可用，将使用CPU")
    print()

def check_gpu_drivers():
    """检查GPU驱动"""
    print("=== GPU驱动检查 ===")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA驱动正常")
            print(result.stdout.split('\n')[0])
        else:
            print("❌ NVIDIA驱动检查失败")
    except FileNotFoundError:
        print("❌ nvidia-smi命令不可用")
        print("请安装NVIDIA驱动程序")
    print()

def check_ports():
    """检查端口占用"""
    print("=== 端口检查 ===")
    ports_to_check = [8888, 5173, 6006]
    
    for port in ports_to_check:
        result = subprocess.run(
            ['netstat', '-ano'], 
            capture_output=True, 
            text=True
        )
        if f':{port}' in result.stdout:
            print(f"⚠️  端口 {port} 被占用")
        else:
            print(f"✅ 端口 {port} 可用")
    print()

def run_performance_test():
    """运行性能测试"""
    print("=== 性能测试 ===")
    
    # CPU测试
    import time
    start = time.time()
    _ = sum(range(1000000))
    cpu_time = time.time() - start
    print(f"CPU计算测试: {cpu_time:.4f}秒")
    
    # PyTorch测试
    if torch.cuda.is_available():
        device = torch.device('cuda')
        x = torch.randn(1000, 1000, device=device)
        start = time.time()
        _ = torch.mm(x, x)
        gpu_time = time.time() - start
        print(f"GPU计算测试: {gpu_time:.4f}秒")
        print(f"GPU速度提升: {cpu_time/gpu_time:.1f}x")
    else:
        print("跳过GPU测试（无可用GPU）")
    print()

def main():
    """主诊断函数"""
    print("🧠 Brain AI Windows 11 环境诊断")
    print("=" * 50)
    
    check_system_info()
    check_python_packages()
    check_pytorch()
    check_gpu_drivers()
    check_ports()
    run_performance_test()
    
    print("=" * 50)
    print("诊断完成！请根据上述信息解决问题。")

if __name__ == "__main__":
    main()
```

```bash
# 运行诊断
python diagnose.py
```

## ✅ 环境验证

### 1. 快速验证脚本

创建 `verify_installation.py`:

```python
#!/usr/bin/env python3
"""
快速验证安装是否成功
"""
import sys
import torch
import numpy as np

def test_basic_imports():
    """测试基础导入"""
    print("测试基础包导入...")
    
    try:
        import numpy as np
        import scipy
        import pandas as pd
        import sklearn
        import matplotlib.pyplot as plt
        print("✅ 基础科学计算包导入成功")
    except ImportError as e:
        print(f"❌ 基础包导入失败: {e}")
        return False
    
    return True

def test_pytorch():
    """测试PyTorch"""
    print("\n测试PyTorch...")
    
    try:
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        
        # 创建测试张量
        x = torch.randn(100, 100)
        y = torch.randn(100, 100)
        z = torch.mm(x, y)
        print("✅ PyTorch张量运算正常")
        
        # GPU测试
        if torch.cuda.is_available():
            x_gpu = x.to('cuda')
            y_gpu = y.to('cuda')
            z_gpu = torch.mm(x_gpu, y_gpu)
            print("✅ GPU计算正常")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")
        return False

def test_brain_ai_modules():
    """测试Brain AI模块"""
    print("\n测试Brain AI模块...")
    
    try:
        # 测试项目模块导入
        import os
        import sys
        
        # 添加项目路径
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        print("✅ 路径配置正常")
        return True
    except Exception as e:
        print(f"❌ Brain AI模块测试失败: {e}")
        return False

def test_performance():
    """测试性能"""
    print("\n性能测试...")
    
    try:
        # 内存测试
        import psutil
        memory = psutil.virtual_memory()
        print(f"内存使用率: {memory.percent:.1f}%")
        
        # 计算性能测试
        start = time.time()
        x = np.random.randn(1000, 1000)
        y = np.random.randn(1000, 1000)
        z = np.dot(x, y)
        numpy_time = time.time() - start
        
        print(f"NumPy矩阵乘法: {numpy_time:.4f}秒")
        print("✅ 性能测试正常")
        return True
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False

def main():
    """主验证函数"""
    print("🧠 Brain AI 安装验证")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_pytorch,
        test_brain_ai_modules,
        test_performance
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"验证结果: {passed}/{len(tests)} 项测试通过")
    
    if passed == len(tests):
        print("🎉 所有测试通过！安装成功！")
    else:
        print("⚠️  部分测试失败，请检查安装")

if __name__ == "__main__":
    import time
    main()
```

### 2. 完整验证流程

```bash
# 按顺序运行验证
python verify_installation.py
python diagnose.py

# 如果所有测试通过，运行演示
python cli_demo.py --mode demo

# 测试Web界面
cd ui/brain-ai-ui
pnpm install
pnpm dev

# 测试Jupyter
jupyter lab
```

## 🎯 总结

恭喜！您已经完成了Windows 11环境下Brain AI系统的完整安装配置。通过本指南，您应该已经：

1. ✅ 安装并配置了Python环境
2. ✅ 安装了所有必要的依赖包
3. ✅ 配置了GPU支持（如果可用）
4. ✅ 设置了Web界面和Jupyter环境
5. ✅ 优化了系统性能
6. ✅ 验证了安装的正确性

### 下一步操作

1. **开始体验**: 运行 `python cli_demo.py --mode demo`
2. **探索功能**: 查看各种演示脚本
3. **深入开发**: 使用Jupyter Lab进行交互式开发
4. **性能调优**: 根据实际使用情况调整配置

### 获取帮助

如果在安装过程中遇到问题：

1. 查看故障排除部分
2. 运行诊断脚本
3. 检查错误日志
4. 参考官方文档

**祝您使用愉快！** 🎉

---

*本指南针对Windows 11优化，如使用其他操作系统，请参考相应的安装文档。*
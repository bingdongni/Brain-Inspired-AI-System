# 🚀 简化安装指南

> **5分钟快速安装脑启发AI系统** - 支持CLI、Web界面、Jupyter集成

## ⚡ 一键安装

### 方式1：使用安装脚本（推荐）

```bash
# 下载并运行安装脚本
curl -fsSL https://raw.githubusercontent.com/brain-ai/brain-inspired-ai/main/install.sh | bash

# 或下载后运行
chmod +x install.sh
./install.sh
```

### 方式2：手动安装

```bash
# 1. 克隆项目
git clone https://github.com/brain-ai/brain-inspired-ai.git
cd brain-inspired-ai

# 2. 一键安装（包含所有依赖）
bash install.sh --dev --viz --yes

# 3. 激活环境
source venv/bin/activate

# 4. 快速验证
python quick_test.py
```

## 🎯 快速体验

### 1. 命令行体验（30秒）

```bash
# 运行基础演示
python cli_demo.py --mode demo

# 交互式体验
python cli_demo.py --mode interactive
```

### 2. Web界面体验（1分钟）

```bash
# 进入Web界面目录
cd ui/brain-ai-ui

# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 访问 http://localhost:5173
```

### 3. Jupyter体验（30秒）

```python
# 在Jupyter中运行
import sys
sys.path.append('/path/to/brain-inspired-ai/ui')
from jupyter_integration import *
show_brain_dashboard()
```

## 📦 安装选项

### 基础安装（CPU版本）
```bash
# 适用于：学习、测试、低配设备
./install.sh
```

### 完整安装（GPU加速）
```bash
# 适用于：开发、生产环境、高性能需求
./install.sh --dev --viz --gpu
```

### 最小安装
```bash
# 适用于：容器化、CI/CD
./install.sh --clean
```

## 🔧 环境配置

### 系统要求
- **操作系统**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Python**: 3.8 - 3.11
- **内存**: 最少 4GB RAM
- **存储**: 至少 2GB 可用空间

### Python环境检查
```bash
# 检查Python版本
python --version  # 应该是3.8+

# 检查pip
pip --version

# 如果需要安装Python
# Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv
# macOS: brew install python3
# Windows: 从 https://python.org 下载
```

## 🐳 Docker安装

### 使用Docker Compose

```bash
# 启动服务
docker-compose up -d

# 进入开发容器
docker exec -it brain-ai-dev bash

# 在容器内运行
python cli_demo.py --mode demo
```

### 手动Docker命令

```bash
# 构建镜像
docker build -t brain-ai .

# 运行容器
docker run -it --rm -p 8888:8888 -v $(pwd):/app brain-ai

# Jupyter服务
docker run -it --rm -p 8888:8888 -v $(pwd):/app brain-ai jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
```

## ❓ 常见安装问题

### 问题1：Python版本不兼容
```bash
# 解决方案：安装Python 3.8+
# 使用pyenv管理多版本
curl https://pyenv.run | bash
pyenv install 3.9.0
pyenv local 3.9.0
```

### 问题2：PyTorch安装失败
```bash
# CPU版本
pip install torch torchvision torchaudio

# GPU版本（CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 问题3：权限错误
```bash
# 使用虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 或者添加--user参数
pip install --user requirements.txt
```

### 问题4：网络连接问题
```bash
# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或设置永久镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题5：内存不足
```bash
# 减少批处理大小
export BATCH_SIZE=16

# 使用CPU模式
python cli_demo.py --mode demo --device cpu

# 检查内存
free -h  # Linux/Mac
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory  # Windows
```

## 🔍 安装验证

### 快速验证
```bash
# 运行系统检查
python quick_test.py

# 检查核心模块
python -c "
import torch
import numpy
import hippocampus
import lifelong_learning
print('✅ 所有核心模块导入成功')
"

# 运行基础演示
python cli_demo.py --mode demo
```

### 详细验证
```bash
# 检查GPU支持
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 检查Web界面依赖
cd ui/brain-ai-ui && npm --version

# 检查Jupyter集成
python -c "import jupyter; import ipywidgets; print('✅ Jupyter依赖正常')"
```

## 📊 安装后配置

### 环境变量设置
```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export BRAIN_AI_HOME="/path/to/brain-inspired-ai"
export PATH="$BRAIN_AI_HOME/bin:$PATH"

# 加载环境
source ~/.bashrc
```

### 配置文件
```bash
# 自动创建在：config/local.yaml
# 可以根据需要修改配置

# 查看配置
cat config/local.yaml
```

### 目录结构
```bash
# 安装后目录结构
brain-inspired-ai/
├── data/           # 数据目录
├── models/         # 模型存储
├── results/        # 结果输出
├── logs/          # 日志文件
├── config/        # 配置文件
└── ui/            # Web界面
```

## 🚀 下一步

安装完成后，推荐的学习路径：

1. **立即体验** (5分钟)
   ```bash
   python cli_demo.py --mode demo
   ```

2. **Web界面** (1分钟)
   ```bash
   cd ui/brain-ai-ui && npm run dev
   ```

3. **深入学习** (30分钟)
   - 阅读 [完整快速指南](../docs/quick_start_guide.md)
   - 运行所有演示程序

4. **开发实践** (1小时)
   - 查看 [用户手册](docs/USER_MANUAL.md)
   - 学习 [API文档](docs/api/API_REFERENCE.md)

## 📞 获取帮助

- **🐛 问题反馈**: [GitHub Issues](https://github.com/brain-ai/brain-inspired-ai/issues)
- **💬 技术讨论**: [GitHub Discussions](https://github.com/brain-ai/brain-inspired-ai/discussions)
- **📖 完整文档**: [docs/](docs/)
- **📧 邮箱支持**: support@brain-ai.org

---

**🎉 恭喜！您已完成脑启发AI系统的安装！**

现在可以开始您的AI探索之旅了！🧠✨
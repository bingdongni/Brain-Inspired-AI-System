#!/bin/bash

# Brain-Inspired AI Framework 快速安装脚本
# =========================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 打印带颜色的信息
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${PURPLE}$1${NC}"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# 显示欢迎横幅
show_banner() {
    echo -e "${PURPLE}"
    cat << 'EOF'
██╗   ██╗██████╗ ███████╗████████╗███████╗███╗   ██╗
██║   ██║██╔══██╗██╔════╝╚══██╔══╝██╔════╝████╗  ██║
██║   ██║██████╔╝█████╗     ██║   █████╗  ██╔██╗ ██║
██║   ██║██╔══██╗██╔══╝     ██║   ██╔══╝  ██║╚██╗██║
╚██████╔╝██████╔╝███████╗   ██║   ███████╗██║ ╚████║
 ╚═════╝ ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═══╝
                                       
Brain-Inspired AI Framework
基于生物大脑启发的深度学习框架
EOF
    echo -e "${NC}"
}

# 检查命令是否存在
check_command() {
    if ! command -v "$1" &> /dev/null; then
        return 1
    fi
    return 0
}

# 检测系统类型
detect_system() {
    log_step "检测系统环境..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if check_command "lsb_release"; then
            DISTRO=$(lsb_release -si)
            VERSION=$(lsb_release -sr)
            SYSTEM="linux-$DISTRO-$VERSION"
        else
            SYSTEM="linux-unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        MACOS_VERSION=$(sw_vers -productVersion)
        SYSTEM="macos-$MACOS_VERSION"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        SYSTEM="windows"
    else
        SYSTEM="unknown"
    fi
    
    log_info "操作系统: $SYSTEM"
    log_info "架构: $(uname -m)"
    log_info "CPU核心数: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "1")"
}

# 检查Python环境
check_python() {
    log_step "检查Python环境..."
    
    # 寻找Python
    PYTHON_CMD=""
    for cmd in python3 python; do
        if check_command "$cmd"; then
            version=$($cmd --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
            major=$(echo $version | cut -d. -f1)
            minor=$(echo $version | cut -d. -f2)
            
            if [ "$major" -eq 3 ] && [ "$minor" -ge 8 ]; then
                PYTHON_CMD="$cmd"
                PYTHON_VERSION="$version"
                break
            fi
        fi
    done
    
    if [ -z "$PYTHON_CMD" ]; then
        log_error "未找到Python 3.8+，请先安装Python"
        log_info "安装建议："
        log_info "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
        log_info "  macOS: brew install python3"
        log_info "  Windows: https://python.org 下载安装"
        exit 1
    fi
    
    log_success "Python版本: $PYTHON_VERSION ($PYTHON_CMD)"
    
    # 检查pip
    if ! $PYTHON_CMD -m pip --version &> /dev/null; then
        log_warning "pip未安装，尝试安装..."
        if check_command "apt"; then
            sudo apt install -y python3-pip
        elif check_command "brew"; then
            brew install python3
        else
            log_error "无法自动安装pip，请手动安装"
            exit 1
        fi
    fi
    
    log_success "pip版本: $($PYTHON_CMD -m pip --version | cut -d' ' -f2)"
}

# 检查GPU支持
check_gpu() {
    log_step "检查GPU支持..."
    
    GPU_AVAILABLE=false
    if check_command "nvidia-smi"; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        GPU_NAME=$(echo $GPU_INFO | cut -d',' -f1 | xargs)
        GPU_MEMORY=$(echo $GPU_INFO | cut -d',' -f2 | xargs)
        
        log_success "检测到NVIDIA GPU: $GPU_NAME (${GPU_MEMORY}MB)"
        GPU_AVAILABLE=true
        
        # 检查CUDA
        if check_command "nvcc"; then
            CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
            log_info "CUDA版本: $CUDA_VERSION"
        fi
    else
        log_info "未检测到NVIDIA GPU，将使用CPU版本"
    fi
    
    # 导出GPU标志
    if [ "$GPU_AVAILABLE" = true ]; then
        export GPU_SUPPORT=true
    else
        export GPU_SUPPORT=false
    fi
}

# 创建虚拟环境
setup_venv() {
    log_step "设置虚拟环境..."
    
    VENV_NAME="${1:-venv}"
    
    if [ -d "$VENV_NAME" ]; then
        log_warning "虚拟环境 $VENV_NAME 已存在"
        read -p "是否重新创建? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "删除现有虚拟环境..."
            rm -rf "$VENV_NAME"
        else
            log_info "使用现有虚拟环境"
            return 0
        fi
    fi
    
    log_info "创建虚拟环境: $VENV_NAME"
    $PYTHON_CMD -m venv "$VENV_NAME"
    
    # 激活虚拟环境
    source "$VENV_NAME/bin/activate" 2>/dev/null || source "$VENV_NAME/Scripts/activate" 2>/dev/null || {
        log_error "无法激活虚拟环境"
        exit 1
    }
    
    # 升级pip
    log_info "升级pip..."
    python -m pip install --upgrade pip setuptools wheel
    
    log_success "虚拟环境创建完成"
}

# 安装依赖
install_deps() {
    log_step "安装项目依赖..."
    
    # 确保在虚拟环境中
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        log_error "请在虚拟环境中运行此脚本"
        exit 1
    fi
    
    # 基础依赖
    if [ -f "requirements.txt" ]; then
        log_info "安装基础依赖..."
        python -m pip install -r requirements.txt
    else
        log_warning "requirements.txt不存在，安装核心依赖..."
        python -m pip install torch torchvision numpy scipy pandas scikit-learn
    fi
    
    # GPU支持
    if [ "$GPU_SUPPORT" = true ] && [ "$INSTALL_GPU" = true ]; then
        log_info "安装GPU支持..."
        python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    fi
    
    # 开发依赖
    if [ "$INSTALL_DEV" = true ]; then
        log_info "安装开发依赖..."
        python -m pip install pytest pytest-cov black flake8 isort mypy jupyter
    fi
    
    # 可视化依赖
    if [ "$INSTALL_VIZ" = true ]; then
        log_info "安装可视化依赖..."
        python -m pip install matplotlib seaborn plotly bokeh
    fi
    
    log_success "依赖安装完成"
}

# 安装项目
install_project() {
    log_step "安装Brain-Inspired AI项目..."
    
    if [ -f "setup.py" ]; then
        python -m pip install -e .
    elif [ -f "pyproject.toml" ]; then
        python -m pip install -e .
    else
        log_warning "未找到安装配置文件，创建requirements.txt..."
        cat > requirements.txt << EOF
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
click>=8.1.0
pyyaml>=6.0
rich>=13.0.0
tqdm>=4.66.0
EOF
        python -m pip install -r requirements.txt
    fi
    
    log_success "项目安装完成"
}

# 验证安装
verify_installation() {
    log_step "验证安装..."
    
    # 测试导入
    python -c "
import brain_ai
print(f'版本: {brain_ai.__version__}')
print('导入测试通过')
" || {
        log_error "导入测试失败"
        return 1
    }
    
    # 运行简单测试
    python main.py info &> /dev/null || {
        log_warning "系统信息测试失败，但不影响基本功能"
    }
    
    log_success "安装验证通过"
}

# 运行演示
run_demo() {
    if [ "$RUN_DEMO" != true ]; then
        return 0
    fi
    
    log_step "运行演示程序..."
    
    echo
    echo "请选择演示类型:"
    echo "1) 基础演示 - 快速体验核心功能"
    echo "2) 高级演示 - 完整功能展示"
    echo "3) 交互式演示 - 可视化界面"
    echo "4) 跳过演示"
    echo
    
    read -p "请选择 (1-4): " DEMO_CHOICE
    
    case $DEMO_CHOICE in
        1)
            log_info "运行基础演示..."
            python main.py demo --demo-type basic
            ;;
        2)
            log_info "运行高级演示..."
            python main.py demo --demo-type advanced
            ;;
        3)
            log_info "运行交互式演示..."
            python main.py demo --interactive
            ;;
        4)
            log_info "跳过演示"
            ;;
        *)
            log_warning "无效选择，跳过演示"
            ;;
    esac
}

# 创建目录结构
create_directories() {
    log_step "创建目录结构..."
    
    local dirs=(
        "data/datasets"
        "data/models" 
        "data/results"
        "data/cache"
        "output/models"
        "output/reports"
        "output/visualizations"
        "logs"
        "temp/cache"
        "temp/tmp_models"
        "temp/downloads"
        "docs/images"
        "config"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done
    
    log_success "目录结构创建完成"
}

# 创建配置文件
create_config() {
    log_step "创建配置文件..."
    
    # 主配置文件
    cat > config/local.yaml << 'EOF'
# Brain-Inspired AI 本地配置文件
# 本文件会被版本控制系统忽略，可以安全修改

system:
  name: "Brain-Inspired AI Local"
  version: "1.0.0"
  debug: true
  log_level: "INFO"
  device: "auto"  # auto, cpu, cuda

model:
  type: "brain_system"
  hidden_dim: 512
  num_layers: 6
  dropout: 0.1
  activation: "relu"

hippocampus:
  input_dim: 512
  hidden_dim: 256
  memory_dim: 256
  num_transformer_layers: 6
  num_attention_heads: 8
  storage_capacity: 10000
  retrieval_threshold: 0.7
  pattern_separation_threshold: 0.5

neocortex:
  input_dim: 512
  hidden_dim: 1024
  num_layers: 12
  num_attention_heads: 16
  abstraction_levels: 4
  sparsity_ratio: 0.05

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  optimizer: "adam"
  weight_decay: 0.0001
  early_stopping: true
  patience: 10
  mixed_precision: false

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./logs/brain_ai.log"
  max_size: "100MB"
  backup_count: 5

output:
  dir: "./output"
  save_models: true
  save_metrics: true
  save_visualizations: true
  model_format: "pkl"  # pkl, pt, onnx

data:
  dataset_dir: "./data/datasets"
  cache_dir: "./data/cache"
  model_dir: "./data/models"
  result_dir: "./data/results"
EOF

    # Git忽略文件
    cat > .gitignore_local << 'EOF'
# 本地配置文件
config/local.yaml
.env

# 数据目录
data/datasets/
data/models/
data/results/
data/cache/

# 输出目录
output/
logs/
temp/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
ENV/
env/

# Jupyter Notebook
.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF

    log_success "配置文件创建完成: config/local.yaml"
}

# 清理函数
cleanup() {
    log_step "清理临时文件..."
    
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
    
    log_success "清理完成"
}

# 显示帮助
show_help() {
    cat << EOF
Brain-Inspired AI Framework 快速安装脚本

用法: $0 [选项]

选项:
  -h, --help          显示此帮助信息
  -v, --verbose       详细输出
  -y, --yes           自动确认所有提示
  --venv <name>       指定虚拟环境名称 (默认: venv)
  --no-demo           跳过演示程序
  --dev               安装开发依赖
  --gpu               安装GPU支持 (如果有GPU)
  --viz               安装可视化依赖
  --clean             安装前清理缓存
  --config-only       仅创建配置文件
  --docker            使用Docker部署

示例:
  $0                          # 基础安装
  $0 --dev --demo             # 安装开发依赖并运行演示
  $0 --gpu --viz --yes        # 安装GPU和可视化，自动确认
  $0 --clean --config-only    # 清理并仅创建配置

环境变量:
  INSTALL_DEV=true            # 自动安装开发依赖
  INSTALL_GPU=true            # 自动安装GPU支持
  INSTALL_VIZ=true            # 自动安装可视化依赖
  RUN_DEMO=true               # 自动运行演示
  AUTO_YES=true               # 自动确认所有提示

EOF
}

# 主函数
main() {
    # 默认参数
    INSTALL_DEV=false
    INSTALL_GPU=false
    INSTALL_VIZ=false
    RUN_DEMO=false
    AUTO_YES=false
    CLEAN=false
    CONFIG_ONLY=false
    USE_DOCKER=false
    VENV_NAME="venv"
    VERBOSE=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -y|--yes)
                AUTO_YES=true
                shift
                ;;
            --venv)
                VENV_NAME="$2"
                shift 2
                ;;
            --no-demo)
                RUN_DEMO=false
                shift
                ;;
            --dev)
                INSTALL_DEV=true
                shift
                ;;
            --gpu)
                INSTALL_GPU=true
                shift
                ;;
            --viz)
                INSTALL_VIZ=true
                shift
                ;;
            --clean)
                CLEAN=true
                shift
                ;;
            --config-only)
                CONFIG_ONLY=true
                shift
                ;;
            --docker)
                USE_DOCKER=true
                shift
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查环境变量
    [ "$INSTALL_DEV" = "true" ] && INSTALL_DEV=true
    [ "$INSTALL_GPU" = "true" ] && INSTALL_GPU=true
    [ "$INSTALL_VIZ" = "true" ] && INSTALL_VIZ=true
    [ "$RUN_DEMO" = "true" ] && RUN_DEMO=true
    [ "$AUTO_YES" = "true" ] && AUTO_YES=true
    
    # 开始安装
    show_banner
    echo
    
    detect_system
    check_python
    check_gpu
    
    if [ "$CLEAN" = true ]; then
        cleanup
    fi
    
    if [ "$CONFIG_ONLY" = true ]; then
        create_directories
        create_config
        log_success "配置创建完成！"
        exit 0
    fi
    
    setup_venv "$VENV_NAME"
    install_deps
    install_project
    
    if [ "$AUTO_YES" = false ] && [ "$RUN_DEMO" != true ]; then
        echo
        read -p "是否运行演示程序? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            RUN_DEMO=true
        fi
    fi
    
    verify_installation
    create_directories
    create_config
    run_demo
    
    echo
    log_success "🎉 Brain-Inspired AI 安装完成！"
    echo
    log_info "下一步操作："
    echo "  python main.py --help          # 查看命令行帮助"
    echo "  python main.py demo --demo-type basic  # 运行基础演示"
    echo "  python main.py info            # 查看系统信息"
    echo
    log_info "文档文件："
    echo "  README.md              # 项目说明"
    echo "  快速开始.md            # 快速开始指南"
    echo "  安装指南.md            # 详细安装指南"
    echo "  使用说明.md            # 使用说明文档"
    echo
    echo "========================================"
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
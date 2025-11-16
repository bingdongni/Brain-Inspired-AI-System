#!/bin/bash

# Brain-Inspired AI 项目部署脚本
# ================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 未安装，请先安装 $1"
        exit 1
    fi
}

# 检查Python版本
check_python_version() {
    python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
    major_version=$(echo $python_version | cut -d. -f1)
    minor_version=$(echo $python_version | cut -d. -f2)
    
    if [ "$major_version" -lt 3 ] || ([ "$major_version" -eq 3 ] && [ "$minor_version" -lt 8 ]); then
        print_error "Python版本过低，需要Python 3.8+。当前版本: $python_version"
        exit 1
    fi
    
    print_success "Python版本检查通过: $python_version"
}

# 创建虚拟环境
create_virtual_env() {
    local venv_name="$1"
    
    if [ ! -d "$venv_name" ]; then
        print_info "创建虚拟环境: $venv_name"
        python3 -m venv $venv_name
    else
        print_warning "虚拟环境 $venv_name 已存在"
    fi
}

# 激活虚拟环境
activate_virtual_env() {
    local venv_name="$1"
    
    if [ -f "$venv_name/bin/activate" ]; then
        print_info "激活虚拟环境: $venv_name"
        source $venv_name/bin/activate
    else
        print_error "无法找到虚拟环境激活脚本: $venv_name/bin/activate"
        exit 1
    fi
}

# 安装依赖
install_dependencies() {
    local requirements_file="$1"
    
    if [ -f "$requirements_file" ]; then
        print_info "安装依赖包: $requirements_file"
        pip install --upgrade pip
        pip install -r $requirements_file
        print_success "依赖安装完成"
    else
        print_error "依赖文件不存在: $requirements_file"
        exit 1
    fi
}

# 安装项目
install_project() {
    print_info "安装Brain-Inspired AI项目"
    
    # 进入项目根目录
    if [ -f "setup.py" ]; then
        pip install -e .
        print_success "项目安装完成"
    else
        print_error "未找到setup.py文件"
        exit 1
    fi
}

# 运行测试
run_tests() {
    print_info "运行项目测试"
    
    if [ -f "pytest.ini" ] || [ -d "tests" ]; then
        python -m pytest tests/ -v --tb=short
        print_success "测试运行完成"
    else
        print_warning "未找到测试文件，跳过测试"
    fi
}

# 构建文档
build_docs() {
    local docs_dir="$1"
    
    if [ -d "$docs_dir" ]; then
        print_info "构建文档"
        cd $docs_dir
        
        if [ -f "Makefile" ]; then
            make html
            print_success "文档构建完成"
        elif [ -f "conf.py" ]; then
            sphinx-build -b html . _build/html
            print_success "文档构建完成"
        else
            print_warning "未找到文档构建配置"
        fi
        
        cd ..
    else
        print_warning "文档目录不存在: $docs_dir"
    fi
}

# 创建系统服务（Linux）
create_systemd_service() {
    local service_name="$1"
    local script_path="$2"
    local user="$3"
    
    local service_file="/etc/systemd/system/${service_name}.service"
    
    sudo tee $service_file > /dev/null <<EOF
[Unit]
Description=Brain-Inspired AI $service_name
After=network.target

[Service]
Type=simple
User=$user
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/python3 $script_path
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable $service_name
    
    print_success "系统服务创建完成: $service_name"
}

# 启动服务
start_service() {
    local service_name="$1"
    
    print_info "启动服务: $service_name"
    sudo systemctl start $service_name
    
    if sudo systemctl is-active --quiet $service_name; then
        print_success "服务启动成功"
    else
        print_error "服务启动失败"
        exit 1
    fi
}

# 停止服务
stop_service() {
    local service_name="$1"
    
    print_info "停止服务: $service_name"
    sudo systemctl stop $service_name
    print_success "服务已停止"
}

# 显示服务状态
service_status() {
    local service_name="$1"
    
    sudo systemctl status $service_name
}

# 部署函数
deploy() {
    local environment="${1:-development}"
    local service_name="brain-ai-${environment}"
    
    print_info "开始部署Brain-Inspired AI (环境: $environment)"
    
    # 检查依赖
    check_command "python3"
    check_command "pip"
    check_python_version
    
    # 设置变量
    case $environment in
        "development")
            venv_name="venv"
            requirements_file="requirements.txt"
            ;;
        "production")
            venv_name="venv_prod"
            requirements_file="requirements.txt"
            ;;
        "testing")
            venv_name="venv_test"
            requirements_file="requirements.txt"
            ;;
        *)
            print_error "未知环境: $environment"
            exit 1
            ;;
    esac
    
    # 创建和激活虚拟环境
    create_virtual_env $venv_name
    activate_virtual_env $venv_name
    
    # 安装依赖
    install_dependencies $requirements_file
    
    # 安装项目
    install_project
    
    # 运行测试
    if [ "$environment" != "production" ]; then
        run_tests
    fi
    
    # 构建文档
    if [ -d "docs" ]; then
        build_docs "docs"
    fi
    
    print_success "部署完成 (环境: $environment)"
}

# 清理函数
cleanup() {
    local environment="$1"
    
    print_info "清理环境: $environment"
    
    case $environment in
        "development")
            venv_name="venv"
            ;;
        "production")
            venv_name="venv_prod"
            ;;
        "testing")
            venv_name="venv_test"
            ;;
        "all")
            for venv_name in venv venv_prod venv_test; do
                if [ -d "$venv_name" ]; then
                    print_info "删除虚拟环境: $venv_name"
                    rm -rf $venv_name
                fi
            done
            
            # 清理构建文件
            if [ -d "build" ]; then
                rm -rf build
            fi
            if [ -d "dist" ]; then
                rm -rf dist
            fi
            if [ -d "__pycache__" ]; then
                find . -name "__pycache__" -type d -exec rm -rf {} +
            fi
            if [ -d ".pytest_cache" ]; then
                rm -rf .pytest_cache
            fi
            
            print_success "清理完成"
            return
            ;;
        *)
            print_error "未知环境: $environment"
            exit 1
            ;;
    esac
    
    if [ -d "$venv_name" ]; then
        print_info "删除虚拟环境: $venv_name"
        rm -rf $venv_name
    fi
    
    print_success "清理完成"
}

# 启动开发服务器
start_dev_server() {
    print_info "启动开发服务器"
    
    # 检查是否在虚拟环境中
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "未检测到虚拟环境，建议先运行部署脚本"
    fi
    
    # 检查端口是否被占用
    port=8080
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        print_warning "端口 $port 已被占用"
        read -p "请输入其他端口 (默认: 8081): " port
        port=${port:-8081}
    fi
    
    # 启动服务
    print_info "在端口 $port 启动服务"
    python -m brain_ai.scripts.dev_server --port $port
}

# 运行CLI命令
run_cli() {
    local command="$1"
    shift
    
    case $command in
        "train")
            python -m brain_ai.scripts.train "$@"
            ;;
        "eval")
            python -m brain_ai.scripts.evaluate "$@"
            ;;
        "demo")
            python -c "from brain_ai.cli import demo; demo()"
            ;;
        "config")
            python -c "from brain_ai.cli import config; config()"
            ;;
        "info")
            python -c "from brain_ai.cli import info; info()"
            ;;
        *)
            print_error "未知命令: $command"
            echo "可用命令: train, eval, demo, config, info"
            exit 1
            ;;
    esac
}

# 显示帮助信息
show_help() {
    echo "Brain-Inspired AI 部署脚本"
    echo ""
    echo "用法: $0 [选项] [命令]"
    echo ""
    echo "命令:"
    echo "  deploy [environment]    部署项目 (development|production|testing)"
    echo "  cleanup [environment]   清理环境 (development|production|testing|all)"
    echo "  start-dev              启动开发服务器"
    echo "  run-cli <command>       运行CLI命令 (train|eval|demo|config|info)"
    echo "  create-service <name>   创建系统服务"
    echo "  start-service <name>    启动系统服务"
    echo "  stop-service <name>     停止系统服务"
    echo "  service-status <name>   查看服务状态"
    echo "  help                   显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 deploy development"
    echo "  $0 deploy production"
    echo "  $0 start-dev"
    echo "  $0 run-cli train --epochs 100"
    echo "  $0 cleanup all"
}

# 主函数
main() {
    case "${1:-help}" in
        "deploy")
            deploy "${2:-development}"
            ;;
        "cleanup")
            cleanup "${2:-development}"
            ;;
        "start-dev")
            start_dev_server
            ;;
        "run-cli")
            if [ -z "$2" ]; then
                print_error "请指定CLI命令"
                exit 1
            fi
            run_cli "$2" "${@:3}"
            ;;
        "create-service")
            if [ -z "$2" ]; then
                print_error "请指定服务名称"
                exit 1
            fi
            if [ -z "$3" ]; then
                script_path="brain_ai_serve.py"
            else
                script_path="$3"
            fi
            create_systemd_service "$2" "$script_path" "${USER:-root}"
            ;;
        "start-service")
            if [ -z "$2" ]; then
                print_error "请指定服务名称"
                exit 1
            fi
            start_service "$2"
            ;;
        "stop-service")
            if [ -z "$2" ]; then
                print_error "请指定服务名称"
                exit 1
            fi
            stop_service "$2"
            ;;
        "service-status")
            if [ -z "$2" ]; then
                print_error "请指定服务名称"
                exit 1
            fi
            service_status "$2"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"
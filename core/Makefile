# Brain-Inspired AI Makefile
# =========================

.PHONY: help install dev-install test lint format clean build docker-build docker-run docs serve demo github-setup release-patch release-minor release-major docs-deploy deploy-all

# 默认目标
help:
	@echo "Brain-Inspired AI Framework - 可用命令:"
	@echo ""
	@echo "环境设置:"
	@echo "  install           - 安装项目依赖"
	@echo "  dev-install       - 安装开发依赖"
	@echo "  clean            - 清理构建文件和缓存"
	@echo ""
	@echo "开发:"
	@echo "  test             - 运行测试"
	@echo "  test-coverage    - 运行测试并生成覆盖率报告"
	@echo "  lint             - 代码质量检查"
	@echo "  format           - 代码格式化"
	@echo "  type-check       - 类型检查"
	@echo ""
	@echo "构建和部署:"
	@echo "  build            - 构建项目"
	@echo "  docker-build     - 构建Docker镜像"
	@echo "  docker-run       - 运行Docker容器"
	@echo "  serve            - 启动开发服务器"
	@echo "  demo             - 运行演示程序"
	@echo ""
	@echo "文档:"
	@echo "  docs             - 生成文档"
	@echo "  docs-serve       - 启动文档服务器"
	@echo ""

# 环境变量
PYTHON := python3
PIP := pip
VENV_DIR := venv
PYTEST := pytest
BLACK := black
FLAKE8 := flake8
MYPY := mypy

# 项目信息
PROJECT_NAME := brain_ai
SRC_DIR := src
TESTS_DIR := tests
DOCS_DIR := docs
CONFIG_DIR := config

# 安装依赖
install:
	@echo "📦 安装项目依赖..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

dev-install:
	@echo "🛠️ 安装开发依赖..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	$(PIP) install -r requirements-dev.txt

# 创建虚拟环境
venv:
	@echo "🐍 创建虚拟环境..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "虚拟环境已创建。请运行: source $(VENV_DIR)/bin/activate"

# 清理
clean:
	@echo "🧹 清理构建文件..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf logs/*

clean-all: clean
	@echo "🧹 清理所有文件..."
	rm -rf $(VENV_DIR)/
	rm -rf models/*
	rm -rf data/cache/*
	rm -rf output/*

# 测试
test:
	@echo "🧪 运行测试..."
	$(PYTEST) $(TESTS_DIR)/ -v

test-coverage:
	@echo "📊 运行测试并生成覆盖率报告..."
	$(PYTEST) $(TESTS_DIR)/ -v --cov=$(SRC_DIR)/$(PROJECT_NAME) --cov-report=html --cov-report=term

test-fast:
	@echo "⚡ 快速测试..."
	$(PYTEST) $(TESTS_DIR)/ -v -x --tb=short

# 代码质量
lint:
	@echo "🔍 代码质量检查..."
	$(FLAKE8) $(SRC_DIR)/$(PROJECT_NAME) $(TESTS_DIR)/ --max-line-length=88 --extend-ignore=E203,W503

format:
	@echo "✨ 代码格式化..."
	$(BLACK) $(SRC_DIR)/$(PROJECT_NAME) $(TESTS_DIR)/ --line-length=88

format-check:
	@echo "🔍 检查代码格式..."
	$(BLACK) $(SRC_DIR)/$(PROJECT_NAME) $(TESTS_DIR)/ --check --line-length=88

type-check:
	@echo "🔍 类型检查..."
	$(MYPY) $(SRC_DIR)/$(PROJECT_NAME)/

quality: lint format-check type-check
	@echo "✅ 代码质量检查完成"

# 构建
build:
	@echo "🏗️ 构建项目..."
	$(PYTHON) setup.py sdist bdist_wheel

build-clean: clean build

# Docker
docker-build:
	@echo "🐳 构建Docker镜像..."
	docker build -t brain-ai:latest .

docker-run:
	@echo "🚀 运行Docker容器..."
	docker run -p 8080:8080 -v $$(pwd)/data:/app/data -v $$(pwd)/models:/app/models brain-ai:latest

docker-compose-up:
	@echo "🚀 启动Docker Compose服务..."
	docker-compose up -d

docker-compose-down:
	@echo "🛑 停止Docker Compose服务..."
	docker-compose down

# 服务器
serve:
	@echo "🌐 启动开发服务器..."
	$(PYTHON) -m brain_ai.scripts.serve --config $(CONFIG_DIR)/development.yaml --host 127.0.0.1 --port 8000

serve-prod:
	@echo "🌐 启动生产服务器..."
	$(PYTHON) -m brain_ai.scripts.serve --config $(CONFIG_DIR)/production.yaml --host 0.0.0.0 --port 8080

# 演示
demo:
	@echo "🎯 运行演示程序..."
	$(PYTHON) -m brain_ai.scripts.demo --type basic --interactive

demo-advanced:
	@echo "🎯 运行高级演示..."
	$(PYTHON) -m brain_ai.scripts.demo --type advanced

demo-full:
	@echo "🎯 运行完整演示..."
	$(PYTHON) -m brain_ai.scripts.demo --type full

# 训练和评估
train:
	@echo "🏋️ 开始训练..."
	$(PYTHON) -m brain_ai.scripts.train \
		--model-type hippocampus \
		--epochs 50 \
		--batch-size 16 \
		--learning-rate 0.001 \
		--output-dir ./output

eval:
	@echo "📊 开始评估..."
	$(PYTHON) -m brain_ai.scripts.evaluate \
		--model-path ./output/hippocampus_model.pkl \
		--test-data ./data/test.csv \
		--output-dir ./evaluation_results

# 文档
docs:
	@echo "📚 生成文档..."
	cd $(DOCS_DIR) && make html

docs-serve:
	@echo "📚 启动文档服务器..."
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8081

docs-clean:
	@echo "🧹 清理文档..."
	cd $(DOCS_DIR) && make clean

# 数据处理
data-download:
	@echo "📥 下载示例数据..."
	$(PYTHON) -m brain_ai.scripts.data download --url https://example.com/dataset.zip

data-process:
	@echo "🔄 处理数据..."
	$(PYTHON) -m brain_ai.scripts.data process --input ./data/raw --output ./data/processed

# 监控
logs:
	@echo "📋 查看日志..."
	tail -f logs/brain_ai.log

logs-errors:
	@echo "❌ 查看错误日志..."
	grep "ERROR" logs/brain_ai.log

status:
	@echo "📈 系统状态检查..."
	$(PYTHON) -m brain_ai.scripts.status

# 性能分析
benchmark:
	@echo "⚡ 运行性能基准测试..."
	$(PYTHON) -m brain_ai.scripts.benchmark --model hippocampus --dataset test_data

profile:
	@echo "📊 性能分析..."
	$(PYTHON) -m brain_ai.scripts.profile --memory --cpu --model neocortex

# 配置管理
config-show:
	@echo "⚙️ 显示当前配置..."
	$(PYTHON) -m brain_ai.scripts.config show

config-validate:
	@echo "✅ 验证配置文件..."
	$(PYTHON) -m brain_ai.scripts.config validate --config $(CONFIG_DIR)/development.yaml

# 模型管理
model-list:
	@echo "📋 列出可用模型..."
	$(PYTHON) -m brain_ai.scripts.model list

model-info:
	@echo "ℹ️ 模型信息..."
	$(PYTHON) -m brain_ai.scripts.model info --model hippocampus

model-convert:
	@echo "🔄 转换模型格式..."
	$(PYTHON) -m brain_ai.scripts.model convert \
		--input ./models/hippocampus.pth \
		--output ./models/hippocortex.onnx \
		--format onnx

# 清理和维护
update-deps:
	@echo "🔄 更新依赖包..."
	$(PIP) list --outdated
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

backup:
	@echo "💾 备份数据..."
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz \
		--exclude='.git' \
		--exclude='__pycache__' \
		--exclude='*.pyc' \
		--exclude='logs/*.log' \
		--exclude='data/cache/*' \
		data/ models/ output/ config/

# 开发工作流
dev-setup: clean venv dev-install
	@echo "🚀 开发环境设置完成！"
	@echo "请运行: source $(VENV_DIR)/bin/activate"

dev-test: lint format-check type-check test
	@echo "✅ 开发测试通过！"

pre-commit: clean lint format-check type-check test
	@echo "✅ 提交前检查完成！"

# 生产部署
prod-build: clean build
	@echo "🏗️ 生产版本构建完成！"

prod-deploy:
	@echo "🚀 开始生产部署..."
	./scripts/deploy.sh deploy production

# 快速启动
quick-start: install
	@echo "🎯 快速启动..."
	$(PYTHON) -c "from brain_ai import HippocampusSimulator; h = HippocampusSimulator(); print('✅ Brain-Inspired AI 安装成功！')"

# 清理所有Docker相关
docker-clean:
	@echo "🐳 清理Docker资源..."
	docker system prune -f
	docker volume prune -f

# 显示项目信息
info:
	@echo "🧠 Brain-Inspired AI Framework"
	@echo "版本: $$(python -c "import sys; sys.path.insert(0, 'src'); from brain_ai import __version__; print(__version__)")"
	@echo "Python版本: $$(python --version)"
	@echo "PyTorch版本: $$(python -c 'import torch; print(torch.__version__)')"
	@echo "CUDA可用: $$(python -c 'import torch; print(torch.cuda.is_available())')"
	@echo "GPU数量: $$(python -c 'import torch; print(torch.cuda.device_count())')"

# GitHub部署
github-setup:
	@echo "🚀 设置GitHub仓库结构..."
	python scripts/setup_github_repo.py --comprehensive
	@echo "✅ GitHub仓库设置完成！"

release-patch:
	@echo "📦 发布补丁版本..."
	python scripts/release/manager.py release --auto
	@echo "✅ 补丁版本发布完成！"

release-minor:
	@echo "📦 发布次要版本..."
	python scripts/release/manager.py version --increment minor
	@echo "✅ 次要版本: $$(python scripts/release/manager.py version --show)"
	@echo "⚠️  请提交更改并手动创建发布"

release-major:
	@echo "📦 发布主要版本..."
	python scripts/release/manager.py version --increment major
	@echo "✅ 主要版本: $$(python scripts/release/manager.py version --show)"
	@echo "⚠️  请提交更改并手动创建发布"

docs-deploy:
	@echo "🚀 部署文档到GitHub Pages..."
	@echo "请确保已配置GitHub Pages并运行: mkdocs gh-deploy"

deploy-all: clean github-setup test lint format-check build docs docs-deploy
	@echo "🎉 完整部署流程完成！"
	@echo ""
	@echo "📋 后续步骤:"
	@echo "1. 审查生成的配置文件"
	@echo "2. 提交并推送更改"
	@echo "3. 配置GitHub仓库设置"
	@echo "4. 启用GitHub Pages"
	@echo "5. 设置仓库密钥"
	@echo "6. 配置分支保护规则"

# 显示帮助信息
.DEFAULT_GOAL := help
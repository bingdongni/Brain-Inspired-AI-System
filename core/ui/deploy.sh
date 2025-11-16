#!/bin/bash

# 脑启发AI系统UI部署脚本
# ==============================================

echo "🧠 脑启发AI系统用户界面部署"
echo "=============================="

# 检查Node.js环境
if ! command -v node &> /dev/null; then
    echo "❌ Node.js未安装，请先安装Node.js"
    exit 1
fi

# 检查npm/pnpm
if ! command -v pnpm &> /dev/null; then
    echo "⚠️  pnpm未安装，使用npm安装依赖..."
    PACKAGE_MANAGER="npm"
else
    PACKAGE_MANAGER="pnpm"
fi

echo "📦 使用包管理器: $PACKAGE_MANAGER"

# 进入Web应用目录
cd "$(dirname "$0")/brain-ai-ui"

echo "📂 工作目录: $(pwd)"

# 安装依赖
echo "🔧 安装依赖包..."
if [ "$PACKAGE_MANAGER" = "pnpm" ]; then
    pnpm install
else
    npm install
fi

if [ $? -ne 0 ]; then
    echo "❌ 依赖安装失败"
    exit 1
fi

echo "✅ 依赖安装完成"

# 构建生产版本
echo "🏗️  构建生产版本..."
if [ "$PACKAGE_MANAGER" = "pnpm" ]; then
    pnpm run build
else
    npm run build
fi

if [ $? -ne 0 ]; then
    echo "❌ 构建失败"
    exit 1
fi

echo "✅ 构建完成"

# 部署到静态服务器（可选）
echo ""
echo "🚀 部署选项："
echo "1. 开发模式: $PACKAGE_MANAGER run dev"
echo "2. 静态服务器: 使用任何静态文件服务器托管 dist/ 目录"
echo "3. 生产环境: 将 dist/ 目录上传到Web服务器"
echo ""

# 启动开发服务器（如果在开发环境）
if [ "$1" = "dev" ]; then
    echo "🖥️  启动开发服务器..."
    if [ "$PACKAGE_MANAGER" = "pnpm" ]; then
        pnpm run dev --host 0.0.0.0 --port 5173
    else
        npm run dev -- --host 0.0.0.0 --port 5173
    fi
fi

echo ""
echo "🎉 部署完成！"
echo ""
echo "📋 下一步："
echo "1. 确保Jupyter环境已安装所需依赖: pip install -r requirements.txt"
echo "2. 在Jupyter中测试: jupyter notebook 界面使用演示.ipynb"
echo "3. 访问Web界面: http://localhost:5173"
echo ""
echo "📚 更多信息请查看 README.md 文件"
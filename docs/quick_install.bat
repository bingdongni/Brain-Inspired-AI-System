@echo off
chcp 65001 >nul
title Brain AI 快速安装向导
color 0A

echo.
echo ██████╗ ███████╗███████╗████████╗███████╗ ██████╗████████╗
echo ██╔══██╗██╔════╝██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝
echo ██████╔╝███████╗█████╗     ██║   █████╗  ██║        ██║   
echo ██╔══██╗╚════██║██╔══╝     ██║   ██╔══╝  ██║        ██║   
echo ██║  ██║███████║███████╗   ██║   ███████╗╚██████╗   ██║   
echo ╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝   
echo.
echo                    Windows 11 快速安装向导
echo.
echo ================================================================
echo.

:: 检查管理员权限
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ⚠️  建议以管理员身份运行此脚本以获得最佳效果
    echo.
)

:: 检查Python
echo [检查 1/4] Python环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python未找到，请先安装Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    echo.
    set /p continue="是否继续安装其他组件? (y/N): "
    if /i not "%continue%"=="y" exit /b 1
) else (
    echo ✅ Python已安装
    python --version
)
echo.

:: 检查pip
echo [检查 2/4] pip包管理器...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip未安装，请重新安装Python并确保包含pip
) else (
    echo ✅ pip已安装
)
echo.

:: 检查Node.js
echo [检查 3/4] Node.js环境...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js未找到
    echo 可选: 是否安装Node.js? (用于Web界面)
    set /p install_node="安装Node.js? (y/N): "
    if /i "%install_node%"=="y" (
        echo 正在安装Node.js...
        winget install OpenJS.NodeJS
    )
) else (
    echo ✅ Node.js已安装
)
echo.

:: 检查项目目录
echo [检查 4/4] 项目目录...
if exist "brain-inspired-ai" (
    echo ✅ 找到项目目录: brain-inspired-ai
    set project_dir=brain-inspired-ai
) else (
    echo ❌ 未找到项目目录: brain-inspired-ai
    echo 请确保在正确的目录下运行此脚本
    set /p project_path="请输入项目路径或按Enter跳过: "
    if defined project_path set project_dir=%project_path%
)
echo.

:: 开始安装
echo ================================================================
echo 开始安装依赖包...
echo.

:: 升级pip
echo 升级pip...
python -m pip install --upgrade pip

:: 创建虚拟环境
if defined project_dir (
    echo.
    echo 创建虚拟环境...
    cd /d "%project_dir%"
    
    if exist brain_ai_env (
        echo 虚拟环境已存在，跳过创建
    ) else (
        python -m venv brain_ai_env
        echo 虚拟环境创建完成
    )
    
    :: 激活虚拟环境
    echo 激活虚拟环境...
    call brain_ai_env\Scripts\activate.bat
    
    :: 安装Python依赖
    echo.
    echo 安装Python依赖包...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install numpy scipy pandas scikit-learn
    pip install matplotlib seaborn pillow opencv-python
    pip install jupyter jupyterlab ipywidgets
    pip install click pyyaml rich tqdm
    pip install wandb tensorboard
    pip install pytest black flake8
    
    :: 安装项目依赖
    if exist "requirements.txt" (
        echo.
        echo 安装项目特定依赖...
        pip install -r requirements.txt
        pip install -e .
    )
)

echo.
echo ================================================================
echo 安装完成！

:: 提供后续步骤
echo.
echo 🎉 安装完成！后续步骤：
echo.
echo 1. 激活虚拟环境:
echo    cd %project_dir%
echo    brain_ai_env\Scripts\activate
echo.
echo 2. 运行演示:
echo    python cli_demo.py --mode demo
echo.
echo 3. 启动Jupyter:
echo    jupyter lab
echo.
if exist "%project_dir%\ui\brain-ai-ui" (
    echo 4. 启动Web界面:
    echo    cd %project_dir%\ui\brain-ai-ui
    echo    npm install
    echo    npm run dev
    echo.
)

:: GPU支持提示
echo 5. GPU支持 (如果有NVIDIA GPU):
echo    安装CUDA版本的PyTorch:
echo    pip uninstall torch torchvision torchaudio
echo    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo.

set /p open_docs="是否打开详细安装文档? (y/N): "
if /i "%open_docs%"=="y" start docs\windows11_installation.md

echo.
echo 感谢使用Brain AI！🚀
pause
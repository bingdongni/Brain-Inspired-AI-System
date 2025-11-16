@echo off
setlocal enabledelayedexpansion

:: 脑启发AI系统 - Windows快速安装验证
:: Brain-Inspired AI System - Windows Quick Installation Check

echo.
echo ================================================================
echo 🧠 脑启发AI系统 - Windows安装验证
echo ================================================================
echo.

:: 设置颜色（如果支持）
if "%TERM%"=="dumb" goto skip_color
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"
:skip_color

echo 📋 步骤1: 检查Python环境
echo ------------------------------------------------

:: 检查Python
python --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    echo ✅ Python版本: !PYTHON_VERSION ✓
) else (
    echo ❌ Python未安装或未添加到PATH
    echo 请从 https://python.org 下载安装Python 3.8+
    goto :error
)

:: 检查pip
pip --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ pip包管理器可用 ✓
) else (
    echo ❌ pip未安装
    echo 重新安装Python并确保勾选"Add Python to PATH"
    goto :error
)

echo.
echo 📋 步骤2: 检查核心依赖包
echo ------------------------------------------------

:: 检查核心包
set CORE_PACKAGES=numpy scipy pandas torch matplotlib sklearn yaml click

for %%p in (%CORE_PACKAGES%) do (
    python -c "import %%p; print('%%p v' + %%p.__version__)" >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "tokens=3" %%v in ('python -c "import %%p; print(%%p.__version__)" 2^>^&1') do echo ✅ %%p (v%%v) ✓
    ) else (
        echo ❌ %%p - 未安装
        set MISSING_PACKAGES=!MISSING_PACKAGES! %%p
    )
)

if defined MISSING_PACKAGES (
    echo.
    echo ⚠️  缺少依赖包:!MISSING_PACKAGES!
    echo 安装命令: pip install !MISSING_PACKAGES!
    echo.
    echo 🔄 正在尝试自动安装...
    pip install !MISSING_PACKAGES!
    if !errorlevel! neq 0 (
        echo ❌ 自动安装失败，请手动安装
        goto :error
    )
) else (
    echo ✅ 所有核心依赖包已安装 ✓
)

echo.
echo 📋 步骤3: 检查项目模块
echo ------------------------------------------------

:: 检查项目模块
set PROJECT_MODULES=hippocampus brain_ai lifelong_learning memory_interface

for %%m in (%PROJECT_MODULES%) do (
    python -c "import %%m" >nul 2>&1
    if !errorlevel! equ 0 (
        echo ✅ %%m ✓
    ) else (
        echo ❌ %%m - 未安装或不可用
        set MISSING_MODULES=!MISSING_MODULES! %%m
    )
)

if defined MISSING_MODULES (
    echo ⚠️  缺少项目模块:!MISSING_MODULES!
    echo 请确保在项目根目录运行此脚本
) else (
    echo ✅ 所有项目核心模块可用 ✓
)

echo.
echo 📋 步骤4: 检查系统兼容性
echo ------------------------------------------------

:: 系统信息
for /f "tokens=*" %%i in ('ver') do set SYSTEM_VERSION=%%i
echo ✅ 操作系统: !SYSTEM_VERSION!

:: Python架构
for /f "tokens=2 delims= " %%i in ('python -c "import platform; print(platform.machine())"') do set ARCH=%%i
echo ✅ 系统架构: !ARCH!

:: 检查GPU支持
python -c "import torch; print('CUDA available:', torch.cuda.is_available())" >nul 2>&1
if !errorlevel! equ 0 (
    for /f "tokens=2" %%a in ('python -c "import torch; print(torch.cuda.is_available())"') do (
        if %%a==True (
            echo ✅ GPU支持: 可用
            for /f "tokens=1" %%d in ('python -c "import torch; print(torch.cuda.device_count())"') do echo ✅ GPU设备: %%d个
        ) else (
            echo ⚠️  GPU支持: 不可用（将使用CPU模式）
        )
    )
) else (
    echo ⚠️  GPU支持检查失败
)

echo.
echo 📋 步骤5: 检查UI组件
echo ------------------------------------------------

:: 检查CLI工具
if exist "cli_demo.py" (
    echo ✅ CLI工具: cli_demo.py ✓
) else (
    echo ❌ CLI工具: cli_demo.py - 未找到
)

if exist "main.py" (
    echo ✅ CLI工具: main.py ✓
) else (
    echo ❌ CLI工具: main.py - 未找到
)

:: 检查Web界面
if exist "ui\brain-ai-ui" (
    if exist "ui\brain-ai-ui\package.json" (
        echo ✅ Web界面: React项目结构 ✓
    ) else (
        echo ⚠️  Web界面: 目录存在但缺少package.json
    )
) else (
    echo ⚠️  Web界面: ui\brain-ai-ui目录未找到
)

:: 检查Jupyter集成
if exist "ui\jupyter_integration.py" (
    echo ✅ Jupyter集成: jupyter_integration.py ✓
) else (
    echo ❌ Jupyter集成: jupyter_integration.py未找到
)

if exist "ui\界面使用演示.ipynb" (
    echo ✅ Jupyter演示: 界面使用演示.ipynb ✓
) else (
    echo ❌ Jupyter演示: 界面使用演示.ipynb未找到
)

echo.
echo 📋 步骤6: 检查数据目录结构
echo ------------------------------------------------

:: 检查并创建必要的目录
set DATA_DIRS=data\datasets data\models data\results logs config examples

for %%d in (%DATA_DIRS%) do (
    if exist "%%d" (
        echo ✅ 目录存在: %%d ✓
    ) else (
        echo ⚠️  目录不存在: %%d
        mkdir "%%d" >nul 2>&1
        if !errorlevel! equ 0 (
            echo ✅ 已创建目录: %%d
        ) else (
            echo ❌ 无法创建目录 %%d
        )
    )
)

echo.
echo ================================================================
echo 📊 检查总结
echo ================================================================

:: 重新检查一次核心功能
echo 📋 运行基本功能测试...
python -c "
try:
    from hippocampus import HippocampusSimulator
    import numpy as np
    h = HippocampusSimulator(memory_capacity=10)
    print('✅ 海马体系统创建成功 ✓')
    print('✅ 基本功能测试通过 ✓')
except Exception as e:
    print('❌ 功能测试失败:', str(e))
    exit(1)
" >nul 2>&1

if !errorlevel! equ 0 (
    echo ✅ 所有检查通过！系统已准备就绪！
    echo.
    echo 🚀 推荐开始方式:
    echo   1. 运行基础演示:
    echo      python cli_demo.py --mode demo
    echo   2. 启动Web界面:
    echo      cd ui\brain-ai-ui ^&^& npm run dev
    echo   3. 查看完整文档:
    echo      docs\quick_start_guide.md
    echo.
    echo 🎉 恭喜！脑启发AI系统安装成功！
    goto :success
) else (
    echo ⚠️  安装存在一些问题
    echo.
    echo 🔧 修复建议:
    echo   1. 重新安装缺失的依赖:
    echo      pip install -r requirements.txt
    echo   2. 确保在正确的虚拟环境中
    echo   3. 重新运行安装脚本:
    echo      install.bat
    echo.
    echo 📞 获取帮助:
    echo   - 查看完整文档: docs\quick_start_guide.md
    echo   - GitHub Issues: 提交问题报告
    goto :error
)

:success
echo.
echo ================================================================
echo 🎉 验证完成！系统正常运行！
echo ================================================================
pause
exit /b 0

:error
echo.
echo ================================================================
echo ❌ 验证失败！请检查上述错误信息！
echo ================================================================
pause
exit /b 1
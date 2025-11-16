@echo off
chcp 65001 >nul
echo 🧠 Windows 11 Brain AI 环境检查
echo ========================================
echo.

:: 检查Python安装
echo [1/6] 检查Python安装...
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Python 已安装
    python --version
) else (
    echo ❌ Python 未安装或未添加到PATH
)
echo.

:: 检查pip版本
echo [2/6] 检查pip...
pip --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ pip 已安装
    pip --version
) else (
    echo ❌ pip 未安装
)
echo.

:: 检查Node.js安装
echo [3/6] 检查Node.js...
node --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Node.js 已安装
    node --version
    npm --version
) else (
    echo ❌ Node.js 未安装
)
echo.

:: 检查NVIDIA GPU
echo [4/6] 检查NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ NVIDIA驱动已安装
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
) else (
    echo ❌ NVIDIA驱动未安装或无可用GPU
)
echo.

:: 检查端口占用
echo [5/6] 检查关键端口...
netstat -ano | findstr ":8888" >nul
if %errorlevel% equ 0 (
    echo ⚠️  端口8888 (Jupyter) 被占用
) else (
    echo ✅ 端口8888 (Jupyter) 可用
)

netstat -ano | findstr ":5173" >nul
if %errorlevel% equ 0 (
    echo ⚠️  端口5173 (Web界面) 被占用
) else (
    echo ✅ 端口5173 (Web界面) 可用
)
echo.

:: 检查GPU内存
echo [6/6] 测试PyTorch和GPU...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" 2>nul
if %errorlevel% neq 0 (
    echo ❌ PyTorch未安装
)
echo.

:: 磁盘空间检查
echo [磁盘空间检查]
for /f "tokens=3" %%a in ('dir C:\ /-c ^| find "bytes free"') do set free_space=%%a
echo 可用空间约: %free_space% bytes
echo.

:: 内存检查
echo [内存检查]
for /f "skip=1 tokens=2 delims=:" %%a in ('wmic OS get TotalVisibleMemorySize /value') do set total_mem=%%a
for /f "skip=1 tokens=2 delims=:" %%a in ('wmic OS get FreePhysicalMemory /value') do set free_mem=%%a
set /a used_mem=total_mem-free_mem
set /a mem_percent=used_mem*100/total_mem
echo 总内存: %total_mem% KB
echo 已用内存: %used_mem% KB (%mem_percent%%%)
echo.

echo ========================================
echo 环境检查完成！
echo.
echo 建议操作:
echo 1. 如果有❌标记，请安装相应的软件
echo 2. 确保所有端口可用
echo 3. 运行 optimize_windows.ps1 进行性能优化
echo.
pause
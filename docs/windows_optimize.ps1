# Windows 11 Brain AI 性能优化脚本
# 需要以管理员权限运行

Write-Host "🧠 开始Windows 11性能优化..." -ForegroundColor Green

# 设置高性能电源计划
Write-Host "设置高性能电源计划..." -ForegroundColor Yellow
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# 禁用Windows Defender实时保护 (可选)
# Set-MpPreference -DisableRealtimeMonitoring $true

# 设置环境变量
Write-Host "配置环境变量..." -ForegroundColor Yellow
[Environment]::SetEnvironmentVariable("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512", "Machine")
[Environment]::SetEnvironmentVariable("CUDA_CACHE_MAXSIZE", "2147483648", "Machine")
[Environment]::SetEnvironmentVariable("PYTHONHASHSEED", "0", "User")
[Environment]::SetEnvironmentVariable("PYTHONDONTWRITEBYTECODE", "1", "User")

# 启用开发者模式
Write-Host "启用开发者模式..." -ForegroundColor Yellow
reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock" /t REG_DWORD /f /v "AllowDevelopmentWithoutDevLicense" /d 1

# 优化虚拟内存
Write-Host "优化虚拟内存设置..." -ForegroundColor Yellow
$computerSystem = Get-WmiObject Win32_ComputerSystem
$totalRAM = [math]::Round($computerSystem.TotalPhysicalMemory / 1GB)
$pageFileSize = [math]::Round($totalRAM * 1.5 * 1024) # 1.5x RAM in MB

# 设置页面文件
$cs = Get-WmiObject -Class Win32_ComputerSystem -EnableAllPrivileges
$cs.AutomaticManagedPagefile = $false
$cs.Put()

$pagefile = Get-WmiObject -Class Win32_PageFileSetting
if ($pagefile) {
    $pagefile.InitialSize = $pageFileSize
    $pagefile.MaximumSize = $pageFileSize
    $pagefile.Put()
}

# 清理临时文件
Write-Host "清理临时文件..." -ForegroundColor Yellow
Remove-Item -Path "$env:TEMP\*" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$env:SystemRoot\Temp\*" -Recurse -Force -ErrorAction SilentlyContinue

# 优化网络设置
Write-Host "优化网络设置..." -ForegroundColor Yellow
netsh int tcp set global autotuninglevel=normal
netsh int tcp set global chimney=enabled
netsh int tcp set global rss=enabled
netsh int tcp set global netdma=enabled

# 禁用不必要的服务
Write-Host "禁用不必要的服务..." -ForegroundColor Yellow
$servicesToDisable = @(
    "XblAuthManager",
    "XblGameSave",
    "XboxGipSvc",
    "XboxNetApiSvc"
)

foreach ($service in $servicesToDisable) {
    try {
        Set-Service -Name $service -StartupType Disabled -ErrorAction SilentlyContinue
        Stop-Service -Name $service -Force -ErrorAction SilentlyContinue
        Write-Host "已禁用服务: $service" -ForegroundColor Green
    }
    catch {
        Write-Host "无法禁用服务: $service" -ForegroundColor Red
    }
}

# 启用Windows功能
Write-Host "启用Hyper-V..." -ForegroundColor Yellow
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All -NoRestart

# 设置注册表优化
Write-Host "应用注册表优化..." -ForegroundColor Yellow

# 优化文件缓存
reg add "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management" /v "LargeSystemCache" /t REG_DWORD /d 1 /f

# 禁用启动时间优化 (可选)
# reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer" /v "Max Cached Icons" /t REG_SZ /d 4096 /f

Write-Host ""
Write-Host "✅ Windows 11优化完成！" -ForegroundColor Green
Write-Host ""
Write-Host "重启建议:" -ForegroundColor Yellow
Write-Host "为了使所有更改生效，建议重启计算机。" -ForegroundColor Yellow
Write-Host ""
Write-Host "手动优化建议:" -ForegroundColor Yellow
Write-Host "1. 关闭不必要的后台应用"
Write-Host "2. 禁用开机自启动程序"
Write-Host "3. 定期清理磁盘空间"
Write-Host ""

$response = Read-Host "是否现在重启? (y/N)"
if ($response -eq "y" -or $response -eq "Y") {
    Write-Host "正在重启..." -ForegroundColor Yellow
    Restart-Computer -Force
} else {
    Write-Host "请记得稍后重启计算机以应用所有更改。" -ForegroundColor Yellow
}
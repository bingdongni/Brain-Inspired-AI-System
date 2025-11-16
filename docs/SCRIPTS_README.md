# Windows 11 安装脚本使用说明

本目录包含了完整的Windows 11环境下Brain AI系统的安装、配置和诊断脚本。

## 📋 脚本清单

### 1. 主安装指南
- **`windows11_installation.md`** - 完整的Windows 11安装指南（1100+行）
  - 详细的安装步骤说明
  - 所有依赖包安装指南
  - 故障排除方案
  - 性能优化建议

### 2. 一键安装脚本
- **`quick_install.bat`** - 快速安装向导
  - 自动检查Python、Node.js环境
  - 一键安装所有Python依赖
  - 创建和配置虚拟环境
  - 提供后续使用指导

**使用方法:**
```batch
# 以管理员身份运行
quick_install.bat
```

### 3. 环境检查脚本
- **`windows_check.bat`** - Windows环境快速检查
  - 检查Python、pip、Node.js安装
  - 检查GPU驱动状态
  - 检查端口占用情况
  - 磁盘空间和内存状态

**使用方法:**
```batch
# 双击运行或在命令行执行
windows_check.bat
```

### 4. 性能优化脚本
- **`windows_optimize.ps1`** - Windows 11系统性能优化
  - 设置高性能电源计划
  - 配置环境变量
  - 禁用不必要服务
  - 优化虚拟内存设置

**使用方法:**
```powershell
# 以管理员身份运行PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\windows_optimize.ps1
```

### 5. 系统诊断脚本
- **`diagnose.py`** - 完整系统诊断工具
  - 系统信息检查
  - 依赖包状态验证
  - GPU性能测试
  - 网络连接测试
  - 详细诊断报告

**使用方法:**
```bash
python diagnose.py
```

### 6. 安装验证脚本
- **`verify_installation.py`** - 安装验证工具
  - 验证所有组件正常工作
  - 测试PyTorch和GPU功能
  - 检查项目模块导入
  - 生成验证报告

**使用方法:**
```bash
python verify_installation.py
```

## 🚀 推荐使用流程

### 第一次安装
1. **检查环境**: 运行 `windows_check.bat`
2. **快速安装**: 运行 `quick_install.bat`
3. **系统优化**: 以管理员身份运行 `windows_optimize.ps1`
4. **验证安装**: 运行 `verify_installation.py`
5. **详细诊断**: 运行 `diagnose.py`

### 问题排查
1. **快速检查**: 运行 `windows_check.bat`
2. **详细诊断**: 运行 `diagnose.py`
3. **重新验证**: 运行 `verify_installation.py`

### 性能优化
1. **系统优化**: 运行 `windows_optimize.ps1`
2. **性能测试**: 运行 `diagnose.py` (查看性能部分)

## 📊 输出文件

脚本运行后会生成以下报告文件：

- `diagnosis_report.json` - 详细诊断报告
- `verification_report.json` - 安装验证报告

## ⚠️ 注意事项

### 权限要求
- `windows_optimize.ps1` 需要管理员权限
- 其他脚本通常不需要管理员权限

### 安全设置
如果遇到PowerShell执行策略问题：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 防火墙设置
确保以下端口可访问：
- 8888 - Jupyter Lab
- 5173 - Web界面开发服务器
- 6006 - TensorBoard

## 🔧 故障排除

### 常见问题

#### 1. 脚本无法执行
```bash
# 检查执行策略
Get-ExecutionPolicy

# 设置执行策略
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 2. Python命令不可用
- 重新安装Python，确保勾选"Add to PATH"
- 或手动添加Python路径到环境变量

#### 3. GPU相关问题
- 安装最新NVIDIA驱动
- 检查CUDA版本兼容性
- 确认PyTorch版本匹配CUDA版本

#### 4. 端口被占用
```bash
# 查看端口占用
netstat -ano | findstr :8888

# 结束占用进程
taskkill /PID <进程ID> /F
```

## 💡 高级使用

### 自定义配置
可以修改脚本中的配置参数：
- 端口号
- Python版本要求
- 依赖包列表
- 性能阈值

### 批处理使用
创建批处理文件组合多个脚本：
```batch
@echo off
echo 开始Brain AI环境安装和配置...
windows_check.bat
quick_install.bat
verify_installation.py
echo 安装完成！
pause
```

### 自动化部署
将脚本集成到CI/CD流程中：
```yaml
# GitHub Actions示例
- name: Setup Brain AI Environment
  run: |
    python docs/verify_installation.py
    python docs/diagnose.py
```

## 📞 获取帮助

如果遇到问题：
1. 查看详细安装指南：`windows11_installation.md`
2. 运行诊断脚本：`python diagnose.py`
3. 检查生成的报告文件
4. 参考故障排除部分

---

*最后更新: 2025-11-16*
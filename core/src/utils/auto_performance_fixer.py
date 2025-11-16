#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动性能修复脚本
===============

自动检测和修复已识别的性能问题：
1. 低效循环自动优化
2. 内存泄漏自动修复
3. 缓存机制自动插入
4. 性能问题自动报告

作者: Brain-Inspired AI Team
创建时间: 2025-11-16
"""

import os
import re
import ast
import astpretty
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import difflib
import shutil
from dataclasses import dataclass
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CodeIssue:
    """代码问题描述"""
    file_path: str
    line_number: int
    issue_type: str  # 'inefficient_loop', 'memory_leak', 'global_variable', 'missing_cache'
    description: str
    suggestion: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    original_code: str
    optimized_code: str


class AutoPerformanceFixer:
    """自动性能修复器"""
    
    def __init__(self, project_root: str, backup_enabled: bool = True):
        """
        初始化自动性能修复器
        
        Args:
            project_root: 项目根目录
            backup_enabled: 是否创建备份
        """
        self.project_root = Path(project_root)
        self.backup_enabled = backup_enabled
        self.issues_found = []
        self.fixes_applied = []
        self.backup_dir = self.project_root / "performance_fix_backup"
        self.lock = threading.RLock()
        
        # 编译优化正则表达式
        self.compile_patterns()
    
    def compile_patterns(self):
        """编译性能问题检测模式"""
        # 低效循环模式
        self.inefficient_loop_patterns = [
            r'for\s+i\s+in\s+range\(len\(([^)]+)\)\):',  # for i in range(len(x))
            r'for\s+(\w+)\s+in\s+range\(len\(([^)]+)\)\):',  # 更一般的情况
            r'while\s+len\([^)]+\)\s*>\s*0:',  # while len(x) > 0
        ]
        
        # 内存泄漏模式
        self.memory_leak_patterns = [
            r'open\([^)]+\)(?!.*with)',  # 未使用with的文件打开
            r'pickle\.load\([^)]+\)(?!.*with)',  # 未使用with的pickle
            r'\.cache\[',  # 无限制的字典缓存
            r'deque\(maxlen=None\)',  # 无限制的双端队列
        ]
        
        # 全局变量模式
        self.global_variable_patterns = [
            r'global\s+_',  # 全局变量声明
        ]
        
        # 缺少缓存的模式
        self.missing_cache_patterns = [
            r'def\s+(\w+)\([^)]*\):\s*\n\s+return\s+.*\n(?!\s*@)',  # 缺少缓存装饰器的函数
        ]
    
    def scan_project(self, file_extensions: List[str] = None) -> List[CodeIssue]:
        """
        扫描整个项目查找性能问题
        
        Args:
            file_extensions: 要扫描的文件扩展名列表
            
        Returns:
            发现的问题列表
        """
        if file_extensions is None:
            file_extensions = ['.py']
        
        logger.info(f"开始扫描项目: {self.project_root}")
        
        issues = []
        python_files = []
        
        # 查找所有Python文件
        for ext in file_extensions:
            python_files.extend(self.project_root.rglob(f"*{ext}"))
        
        logger.info(f"找到 {len(python_files)} 个Python文件")
        
        # 并行扫描文件
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for file_path in python_files:
                # 跳过某些目录
                if any(skip_dir in str(file_path) for skip_dir in ['.git', '__pycache__', '.pytest_cache', 'venv', 'env']):
                    continue
                
                future = executor.submit(self.scan_file, file_path)
                futures.append((file_path, future))
            
            # 收集结果
            for file_path, future in futures:
                try:
                    file_issues = future.result(timeout=30)
                    with self.lock:
                        issues.extend(file_issues)
                except Exception as e:
                    logger.error(f"扫描文件失败 {file_path}: {e}")
        
        with self.lock:
            self.issues_found = issues
        
        logger.info(f"扫描完成，发现 {len(issues)} 个性能问题")
        
        # 按严重程度分类
        self.categorize_issues()
        
        return issues
    
    def scan_file(self, file_path: Path) -> List[CodeIssue]:
        """扫描单个文件查找性能问题"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"无法读取文件 {file_path}: {e}")
            return issues
        
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # 检测低效循环
            issues.extend(self.detect_inefficient_loops(file_path, line_num, line))
            
            # 检测内存泄漏
            issues.extend(self.detect_memory_leaks(file_path, line_num, line))
            
            # 检测全局变量
            issues.extend(self.detect_global_variables(file_path, line_num, line))
            
            # 检测缺少缓存的函数
            issues.extend(self.detect_missing_cache(file_path, line_num, line, content))
        
        return issues
    
    def detect_inefficient_loops(self, file_path: Path, line_num: int, line: str) -> List[CodeIssue]:
        """检测低效循环"""
        issues = []
        
        for pattern in self.inefficient_loop_patterns:
            match = re.search(pattern, line)
            if match:
                original_code = line.strip()
                
                # 生成优化建议
                if 'for i in range(len(' in original_code:
                    if 'len(' in original_code and '))' in original_code:
                        # 提取变量名
                        var_match = re.search(r'for\s+(\w+)\s+in\s+range\(len\(([^)]+)\)\):', original_code)
                        if var_match:
                            var_name = var_match.group(1)
                            list_name = var_match.group(2)
                            optimized_code = f"for {var_name} in {list_name}:"
                            
                                issues.append(CodeIssue(
                                file_path=str(file_path),
                                line_number=line_num,
                                issue_type='inefficient_loop',
                                description=f"低效的范围循环: {original_code}",
                                suggestion="使用enumerate()或直接迭代替代range(len())",
                                optimized_code=optimized_code,
                                original_code=original_code,
                                severity='medium'
                            ))
                else:
                    issues.append(CodeIssue(
                        file_path=str(file_path),
                        line_number=line_num,
                        issue_type='inefficient_loop',
                        description=f"可能的低效循环: {original_code}",
                        suggestion="考虑向量化操作或使用内置函数",
                        optimized_code="# 需要手动优化",
                        original_code=original_code,
                        severity='low'
                    ))
        
        return issues
    
    def detect_memory_leaks(self, file_path: Path, line_num: int, line: str) -> List[CodeIssue]:
        """检测内存泄漏"""
        issues = []
        
        for pattern in self.memory_leak_patterns:
            match = re.search(pattern, line)
            if match:
                original_code = line.strip()
                severity = 'high'
                suggestion = ""
                optimized_code = ""
                
                if 'open(' in original_code and 'with' not in original_code:
                    suggestion = "使用with语句确保文件正确关闭"
                    optimized_code = re.sub(r'open\(([^)]+)\)', r'with open(\1) as f:', original_code)
                    severity = 'high'
                
                elif 'pickle.load(' in original_code and 'with' not in original_code:
                    suggestion = "使用with语句确保pickle文件正确关闭"
                    optimized_code = re.sub(r'pickle\.load\(([^)]+)\)', r'with open(\1, "rb") as f: pickle.load(f)', original_code)
                    severity = 'medium'
                
                elif '.cache[' in original_code:
                    suggestion = "添加缓存大小限制和过期策略"
                    optimized_code = "# 添加缓存管理"
                    severity = 'medium'
                
                elif 'deque(maxlen=None)' in original_code:
                    suggestion = "为deque指定maxlen防止内存泄漏"
                    optimized_code = original_code.replace('maxlen=None', 'maxlen=1000')
                    severity = 'low'
                
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=line_num,
                    issue_type='memory_leak',
                    description=f"潜在的内存泄漏: {original_code}",
                    suggestion=suggestion,
                    optimized_code=optimized_code,
                    original_code=original_code,
                    severity=severity
                ))
        
        return issues
    
    def detect_global_variables(self, file_path: Path, line_num: int, line: str) -> List[CodeIssue]:
        """检测全局变量"""
        issues = []
        
        for pattern in self.global_variable_patterns:
            match = re.search(pattern, line)
            if match:
                original_code = line.strip()
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=line_num,
                    issue_type='global_variable',
                    description=f"使用全局变量: {original_code}",
                    suggestion="考虑使用单例模式或依赖注入",
                    optimized_code="# 替换为局部变量或单例模式",
                    original_code=original_code,
                    severity='low'
                ))
        
        return issues
    
    def detect_missing_cache(self, file_path: Path, line_num: int, line: str, content: str) -> List[CodeIssue]:
        """检测缺少缓存的函数"""
        issues = []
        
        # 检测可能的计算密集型函数
        if line.strip().startswith('def ') and 'cache' not in line:
            # 简单的启发式检测：检查函数名和参数
            func_match = re.match(r'def\s+(\w+)\([^)]*\):', line)
            if func_match:
                func_name = func_match.group(1)
                
                # 检查函数名是否暗示计算密集型
                if any(keyword in func_name.lower() for keyword in 
                      ['compute', 'calculate', 'process', 'transform', 'analyze']):
                    # 检查下一行是否包含复杂的return语句
                    lines = content.split('\n')
                    if line_num < len(lines):
                        next_line = lines[line_num].strip() if line_num < len(lines) else ""
                        
                        if any(keyword in next_line.lower() for keyword in ['return', 'np.', 'math.']):
                            issues.append(CodeIssue(
                                file_path=str(file_path),
                                line_number=line_num,
                                issue_type='missing_cache',
                                description=f"函数 {func_name} 可能需要缓存",
                                suggestion="添加@functools.lru_cache或内存缓存",
                                optimized_code=f"@functools.lru_cache(maxsize=128)\n{line}",
                                original_code=line,
                                severity='low'
                            ))
        
        return issues
    
    def categorize_issues(self):
        """对问题按严重程度分类"""
        self.critical_issues = [issue for issue in self.issues_found if issue.severity == 'critical']
        self.high_issues = [issue for issue in self.issues_found if issue.severity == 'high']
        self.medium_issues = [issue for issue in self.issues_found if issue.severity == 'medium']
        self.low_issues = [issue for issue in self.issues_found if issue.severity == 'low']
    
    def create_backup(self) -> bool:
        """创建项目备份"""
        if not self.backup_enabled:
            return True
        
        try:
            if self.backup_dir.exists():
                shutil.rmtree(self.backup_dir)
            
            self.backup_dir.mkdir(parents=True)
            
            # 复制源代码目录（排除某些文件）
            src_dirs = ['brain-inspired-ai/src', 'hippocampus', 'memory_interface']
            
            for src_dir in src_dirs:
                src_path = self.project_root / src_dir
                if src_path.exists():
                    backup_path = self.backup_dir / src_dir
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 使用shutil复制目录
                    shutil.copytree(src_path, backup_path, 
                                  ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
            
            logger.info(f"备份已创建: {self.backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"创建备份失败: {e}")
            return False
    
    def apply_safe_fixes(self, severity_threshold: str = 'high') -> List[CodeIssue]:
        """
        应用安全的修复
        
        Args:
            severity_threshold: 修复的严重程度阈值 ('critical', 'high', 'medium', 'low')
            
        Returns:
            成功应用的修复列表
        """
        severity_levels = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        threshold_level = severity_levels.get(severity_threshold, 3)
        
        # 筛选要修复的问题
        issues_to_fix = []
        for issue in self.issues_found:
            issue_level = severity_levels.get(issue.severity, 3)
            if issue_level <= threshold_level:
                issues_to_fix.append(issue)
        
        logger.info(f"准备修复 {len(issues_to_fix)} 个问题（严重程度 >= {severity_threshold}）")
        
        # 按文件分组问题
        issues_by_file = defaultdict(list)
        for issue in issues_to_fix:
            issues_by_file[issue.file_path].append(issue)
        
        applied_fixes = []
        
        # 创建备份
        if not self.create_backup():
            logger.error("无法创建备份，取消修复")
            return applied_fixes
        
        # 应用修复
        for file_path, file_issues in issues_by_file.items():
            try:
                fixes_applied = self.apply_file_fixes(Path(file_path), file_issues)
                applied_fixes.extend(fixes_applied)
            except Exception as e:
                logger.error(f"修复文件失败 {file_path}: {e}")
        
        with self.lock:
            self.fixes_applied = applied_fixes
        
        logger.info(f"成功应用 {len(applied_fixes)} 个修复")
        return applied_fixes
    
    def apply_file_fixes(self, file_path: Path, issues: List[CodeIssue]) -> List[CodeIssue]:
        """修复单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"无法读取文件 {file_path}: {e}")
            return []
        
        lines = content.split('\n')
        applied_fixes = []
        
        # 按行号排序（从后往前修复，避免行号偏移）
        sorted_issues = sorted(issues, key=lambda x: x.line_number, reverse=True)
        
        for issue in sorted_issues:
            if issue.line_number <= len(lines):
                try:
                    # 检查原始代码是否匹配
                    original_line = lines[issue.line_number - 1]
                    
                    if original_line.strip() == issue.original_code.strip():
                        # 应用修复
                        lines[issue.line_number - 1] = issue.optimized_code
                        applied_fixes.append(issue)
                        logger.info(f"已修复 {file_path}:{issue.line_number} - {issue.issue_type}")
                    else:
                        logger.warning(f"代码不匹配，跳过修复 {file_path}:{issue.line_number}")
                        
                except Exception as e:
                    logger.error(f"修复行失败 {file_path}:{issue.line_number} - {e}")
        
        # 写回文件
        try:
            modified_content = '\n'.join(lines)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
        except Exception as e:
            logger.error(f"写回文件失败 {file_path}: {e}")
            return []
        
        return applied_fixes
    
    def generate_fix_report(self, output_path: str = None) -> str:
        """生成修复报告"""
        report_lines = [
            "# 自动性能修复报告",
            f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"项目路径: {self.project_root}",
            "",
            "## 问题统计",
            f"- 总问题数: {len(self.issues_found)}",
            f"- 关键问题: {len(self.critical_issues)}",
            f"- 高优先级问题: {len(self.high_issues)}",
            f"- 中优先级问题: {len(self.medium_issues)}",
            f"- 低优先级问题: {len(self.low_issues)}",
            "",
            f"- 已修复问题: {len(self.fixes_applied)}",
            "",
            "## 修复详情",
            ""
        ]
        
        if self.fixes_applied:
            for i, fix in enumerate(self.fixes_applied, 1):
                report_lines.extend([
                    f"### {i}. {fix.file_path}:{fix.line_number}",
                    f"**问题类型**: {fix.issue_type}",
                    f"**严重程度**: {fix.severity}",
                    f"**描述**: {fix.description}",
                    f"**建议**: {fix.suggestion}",
                    "",
                    "**修复前**:",
                    f"```python",
                    fix.original_code,
                    "```",
                    "",
                    "**修复后**:",
                    f"```python",
                    fix.optimized_code,
                    "```",
                    ""
                ])
        else:
            report_lines.append("没有应用任何修复。")
        
        # 未修复的问题
        unfixed_issues = [issue for issue in self.issues_found 
                         if issue not in self.fixes_applied]
        
        if unfixed_issues:
            report_lines.extend([
                "## 未修复的问题",
                ""
            ])
            
            for issue in unfixed_issues:
                report_lines.extend([
                    f"- **{issue.file_path}:{issue.line_number}** [{issue.severity}] {issue.issue_type}",
                    f"  - {issue.description}",
                    f"  - 建议: {issue.suggestion}",
                    ""
                ])
        
        # 优化建议
        report_lines.extend([
            "## 额外优化建议",
            "",
            "### 代码级别优化",
            "1. 使用NumPy向量化操作替代Python循环",
            "2. 实现适当的缓存机制（LRU缓存）",
            "3. 使用内存池管理频繁的小对象分配",
            "4. 实现批处理机制减少IO操作",
            "",
            "### 架构级别优化",
            "1. 考虑实现微服务架构分解大模块",
            "2. 添加异步处理支持",
            "3. 实现分布式缓存",
            "4. 使用连接池管理数据库连接",
            "",
            "### 监控和调优",
            "1. 集成性能监控工具（如PyTorch Profiler）",
            "2. 实施持续的性能基准测试",
            "3. 添加内存泄漏检测",
            "4. 实施自动扩展机制"
        ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"修复报告已保存到: {output_path}")
            except Exception as e:
                logger.error(f"保存报告失败: {e}")
        
        return report
    
    def rollback_fixes(self) -> bool:
        """回滚修复"""
        if not self.backup_enabled or not self.backup_dir.exists():
            logger.warning("没有备份可供回滚")
            return False
        
        try:
            # 恢复源代码
            for backup_file in self.backup_dir.rglob('*.py'):
                relative_path = backup_file.relative_to(self.backup_dir)
                original_file = self.project_root / relative_path
                
                if original_file.exists():
                    original_file.unlink()
                
                # 复制备份文件
                shutil.copy2(backup_file, original_file)
            
            logger.info("修复已回滚")
            return True
            
        except Exception as e:
            logger.error(f"回滚失败: {e}")
            return False


# 示例使用
if __name__ == "__main__":
    # 创建自动修复器
    fixer = AutoPerformanceFixer("/workspace", backup_enabled=True)
    
    print("开始自动性能问题检测和修复...")
    
    # 扫描项目
    issues = fixer.scan_project()
    
    print(f"\n发现 {len(issues)} 个性能问题:")
    print(f"- 关键问题: {len(fixer.critical_issues)}")
    print(f"- 高优先级: {len(fixer.high_issues)}")
    print(f"- 中优先级: {len(fixer.medium_issues)}")
    print(f"- 低优先级: {len(fixer.low_issues)}")
    
    # 显示一些典型问题
    print("\n典型问题示例:")
    for i, issue in enumerate(issues[:5], 1):
        print(f"{i}. {issue.file_path}:{issue.line_number}")
        print(f"   类型: {issue.issue_type} | 严重程度: {issue.severity}")
        print(f"   原始: {issue.original_code}")
        print(f"   建议: {issue.optimization_suggestion}")
        print()
    
    # 应用高优先级修复
    if fixer.high_issues:
        print("正在应用高优先级修复...")
        applied_fixes = fixer.apply_safe_fixes(severity_threshold='high')
        print(f"成功应用 {len(applied_fixes)} 个修复")
    
    # 生成报告
    report = fixer.generate_fix_report("/tmp/auto_fix_report.md")
    print(f"\n修复报告已生成: /tmp/auto_fix_report.md")
    print(f"报告预览:\n{report[:800]}...")
    
    print("\n自动性能修复完成!")
    
    # 注意：在实际使用中，用户可能希望先查看报告再决定是否应用修复
    # fixer.rollback_fixes()  # 如果需要回滚修复

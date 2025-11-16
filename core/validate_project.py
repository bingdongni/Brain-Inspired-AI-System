#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Brain-Inspired AI Framework 项目完整性验证脚本
==============================================

此脚本用于验证项目结构的完整性和功能的可用性，
确保所有组件都正确配置并且可以正常运行。

验证内容包括:
1. 项目文件结构检查
2. 依赖包验证
3. 模块导入测试
4. 核心功能测试
5. 配置文件验证
6. 文档完整性检查
"""

import sys
import os
import importlib
import importlib.util
import subprocess
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback

# 添加src路径到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

class ProjectValidator:
    """项目验证器"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent
        self.src_dir = self.project_root / "src"
        self.test_results = {}
        self.errors = []
        self.warnings = []
        self.success_count = 0
        self.total_checks = 0
    
    def log_info(self, message: str):
        """记录信息"""
        print(f"[INFO] {message}")
    
    def log_success(self, message: str):
        """记录成功"""
        print(f"[✓] {message}")
        self.success_count += 1
    
    def log_warning(self, message: str):
        """记录警告"""
        print(f"[⚠] {message}")
        self.warnings.append(message)
    
    def log_error(self, message: str):
        """记录错误"""
        print(f"[✗] {message}")
        self.errors.append(message)
    
    def log_header(self, title: str):
        """记录标题"""
        print(f"\n{'=' * 60}")
        print(f" {title}")
        print(f"{'=' * 60}")
    
    def run_check(self, name: str, check_func, *args, **kwargs) -> bool:
        """运行单个检查"""
        self.total_checks += 1
        try:
            result = check_func(*args, **kwargs)
            if result:
                self.log_success(f"{name}: 验证通过")
                self.test_results[name] = {"status": "PASS", "details": ""}
                return True
            else:
                self.log_error(f"{name}: 验证失败")
                self.test_results[name] = {"status": "FAIL", "details": "检查返回False"}
                return False
        except Exception as e:
            self.log_error(f"{name}: 验证异常 - {str(e)}")
            self.test_results[name] = {"status": "ERROR", "details": str(e)}
            return False
    
    def check_project_structure(self) -> bool:
        """检查项目文件结构"""
        required_dirs = [
            "src",
            "src/brain_ai",
            "src/modules",
            "tests",
            "demos",
            "examples",
            "docs",
            "scripts",
            "config",
            "data",
            "output"
        ]
        
        required_files = [
            "README.md",
            "requirements.txt",
            "setup.py",
            "main.py",
            "install.sh",
            "Makefile",
            "config.yaml",
            "LICENSE"
        ]
        
        # 检查目录
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            self.log_warning(f"缺少目录: {', '.join(missing_dirs)}")
        
        # 检查文件
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.log_warning(f"缺少文件: {', '.join(missing_files)}")
        
        # 检查核心__init__.py文件
        init_files = [
            "src/brain_ai/__init__.py",
            "src/modules/dynamic_routing/__init__.py",
            "src/modules/hippocampus/__init__.py",
            "src/modules/neocortex/__init__.py"
        ]
        
        missing_init = []
        for init_file in init_files:
            full_path = self.project_root / init_file
            if not full_path.exists():
                missing_init.append(init_file)
        
        if missing_init:
            self.log_error(f"缺少__init__.py文件: {', '.join(missing_init)}")
            return False
        
        return True
    
    def check_dependencies(self) -> bool:
        """检查依赖包"""
        # 核心依赖
        core_deps = [
            "torch",
            "numpy", 
            "scipy",
            "pandas",
            "scikit-learn",
            "matplotlib",
            "pyyaml",
            "click",
            "rich",
            "tqdm"
        ]
        
        missing_deps = []
        for dep in core_deps:
            try:
                importlib.import_module(dep)
                self.log_info(f"依赖 {dep}: 已安装")
            except ImportError:
                missing_deps.append(dep)
                self.log_error(f"缺少依赖: {dep}")
        
        if missing_deps:
            self.log_warning(f"建议安装缺少的依赖: pip install {' '.join(missing_deps)}")
            # 不返回False，因为某些依赖是可选的
        
        return True
    
    def check_module_imports(self) -> bool:
        """检查模块导入"""
        # 核心模块
        core_modules = [
            "brain_ai",
            "brain_ai.core",
            "brain_ai.cli"
        ]
        
        # 动态路由模块
        dynamic_routing_modules = [
            "modules.dynamic_routing",
            "modules.dynamic_routing.realtime_routing_controller",
            "modules.dynamic_routing.reinforcement_routing.actor_critic"
        ]
        
        # 海马体模块
        hippocampus_modules = [
            "modules.hippocampus",
            "modules.hippocampus.core.simulator",
            "modules.hippocampus.encoders.transformer_encoder"
        ]
        
        # 持续学习模块
        lifelong_learning_modules = [
            "elastic_weight_consolidation",
            "generative_replay",
            "dynamic_expansion",
            "knowledge_transfer"
        ]
        
        all_modules = core_modules + dynamic_routing_modules + hippocampus_modules + lifelong_learning_modules
        
        failed_imports = []
        for module_name in all_modules:
            try:
                importlib.import_module(module_name)
                self.log_success(f"模块 {module_name}: 导入成功")
            except Exception as e:
                failed_imports.append((module_name, str(e)))
                self.log_error(f"模块 {module_name}: 导入失败 - {e}")
        
        return len(failed_imports) == 0
    
    def check_core_functionality(self) -> bool:
        """检查核心功能"""
        try:
            # 测试基本导入
            from brain_ai import HippocampalSimulator, NeocortexArchitecture
            
            # 测试海马体创建
            hippocampus = HippocampalSimulator(
                input_dim=256,
                memory_dim=128,
                storage_capacity=1000
            )
            self.log_success("海马体模块: 创建成功")
            
            # 测试新皮层创建
            neocortex = NeocortexArchitecture(
                input_dim=256,
                hidden_dim=512,
                num_layers=4
            )
            self.log_success("新皮层模块: 创建成功")
            
            # 测试动态路由
            try:
                from modules.dynamic_routing import DynamicRoutingSystem
                router = DynamicRoutingSystem()
                self.log_success("动态路由模块: 创建成功")
            except Exception as e:
                self.log_warning(f"动态路由模块: 创建失败 - {e}")
            
            # 测试持续学习
            try:
                from elastic_weight_consolidation import ElasticWeightConsolidation
                ewc = ElasticWeightConsolidation()
                self.log_success("持续学习模块: 创建成功")
            except Exception as e:
                self.log_warning(f"持续学习模块: 创建失败 - {e}")
            
            return True
            
        except Exception as e:
            self.log_error(f"核心功能检查失败: {e}")
            self.log_error(f"详细错误: {traceback.format_exc()}")
            return False
    
    def check_cli_functionality(self) -> bool:
        """检查CLI功能"""
        try:
            # 测试CLI帮助
            result = subprocess.run([
                sys.executable, "main.py", "--help"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                self.log_success("CLI帮助: 运行正常")
            else:
                self.log_error(f"CLI帮助运行失败: {result.stderr}")
                return False
            
            # 测试info命令
            result = subprocess.run([
                sys.executable, "main.py", "info"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.log_success("CLI info命令: 运行正常")
            else:
                self.log_error(f"CLI info命令失败: {result.stderr}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            self.log_error("CLI命令超时")
            return False
        except Exception as e:
            self.log_error(f"CLI功能检查失败: {e}")
            return False
    
    def check_config_files(self) -> bool:
        """检查配置文件"""
        config_files = [
            "config.yaml",
            "config/development.yaml",
            "config/production.yaml",
            "src/config/development.yaml",
            "src/config/production.yaml"
        ]
        
        config_issues = []
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                            yaml.safe_load(f)
                    self.log_success(f"配置文件 {config_file}: 格式正确")
                except Exception as e:
                    config_issues.append((config_file, str(e)))
                    self.log_error(f"配置文件 {config_file}: 格式错误 - {e}")
            else:
                self.log_warning(f"配置文件不存在: {config_file}")
        
        return len(config_issues) == 0
    
    def check_documentation(self) -> bool:
        """检查文档完整性"""
        doc_files = [
            "README.md",
            "快速开始.md",
            "安装指南.md",
            "使用说明.md",
            "项目结构.md"
        ]
        
        missing_docs = []
        for doc_file in doc_files:
            doc_path = self.project_root / doc_file
            if not doc_path.exists():
                missing_docs.append(doc_file)
                self.log_error(f"缺少文档: {doc_file}")
            else:
                # 检查文档内容长度
                content = doc_path.read_text(encoding='utf-8')
                if len(content) < 100:
                    self.log_warning(f"文档 {doc_file} 内容太少，可能不完整")
                else:
                    self.log_success(f"文档 {doc_file}: 存在且有内容")
        
        return len(missing_docs) == 0
    
    def check_test_files(self) -> bool:
        """检查测试文件"""
        test_dirs = [
            "tests",
            "src/tests"
        ]
        
        test_files_found = 0
        for test_dir in test_dirs:
            test_path = self.project_root / test_dir
            if test_path.exists():
                for test_file in test_path.glob("test_*.py"):
                    test_files_found += 1
        
        if test_files_found > 0:
            self.log_success(f"找到测试文件: {test_files_found} 个")
        else:
            self.log_warning("未找到测试文件")
        
        return True  # 测试文件是可选的
    
    def run_performance_test(self) -> bool:
        """运行性能测试"""
        try:
            import torch
            import time
            
            # 创建小规模测试
            hippocampus = HippocampalSimulator(
                input_dim=128,
                memory_dim=64,
                storage_capacity=100
            )
            
            test_data = torch.randn(10, 128)
            
            start_time = time.time()
            
            # 测试编码速度
            encoding_result = hippocampus.encode_memory(test_data[:1])
            
            encode_time = time.time() - start_time
            
            if encode_time < 1.0:  # 1秒内完成
                self.log_success(f"编码性能: {encode_time:.3f}s (良好)")
            else:
                self.log_warning(f"编码性能: {encode_time:.3f}s (较慢)")
            
            return True
            
        except Exception as e:
            self.log_error(f"性能测试失败: {e}")
            return False
    
    def generate_report(self) -> Dict:
        """生成验证报告"""
        report = {
            "validation_summary": {
                "total_checks": self.total_checks,
                "passed_checks": self.success_count,
                "failed_checks": self.total_checks - self.success_count,
                "success_rate": f"{(self.success_count / self.total_checks * 100):.1f}%" if self.total_checks > 0 else "0%"
            },
            "test_results": self.test_results,
            "errors": self.errors,
            "warnings": self.warnings,
            "recommendations": []
        }
        
        # 生成建议
        if self.errors:
            report["recommendations"].append("修复所有错误以确保项目正常运行")
        
        if self.warnings:
            report["recommendations"].append("处理警告以提高项目质量")
        
        if self.success_count / self.total_checks < 0.8:
            report["recommendations"].append("项目验证通过率较低，建议全面检查")
        
        return report
    
    def run_all_validations(self) -> Dict:
        """运行所有验证"""
        self.log_header("Brain-Inspired AI Framework 项目验证")
        
        # 项目结构检查
        self.run_check("项目文件结构", self.check_project_structure)
        
        # 依赖检查
        self.run_check("依赖包检查", self.check_dependencies)
        
        # 模块导入检查
        self.run_check("模块导入检查", self.check_module_imports)
        
        # 核心功能检查
        self.run_check("核心功能检查", self.check_core_functionality)
        
        # CLI功能检查
        self.run_check("CLI功能检查", self.check_cli_functionality)
        
        # 配置文件检查
        self.run_check("配置文件检查", self.check_config_files)
        
        # 文档检查
        self.run_check("文档完整性检查", self.check_documentation)
        
        # 测试文件检查
        self.run_check("测试文件检查", self.check_test_files)
        
        # 性能测试
        self.run_check("性能测试", self.run_performance_test)
        
        # 生成报告
        self.log_header("验证报告")
        report = self.generate_report()
        
        # 打印报告
        print(f"总检查项: {report['validation_summary']['total_checks']}")
        print(f"通过检查: {report['validation_summary']['passed_checks']}")
        print(f"失败检查: {report['validation_summary']['failed_checks']}")
        print(f"成功率: {report['validation_summary']['success_rate']}")
        
        if self.errors:
            print(f"\n错误列表 ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\n警告列表 ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if report['recommendations']:
            print(f"\n建议:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        # 保存报告到文件
        report_file = self.project_root / "validation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.log_success(f"验证报告已保存: {report_file}")
        
        return report

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Brain-Inspired AI 项目验证工具")
    parser.add_argument("--project-root", help="项目根目录路径", default=".")
    parser.add_argument("--output", help="输出报告文件路径", default="validation_report.json")
    parser.add_argument("--quiet", action="store_true", help="静默模式")
    
    args = parser.parse_args()
    
    try:
        validator = ProjectValidator(args.project_root)
        report = validator.run_all_validations()
        
        # 根据验证结果设置退出码
        if validator.errors:
            print(f"\n{validator.log_error('验证失败! 发现错误')}")
            sys.exit(1)
        elif validator.warnings:
            print(f"\n{validator.log_warning('验证完成，但有警告')}")
            sys.exit(0)
        else:
            print(f"\n{validator.log_success('验证完全通过!')}")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n验证被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n验证过程中发生异常: {e}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    main()
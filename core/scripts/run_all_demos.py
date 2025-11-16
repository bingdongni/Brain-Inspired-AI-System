#!/usr/bin/env python3
"""
运行所有演示脚本
Run All Demos Script

自动运行所有演示和测试，生成完整的演示报告
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import argparse

class DemoRunner:
    """演示运行器"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.demo_results = {}
        self.start_time = None
        self.end_time = None
        
        # 演示脚本列表
        self.demo_scripts = {
            'cli_demo': {
                'script': 'cli_demo.py',
                'args': ['--mode', 'demo', '--dataset', 'synthetic', '--epochs', '10'],
                'description': '交互式命令行界面演示',
                'category': 'core'
            },
            'memory_learning': {
                'script': 'demos/memory_learning_demo.py',
                'args': ['--demo', 'memory'],
                'description': '记忆学习演示',
                'category': 'learning'
            },
            'lifelong_learning': {
                'script': 'demos/lifelong_learning_demo.py',
                'args': ['--demo', 'lifelong', '--tasks', '3'],
                'description': '终身学习演示',
                'category': 'learning'
            },
            'dynamic_routing': {
                'script': 'demos/dynamic_routing_demo.py',
                'args': ['--demo', 'routing'],
                'description': '动态路由演示',
                'category': 'network'
            },
            'benchmark_test': {
                'script': 'scripts/benchmark_test.py',
                'args': ['--test', 'all', '--device', 'auto'],
                'description': '性能基准测试',
                'category': 'performance'
            },
            'automated_testing': {
                'script': 'scripts/automated_testing.py',
                'args': ['--test', 'core', '--quiet'],
                'description': '自动化测试',
                'category': 'testing'
            }
        }
        
    def run_demo(self, demo_name: str) -> Dict[str, Any]:
        """运行单个演示"""
        if demo_name not in self.demo_scripts:
            return {
                'success': False,
                'error': f'未知演示: {demo_name}',
                'execution_time': 0
            }
            
        demo_info = self.demo_scripts[demo_name]
        script_path = self.base_dir / demo_info['script']
        
        print(f"🚀 运行演示: {demo_name}")
        print(f"   描述: {demo_info['description']}")
        print(f"   脚本: {script_path}")
        
        # 检查脚本是否存在
        if not script_path.exists():
            return {
                'success': False,
                'error': f'脚本不存在: {script_path}',
                'execution_time': 0
            }
            
        # 构建命令
        cmd = [sys.executable, str(script_path)] + demo_info['args']
        
        start_time = time.time()
        
        try:
            # 运行脚本
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            execution_time = time.time() - start_time
            
            demo_result = {
                'success': result.returncode == 0,
                'execution_time': execution_time,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'script_path': str(script_path),
                'demo_info': demo_info
            }
            
            if result.returncode == 0:
                print(f"   ✅ 成功完成 (耗时: {execution_time:.2f}s)")
            else:
                print(f"   ❌ 执行失败 (耗时: {execution_time:.2f}s)")
                print(f"   错误: {result.stderr[:200]}...")
                
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            demo_result = {
                'success': False,
                'error': '执行超时 (5分钟)',
                'execution_time': execution_time,
                'script_path': str(script_path),
                'demo_info': demo_info
            }
            print(f"   ⏰ 执行超时")
            
        except Exception as e:
            execution_time = time.time() - start_time
            demo_result = {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'script_path': str(script_path),
                'demo_info': demo_info
            }
            print(f"   ❌ 异常: {e}")
            
        return demo_result
        
    def run_all_demos(self, categories: List[str] = None) -> Dict[str, Any]:
        """运行所有演示"""
        print("🎭 运行所有演示")
        print("=" * 80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"工作目录: {self.base_dir}")
        
        self.start_time = time.time()
        
        # 选择要运行的演示
        demos_to_run = {}
        for name, info in self.demo_scripts.items():
            if categories is None or info['category'] in categories:
                demos_to_run[name] = info
                
        print(f"将运行 {len(demos_to_run)} 个演示")
        print()
        
        # 运行演示
        results = {}
        
        for demo_name in demos_to_run.keys():
            print(f"\n{'='*60}")
            result = self.run_demo(demo_name)
            results[demo_name] = result
            self.demo_results[demo_name] = result
            
        self.end_time = time.time()
        
        # 生成总结报告
        summary = self._generate_summary(results)
        
        # 保存结果
        self._save_results(results, summary)
        
        return {
            'results': results,
            'summary': summary
        }
        
    def run_single_category(self, category: str) -> Dict[str, Any]:
        """运行特定类别的演示"""
        print(f"🎯 运行演示类别: {category}")
        
        demos_in_category = {
            name: info for name, info in self.demo_scripts.items() 
            if info['category'] == category
        }
        
        if not demos_in_category:
            print(f"❌ 没有找到类别 '{category}' 的演示")
            return {'results': {}, 'summary': {'error': '没有找到指定类别的演示'}}
            
        print(f"找到 {len(demos_in_category)} 个演示:")
        for name, info in demos_in_category.items():
            print(f"   - {name}: {info['description']}")
            
        return self.run_all_demos([category])
        
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成演示总结"""
        total_demos = len(results)
        successful_demos = sum(1 for result in results.values() if result.get('success', False))
        failed_demos = total_demos - successful_demos
        
        total_time = sum(result.get('execution_time', 0) for result in results.values())
        avg_time = total_time / total_demos if total_demos > 0 else 0
        
        # 按类别统计
        categories = {}
        for demo_name, result in results.items():
            if demo_name in self.demo_scripts:
                category = self.demo_scripts[demo_name]['category']
                if category not in categories:
                    categories[category] = {'total': 0, 'successful': 0, 'total_time': 0}
                    
                categories[category]['total'] += 1
                categories[category]['total_time'] += result.get('execution_time', 0)
                
                if result.get('success', False):
                    categories[category]['successful'] += 1
                    
        # 计算类别成功率
        for category in categories:
            if categories[category]['total'] > 0:
                categories[category]['success_rate'] = categories[category]['successful'] / categories[category]['total']
                categories[category]['avg_time'] = categories[category]['total_time'] / categories[category]['total']
            else:
                categories[category]['success_rate'] = 0
                categories[category]['avg_time'] = 0
                
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_execution_time': total_time,
            'overall_stats': {
                'total_demos': total_demos,
                'successful_demos': successful_demos,
                'failed_demos': failed_demos,
                'success_rate': successful_demos / total_demos if total_demos > 0 else 0,
                'avg_execution_time': avg_time
            },
            'categories': categories
        }
        
        # 性能评估
        if successful_demos == total_demos:
            summary['performance_grade'] = 'Excellent'
        elif successful_demos >= total_demos * 0.8:
            summary['performance_grade'] = 'Good'
        elif successful_demos >= total_demos * 0.6:
            summary['performance_grade'] = 'Fair'
        else:
            summary['performance_grade'] = 'Poor'
            
        return summary
        
    def _save_results(self, results: Dict[str, Any], summary: Dict[str, Any]):
        """保存演示结果"""
        try:
            os.makedirs('data/results', exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存详细结果
            detailed_file = f"data/results/demos_detailed_{timestamp}.json"
            with open(detailed_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            # 保存总结
            summary_file = f"data/results/demos_summary_{timestamp}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                
            print(f"\n💾 结果已保存:")
            print(f"   详细结果: {detailed_file}")
            print(f"   总结报告: {summary_file}")
            
        except Exception as e:
            print(f"⚠️ 结果保存失败: {e}")
            
    def print_summary_report(self, summary: Dict[str, Any]):
        """打印总结报告"""
        print(f"\n📊 演示总结报告")
        print("=" * 80)
        
        overall = summary['overall_stats']
        print(f"📈 总体统计:")
        print(f"   总演示数: {overall['total_demos']}")
        print(f"   成功演示: {overall['successful_demos']}")
        print(f"   失败演示: {overall['failed_demos']}")
        print(f"   成功率: {overall['success_rate']:.1%}")
        print(f"   总执行时间: {overall['total_execution_time']:.2f}秒")
        print(f"   平均执行时间: {overall['avg_execution_time']:.2f}秒")
        print(f"   性能评级: {summary['performance_grade']}")
        
        print(f"\n📂 按类别统计:")
        for category, stats in summary['categories'].items():
            print(f"   {category}:")
            print(f"     演示数: {stats['total']}")
            print(f"     成功率: {stats['success_rate']:.1%}")
            print(f"     平均时间: {stats['avg_time']:.2f}秒")
            
    def check_demo_availability(self) -> Dict[str, bool]:
        """检查演示可用性"""
        print("🔍 检查演示可用性")
        print("=" * 40)
        
        availability = {}
        
        for demo_name, demo_info in self.demo_scripts.items():
            script_path = self.base_dir / demo_info['script']
            
            if script_path.exists():
                availability[demo_name] = True
                print(f"✅ {demo_name}: {script_path}")
            else:
                availability[demo_name] = False
                print(f"❌ {demo_name}: {script_path} (文件不存在)")
                
        return availability
        
    def create_demo_report(self) -> str:
        """创建演示报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"data/results/demo_report_{timestamp}.md"
        
        try:
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("# 脑启发AI演示系统报告\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 演示列表
                f.write("## 可用演示\n\n")
                for name, info in self.demo_scripts.items():
                    f.write(f"- **{name}**: {info['description']}\n")
                    f.write(f"  - 脚本: `{info['script']}`\n")
                    f.write(f"  - 类别: {info['category']}\n\n")
                    
                # 使用说明
                f.write("## 使用说明\n\n")
                f.write("### 运行所有演示\n")
                f.write("```bash\n")
                f.write("python run_all_demos.py\n")
                f.write("```\n\n")
                
                f.write("### 运行特定类别\n")
                f.write("```bash\n")
                f.write("python run_all_demos.py --category core\n")
                f.write("```\n\n")
                
                f.write("### 运行单个演示\n")
                f.write("```bash\n")
                f.write("python run_all_demos.py --demo cli_demo\n")
                f.write("```\n\n")
                
                print(f"📄 演示报告已创建: {report_file}")
                return report_file
                
        except Exception as e:
            print(f"❌ 报告创建失败: {e}")
            return ""


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行所有演示')
    parser.add_argument('--category', choices=['core', 'learning', 'network', 'performance', 'testing'],
                       help='运行特定类别的演示')
    parser.add_argument('--demo', help='运行单个演示')
    parser.add_argument('--check', action='store_true', help='检查演示可用性')
    parser.add_argument('--report', action='store_true', help='生成演示报告')
    parser.add_argument('--base-dir', help='基础目录路径')
    
    args = parser.parse_args()
    
    runner = DemoRunner(args.base_dir)
    
    if args.check:
        runner.check_demo_availability()
        
    elif args.report:
        runner.create_demo_report()
        
    elif args.demo:
        print(f"🎯 运行单个演示: {args.demo}")
        result = runner.run_demo(args.demo)
        print(f"\n结果: {'成功' if result.get('success') else '失败'}")
        print(f"耗时: {result.get('execution_time', 0):.2f}秒")
        if not result.get('success'):
            print(f"错误: {result.get('error', '未知错误')}")
            
    elif args.category:
        print(f"🎯 运行类别演示: {args.category}")
        result = runner.run_single_category(args.category)
        if 'summary' in result:
            runner.print_summary_report(result['summary'])
            
    else:
        print("🎭 运行所有演示")
        result = runner.run_all_demos()
        
        if 'summary' in result:
            runner.print_summary_report(result['summary'])


if __name__ == "__main__":
    main()
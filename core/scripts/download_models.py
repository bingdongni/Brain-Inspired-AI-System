#!/usr/bin/env python3
"""
预训练模型下载脚本
Pretrained Models Download Script

自动下载和管理预训练模型
"""

import os
import sys
import json
import requests
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import argparse

class ModelDownloader:
    """模型下载器"""
    
    def __init__(self, models_dir: str = "data/models/pretrained"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # 预定义模型列表
        self.available_models = {
            'brain_inspired_v1.0': {
                'url': 'https://github.com/brain-ai/models/releases/download/v1.0/brain_inspired_v1.0.pth',
                'filename': 'brain_inspired_v1.0.pth',
                'description': '完整脑启发模型',
                'size_mb': 50,
                'expected_hash': 'sha256:abc123def456',
                'requirements': ['torch']
            },
            'hippocampus_v1.0': {
                'url': 'https://github.com/brain-ai/models/releases/download/v1.0/hippocampus_v1.0.pth',
                'filename': 'hippocampus_v1.0.pth',
                'description': '海马体专用模型',
                'size_mb': 30,
                'expected_hash': 'sha256:def456ghi789',
                'requirements': ['torch']
            },
            'neocortex_v1.0': {
                'url': 'https://github.com/brain-ai/models/releases/download/v1.0/neocortex_v1.0.pth',
                'filename': 'neocortex_v1.0.pth',
                'description': '新皮层专用模型',
                'size_mb': 40,
                'expected_hash': 'sha256:ghi789jkl012',
                'requirements': ['torch']
            },
            'demo_models_pack': {
                'url': 'https://github.com/brain-ai/models/releases/download/v1.0/demo_models_pack.zip',
                'filename': 'demo_models_pack.zip',
                'description': '演示模型包',
                'size_mb': 100,
                'expected_hash': 'sha256:jkl012mno345',
                'requirements': []
            }
        }
        
    def list_available_models(self) -> Dict:
        """列出所有可用模型"""
        print("📋 可用预训练模型:")
        print("=" * 60)
        
        for model_id, info in self.available_models.items():
            print(f"🆔 {model_id}")
            print(f"   描述: {info['description']}")
            print(f"   大小: {info['size_mb']} MB")
            print(f"   文件: {info['filename']}")
            print(f"   要求: {', '.join(info['requirements']) if info['requirements'] else '无'}")
            print()
            
        return self.available_models
        
    def download_model(self, model_id: str, force: bool = False) -> bool:
        """下载指定模型"""
        if model_id not in self.available_models:
            print(f"❌ 未知模型: {model_id}")
            return False
            
        model_info = self.available_models[model_id]
        local_file = self.models_dir / model_info['filename']
        
        # 检查文件是否已存在
        if local_file.exists() and not force:
            print(f"✅ 模型已存在: {local_file}")
            
            # 验证文件完整性
            if self._verify_file(local_file, model_info.get('expected_hash')):
                print("✅ 文件验证通过")
                return True
            else:
                print("⚠️ 文件验证失败，将重新下载")
                
        # 检查依赖
        if not self._check_requirements(model_info.get('requirements', [])):
            print(f"❌ 缺少依赖，无法下载 {model_id}")
            return False
            
        print(f"📥 开始下载模型: {model_id}")
        print(f"   URL: {model_info['url']}")
        print(f"   目标文件: {local_file}")
        print(f"   预期大小: {model_info['size_mb']} MB")
        
        try:
            # 下载文件
            success = self._download_file(model_info['url'], local_file)
            
            if success:
                print(f"✅ 下载完成: {model_id}")
                
                # 验证文件
                if self._verify_file(local_file, model_info.get('expected_hash')):
                    print("✅ 文件验证通过")
                    return True
                else:
                    print("⚠️ 文件验证失败")
                    return False
            else:
                print(f"❌ 下载失败: {model_id}")
                return False
                
        except Exception as e:
            print(f"❌ 下载异常: {e}")
            return False
            
    def _check_requirements(self, requirements: List[str]) -> bool:
        """检查依赖要求"""
        for req in requirements:
            try:
                if req == 'torch':
                    import torch
                    print(f"✅ {req}: {torch.__version__}")
                elif req == 'tensorflow':
                    import tensorflow
                    print(f"✅ {req}: {tensorflow.__version__}")
                else:
                    __import__(req)
                    print(f"✅ {req}: 已安装")
            except ImportError:
                print(f"❌ {req}: 未安装")
                return False
                
        return True
        
    def _download_file(self, url: str, local_file: Path) -> bool:
        """下载文件"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_file, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # 显示进度
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r   进度: {percent:.1f}% ({downloaded//1024//1024}/{total_size//1024//1024} MB)", end='')
                            
            print()  # 换行
            return True
            
        except requests.RequestException as e:
            print(f"\n❌ 网络错误: {e}")
            return False
        except Exception as e:
            print(f"\n❌ 下载错误: {e}")
            return False
            
    def _verify_file(self, file_path: Path, expected_hash: Optional[str]) -> bool:
        """验证文件完整性"""
        if not expected_hash or not file_path.exists():
            return True
            
        try:
            # 简化的哈希验证
            if expected_hash.startswith('sha256:'):
                expected = expected_hash[7:]  # 去掉 'sha256:' 前缀
                
                # 计算实际哈希
                sha256_hash = hashlib.sha256()
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)
                        
                actual = sha256_hash.hexdigest()
                
                if actual == expected:
                    print(f"✅ 哈希验证通过: {actual[:8]}...")
                    return True
                else:
                    print(f"❌ 哈希验证失败: 期望 {expected[:8]}..., 实际 {actual[:8]}...")
                    return False
            else:
                print("⚠️ 不支持的哈希格式，跳过验证")
                return True
                
        except Exception as e:
            print(f"⚠️ 哈希验证失败: {e}")
            return False
            
    def download_all(self, force: bool = False) -> Dict[str, bool]:
        """下载所有模型"""
        print("📦 下载所有预训练模型")
        print("=" * 50)
        
        results = {}
        for model_id in self.available_models.keys():
            print(f"\n处理模型: {model_id}")
            results[model_id] = self.download_model(model_id, force)
            
        # 总结
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        print(f"\n📊 下载总结:")
        print(f"   成功: {successful}/{total}")
        print(f"   成功率: {successful/total:.1%}")
        
        if successful == total:
            print("🎉 所有模型下载成功!")
        elif successful > 0:
            print("👍 部分模型下载成功")
        else:
            print("😞 所有模型下载失败")
            
        return results
        
    def list_downloaded_models(self) -> Dict[str, Dict]:
        """列出已下载的模型"""
        print("📂 已下载的模型:")
        print("=" * 50)
        
        downloaded = {}
        
        for model_id, info in self.available_models.items():
            local_file = self.models_dir / info['filename']
            
            if local_file.exists():
                file_size = local_file.stat().st_size / 1024 / 1024  # MB
                download_time = local_file.stat().st_mtime
                
                downloaded[model_id] = {
                    'file_path': str(local_file),
                    'file_size_mb': round(file_size, 2),
                    'download_time': download_time,
                    'info': info
                }
                
                print(f"✅ {model_id}")
                print(f"   文件: {local_file}")
                print(f"   大小: {file_size:.1f} MB")
                print(f"   描述: {info['description']}")
                print()
            else:
                print(f"❌ {model_id} (未下载)")
                print()
                
        return downloaded
        
    def delete_model(self, model_id: str) -> bool:
        """删除模型"""
        if model_id not in self.available_models:
            print(f"❌ 未知模型: {model_id}")
            return False
            
        model_info = self.available_models[model_id]
        local_file = self.models_dir / model_info['filename']
        
        if local_file.exists():
            try:
                local_file.unlink()
                print(f"✅ 已删除模型: {model_id}")
                return True
            except Exception as e:
                print(f"❌ 删除失败: {e}")
                return False
        else:
            print(f"⚠️ 模型不存在: {model_id}")
            return True
            
    def cleanup_downloads(self) -> int:
        """清理损坏的下载文件"""
        print("🧹 清理下载文件")
        print("=" * 30)
        
        cleaned = 0
        
        for model_id, info in self.available_models.items():
            local_file = self.models_dir / info['filename']
            
            if local_file.exists():
                # 检查文件是否损坏（大小为0或异常小）
                file_size = local_file.stat().st_size
                expected_size_mb = info['size_mb']
                expected_size = expected_size_mb * 1024 * 1024
                
                if file_size == 0 or file_size < expected_size * 0.1:  # 小于预期的10%
                    try:
                        local_file.unlink()
                        print(f"   清理损坏文件: {model_id}")
                        cleaned += 1
                    except Exception as e:
                        print(f"   清理失败 {model_id}: {e}")
                        
        print(f"✅ 清理完成，删除了 {cleaned} 个文件")
        return cleaned


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='预训练模型下载工具')
    parser.add_argument('--list', action='store_true', help='列出所有可用模型')
    parser.add_argument('--download', help='下载指定模型')
    parser.add_argument('--download-all', action='store_true', help='下载所有模型')
    parser.add_argument('--force', action='store_true', help='强制重新下载')
    parser.add_argument('--installed', action='store_true', help='列出已安装的模型')
    parser.add_argument('--delete', help='删除指定模型')
    parser.add_argument('--cleanup', action='store_true', help='清理损坏的文件')
    parser.add_argument('--models-dir', default='data/models/pretrained', help='模型存储目录')
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.models_dir)
    
    if args.list:
        downloader.list_available_models()
        
    elif args.download:
        downloader.download_model(args.download, args.force)
        
    elif args.download_all:
        downloader.download_all(args.force)
        
    elif args.installed:
        downloader.list_downloaded_models()
        
    elif args.delete:
        downloader.delete_model(args.delete)
        
    elif args.cleanup:
        downloader.cleanup_downloads()
        
    else:
        print("🛠️ 预训练模型下载工具")
        print("使用 --help 查看可用选项")
        print("\n常用命令:")
        print("  python download_models.py --list                    # 列出所有模型")
        print("  python download_models.py --download brain_inspired_v1.0  # 下载指定模型")
        print("  python download_models.py --download-all           # 下载所有模型")
        print("  python download_models.py --installed              # 查看已安装模型")


if __name__ == "__main__":
    main()
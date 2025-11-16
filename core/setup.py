#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脑启发AI项目安装配置
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_file(filename):
    """读取文件内容"""
    with open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8') as f:
        return f.read()

# 读取requirements文件
def read_requirements(filename):
    """读取requirements文件"""
    requirements = []
    with open(os.path.join(os.path.dirname(__file__), filename), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if line and not line.startswith('#'):
                requirements.append(line)
    return requirements

setup(
    name="brain-inspired-ai",
    version="1.0.0",
    author="Brain-Inspired AI Team",
    author_email="team@brain-ai.org",
    description="基于生物大脑启发的深度学习框架",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/brain-ai/brain-inspired-ai",
    project_urls={
        "Bug Reports": "https://github.com/brain-ai/brain-inspired-ai/issues",
        "Source": "https://github.com/brain-ai/brain-inspired-ai",
        "Documentation": "https://brain-ai.readthedocs.io/",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "demos"]),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-xdist>=3.3.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
        "visualization": [
            "plotly>=5.15.0",
            "bokeh>=3.2.0",
            "altair>=5.0.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
            "tensorflow[gpu]>=2.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "brain-ai=brain_ai.cli:main",
            "brain-train=brain_ai.scripts.train:main",
            "brain-eval=brain_ai.scripts.evaluate:main",
        ],
    },
    package_data={
        "brain_ai": [
            "config/*.yaml",
            "config/*.yml",
            "data/*.json",
            "data/*.pkl",
            "models/*.pt",
            "models/*.pth",
        ],
    },
    zip_safe=False,
    keywords="brain-inspired ai neural networks deep learning hippocampus neocortex memory",
)
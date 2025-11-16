# Brain-Inspired AI Dockerfile
# ============================

# 使用官方Python基础镜像
FROM python:3.10-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 升级pip并安装Python依赖
RUN pip install --upgrade pip setuptools wheel

# 安装Python依赖（包括GPU支持）
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他Python依赖
RUN pip install -r requirements.txt

# 复制项目源代码
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/

# 创建必要的目录
RUN mkdir -p logs models data cache output

# 设置文件权限
RUN chmod +x scripts/*.sh

# 创建非root用户
RUN useradd --create-home --shell /bin/bash brain_ai
RUN chown -R brain_ai:brain_ai /app
USER brain_ai

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 默认命令
CMD ["python", "-m", "brain_ai.scripts.serve", "--config", "config/production.yaml"]
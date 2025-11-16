# Brain-Inspired AI 部署指南

本指南详细说明如何在各种环境中部署和配置Brain-Inspired AI Framework。

## 📋 目录

- [环境准备](#环境准备)
- [本地开发部署](#本地开发部署)
- [生产环境部署](#生产环境部署)
- [Docker部署](#docker部署)
- [云平台部署](#云平台部署)
- [监控和维护](#监控和维护)
- [故障排除](#故障排除)

## 环境准备

### 系统要求

#### 最低要求
- **操作系统**: Linux (Ubuntu 20.04+), macOS 10.15+, Windows 10+
- **Python**: 3.8+
- **内存**: 8GB RAM
- **存储**: 10GB 可用空间
- **CPU**: 4核处理器

#### 推荐配置
- **操作系统**: Ubuntu 22.04 LTS
- **Python**: 3.10+
- **内存**: 16GB+ RAM
- **存储**: 50GB+ SSD
- **CPU**: 8核+ 处理器
- **GPU**: NVIDIA RTX 3080+ (可选，用于加速)

#### 生产环境配置
- **内存**: 32GB+ RAM
- **存储**: 100GB+ NVMe SSD
- **CPU**: 16核+ 处理器
- **GPU**: NVIDIA A100/V100 (推荐)
- **网络**: 1Gbps+

### 依赖软件

#### 必需依赖
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip python3-dev
sudo apt install -y build-essential cmake git curl wget
sudo apt install -y libpq-dev libssl-dev libffi-dev

# macOS (使用Homebrew)
brew install python@3.10 cmake git curl wget postgresql

# Windows (使用Chocolatey)
choco install python310 cmake git curl wget postgresql
```

#### GPU支持 (可选)
```bash
# 安装CUDA Toolkit 11.8+
# 下载地址: https://developer.nvidia.com/cuda-downloads

# 验证安装
nvidia-smi
nvcc --version
```

## 本地开发部署

### 方法一：使用Python虚拟环境

```bash
# 1. 克隆项目
git clone https://github.com/brain-ai/brain-inspired-ai.git
cd brain-inspired-ai

# 2. 创建虚拟环境
python3 -m venv venv_dev
source venv_dev/bin/activate  # Linux/Mac
# venv_dev\Scripts\activate  # Windows

# 3. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 4. 验证安装
python -c "from brain_ai import HippocampusSimulator; print('安装成功!')"

# 5. 运行开发服务器
make serve
# 或者
brain-ai serve --config config/development.yaml
```

### 方法二：使用Makefile

```bash
# 克隆项目
git clone https://github.com/brain-ai/brain-inspired-ai.git
cd brain-inspired-ai

# 自动设置开发环境
make dev-setup

# 激活虚拟环境
source venv_dev/bin/activate

# 运行测试
make dev-test

# 启动开发服务器
make serve
```

### 开发环境配置

#### 配置文件 (config/development.yaml)
```yaml
# 开发环境特定配置
system:
  device: "auto"
  num_workers: 2
  batch_size: 16

logging:
  level: "DEBUG"
  file: "logs/brain_ai_dev.log"

server:
  http:
    host: "127.0.0.1"
    port: 8000
    workers: 1

security:
  authentication:
    enabled: false
```

#### 环境变量设置
```bash
# 创建 .env 文件
cat > .env << EOF
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
DATABASE_URL=sqlite:///brain_ai_dev.db
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=dev-secret-key-change-in-production
EOF

# 加载环境变量
source .env
```

### IDE配置

#### VS Code
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv_dev/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.associations": {
        "*.yaml": "yaml",
        "*.yml": "yaml"
    }
}
```

#### PyCharm
1. 设置Python解释器为 `venv_dev/bin/python`
2. 启用代码检查和格式化
3. 配置测试运行器为pytest

## 生产环境部署

### 部署架构

```
                    [Load Balancer]
                           |
                   [Nginx Reverse Proxy]
                           |
                [Brain-AI API Servers]
                           |
    [Database] [Redis Cache] [Message Queue] [Storage]
```

### 前置条件

#### 系统优化
```bash
# 增加文件描述符限制
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# 优化网络参数
echo "net.core.somaxconn = 65536" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 65536" >> /etc/sysctl.conf
sysctl -p

# 设置时区
timedatectl set-timezone Asia/Shanghai
```

#### 用户和权限
```bash
# 创建专用用户
sudo useradd -r -s /bin/bash brain-ai
sudo usermod -aG sudo brain-ai

# 创建应用目录
sudo mkdir -p /opt/brain-ai
sudo chown brain-ai:brain-ai /opt/brain-ai
```

### 部署步骤

#### 1. 安装应用
```bash
# 切换到brain-ai用户
sudo su - brain-ai

# 克隆项目
cd /opt/brain-ai
git clone https://github.com/brain-ai/brain-inspired-ai.git .
git checkout v1.0.0  # 使用稳定版本

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 设置权限
chmod +x scripts/*.sh
```

#### 2. 配置环境
```bash
# 创建生产环境配置
cp config/production.yaml config/production-local.yaml

# 编辑配置文件
vim config/production-local.yaml
```

```yaml
# 生产环境配置示例
system:
  device: "cuda"
  num_workers: 8
  batch_size: 64

database:
  primary:
    type: "postgresql"
    host: "localhost"
    port: 5432
    database: "brain_ai_prod"
    username: "brain_ai"
    password: "secure_password_123"
    pool_size: 20
    max_overflow: 30

redis:
  host: "localhost"
  port: 6379
  db: 0
  password: "redis_password_123"

logging:
  level: "INFO"
  file: "/var/log/brain-ai/brain_ai.log"
  max_size: "100MB"
  backup_count: 10

server:
  http:
    host: "0.0.0.0"
    port: 8080
    workers: 4
    worker_class: "uvicorn.workers.UvicornWorker"
    max_requests: 10000
    timeout: 30

security:
  authentication:
    enabled: true
    secret_key: "your-production-secret-key"
    access_token_expire_minutes: 30
```

#### 3. 初始化数据库
```bash
# 创建PostgreSQL数据库
sudo -u postgres psql << EOF
CREATE DATABASE brain_ai_prod;
CREATE USER brain_ai WITH PASSWORD 'secure_password_123';
GRANT ALL PRIVILEGES ON DATABASE brain_ai_prod TO brain_ai;
ALTER USER brain_ai CREATEDB;
EOF

# 初始化数据库表
python -m brain_ai.scripts.init_db --config config/production-local.yaml
```

#### 4. 创建系统服务
```bash
# 创建systemd服务文件
sudo tee /etc/systemd/system/brain-ai.service > /dev/null << EOF
[Unit]
Description=Brain-Inspired AI Service
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=exec
User=brain-ai
Group=brain-ai
WorkingDirectory=/opt/brain-ai
Environment=PATH=/opt/brain-ai/venv/bin
EnvironmentFile=/opt/brain-ai/.env
ExecStart=/opt/brain-ai/venv/bin/python -m brain_ai.scripts.serve --config config/production-local.yaml
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=5

# 安全设置
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/brain-ai/logs /opt/brain-ai/data /opt/brain-ai/models

[Install]
WantedBy=multi-user.target
EOF

# 重新加载systemd并启用服务
sudo systemctl daemon-reload
sudo systemctl enable brain-ai
```

#### 5. 设置Nginx反向代理
```bash
# 安装Nginx
sudo apt install nginx

# 创建Nginx配置
sudo tee /etc/nginx/sites-available/brain-ai > /dev/null << EOF
server {
    listen 80;
    server_name your-domain.com;

    # 重定向到HTTPS
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL配置
    ssl_certificate /etc/ssl/certs/brain-ai.crt;
    ssl_certificate_key /etc/ssl/private/brain-ai.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # 安全头
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # 限制请求大小
    client_max_body_size 100M;

    # 代理到Brain-AI应用
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket支持
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # 超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # 静态文件缓存
    location /static/ {
        alias /opt/brain-ai/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # 健康检查
    location /health {
        proxy_pass http://127.0.0.1:8080/health;
        access_log off;
    }
}
EOF

# 启用站点
sudo ln -s /etc/nginx/sites-available/brain-ai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### 6. 启动服务
```bash
# 启动所有服务
sudo systemctl start brain-ai
sudo systemctl start postgresql
sudo systemctl start redis

# 检查服务状态
sudo systemctl status brain-ai
sudo systemctl status postgresql
sudo systemctl status redis

# 查看日志
sudo journalctl -u brain-ai -f
```

### 使用部署脚本

项目提供了自动化的部署脚本：

```bash
# 部署到开发环境
./scripts/deploy.sh deploy development

# 部署到生产环境
./scripts/deploy.sh deploy production

# 启动开发服务器
./scripts/deploy.sh start-dev

# 清理环境
./scripts/deploy.sh cleanup all
```

## Docker部署

### 单容器部署

```bash
# 构建镜像
docker build -t brain-ai:latest .

# 运行容器
docker run -d \
  --name brain-ai \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --gpus all \
  brain-ai:latest
```

### Docker Compose部署

#### 基础服务
```bash
# 启动核心服务
docker-compose up brain-ai redis postgres -d

# 查看日志
docker-compose logs -f brain-ai
```

#### 完整服务栈
```bash
# 启动所有服务（包括监控）
docker-compose --profile monitoring up -d

# 仅启动特定服务
docker-compose up brain-ai influxdb grafana -d
```

#### 生产环境Docker Compose
```bash
# 使用生产配置
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 扩展服务
docker-compose up --scale brain-ai=3 -d
```

### Kubernetes部署

#### 部署清单
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: brain-ai

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: brain-ai-config
  namespace: brain-ai
data:
  production.yaml: |
    # 生产配置内容

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: brain-ai
  namespace: brain-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: brain-ai
  template:
    metadata:
      labels:
        app: brain-ai
    spec:
      containers:
      - name: brain-ai
        image: brain-ai:latest
        ports:
        - containerPort: 8080
        env:
        - name: CONFIG_FILE
          value: "/app/config/production.yaml"
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: data
          mountPath: /app/data
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
      volumes:
      - name: config
        configMap:
          name: brain-ai-config
      - name: data
        persistentVolumeClaim:
          claimName: brain-ai-data

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: brain-ai-service
  namespace: brain-ai
spec:
  selector:
    app: brain-ai
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: brain-ai-ingress
  namespace: brain-ai
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.brain-ai.org
    secretName: brain-ai-tls
  rules:
  - host: api.brain-ai.org
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: brain-ai-service
            port:
              number: 80
```

#### 部署到Kubernetes
```bash
# 创建命名空间
kubectl apply -f k8s/namespace.yaml

# 部署应用
kubectl apply -f k8s/

# 检查部署状态
kubectl get pods -n brain-ai
kubectl get services -n brain-ai
kubectl get ingress -n brain-ai

# 查看日志
kubectl logs -f deployment/brain-ai -n brain-ai
```

## 云平台部署

### AWS部署

#### 使用ECS
```bash
# 构建并推送Docker镜像到ECR
aws ecr create-repository --repository-name brain-ai
$(aws ecr get-login --no-include-email --region us-west-2)
docker build -t brain-ai .
docker tag brain-ai:latest 123456789.dkr.ecr.us-west-2.amazonaws.com/brain-ai:latest
docker push 123456789.dkr.ecr.us-west-2.amazonaws.com/brain-ai:latest

# 创建ECS任务定义
aws ecs register-task-definition --cli-input-json file://aws/task-definition.json

# 创建ECS服务
aws ecs create-service \
  --cluster brain-ai-cluster \
  --service-name brain-ai-service \
  --task-definition brain-ai:1 \
  --desired-count 3
```

#### 使用Lambda (适用于无服务器)
```python
# lambda_function.py
import json
import boto3
from brain_ai import HippocampusSimulator

def lambda_handler(event, context):
    # 初始化海马体模拟器
    hippocampus = HippocampusSimulator()
    
    # 处理请求
    input_data = event.get('input_data')
    result = hippocampus.process(input_data)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'result': result
        })
    }
```

### GCP部署

#### 使用Cloud Run
```bash
# 构建并部署到Cloud Run
gcloud builds submit --tag gcr.io/PROJECT-ID/brain-ai
gcloud run deploy brain-ai \
  --image gcr.io/PROJECT-ID/brain-ai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

### Azure部署

#### 使用Container Instances
```bash
# 创建资源组
az group create --name brain-ai-rg --location eastus

# 创建容器实例
az container create \
  --resource-group brain-ai-rg \
  --name brain-ai \
  --image brain-ai:latest \
  --dns-name-label brain-ai \
  --ports 8080 \
  --memory 2 \
  --cpu 2
```

## 监控和维护

### 健康检查

#### 端点检查
```bash
# 基础健康检查
curl -f http://localhost:8080/health

# 详细状态检查
curl -f http://localhost:8080/status

# API可用性测试
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"input": "测试数据"}'
```

#### 系统监控脚本
```bash
#!/bin/bash
# monitor.sh

# 检查服务状态
check_service() {
    if ! systemctl is-active --quiet brain-ai; then
        echo "错误: Brain-AI服务未运行"
        return 1
    fi
    
    # 检查端口
    if ! netstat -ln | grep -q ":8080 "; then
        echo "错误: 端口8080未监听"
        return 1
    fi
    
    # 检查健康端点
    if ! curl -f http://localhost:8080/health > /dev/null 2>&1; then
        echo "错误: 健康检查失败"
        return 1
    fi
    
    echo "服务状态正常"
    return 0
}

# 检查资源使用
check_resources() {
    # CPU使用率
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    echo "CPU使用率: ${cpu_usage}%"
    
    # 内存使用
    memory_info=$(free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2}')
    echo "内存使用: ${memory_info}"
    
    # 磁盘使用
    disk_usage=$(df -h / | awk 'NR==2{print $5}')
    echo "磁盘使用: ${disk_usage}"
}

# 发送告警
send_alert() {
    local message="$1"
    # 这里可以集成邮件、Slack等告警方式
    echo "$(date): $message" >> /var/log/brain-ai/alerts.log
}

# 主检查逻辑
main() {
    echo "开始系统监控检查..."
    
    if ! check_service; then
        send_alert "服务检查失败"
        exit 1
    fi
    
    check_resources
    echo "监控检查完成"
}

main "$@"
```

### 日志管理

#### 日志轮转配置
```bash
# /etc/logrotate.d/brain-ai
/var/log/brain-ai/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    sharedscripts
    postrotate
        systemctl reload brain-ai
    endscript
}
```

#### 集中化日志
```yaml
# 使用ELK Stack
version: '3.8'
services:
  elasticsearch:
    image: elasticsearch:7.15.0
    environment:
      - discovery.type=single-node
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
  
  logstash:
    image: logstash:7.15.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch
  
  kibana:
    image: kibana:7.15.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

### 备份策略

#### 数据库备份
```bash
#!/bin/bash
# backup_db.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/backups"
DB_NAME="brain_ai_prod"

# 创建备份目录
mkdir -p $BACKUP_DIR

# PostgreSQL备份
pg_dump -h localhost -U brain-ai -d $DB_NAME \
  --format=custom --verbose --file="$BACKUP_DIR/brain_ai_$DATE.backup"

# 压缩备份
gzip "$BACKUP_DIR/brain_ai_$DATE.backup"

# 清理旧备份（保留7天）
find $BACKUP_DIR -name "brain_ai_*.backup.gz" -mtime +7 -delete

echo "数据库备份完成: brain_ai_$DATE.backup.gz"
```

#### 文件系统备份
```bash
#!/bin/bash
# backup_fs.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/backups/files"
SOURCE_DIR="/opt/brain-ai"

# 创建备份
tar -czf "$BACKUP_DIR/brain_ai_files_$DATE.tar.gz" \
  -C "$SOURCE_DIR" \
  --exclude='venv' \
  --exclude='.git' \
  --exclude='logs' \
  --exclude='__pycache__' \
  data/ models/ config/

echo "文件备份完成: brain_ai_files_$DATE.tar.gz"
```

### 性能优化

#### 系统优化
```bash
# /etc/sysctl.conf 优化
# 网络优化
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 65536 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 5000

# 文件系统优化
fs.file-max = 2097152
vm.swappiness = 10

# 应用优化
kernel.sched_migration_cost_ns = 5000000
kernel.sched_autogroup_enabled = 0
```

#### 应用优化
```yaml
# 生产配置优化
training:
  batch_size: 64
  num_workers: 8
  pin_memory: true
  mixed_precision:
    enabled: true

server:
  http:
    workers: 4
    worker_class: "uvicorn.workers.UvicornWorker"
    max_requests: 10000
    max_requests_jitter: 1000
    timeout: 30
    keepalive: 5
```

## 故障排除

### 常见问题

#### 1. 服务启动失败
```bash
# 检查日志
sudo journalctl -u brain-ai -n 50

# 检查配置文件
python -m brain_ai.scripts.config validate --config config/production.yaml

# 检查依赖
pip check

# 权限检查
ls -la /opt/brain-ai/
```

#### 2. 内存不足
```bash
# 检查内存使用
free -h
ps aux --sort=-%mem | head

# 调整批次大小
export BRAIN_AI_BATCH_SIZE=16

# 启用内存优化
export BRAIN_AI_MEMORY_OPTIMIZATION=true
```

#### 3. GPU不可用
```bash
# 检查CUDA安装
nvcc --version
nvidia-smi

# 检查PyTorch GPU支持
python -c "import torch; print(torch.cuda.is_available())"

# 强制使用CPU
export BRAIN_AI_DEVICE=cpu
```

#### 4. 数据库连接失败
```bash
# 检查PostgreSQL状态
sudo systemctl status postgresql

# 测试连接
psql -h localhost -U brain-ai -d brain_ai_prod

# 检查配置
grep DATABASE_URL config/production.yaml
```

### 调试模式

#### 启用调试
```bash
# 设置调试环境变量
export BRAIN_AI_DEBUG=true
export BRAIN_AI_LOG_LEVEL=DEBUG

# 以调试模式启动
python -m brain_ai.scripts.serve --debug --config config/development.yaml
```

#### 性能分析
```bash
# CPU分析
python -m cProfile -o profile.stats -m brain_ai.scripts.serve

# 内存分析
python -m memory_profiler -m brain_ai.scripts.serve

# 使用Py-Spy
py-spy top --pid $(pgrep -f brain_ai)
```

### 恢复操作

#### 服务恢复
```bash
# 重启服务
sudo systemctl restart brain-ai

# 回滚到上一个版本
cd /opt/brain-ai
git checkout HEAD~1
sudo systemctl restart brain-ai

# 从备份恢复
# 恢复数据库
pg_restore -h localhost -U brain-ai -d brain_ai_prod /opt/backups/brain_ai_latest.backup

# 恢复文件
tar -xzf /opt/backups/brain_ai_files_latest.tar.gz -C /opt/brain-ai/
```

## 总结

本部署指南涵盖了Brain-Inspired AI Framework在各种环境下的部署方法。建议：

1. **开发环境**: 使用本地Python环境或Docker
2. **测试环境**: 使用Docker Compose
3. **生产环境**: 使用systemd + Nginx或Kubernetes
4. **云平台**: 使用托管服务如AWS ECS、GCP Cloud Run等

选择最适合您需求和基础设施的部署方案，并确保实施适当的监控、备份和安全措施。

如有问题，请参考项目的[GitHub Issues](https://github.com/brain-ai/brain-inspired-ai/issues)或联系开发团队。
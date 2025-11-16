# UI集成模块文档

## 概述

UI集成模块为脑启发AI框架提供了完整的用户界面解决方案，包括React组件库、Jupyter集成、Web界面部署等功能，让用户可以通过直观的方式使用和测试AI系统。

## 核心组件

### 1. React组件库 (@brain-ai/react)

基于React的UI组件库，提供脑启发AI系统的可视化界面。

#### 安装和设置

```bash
# 安装依赖
npm install @brain-ai/react @brain-ai/core
# 或
yarn add @brain-ai/react @brain-ai/core
```

#### 基础使用

```jsx
import React from 'react';
import { BrainSystem, MemoryVisualizer, PerformanceMonitor } from '@brain-ai/react';

function BrainAIApp() {
  return (
    <div className="brain-ai-app">
      <BrainSystem 
        config={{
          hippocampus: { memory_capacity: 10000 },
          neocortex: { layers: 8 }
        }}
        onInitialize={(system) => {
          console.log('Brain system initialized:', system);
        }}
      />
      
      <MemoryVisualizer 
        system={brainSystem}
        viewMode="3d"
        showConnections={true}
      />
      
      <PerformanceMonitor 
        realTime={true}
        metrics={['cpu_usage', 'memory_usage', 'accuracy']}
      />
    </div>
  );
}
```

### 2. 核心组件详解

#### 2.1 BrainSystem 组件

大脑系统的主控制组件。

```jsx
import { BrainSystem } from '@brain-ai/react';

<BrainSystem
  // 必需属性
  config={{
    hippocampus: { memory_capacity: 10000 },
    neocortex: { layers: 8, hierarchical_levels: 4 },
    attention: { num_heads: 8 }
  }}
  
  // 可选属性
  theme="dark" // "light" | "dark" | "auto"
  language="zh-CN" // "en" | "zh-CN"
  autoStart={true}
  
  // 事件回调
  onInitialize={(system) => {}}
  onProcess={(input, output) => {}}
  onError={(error) => {}}
  onStatusChange={(status) => {}}
/>
```

**属性说明**:
- `config`: 系统配置对象
- `theme`: UI主题风格
- `language`: 界面语言
- `autoStart`: 是否自动启动系统
- `onInitialize`: 初始化完成回调
- `onProcess`: 数据处理回调
- `onError`: 错误处理回调
- `onStatusChange`: 状态变化回调

#### 2.2 MemoryVisualizer 组件

记忆系统的可视化组件。

```jsx
import { MemoryVisualizer } from '@brain-ai/react';

<MemoryVisualizer
  system={brainSystem}
  viewMode="3d" // "2d" | "3d" | "network" | "heatmap"
  showConnections={true}
  showActivations={true}
  animate={true}
  memoryTypes={["episodic", "semantic", "procedural"]}
  timeRange={{ start: 0, end: 1000 }}
  
  // 交互功能
  onMemorySelect={(memory) => {}}
  onTimeChange={(time) => {}}
  onRegionFocus={(region) => {}}
/>
```

**视图模式**:
- **2D**: 平面视图，适合概览
- **3D**: 三维视图，立体展示
- **Network**: 网络图，显示连接关系
- **Heatmap**: 热力图，显示激活强度

#### 2.3 PerformanceMonitor 组件

实时性能监控组件。

```jsx
import { PerformanceMonitor } from '@brain-ai/react';

<PerformanceMonitor
  realTime={true}
  refreshInterval={1000} // ms
  metrics={[
    'cpu_usage',
    'memory_usage', 
    'gpu_usage',
    'inference_time',
    'accuracy',
    'loss'
  ]}
  charts={[
    { type: 'line', metric: 'accuracy', title: '准确率变化' },
    { type: 'gauge', metric: 'cpu_usage', title: 'CPU使用率' },
    { type: 'bar', metric: 'memory_usage', title: '内存使用' }
  ]}
  
  // 告警设置
  alerts={{
    cpu_usage: { threshold: 80, action: 'warning' },
    memory_usage: { threshold: 85, action: 'critical' }
  }}
  
  // 数据导出
  exportFormat="csv"
  onExport={(data) => {}}
/>
```

#### 2.4 TrainingInterface 组件

训练过程控制界面。

```jsx
import { TrainingInterface } from '@brain-ai/react';

<TrainingInterface
  model={neuralNetwork}
  dataLoader={trainLoader}
  config={{
    epochs: 100,
    batchSize: 32,
    learningRate: 0.001,
    optimizer: 'adam'
  }}
  
  // 训练控制
  onStart={() => {}}
  onPause={() => {}}
  onStop={() => {}}
  onEpochComplete={(epoch, metrics) => {}}
  
  // 超参数调优
  hyperparameterTuning={{
    enabled: true,
    method: 'bayesian', // "grid" | "random" | "bayesian"
    parameters: ['learning_rate', 'batch_size', 'epochs']
  }}
/>
```

### 3. Jupyter集成

在Jupyter环境中使用脑启发AI系统。

#### 3.1 安装Jupyter扩展

```bash
pip install brain-ai[jupyter]
jupyter nbextension install --py brain_ai.jupyter
jupyter nbextension enable --py brain_ai.jupyter
```

#### 3.2 基础使用

```python
from brain_ai.jupyter import BrainAIDisplay, MemoryViz, TrainingMonitor

# 创建显示对象
display = BrainAIDisplay()

# 可视化记忆系统
memory_viz = MemoryViz()
memory_viz.show_hippocampus_network()

# 训练监控
train_monitor = TrainingMonitor()
train_monitor.start_monitoring(model, data_loader)
```

#### 3.3 交互式组件

```python
from brain_ai.jupyter import InteractiveBrainSystem
from ipywidgets import VBox, HBox

# 创建交互式大脑系统
brain_system = InteractiveBrainSystem()

# 创建控制面板
controls = VBox([
    brain_system.config_panel,
    brain_system.start_button,
    brain_system.status_display
])

# 创建可视化区域
visualization = VBox([
    brain_system.memory_viz,
    brain_system.performance_chart
])

# 组合界面
dashboard = HBox([controls, visualization])
display(dashboard)
```

#### 3.4 预制仪表板

```python
from brain_ai.jupyter import create_brain_dashboard

# 创建完整的脑AI仪表板
dashboard = create_brain_dashboard(
    brain_system=my_brain,
    show_memory=True,
    show_performance=True,
    show_training=True,
    theme="dark"
)

# 显示仪表板
dashboard
```

### 4. Web界面部署

完整的Web应用程序部署方案。

#### 4.1 项目结构

```
brain-ai-ui/
├── src/
│   ├── components/          # React组件
│   ├── pages/              # 页面组件
│   ├── hooks/              # 自定义Hook
│   ├── utils/              # 工具函数
│   └── api/                # API调用
├── public/
│   └── index.html
├── package.json
├── vite.config.js
└── README.md
```

#### 4.2 快速启动

```bash
# 进入UI目录
cd brain-ai-ui

# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 构建生产版本
npm run build

# 预览生产版本
npm run preview
```

#### 4.3 主要页面

##### 4.3.1 系统概览页

```jsx
// src/pages/Dashboard.jsx
import React, { useState, useEffect } from 'react';
import { BrainSystem, MemoryVisualizer, PerformanceMonitor } from '@brain-ai/react';

export default function Dashboard() {
  const [brainSystem, setBrainSystem] = useState(null);
  const [status, setStatus] = useState('initializing');

  useEffect(() => {
    // 初始化大脑系统
    const initBrain = async () => {
      const system = new BrainSystem({
        config: {
          hippocampus: { memory_capacity: 10000 },
          neocortex: { layers: 8 }
        }
      });
      
      await system.initialize();
      setBrainSystem(system);
      setStatus('ready');
    };

    initBrain();
  }, []);

  return (
    <div className="dashboard">
      <header>
        <h1>脑启发AI系统</h1>
        <StatusIndicator status={status} />
      </header>
      
      <main>
        <section className="brain-visualization">
          <h2>大脑可视化</h2>
          {brainSystem && (
            <MemoryVisualizer 
              system={brainSystem}
              viewMode="3d"
              showConnections={true}
            />
          )}
        </section>
        
        <section className="performance">
          <h2>性能监控</h2>
          <PerformanceMonitor 
            realTime={true}
            metrics={['cpu_usage', 'memory_usage', 'accuracy']}
          />
        </section>
      </main>
    </div>
  );
}
```

##### 4.3.2 训练页面

```jsx
// src/pages/Training.jsx
import React, { useState } from 'react';
import { TrainingInterface, TrainingChart } from '@brain-ai/react';

export default function Training() {
  const [trainingConfig, setTrainingConfig] = useState({
    epochs: 100,
    batchSize: 32,
    learningRate: 0.001
  });

  const handleTrainingStart = (config) => {
    console.log('开始训练:', config);
  };

  return (
    <div className="training-page">
      <TrainingInterface
        config={trainingConfig}
        onConfigChange={setTrainingConfig}
        onStart={handleTrainingStart}
      />
      
      <TrainingChart 
        height={400}
        showValidation={true}
        metrics={['loss', 'accuracy', 'learning_rate']}
      />
    </div>
  );
}
```

### 5. 自定义组件开发

#### 5.1 创建自定义可视化组件

```jsx
// src/components/CustomMemoryViz.jsx
import React, { useMemo } from 'react';
import { D3Visualization } from '@brain-ai/react';

export function CustomMemoryViz({ memoryData, width = 800, height = 600 }) {
  const visualizationData = useMemo(() => {
    return memoryData.map(memory => ({
      id: memory.id,
      x: memory.position.x,
      y: memory.position.y,
      z: memory.position.z,
      strength: memory.strength,
      type: memory.type
    }));
  }, [memoryData]);

  return (
    <D3Visualization
      data={visualizationData}
      width={width}
      height={height}
      render={(container, data) => {
        // 使用D3.js创建自定义可视化
        const svg = d3.select(container)
          .append('svg')
          .attr('width', width)
          .attr('height', height);

        // 绘制记忆节点
        svg.selectAll('circle')
          .data(data)
          .enter()
          .append('circle')
          .attr('cx', d => d.x)
          .attr('cy', d => d.y)
          .attr('r', d => d.strength * 10)
          .attr('fill', d => getColorByType(d.type))
          .attr('opacity', 0.7);
      }}
    />
  );
}
```

#### 5.2 创建自定义Hook

```jsx
// src/hooks/useBrainSystem.js
import { useState, useEffect, useCallback } from 'react';

export function useBrainSystem(config) {
  const [brainSystem, setBrainSystem] = useState(null);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState(null);
  const [metrics, setMetrics] = useState({});

  useEffect(() => {
    if (!config) return;

    const initSystem = async () => {
      try {
        setStatus('initializing');
        
        // 创建大脑系统实例
        const system = new window.BrainSystem(config);
        await system.initialize();
        
        setBrainSystem(system);
        setStatus('ready');
        
        // 开始监控
        const intervalId = setInterval(() => {
          const currentMetrics = system.getMetrics();
          setMetrics(currentMetrics);
        }, 1000);
        
        return () => clearInterval(intervalId);
        
      } catch (err) {
        setError(err);
        setStatus('error');
      }
    };

    initSystem();
  }, [config]);

  const processData = useCallback(async (inputData) => {
    if (!brainSystem || status !== 'ready') {
      throw new Error('Brain system not ready');
    }
    
    return await brainSystem.process(inputData);
  }, [brainSystem, status]);

  const cleanup = useCallback(() => {
    if (brainSystem) {
      brainSystem.cleanup();
      setBrainSystem(null);
      setStatus('idle');
    }
  }, [brainSystem]);

  return {
    brainSystem,
    status,
    error,
    metrics,
    processData,
    cleanup
  };
}
```

### 6. 主题和样式

#### 6.1 自定义主题

```jsx
// src/theme/brainTheme.js
export const brainTheme = {
  colors: {
    primary: '#2563eb',
    secondary: '#7c3aed',
    accent: '#06b6d4',
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    background: '#0f172a',
    surface: '#1e293b',
    text: '#f8fafc',
    textSecondary: '#94a3b8'
  },
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem', 
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem'
  },
  borderRadius: {
    sm: '0.25rem',
    md: '0.5rem',
    lg: '0.75rem'
  },
  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1)'
  }
};

// 使用主题
import { ThemeProvider } from 'styled-components';
import { brainTheme } from './theme/brainTheme';

function App() {
  return (
    <ThemeProvider theme={brainTheme}>
      <BrainSystem />
    </ThemeProvider>
  );
}
```

#### 6.2 CSS-in-JS样式

```jsx
import styled from 'styled-components';

export const BrainContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: ${props => props.theme.colors.background};
  color: ${props => props.theme.colors.text};
`;

export const BrainHeader = styled.header`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: ${props => props.theme.spacing.lg};
  background: ${props => props.theme.colors.surface};
  border-bottom: 1px solid ${props => props.theme.colors.textSecondary}20;
`;

export const BrainVisualization = styled.div`
  flex: 1;
  position: relative;
  overflow: hidden;
  
  canvas {
    display: block;
    width: 100%;
    height: 100%;
  }
`;

export const StatusIndicator = styled.div`
  display: inline-flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  border-radius: ${props => props.theme.borderRadius.md};
  background: ${props => {
    switch (props.status) {
      case 'ready': return props.theme.colors.success;
      case 'error': return props.theme.colors.error;
      case 'initializing': return props.theme.colors.warning;
      default: return props.theme.colors.textSecondary;
    }
  }};
  color: white;
  font-weight: 500;
`;
```

### 7. 部署指南

#### 7.1 Docker部署

```dockerfile
# Dockerfile
FROM node:18-alpine

WORKDIR /app

# 安装依赖
COPY package*.json ./
RUN npm ci --only=production

# 复制源代码
COPY . .

# 构建应用
RUN npm run build

# 暴露端口
EXPOSE 3000

# 启动应用
CMD ["npm", "start"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  brain-ai-ui:
    build: .
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8080
      - REACT_APP_BRAIN_ENGINE_URL=http://localhost:8081
    volumes:
      - ./public:/app/public
    restart: unless-stopped

  brain-ai-backend:
    image: brain-ai/backend:latest
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/brain_ai
    depends_on:
      - db

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=brain_ai
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

#### 7.2 生产环境配置

```javascript
// vite.config.js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    sourcemap: false,
    minify: 'terser',
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          brain: ['@brain-ai/react', '@brain-ai/core'],
          charts: ['d3', 'recharts']
        }
      }
    }
  },
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://localhost:8080',
      '/ws': {
        target: 'ws://localhost:8080',
        ws: true
      }
    }
  }
});
```

#### 7.3 性能优化

```jsx
// 组件懒加载
const MemoryVisualizer = React.lazy(() => 
  import('@brain-ai/react').then(module => ({
    default: module.MemoryVisualizer
  }))
);

// 虚拟化大列表
import { FixedSizeList as List } from 'react-window';

function VirtualizedMemoryList({ memories }) {
  const Row = ({ index, style }) => (
    <div style={style}>
      <MemoryItem memory={memories[index]} />
    </div>
  );

  return (
    <List
      height={600}
      itemCount={memories.length}
      itemSize={50}
      width="100%"
    >
      {Row}
    </List>
  );
}

// WebWorker处理计算密集任务
const worker = new Worker('/brain-worker.js');

function useBrainWorker() {
  const processMemory = useCallback(async (memoryData) => {
    return new Promise((resolve) => {
      worker.postMessage({ type: 'PROCESS_MEMORY', data: memoryData });
      worker.onmessage = (e) => resolve(e.data);
    });
  }, []);

  return { processMemory };
}
```

## API 参考

详细的React组件API文档请参考 `docs/api/react_components_api.md`

Jupyter扩展API文档请参考 `docs/api/jupyter_integration_api.md`

---

**作者**: Brain-Inspired AI Team  
**版本**: 1.0.0  
**最后更新**: 2025-11-16

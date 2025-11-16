"""
脑启发AI系统Jupyter集成模块
=========================

提供在Jupyter notebook中使用脑启发AI系统UI的功能。

主要功能：
- 在notebook中嵌入Web界面
- 实时显示系统状态
- 交互式训练控制
- 性能监控仪表板
- 系统架构可视化

作者: Brain-Inspired AI Team
创建时间: 2025-11-16
"""

import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from IPython.display import HTML, display, Javascript
import threading
import random

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import ipywidgets as widgets
    from IPython.display import clear_output
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False


@dataclass
class BrainRegionStatus:
    """大脑区域状态"""
    id: str
    name: str
    function: str
    activity: float
    connections: int
    neurons: int
    status: str


@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float
    epoch_time: float


class JupyterUIIntegration:
    """Jupyter界面集成类"""
    
    def __init__(self):
        self._callbacks: Dict[str, List[Callable]] = {}
        self._status_data = {}
        self._training_data = []
        self._running = False
        
    def register_callback(self, event_type: str, callback: Callable):
        """注册事件回调"""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)
    
    def emit_event(self, event_type: str, data: Any = None):
        """触发事件"""
        if event_type in self._callbacks:
            for callback in self._callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Callback error: {e}")
    
    def embed_ui_dashboard(self) -> HTML:
        """嵌入主仪表板界面"""
        ui_html = """
        <div id="brain-ai-dashboard" style="width: 100%; height: 600px; border: 1px solid #ccc; border-radius: 8px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h2 style="color: white; text-align: center; margin-bottom: 20px;">🧠 脑启发AI系统仪表板</h2>
            
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 20px;">
                <div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 8px; text-align: center;">
                    <h3 style="margin: 0; color: #333;">CPU使用率</h3>
                    <div id="cpu-usage" style="font-size: 2em; font-weight: bold; color: #4CAF50;">45.2%</div>
                </div>
                <div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 8px; text-align: center;">
                    <h3 style="margin: 0; color: #333;">内存使用</h3>
                    <div id="memory-usage" style="font-size: 2em; font-weight: bold; color: #2196F3;">67.8%</div>
                </div>
                <div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 8px; text-align: center;">
                    <h3 style="margin: 0; color: #333;">意识水平</h3>
                    <div id="consciousness-level" style="font-size: 2em; font-weight: bold; color: #9C27B0;">78%</div>
                </div>
            </div>
            
            <div style="background: rgba(255,255,255,0.9); padding: 20px; border-radius: 8px;">
                <h3 style="color: #333; margin-bottom: 15px;">大脑区域状态</h3>
                <div id="brain-regions" style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <!-- 区域状态将通过JavaScript动态更新 -->
                </div>
            </div>
        </div>
        
        <script>
        (function() {
            function updateDashboard() {
                // 模拟实时数据更新
                const cpuUsage = document.getElementById('cpu-usage');
                const memoryUsage = document.getElementById('memory-usage');
                const consciousnessLevel = document.getElementById('consciousness-level');
                
                if (cpuUsage) {
                    const cpu = (40 + Math.random() * 20).toFixed(1);
                    cpuUsage.textContent = cpu + '%';
                }
                
                if (memoryUsage) {
                    const mem = (60 + Math.random() * 20).toFixed(1);
                    memoryUsage.textContent = mem + '%';
                }
                
                if (consciousnessLevel) {
                    const consciousness = (70 + Math.random() * 20).toFixed(0);
                    consciousnessLevel.textContent = consciousness + '%';
                }
                
                // 更新大脑区域状态
                updateBrainRegions();
            }
            
            function updateBrainRegions() {
                const regionsContainer = document.getElementById('brain-regions');
                if (!regionsContainer) return;
                
                const regions = [
                    { name: '前额叶', activity: 85, status: 'active' },
                    { name: '皮层', activity: 92, status: 'active' },
                    { name: '海马体', activity: 78, status: 'processing' },
                    { name: '内嗅皮层', activity: 65, status: 'active' }
                ];
                
                regionsContainer.innerHTML = regions.map(region => `
                    <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
                        <h4 style="margin: 0 0 10px 0; color: #333;">${region.name}</h4>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #666;">活跃度: ${region.activity}%</span>
                            <span style="padding: 2px 8px; border-radius: 10px; background: ${region.status === 'active' ? '#4CAF50' : '#FF9800'}; color: white; font-size: 12px;">
                                ${region.status === 'active' ? '活跃' : '处理中'}
                            </span>
                        </div>
                        <div style="width: 100%; background: #eee; border-radius: 10px; height: 8px; margin-top: 8px;">
                            <div style="width: ${region.activity}%; height: 100%; background: linear-gradient(90deg, #4CAF50, #8BC34A); border-radius: 10px; transition: width 0.5s ease;"></div>
                        </div>
                    </div>
                `).join('');
            }
            
            // 每2秒更新一次数据
            setInterval(updateDashboard, 2000);
            
            // 立即更新一次
            updateDashboard();
        })();
        </script>
        """
        return HTML(ui_html)
    
    def embed_training_interface(self) -> HTML:
        """嵌入训练界面"""
        ui_html = """
        <div id="training-interface" style="width: 100%; border: 1px solid #ccc; border-radius: 8px; padding: 20px; background: #f9f9f9;">
            <h2 style="color: #333; text-align: center; margin-bottom: 20px;">🎯 模型训练控制台</h2>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <!-- 控制面板 -->
                <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h3 style="color: #333; margin-bottom: 15px;">训练控制</h3>
                    
                    <div style="margin-bottom: 15px;">
                        <label style="display: block; margin-bottom: 5px; color: #666;">训练轮数:</label>
                        <input type="number" id="epochs" value="100" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                    </div>
                    
                    <div style="margin-bottom: 15px;">
                        <label style="display: block; margin-bottom: 5px; color: #666;">学习率:</label>
                        <input type="number" id="learning-rate" value="0.001" step="0.0001" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                    </div>
                    
                    <div style="margin-bottom: 15px;">
                        <label style="display: block; margin-bottom: 5px; color: #666;">批次大小:</label>
                        <select id="batch-size" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                            <option value="16">16</option>
                            <option value="32" selected>32</option>
                            <option value="64">64</option>
                            <option value="128">128</option>
                        </select>
                    </div>
                    
                    <div style="display: flex; gap: 10px; margin-top: 20px;">
                        <button onclick="startTraining()" style="flex: 1; padding: 10px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer;">开始训练</button>
                        <button onclick="pauseTraining()" style="flex: 1; padding: 10px; background: #FF9800; color: white; border: none; border-radius: 4px; cursor: pointer;">暂停</button>
                        <button onclick="stopTraining()" style="flex: 1; padding: 10px; background: #f44336; color: white; border: none; border-radius: 4px; cursor: pointer;">停止</button>
                    </div>
                </div>
                
                <!-- 实时指标 -->
                <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h3 style="color: #333; margin-bottom: 15px;">实时指标</h3>
                    
                    <div id="training-metrics" style="space-y: 10px;">
                        <div style="display: flex; justify-content: space-between; padding: 10px; background: #f5f5f5; border-radius: 4px; margin-bottom: 8px;">
                            <span>当前轮数:</span>
                            <span id="current-epoch">0</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 10px; background: #f5f5f5; border-radius: 4px; margin-bottom: 8px;">
                            <span>训练损失:</span>
                            <span id="train-loss">0.000</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 10px; background: #f5f5f5; border-radius: 4px; margin-bottom: 8px;">
                            <span>验证损失:</span>
                            <span id="val-loss">0.000</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 10px; background: #f5f5f5; border-radius: 4px; margin-bottom: 8px;">
                            <span>训练准确率:</span>
                            <span id="train-accuracy">0.0%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 10px; background: #f5f5f5; border-radius: 4px; margin-bottom: 8px;">
                            <span>验证准确率:</span>
                            <span id="val-accuracy">0.0%</span>
                        </div>
                    </div>
                    
                    <div style="margin-top: 20px;">
                        <div style="background: #e0e0e0; border-radius: 10px; height: 20px;">
                            <div id="progress-bar" style="width: 0%; height: 100%; background: linear-gradient(90deg, #4CAF50, #8BC34A); border-radius: 10px; transition: width 0.5s ease;"></div>
                        </div>
                        <div style="text-align: center; margin-top: 5px; color: #666; font-size: 14px;">
                            <span id="progress-text">进度: 0%</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
        let trainingInterval;
        let currentEpoch = 0;
        let maxEpochs = 100;
        
        function startTraining() {
            maxEpochs = parseInt(document.getElementById('epochs').value) || 100;
            currentEpoch = 0;
            
            trainingInterval = setInterval(() => {
                if (currentEpoch < maxEpochs) {
                    currentEpoch++;
                    updateMetrics();
                } else {
                    stopTraining();
                }
            }, 1000);
        }
        
        function pauseTraining() {
            clearInterval(trainingInterval);
        }
        
        function stopTraining() {
            clearInterval(trainingInterval);
            currentEpoch = 0;
            updateMetrics();
        }
        
        function updateMetrics() {
            const progress = (currentEpoch / maxEpochs) * 100;
            
            document.getElementById('current-epoch').textContent = currentEpoch;
            document.getElementById('train-loss').textContent = (Math.random() * 2).toFixed(3);
            document.getElementById('val-loss').textContent = (Math.random() * 2.5).toFixed(3);
            document.getElementById('train-accuracy').textContent = ((0.5 + currentEpoch * 0.01) * 100).toFixed(1) + '%';
            document.getElementById('val-accuracy').textContent = ((0.45 + currentEpoch * 0.008) * 100).toFixed(1) + '%';
            
            document.getElementById('progress-bar').style.width = progress + '%';
            document.getElementById('progress-text').textContent = '进度: ' + progress.toFixed(1) + '%';
        }
        </script>
        """
        return HTML(ui_html)
    
    def embed_performance_monitor(self) -> HTML:
        """嵌入性能监控界面"""
        ui_html = """
        <div id="performance-monitor" style="width: 100%; border: 1px solid #ccc; border-radius: 8px; padding: 20px; background: #f9f9f9;">
            <h2 style="color: #333; text-align: center; margin-bottom: 20px;">📊 性能监控仪表板</h2>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                <!-- CPU使用率 -->
                <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h3 style="color: #333; margin-bottom: 15px;">CPU使用率</h3>
                    <div style="text-align: center;">
                        <div id="cpu-gauge" style="width: 150px; height: 75px; margin: 0 auto; position: relative;">
                            <div style="width: 150px; height: 150px; border-radius: 50%; border: 15px solid #e0e0e0; border-bottom-color: transparent; border-left-color: transparent; transform: rotate(45deg);"></div>
                            <div id="cpu-fill" style="width: 150px; height: 150px; border-radius: 50%; border: 15px solid #4CAF50; border-bottom-color: transparent; border-left-color: transparent; transform: rotate(45deg); position: absolute; top: 0; left: 0;"></div>
                        </div>
                        <div id="cpu-percentage" style="font-size: 2em; font-weight: bold; color: #4CAF50; margin-top: 10px;">45%</div>
                    </div>
                </div>
                
                <!-- 内存使用 -->
                <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h3 style="color: #333; margin-bottom: 15px;">内存使用</h3>
                    <div style="text-align: center;">
                        <div id="memory-gauge" style="width: 150px; height: 75px; margin: 0 auto; position: relative;">
                            <div style="width: 150px; height: 150px; border-radius: 50%; border: 15px solid #e0e0e0; border-bottom-color: transparent; border-left-color: transparent; transform: rotate(45deg);"></div>
                            <div id="memory-fill" style="width: 150px; height: 150px; border-radius: 50%; border: 15px solid #2196F3; border-bottom-color: transparent; border-left-color: transparent; transform: rotate(45deg); position: absolute; top: 0; left: 0;"></div>
                        </div>
                        <div id="memory-percentage" style="font-size: 2em; font-weight: bold; color: #2196F3; margin-top: 10px;">68%</div>
                    </div>
                </div>
                
                <!-- 网络延迟 -->
                <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h3 style="color: #333; margin-bottom: 15px;">网络延迟</h3>
                    <div style="text-align: center;">
                        <div id="network-chart" style="width: 100%; height: 120px; background: linear-gradient(to top, #4CAF50, #FFC107, #f44336); border-radius: 8px; position: relative;">
                            <div id="network-indicator" style="width: 4px; height: 40px; background: white; position: absolute; left: 50%; bottom: 0; border-radius: 2px; transform: translateX(-50%);"></div>
                        </div>
                        <div id="network-latency" style="font-size: 2em; font-weight: bold; color: #4CAF50; margin-top: 10px;">12ms</div>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 20px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="color: #333; margin-bottom: 15px;">系统资源趋势</h3>
                <div id="resource-chart" style="width: 100%; height: 200px; background: #f5f5f5; border-radius: 8px; position: relative;">
                    <canvas id="trend-canvas" style="width: 100%; height: 100%;"></canvas>
                </div>
            </div>
        </div>
        
        <script>
        let trendData = [];
        
        function updatePerformance() {
            // 更新CPU
            const cpuPercent = 40 + Math.random() * 30;
            updateGauge('cpu', cpuPercent);
            
            // 更新内存
            const memoryPercent = 60 + Math.random() * 20;
            updateGauge('memory', memoryPercent);
            
            // 更新网络延迟
            const latency = 8 + Math.random() * 15;
            document.getElementById('network-latency').textContent = latency.toFixed(1) + 'ms';
            
            // 更新趋势数据
            trendData.push({
                time: Date.now(),
                cpu: cpuPercent,
                memory: memoryPercent,
                latency: latency
            });
            
            if (trendData.length > 20) {
                trendData.shift();
            }
            
            drawTrendChart();
        }
        
        function updateGauge(type, percent) {
            const rotation = (percent / 100) * 270 - 135;
            const fillElement = document.getElementById(type + '-fill');
            const percentageElement = document.getElementById(type + '-percentage');
            
            if (fillElement) {
                fillElement.style.transform = 'rotate(' + rotation + 'deg)';
            }
            if (percentageElement) {
                percentageElement.textContent = percent.toFixed(1) + '%';
                percentageElement.style.color = percent > 80 ? '#f44336' : percent > 60 ? '#FF9800' : '#4CAF50';
            }
        }
        
        function drawTrendChart() {
            const canvas = document.getElementById('trend-canvas');
            if (!canvas || trendData.length === 0) return;
            
            const ctx = canvas.getContext('2d');
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = rect.height * window.devicePixelRatio;
            
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
            
            const width = rect.width;
            const height = rect.height;
            
            ctx.clearRect(0, 0, width, height);
            
            // 绘制网格
            ctx.strokeStyle = '#e0e0e0';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {
                const y = (height / 4) * i;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(width, y);
                ctx.stroke();
            }
            
            // 绘制CPU趋势线
            if (trendData.length > 1) {
                ctx.strokeStyle = '#4CAF50';
                ctx.lineWidth = 2;
                ctx.beginPath();
                trendData.forEach((point, index) => {
                    const x = (index / (trendData.length - 1)) * width;
                    const y = height - (point.cpu / 100) * height;
                    if (index === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                });
                ctx.stroke();
            }
        }
        
        // 每3秒更新一次数据
        setInterval(updatePerformance, 3000);
        updatePerformance();
        </script>
        """
        return HTML(ui_html)
    
    def embed_system_architecture(self) -> HTML:
        """嵌入系统架构图"""
        ui_html = """
        <div id="system-architecture" style="width: 100%; border: 1px solid #ccc; border-radius: 8px; padding: 20px; background: #f9f9f9;">
            <h2 style="color: #333; text-align: center; margin-bottom: 20px;">🧠 系统架构图</h2>
            
            <div style="background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <svg id="brain-architecture" width="100%" height="400" style="border: 1px solid #e0e0e0; border-radius: 8px;">
                    <!-- 前额叶 -->
                    <g id="prefrontal" class="brain-region" cursor="pointer">
                        <circle cx="300" cy="80" r="40" fill="#3B82F6" opacity="0.8" stroke="white" stroke-width="2"/>
                        <text x="300" y="85" text-anchor="middle" fill="white" font-size="12" font-weight="bold">前额叶</text>
                        <text x="300" y="140" text-anchor="middle" fill="#666" font-size="10">执行控制</text>
                    </g>
                    
                    <!-- 皮层 -->
                    <g id="cortex" class="brain-region" cursor="pointer">
                        <circle cx="150" cy="180" r="50" fill="#10B981" opacity="0.8" stroke="white" stroke-width="2"/>
                        <text x="150" y="185" text-anchor="middle" fill="white" font-size="12" font-weight="bold">皮层</text>
                        <text x="150" y="250" text-anchor="middle" fill="#666" font-size="10">感知处理</text>
                    </g>
                    
                    <!-- 海马体 -->
                    <g id="hippocampus" class="brain-region" cursor="pointer">
                        <circle cx="450" cy="180" r="35" fill="#8B5CF6" opacity="0.8" stroke="white" stroke-width="2"/>
                        <text x="450" y="185" text-anchor="middle" fill="white" font-size="12" font-weight="bold">海马体</text>
                        <text x="450" y="230" text-anchor="middle" fill="#666" font-size="10">记忆巩固</text>
                    </g>
                    
                    <!-- 内嗅皮层 -->
                    <g id="entorhinal" class="brain-region" cursor="pointer">
                        <circle cx="450" cy="280" r="30" fill="#F59E0B" opacity="0.8" stroke="white" stroke-width="2"/>
                        <text x="450" y="285" text-anchor="middle" fill="white" font-size="11" font-weight="bold">内嗅皮层</text>
                        <text x="450" y="320" text-anchor="middle" fill="#666" font-size="9">空间导航</text>
                    </g>
                    
                    <!-- 丘脑 -->
                    <g id="thalamus" class="brain-region" cursor="pointer">
                        <circle cx="300" cy="280" r="25" fill="#EF4444" opacity="0.8" stroke="white" stroke-width="2"/>
                        <text x="300" y="285" text-anchor="middle" fill="white" font-size="11" font-weight="bold">丘脑</text>
                        <text x="300" y="315" text-anchor="middle" fill="#666" font-size="9">信息中继</text>
                    </g>
                    
                    <!-- 杏仁核 -->
                    <g id="amygdala" class="brain-region" cursor="pointer">
                        <circle cx="150" cy="280" r="28" fill="#EC4899" opacity="0.8" stroke="white" stroke-width="2"/>
                        <text x="150" y="285" text-anchor="middle" fill="white" font-size="11" font-weight="bold">杏仁核</text>
                        <text x="150" y="318" text-anchor="middle" fill="#666" font-size="9">情感处理</text>
                    </g>
                    
                    <!-- 连接线 -->
                    <defs>
                        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                            <polygon points="0 0, 10 3.5, 0 7" fill="#666" opacity="0.7"/>
                        </marker>
                    </defs>
                    
                    <!-- 连接线 -->
                    <line x1="260" y1="95" x2="190" y2="160" stroke="#666" stroke-width="2" opacity="0.7" marker-end="url(#arrowhead)"/>
                    <line x1="340" y1="95" x2="415" y2="160" stroke="#666" stroke-width="2" opacity="0.7" marker-end="url(#arrowhead)"/>
                    <line x1="170" y1="210" x2="430" y2="195" stroke="#666" stroke-width="2" opacity="0.7" marker-end="url(#arrowhead)"/>
                    <line x1="440" y1="215" x2="325" y2="265" stroke="#666" stroke-width="2" opacity="0.7" marker-end="url(#arrowhead)"/>
                    <line x1="170" y1="250" x2="430" y2="265" stroke="#666" stroke-width="2" opacity="0.7" marker-end="url(#arrowhead)"/>
                    <line x1="325" y1="255" x2="430" y2="265" stroke="#666" stroke-width="2" opacity="0.7" marker-end="url(#arrowhead)"/>
                </svg>
            </div>
            
            <div id="region-details" style="margin-top: 20px; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="color: #333; margin-bottom: 10px;">区域详情</h3>
                <p style="color: #666;">点击大脑区域查看详细信息</p>
            </div>
        </div>
        
        <script>
        const regionInfo = {
            prefrontal: {
                name: '前额叶',
                function: '执行控制',
                description: '负责高级认知功能，包括决策制定、工作记忆和注意力控制',
                activity: 85,
                connections: 1234
            },
            cortex: {
                name: '皮层',
                function: '感知处理',
                description: '处理感觉输入，产生感知和认知体验',
                activity: 92,
                connections: 2156
            },
            hippocampus: {
                name: '海马体',
                function: '记忆巩固',
                description: '将短期记忆转换为长期记忆，是记忆形成的关键区域',
                activity: 78,
                connections: 987
            },
            entorhinal: {
                name: '内嗅皮层',
                function: '空间导航',
                description: '参与空间认知和记忆检索',
                activity: 65,
                connections: 543
            },
            thalamus: {
                name: '丘脑',
                function: '信息中继',
                description: '大脑的信息中继站，控制意识状态',
                activity: 71,
                connections: 1876
            },
            amygdala: {
                name: '杏仁核',
                function: '情感处理',
                description: '处理恐惧、情感记忆和情感学习',
                activity: 23,
                connections: 432
            }
        };
        
        document.querySelectorAll('.brain-region').forEach(region => {
            region.addEventListener('click', function() {
                const regionId = this.id;
                const info = regionInfo[regionId];
                
                if (info) {
                    const detailsDiv = document.getElementById('region-details');
                    detailsDiv.innerHTML = `
                        <h3 style="color: #333; margin-bottom: 10px;">${info.name}</h3>
                        <p style="color: #666; margin-bottom: 10px;">${info.description}</p>
                        <div style="display: flex; justify-content: space-between;">
                            <span><strong>功能:</strong> ${info.function}</span>
                            <span><strong>活跃度:</strong> ${info.activity}%</span>
                            <span><strong>连接数:</strong> ${info.connections}</span>
                        </div>
                    `;
                }
            });
            
            region.addEventListener('mouseenter', function() {
                this.style.opacity = '1';
                this.style.transform = 'scale(1.1)';
                this.style.transition = 'all 0.2s ease';
            });
            
            region.addEventListener('mouseleave', function() {
                this.style.opacity = '0.8';
                this.style.transform = 'scale(1)';
            });
        });
        </script>
        """
        return HTML(ui_html)


class NotebookUI:
    """Notebook UI界面管理器"""
    
    def __init__(self):
        self.integration = JupyterUIIntegration()
    
    def show_dashboard(self):
        """显示主仪表板"""
        return self.integration.embed_ui_dashboard()
    
    def show_training_interface(self):
        """显示训练界面"""
        return self.integration.embed_training_interface()
    
    def show_performance_monitor(self):
        """显示性能监控"""
        return self.integration.embed_performance_monitor()
    
    def show_system_architecture(self):
        """显示系统架构"""
        return self.integration.embed_system_architecture()
    
    def create_brain_state_widget(self) -> widgets.Widget:
        """创建大脑状态监控小部件"""
        if not IPYWIDGETS_AVAILABLE:
            raise ImportError("ipywidgets is required for brain state widgets")
        
        # 创建状态显示
        status_display = widgets.HTML(value="""
        <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; text-align: center;">
            <h2>🧠 大脑状态监控</h2>
            <p>实时监控各区域状态</p>
        </div>
        """)
        
        # 创建控制按钮
        start_button = widgets.Button(description="开始监控", button_style="success")
        stop_button = widgets.Button(description="停止监控", button_style="warning")
        reset_button = widgets.Button(description="重置", button_style="info")
        
        # 创建状态指示器
        cpu_indicator = widgets.IntProgress(value=45, min=0, max=100, description="CPU:")
        memory_indicator = widgets.IntProgress(value=67, min=0, max=100, description="内存:")
        consciousness_indicator = widgets.FloatSlider(value=0.78, min=0, max=1, description="意识水平:")
        
        # 布局
        controls = widgets.HBox([start_button, stop_button, reset_button])
        metrics = widgets.VBox([cpu_indicator, memory_indicator, consciousness_indicator])
        
        # 更新函数
        def update_metrics():
            import time
            import random
            
            for _ in range(10):  # 更新10次
                cpu_indicator.value = int(40 + random.random() * 30)
                memory_indicator.value = int(60 + random.random() * 20)
                consciousness_indicator.value = 0.5 + random.random() * 0.4
                time.sleep(1)
        
        def start_monitoring(_):
            import threading
            thread = threading.Thread(target=update_metrics)
            thread.daemon = True
            thread.start()
        
        start_button.on_click(start_monitoring)
        
        return widgets.VBox([status_display, controls, metrics])
    
    def create_training_widget(self) -> widgets.Widget:
        """创建训练控制小部件"""
        if not IPYWIDGETS_AVAILABLE:
            raise ImportError("ipywidgets is required for training widgets")
        
        # 配置控件
        epochs_slider = widgets.IntSlider(value=100, min=10, max=1000, description="训练轮数:")
        learning_rate_slider = widgets.FloatSlider(value=0.001, min=0.0001, max=0.1, description="学习率:")
        batch_size_dropdown = widgets.Dropdown(options=[16, 32, 64, 128], value=32, description="批次大小:")
        
        # 控制按钮
        start_training_btn = widgets.Button(description="开始训练", button_style="success")
        pause_training_btn = widgets.Button(description="暂停", button_style="warning")
        stop_training_btn = widgets.Button(description="停止", button_style="danger")
        
        # 进度和指标
        progress_bar = widgets.IntProgress(value=0, min=0, max=100, description="进度:")
        loss_text = widgets.HTML(value="训练损失: 0.000")
        accuracy_text = widgets.HTML(value="准确率: 0.0%")
        
        # 布局
        config_panel = widgets.VBox([
            widgets.HTML("<h3>训练配置</h3>"),
            epochs_slider,
            learning_rate_slider,
            batch_size_dropdown
        ])
        
        control_panel = widgets.VBox([
            widgets.HTML("<h3>训练控制</h3>"),
            widgets.HBox([start_training_btn, pause_training_btn, stop_training_btn]),
            progress_bar,
            loss_text,
            accuracy_text
        ])
        
        return widgets.HBox([config_panel, control_panel])
    
    def create_performance_chart(self):
        """创建性能图表"""
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly is required for performance charts")
        
        # 模拟数据
        import numpy as np
        
        time_points = np.arange(0, 100, 1)
        cpu_usage = 45 + 10 * np.sin(time_points * 0.1) + np.random.normal(0, 5, len(time_points))
        memory_usage = 60 + 8 * np.cos(time_points * 0.08) + np.random.normal(0, 3, len(time_points))
        
        # 创建图表
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("CPU使用率", "内存使用", "网络延迟", "GPU使用率"),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        # CPU指示器
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=45,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "CPU使用率"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # 内存指示器
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=68,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "内存使用率"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 70], 'color': "lightgray"},
                        {'range': [70, 90], 'color': "yellow"},
                        {'range': [90, 100], 'color': "red"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        # 时间序列图
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=cpu_usage,
                mode='lines',
                name='CPU使用率',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        # GPU指示器
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=78,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "GPU使用率"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "purple"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 85], 'color': "yellow"},
                        {'range': [85, 100], 'color': "red"}
                    ]
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="系统性能监控仪表板",
            title_x=0.5,
            showlegend=False
        )
        
        return fig


# 创建全局实例
notebook_ui = NotebookUI()

# 便捷函数
def show_brain_dashboard():
    """显示大脑系统仪表板"""
    display(notebook_ui.show_dashboard())

def show_training_console():
    """显示训练控制台"""
    display(notebook_ui.show_training_interface())

def show_performance_dashboard():
    """显示性能仪表板"""
    display(notebook_ui.show_performance_monitor())

def show_system_diagram():
    """显示系统架构图"""
    display(notebook_ui.show_system_architecture())

def create_brain_monitor_widget():
    """创建大脑监控小部件"""
    return notebook_ui.create_brain_state_widget()

def create_training_widget():
    """创建训练控制小部件"""
    return notebook_ui.create_training_widget()

def create_performance_chart():
    """创建性能图表"""
    return notebook_ui.create_performance_chart()


if __name__ == "__main__":
    # 在Jupyter中运行时的测试代码
    print("脑启发AI系统Jupyter集成模块已加载")
    print("可用函数:")
    print("- show_brain_dashboard(): 显示主仪表板")
    print("- show_training_console(): 显示训练控制台")
    print("- show_performance_dashboard(): 显示性能仪表板")
    print("- show_system_diagram(): 显示系统架构图")
    print("- create_brain_monitor_widget(): 创建监控小部件")
    print("- create_training_widget(): 创建训练小部件")
    print("- create_performance_chart(): 创建性能图表")
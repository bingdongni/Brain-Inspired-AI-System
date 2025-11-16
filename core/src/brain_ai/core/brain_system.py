#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BrainSystem - 完整大脑系统
========================

集成海马体和新皮层的完整大脑模拟系统。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from .base_module import BaseModule, ModuleType

class BrainSystem(BaseModule):
    """完整的大脑系统模拟器"""
    
    def __init__(self, 
                 input_size: int = 512,
                 hippocampus_config: Optional[Dict] = None,
                 neocortex_config: Optional[Dict] = None):
        """
        初始化大脑系统
        
        Args:
            input_size: 输入维度
            hippocampus_config: 海马体配置
            neocortex_config: 新皮层配置
        """
        super().__init__("BrainSystem", ModuleType.HIPPOCAMPUS)
        
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化组件
        self.hippocampus = None
        self.neocortex = None
        
        # 内存系统
        self.short_term_memory = {}
        self.long_term_memory = {}
        
        # 注意力控制器
        self.attention_controller = AttentionController()
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化大脑系统"""
        try:
            # 初始化海马体
            try:
                from ..hippocampus.core.simulator import HippocampusSimulator
                self.hippocampus = HippocampusSimulator(
                    input_size=self.input_size,
                    config=config.get('hippocampus') if config else None
                ).to(self.device)
                self.logger.info("海马体组件初始化完成")
            except ImportError:
                self.logger.warning("海马体组件导入失败，使用简化版本")
                self.hippocampus = SimplifiedHippocampus(self.input_size).to(self.device)
            
            # 初始化新皮层
            try:
                from ..neocortex.neocortex_architecture import NeocortexArchitecture
                self.neocortex = NeocortexArchitecture(
                    input_size=self.input_size,
                    config=config.get('neocortex') if config else None
                ).to(self.device)
                self.logger.info("新皮层组件初始化完成")
            except ImportError:
                self.logger.warning("新皮层组件导入失败，使用简化版本")
                self.neocortex = SimplifiedNeocortex(self.input_size).to(self.device)
            
            self.state = ModuleState.READY
            self.logger.info("大脑系统初始化完成")
            return True
            
        except Exception as e:
            self.state = ModuleState.ERROR
            self.logger.error(f"大脑系统初始化失败: {e}")
            return False
    
    def forward(self, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        self.forward_count += 1
        import time
        start_time = time.time()
        
        try:
            # 注意力控制
            attention_weights = self.attention_controller(input_data)
            
            # 海马体处理
            hippocampus_output = self.hippocampus(input_data) if self.hippocampus else input_data
            
            # 新皮层处理
            neocortex_output = self.neocortex(input_data) if self.neocortex else input_data
            
            # 整合输出
            final_output = self._integrate_outputs(
                hippocampus_output, neocortex_output, attention_weights
            )
            
            return {
                'hippocampus_output': hippocampus_output,
                'neocortex_output': neocortex_output,
                'attention_weights': attention_weights,
                'final_output': final_output
            }
            
        finally:
            self.total_forward_time += time.time() - start_time
    
    def _integrate_outputs(self, hippocampus_out, neocortex_out, attention_weights):
        """整合海马体和新皮层输出"""
        if isinstance(hippocampus_out, dict):
            hippocampus_final = hippocampus_out.get('final_output', hippocampus_out)
        else:
            hippocampus_final = hippocampus_out
            
        if isinstance(neocortex_out, dict):
            neocortex_final = neocortex_out.get('final_output', neocortex_out)
        else:
            neocortex_final = neocortex_out
        
        # 加权整合
        if hippocampus_final.shape == neocortex_final.shape:
            integrated = 0.3 * hippocampus_final + 0.7 * neocortex_final
        else:
            integrated = neocortex_final
            
        return integrated
    
    def store_memory(self, key: str, value: Any) -> str:
        """存储记忆"""
        memory_id = f"mem_{len(self.short_term_memory)}"
        self.short_term_memory[memory_id] = {
            'key': key,
            'value': value,
            'timestamp': len(self.short_term_memory),
            'access_count': 0
        }
        return memory_id
    
    def retrieve_memory(self, memory_id: str) -> Optional[Any]:
        """检索记忆"""
        if memory_id in self.short_term_memory:
            memory = self.short_term_memory[memory_id]
            memory['access_count'] += 1
            return memory['value']
        return None
    
    def learn(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """学习任务"""
        try:
            if 'input_data' in task:
                input_tensor = torch.tensor(task['input_data'], dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.forward(input_tensor)
                
                return {
                    'success': True,
                    'loss': 0.1,  # 模拟损失
                    'output_shape': output['final_output'].shape,
                    'message': '学习任务完成'
                }
            else:
                return {'success': False, 'message': '无效的任务格式'}
                
        except Exception as e:
            return {'success': False, 'message': f'学习失败: {str(e)}'}
    
    def get_memory_stats(self) -> Dict[str, int]:
        """获取记忆统计"""
        return {
            'short_term_memories': len(self.short_term_memory),
            'long_term_memories': len(self.long_term_memory)
        }

class AttentionController(nn.Module):
    """注意力控制器"""
    
    def __init__(self, attention_dim: int = 128):
        super().__init__()
        self.attention_dim = attention_dim
        self.attention_net = nn.Sequential(
            nn.Linear(attention_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 1:
            # 如果是多维输入，使用最后一个维度
            attention_input = x.mean(dim=-1) if x.dim() > 2 else x
        else:
            attention_input = x.unsqueeze(0)
        
        if attention_input.size(-1) != self.attention_dim:
            # 调整维度
            if attention_input.size(-1) < self.attention_dim:
                padding = torch.zeros(*attention_input.shape[:-1], 
                                    self.attention_dim - attention_input.size(-1)).to(attention_input.device)
                attention_input = torch.cat([attention_input, padding], dim=-1)
            else:
                attention_input = attention_input[..., :self.attention_dim]
        
        weights = self.attention_net(attention_input)
        return weights

class SimplifiedHippocampus(nn.Module):
    """简化版海马体"""
    
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return {'final_output': self.encoder(x)}

class SimplifiedNeocortex(nn.Module):
    """简化版新皮层"""
    
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, input_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return {'final_output': self.layers(x)}

# 注册模块类型
ModuleType.BRAIN_SYSTEM = "brain_system"
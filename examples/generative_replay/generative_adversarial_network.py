"""
生成对抗网络模块

实现用于生成式回放的生成对抗网络，包括条件GAN、VAE-GAN和双判别器架构。
用于生成高质量的历史样本，支持持续学习的经验重放。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseGenerator(nn.Module, ABC):
    """
    基础生成器类
    """
    def __init__(self, latent_dim: int, output_shape: Tuple[int, ...]):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
    
    @abstractmethod
    def forward(self, z: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass
    
    @abstractmethod
    def generate(self, num_samples: int, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass


class BaseDiscriminator(nn.Module, ABC):
    """
    基础判别器类
    """
    def __init__(self, input_shape: Tuple[int, ...]):
        super().__init__()
        self.input_shape = input_shape
    
    @abstractmethod
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass


class ConditionalGenerator(BaseGenerator):
    """
    条件生成器
    
    支持标签条件的生成，可用于类别条件生成式回放。
    """
    
    def __init__(self,
                 latent_dim: int,
                 output_shape: Tuple[int, ...],
                 num_classes: int,
                 condition_dim: int = 128,
                 hidden_dims: List[int] = [512, 256, 128]):
        """
        初始化条件生成器
        
        Args:
            latent_dim: 噪声维度
            output_shape: 输出形状
            num_classes: 类别数量
            condition_dim: 条件嵌入维度
            hidden_dims: 隐藏层维度列表
        """
        super().__init__(latent_dim, output_shape)
        
        self.num_classes = num_classes
        
        # 标签嵌入
        self.label_embedding = nn.Embedding(num_classes, condition_dim)
        
        # 计算输入维度
        input_dim = latent_dim + condition_dim
        
        # 构建网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        output_size = np.prod(output_shape)
        layers.append(nn.Linear(prev_dim, output_size))
        
        self.net = nn.Sequential(*layers)
        
        # 输出重塑
        self.output_size = output_size
    
    def forward(self, z: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            z: 噪声输入 [batch_size, latent_dim]
            condition: 条件标签 [batch_size] 或 None
            
        Returns:
            生成样本 [batch_size, *output_shape]
        """
        if condition is not None:
            # 标签嵌入
            label_emb = self.label_embedding(condition)
            # 拼接噪声和条件
            combined_input = torch.cat([z, label_emb], dim=1)
        else:
            combined_input = z
        
        # 生成
        output = self.net(combined_input)
        
        # 重塑到目标形状
        batch_size = output.size(0)
        output = output.view(batch_size, *self.output_shape)
        
        return output
    
    def generate(self, num_samples: int, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成样本
        
        Args:
            num_samples: 生成样本数量
            condition: 条件标签，如果为None则随机选择
            
        Returns:
            生成的样本
        """
        self.eval()
        with torch.no_grad():
            # 随机噪声
            z = torch.randn(num_samples, self.latent_dim, device=z.device if hasattr(self, 'device') else 'cpu')
            
            # 随机条件（如果未提供）
            if condition is None and self.num_classes > 0:
                condition = torch.randint(0, self.num_classes, (num_samples,))
            
            # 生成
            samples = self.forward(z, condition)
        
        return samples


class ConditionalDiscriminator(BaseDiscriminator):
    """
    条件判别器
    
    用于判断样本真实性，支持标签条件。
    """
    
    def __init__(self,
                 input_shape: Tuple[int, ...],
                 num_classes: int,
                 condition_dim: int = 128,
                 hidden_dims: List[int] = [128, 256, 512]):
        """
        初始化条件判别器
        
        Args:
            input_shape: 输入形状
            num_classes: 类别数量
            condition_dim: 条件嵌入维度
            hidden_dims: 隐藏层维度列表
        """
        super().__init__(input_shape)
        
        self.num_classes = num_classes
        
        # 标签嵌入
        self.label_embedding = nn.Embedding(num_classes, condition_dim)
        
        # 计算输入维度
        input_size = np.prod(input_shape)
        
        # 构建网络
        layers = []
        prev_dim = input_size + condition_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # 输出层 (真实度)
        self.output_head = nn.Linear(prev_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入样本 [batch_size, *input_shape]
            condition: 条件标签 [batch_size] 或 None
            
        Returns:
            真实度分数 [batch_size, 1]
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        if condition is not None:
            # 标签嵌入
            label_emb = self.label_embedding(condition)
            # 拼接输入和条件
            combined_input = torch.cat([x_flat, label_emb], dim=1)
        else:
            combined_input = x_flat
        
        # 特征提取
        features = self.net(combined_input)
        
        # 真实度判断
        validity = self.output_head(features)
        validity = self.sigmoid(validity)
        
        return validity


class VAEGenerator(BaseGenerator):
    """
    VAE生成器
    
    变分自编码器生成器，提供更稳定的训练和更好的潜在空间控制。
    """
    
    def __init__(self,
                 latent_dim: int,
                 output_shape: Tuple[int, ...],
                 hidden_dims: List[int] = [512, 256]):
        """
        初始化VAE生成器
        
        Args:
            latent_dim: 潜在空间维度
            output_shape: 输出形状
            hidden_dims: 隐藏层维度
        """
        super().__init__(latent_dim, output_shape)
        
        # 编码器部分（用于训练时的重建）
        encoder_layers = []
        input_size = np.prod(output_shape)
        prev_dim = input_size
        
        for hidden_dim in hidden_dims[::-1]:  # 反向构建
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 均值和对数方差层
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # 解码器部分
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        output_size = np.prod(output_shape)
        decoder_layers.append(nn.Linear(prev_dim, output_size))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码
        
        Args:
            x: 输入样本
            
        Returns:
            (mu, logvar) 元组
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        h = self.encoder(x_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧
        
        Args:
            mu: 均值
            logvar: 对数方差
            
        Returns:
            采样潜在向量
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码
        
        Args:
            z: 潜在向量
            
        Returns:
            重建样本
        """
        batch_size = z.size(0)
        output = self.decoder(z)
        output = output.view(batch_size, *self.output_shape)
        return output
    
    def forward(self, z: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播（用于生成）
        
        Args:
            z: 噪声输入
            condition: 条件（未在此实现中使用）
            
        Returns:
            生成样本
        """
        # 对于无条件生成，直接解码
        return self.decode(z)
    
    def generate(self, num_samples: int, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成样本
        
        Args:
            num_samples: 生成样本数量
            condition: 条件（未在此实现中使用）
            
        Returns:
            生成的样本
        """
        self.eval()
        with torch.no_grad():
            # 从标准正态分布采样
            z = torch.randn(num_samples, self.latent_dim, device=z.device if hasattr(self, 'device') else 'cpu')
            samples = self.decode(z)
        
        return samples


class GenerativeAdversarialNetwork:
    """
    生成对抗网络
    
    集成多种生成器类型的完整GAN系统，支持训练和生成功能。
    """
    
    def __init__(self,
                 generator_type: str = 'conditional',
                 latent_dim: int = 100,
                 output_shape: Optional[Tuple[int, ...]] = None,
                 num_classes: int = 10,
                 device: Optional[torch.device] = None,
                 gan_config: Optional[Dict] = None):
        """
        初始化GAN
        
        Args:
            generator_type: 生成器类型 ('conditional', 'vae', 'unconditional')
            latent_dim: 潜在空间维度
            output_shape: 输出形状（可选）
            num_classes: 类别数量
            device: 计算设备
            gan_config: GAN配置字典
        """
        self.generator_type = generator_type
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 默认配置
        default_config = {
            'learning_rate_g': 0.0002,
            'learning_rate_d': 0.0002,
            'beta1': 0.5,
            'beta2': 0.999,
            'weight_decay': 1e-4,
            'gradient_penalty': True,
            'spectral_norm': True
        }
        default_config.update(gan_config or {})
        self.config = default_config
        
        # 创建生成器和判别器
        self.generator = self._create_generator(output_shape)
        self.discriminator = self._create_discriminator(output_shape)
        
        # 移动到设备
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # 创建优化器
        self.optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config['learning_rate_g'],
            betas=(self.config['beta1'], self.config['beta2']),
            weight_decay=self.config['weight_decay']
        )
        
        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config['learning_rate_d'],
            betas=(self.config['beta1'], self.config['beta2']),
            weight_decay=self.config['weight_decay']
        )
        
        # 损失函数
        self.adversarial_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
        # 训练统计
        self.training_stats = {
            'g_losses': [],
            'd_losses': [],
            'gp_losses': [] if self.config['gradient_penalty'] else None
        }
        
        logger.info(f"初始化GAN成功，类型: {generator_type}")
    
    def _create_generator(self, output_shape: Optional[Tuple[int, ...]]) -> BaseGenerator:
        """
        创建生成器
        """
        if self.generator_type == 'conditional':
            if output_shape is None:
                raise ValueError("条件生成器需要指定输出形状")
            return ConditionalGenerator(
                latent_dim=self.latent_dim,
                output_shape=output_shape,
                num_classes=self.num_classes
            )
        elif self.generator_type == 'vae':
            if output_shape is None:
                raise ValueError("VAE生成器需要指定输出形状")
            return VAEGenerator(
                latent_dim=self.latent_dim,
                output_shape=output_shape
            )
        else:
            raise ValueError(f"不支持的生成器类型: {self.generator_type}")
    
    def _create_discriminator(self, output_shape: Optional[Tuple[int, ...]]) -> BaseDiscriminator:
        """
        创建判别器
        """
        if output_shape is None:
            raise ValueError("判别器需要指定输入形状")
        
        return ConditionalDiscriminator(
            input_shape=output_shape,
            num_classes=self.num_classes if self.generator_type == 'conditional' else 0
        )
    
    def train_step(self,
                  real_data: torch.Tensor,
                  real_labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            real_data: 真实数据
            real_labels: 真实标签
            
        Returns:
            训练统计字典
        """
        batch_size = real_data.size(0)
        
        # 标签
        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)
        
        # ====================
        # 训练判别器
        # ====================
        self.optimizer_d.zero_grad()
        
        # 真实数据
        real_validity = self.discriminator(real_data, real_labels)
        d_real_loss = self.adversarial_loss(real_validity, valid)
        
        # 生成假数据
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device) if real_labels is None else real_labels
        fake_data = self.generator(noise, fake_labels if self.generator_type == 'conditional' else None)
        
        # 假数据
        fake_validity = self.discriminator(fake_data.detach(), fake_labels if self.generator_type == 'conditional' else None)
        d_fake_loss = self.adversarial_loss(fake_validity, fake)
        
        # 总判别器损失
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        # 梯度惩罚
        gp_loss = torch.tensor(0.0, device=self.device)
        if self.config['gradient_penalty']:
            gp_loss = self._compute_gradient_penalty(real_data, fake_data, real_labels if self.generator_type == 'conditional' else None)
            d_loss += 10 * gp_loss
        
        d_loss.backward()
        self.optimizer_d.step()
        
        # ====================
        # 训练生成器
        # ====================
        self.optimizer_g.zero_grad()
        
        # 生成假数据
        gen_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device) if real_labels is None else real_labels
        generated_data = self.generator(noise, gen_labels if self.generator_type == 'conditional' else None)
        
        # 判别器对生成数据的判断
        validity = self.discriminator(generated_data, gen_labels if self.generator_type == 'conditional' else None)
        g_loss = self.adversarial_loss(validity, valid)
        
        # 如果是VAE，添加重建损失
        if self.generator_type == 'vae' and isinstance(self.generator, VAEGenerator):
            # 重建损失
            reconstructed = self.generator.decode(self.generator.encode(real_data)[0])
            recon_loss = self.mse_loss(reconstructed, real_data)
            g_loss += 0.1 * recon_loss
        
        g_loss.backward()
        self.optimizer_g.step()
        
        # 更新统计
        self.training_stats['g_losses'].append(g_loss.item())
        self.training_stats['d_losses'].append(d_loss.item())
        if self.config['gradient_penalty'] and gp_loss is not None:
            self.training_stats['gp_losses'].append(gp_loss.item())
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'gp_loss': gp_loss.item() if self.config['gradient_penalty'] and gp_loss is not None else 0.0
        }
    
    def _compute_gradient_penalty(self,
                                real_data: torch.Tensor,
                                fake_data: torch.Tensor,
                                labels: Optional[torch.Tensor]) -> torch.Tensor:
        """
        计算梯度惩罚（WGAN-GP）
        """
        batch_size = real_data.size(0)
        
        # 随机权重
        alpha = torch.rand(batch_size, 1, device=self.device)
        alpha = alpha.expand_as(real_data)
        
        # 插值样本
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)
        
        # 判别器对插值样本的判断
        d_interpolates = self.discriminator(interpolates, labels)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # 计算梯度范数
        gradients = gradients.view(batch_size, -1)
        gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        
        # 梯度惩罚
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        
        return gradient_penalty
    
    def generate_samples(self,
                        num_samples: int,
                        labels: Optional[torch.Tensor] = None,
                        return_raw: bool = False) -> torch.Tensor:
        """
        生成样本
        
        Args:
            num_samples: 生成样本数量
            labels: 条件标签，None表示随机
            return_raw: 是否返回原始输出
            
        Returns:
            生成的样本
        """
        self.generator.eval()
        with torch.no_grad():
            if labels is None:
                labels = torch.randint(0, self.num_classes, (num_samples,), device=self.device)
            
            # 随机噪声
            noise = torch.randn(num_samples, self.latent_dim, device=self.device)
            
            # 生成
            if self.generator_type == 'conditional':
                samples = self.generator(noise, labels)
            else:
                samples = self.generator(noise, None)
            
            if return_raw:
                return samples
            else:
                # 归一化到[0,1]（假设原始输出在[-1,1]）
                samples = (samples + 1) / 2
                samples = torch.clamp(samples, 0, 1)
                return samples
    
    def save_model(self, filepath: str) -> None:
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        save_data = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'config': self.config,
            'generator_type': self.generator_type,
            'training_stats': self.training_stats
        }
        
        torch.save(save_data, filepath)
        logger.info(f"GAN模型已保存到: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        加载模型
        
        Args:
            filepath: 模型路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.config.update(checkpoint.get('config', {}))
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        
        logger.info(f"GAN模型已从 {filepath} 加载")
    
    def get_training_statistics(self) -> Dict[str, float]:
        """
        获取训练统计信息
        
        Returns:
            统计信息字典
        """
        stats = {}
        
        if self.training_stats['g_losses']:
            stats['avg_g_loss'] = np.mean(self.training_stats['g_losses'])
            stats['std_g_loss'] = np.std(self.training_stats['g_losses'])
        
        if self.training_stats['d_losses']:
            stats['avg_d_loss'] = np.mean(self.training_stats['d_losses'])
            stats['std_d_loss'] = np.std(self.training_stats['d_losses'])
        
        if self.training_stats['gp_losses']:
            stats['avg_gp_loss'] = np.mean(self.training_stats['gp_losses'])
        
        stats['total_steps'] = len(self.training_stats['g_losses'])
        
        return stats


class DualDiscriminatorGAN:
    """
    双判别器GAN
    
    使用两个判别器：一个用于真实性判断，一个用于质量判断。
    """
    
    def __init__(self,
                 generator: BaseGenerator,
                 real_discriminator: BaseDiscriminator,
                 quality_discriminator: BaseDiscriminator,
                 device: Optional[torch.device] = None):
        """
        初始化双判别器GAN
        
        Args:
            generator: 生成器
            real_discriminator: 真实性判别器
            quality_discriminator: 质量判别器
            device: 计算设备
        """
        self.generator = generator
        self.real_discriminator = real_discriminator
        self.quality_discriminator = quality_discriminator
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 移动到设备
        self.generator.to(self.device)
        self.real_discriminator.to(self.device)
        self.quality_discriminator.to(self.device)
        
        # 优化器
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        self.optimizer_dr = torch.optim.Adam(self.real_discriminator.parameters(), lr=0.0002)
        self.optimizer_dq = torch.optim.Adam(self.quality_discriminator.parameters(), lr=0.0002)
        
        # 损失函数
        self.adversarial_loss = nn.BCELoss()
        
    def train_step(self,
                  real_data: torch.Tensor,
                  real_labels: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        训练步骤
        """
        batch_size = real_data.size(0)
        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)
        
        # ====================
        # 训练判别器
        # ====================
        self.optimizer_dr.zero_grad()
        self.optimizer_dq.zero_grad()
        
        # 真实数据
        real_validity = self.real_discriminator(real_data, real_labels)
        dr_real_loss = self.adversarial_loss(real_validity, valid)
        
        # 生成假数据
        noise = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
        fake_data = self.generator(noise, real_labels)
        
        # 假数据
        fake_validity = self.real_discriminator(fake_data.detach(), real_labels)
        dr_fake_loss = self.adversarial_loss(fake_validity, fake)
        
        # 总真实性判别器损失
        dr_loss = (dr_real_loss + dr_fake_loss) / 2
        dr_loss.backward()
        self.optimizer_dr.step()
        
        # 质量判别器
        quality_real = self.quality_discriminator(real_data, real_labels)
        dq_real_loss = self.adversarial_loss(quality_real, valid)
        
        quality_fake = self.quality_discriminator(fake_data.detach(), real_labels)
        dq_fake_loss = self.adversarial_loss(quality_fake, fake)
        
        dq_loss = (dq_real_loss + dq_fake_loss) / 2
        dq_loss.backward()
        self.optimizer_dq.step()
        
        # ====================
        # 训练生成器
        # ====================
        self.optimizer_g.zero_grad()
        
        # 重新生成
        generated_data = self.generator(noise, real_labels)
        
        # 判别器判断
        real_validity = self.real_discriminator(generated_data, real_labels)
        quality = self.quality_discriminator(generated_data, real_labels)
        
        # 生成器损失：既要骗过真实性判别器，也要通过质量判别器
        g_loss = self.adversarial_loss(real_validity, valid) + self.adversarial_loss(quality, valid)
        g_loss.backward()
        self.optimizer_g.step()
        
        return {
            'g_loss': g_loss.item(),
            'dr_loss': dr_loss.item(),
            'dq_loss': dq_loss.item()
        }
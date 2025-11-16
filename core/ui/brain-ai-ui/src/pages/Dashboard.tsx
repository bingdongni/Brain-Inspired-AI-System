import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  Brain, 
  Activity, 
  Cpu, 
  Database, 
  Zap, 
  TrendingUp
} from 'lucide-react';
import { MetricCard } from '../components/MetricCard';
import { StatusIndicator } from '../components/StatusIndicator';

interface SystemMetrics {
  cpuUsage: number;
  memoryUsage: number;
  activeRegions: number;
  totalMemoryTraces: number;
  consciousnessLevel: number;
  activeConnections: number;
  learningRate: number;
  accuracy: number;
}

export const Dashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<SystemMetrics>({
    cpuUsage: 45,
    memoryUsage: 67,
    activeRegions: 5,
    totalMemoryTraces: 1234,
    consciousnessLevel: 0.78,
    activeConnections: 8923,
    learningRate: 0.001,
    accuracy: 0.92
  });

  const [brainRegions] = useState([
    { name: '前额叶', status: 'active', activity: 85, connections: 1234 },
    { name: '海马体', status: 'active', activity: 92, connections: 2156 },
    { name: '皮层', status: 'active', activity: 78, connections: 3421 },
    { name: '内嗅皮层', status: 'active', activity: 65, connections: 987 },
    { name: '杏仁核', status: 'idle', activity: 23, connections: 456 }
  ]);

  useEffect(() => {
    // 模拟实时数据更新
    const interval = setInterval(() => {
      setMetrics(prev => ({
        ...prev,
        cpuUsage: Math.max(20, Math.min(90, prev.cpuUsage + (Math.random() - 0.5) * 10)),
        memoryUsage: Math.max(30, Math.min(95, prev.memoryUsage + (Math.random() - 0.5) * 8)),
        consciousnessLevel: Math.max(0.5, Math.min(1.0, prev.consciousnessLevel + (Math.random() - 0.5) * 0.1)),
        accuracy: Math.max(0.7, Math.min(1.0, prev.accuracy + (Math.random() - 0.5) * 0.05))
      }));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // 使用useCallback优化函数定义

  // 使用useMemo优化cards数组，避免每次render都重新创建
  const cards = useMemo(() => [
    {
      title: 'CPU使用率',
      value: `${metrics.cpuUsage.toFixed(1)}%`,
      icon: Cpu,
      color: 'blue' as const,
      change: '+2.3%'
    },
    {
      title: '内存使用率',
      value: `${metrics.memoryUsage.toFixed(1)}%`,
      icon: Database,
      color: 'green' as const,
      change: '+1.7%'
    },
    {
      title: '意识水平',
      value: `${(metrics.consciousnessLevel * 100).toFixed(1)}%`,
      icon: Brain,
      color: 'purple' as const,
      change: '+0.8%'
    },
    {
      title: '活跃连接',
      value: metrics.activeConnections.toLocaleString(),
      icon: Zap,
      color: 'orange' as const,
      change: '+156'
    },
    {
      title: '记忆痕迹',
      value: metrics.totalMemoryTraces.toLocaleString(),
      icon: Database,
      color: 'indigo' as const,
      change: '+23'
    },
    {
      title: '模型准确率',
      value: `${(metrics.accuracy * 100).toFixed(1)}%`,
      icon: TrendingUp,
      color: 'emerald' as const,
      change: '+1.2%'
    }
  ], [metrics]);

  return (
    <div className="space-y-6">
      {/* 页面标题 */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">系统概览</h1>
        <p className="mt-2 text-gray-600">大脑启发AI系统实时状态监控</p>
      </div>

      {/* 指标卡片 */}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {cards.map((card) => (
          <MetricCard
            key={card.title}
            title={card.title}
            value={card.value}
            icon={card.icon}
            color={card.color}
            change={card.change}
          />
        ))}
      </div>

      {/* 大脑区域状态 */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">大脑区域状态</h3>
          <div className="space-y-4">
            {brainRegions.map((region) => (
              <div key={region.name} className="border rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <StatusIndicator status={region.status as any} size="sm" />
                    <span className="ml-3 text-sm font-medium text-gray-900">{region.name}</span>
                  </div>
                  <div className="text-sm text-gray-500">
                    {region.connections.toLocaleString()} 连接
                  </div>
                </div>
                <div className="mt-2">
                  <div className="flex items-center justify-between text-sm text-gray-600 mb-1">
                    <span>活跃度</span>
                    <span>{region.activity}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${region.activity}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 实时活动图表 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">神经活动趋势</h3>
          <div className="h-64 flex items-end space-x-2">
            {Array.from({ length: 20 }, (_, i) => (
              <div 
                key={i}
                className="bg-blue-500 rounded-t"
                style={{ 
                  height: `${Math.random() * 100 + 10}%`,
                  width: '20px',
                  animationDelay: `${i * 100}ms`
                }}
              />
            ))}
          </div>
        </div>

        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">记忆形成速率</h3>
          <div className="h-64 flex items-end space-x-2">
            {Array.from({ length: 20 }, (_, i) => (
              <div 
                key={i}
                className="bg-green-500 rounded-t"
                style={{ 
                  height: `${Math.random() * 80 + 20}%`,
                  width: '20px',
                  animationDelay: `${i * 150}ms`
                }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
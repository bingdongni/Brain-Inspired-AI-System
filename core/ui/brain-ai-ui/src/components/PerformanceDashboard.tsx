import React, { useState, useEffect } from 'react';
import { 
  Cpu, 
  Database, 
  Network, 
  Activity,
  Clock,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  RefreshCw
} from 'lucide-react';

interface PerformanceMetric {
  name: string;
  value: number;
  unit: string;
  status: 'normal' | 'warning' | 'critical';
  trend: 'up' | 'down' | 'stable';
  history: number[];
}

interface SystemResource {
  name: string;
  current: number;
  total: number;
  usage: number;
  status: 'normal' | 'warning' | 'critical';
}

export const PerformanceDashboard: React.FC = () => {
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [autoRefresh, setAutoRefresh] = useState(true);

  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetric[]>([
    {
      name: 'CPU 使用率',
      value: 45.2,
      unit: '%',
      status: 'normal',
      trend: 'up',
      history: [40, 42, 45, 43, 46, 45, 45]
    },
    {
      name: '内存使用率',
      value: 67.8,
      unit: '%',
      status: 'warning',
      trend: 'up',
      history: [65, 66, 68, 67, 69, 68, 68]
    },
    {
      name: '网络延迟',
      value: 12.5,
      unit: 'ms',
      status: 'normal',
      trend: 'stable',
      history: [13, 12, 11, 13, 12, 12, 13]
    },
    {
      name: '磁盘I/O',
      value: 234.7,
      unit: 'MB/s',
      status: 'normal',
      trend: 'up',
      history: [220, 225, 230, 235, 232, 234, 235]
    },
    {
      name: 'GPU 使用率',
      value: 78.9,
      unit: '%',
      status: 'warning',
      trend: 'up',
      history: [70, 72, 75, 78, 77, 79, 79]
    },
    {
      name: '活跃连接',
      value: 1234,
      unit: '',
      status: 'normal',
      trend: 'stable',
      history: [1200, 1220, 1240, 1230, 1235, 1234, 1234]
    }
  ]);

  const [systemResources, setSystemResources] = useState<SystemResource[]>([
    { name: 'CPU 核心', current: 3.6, total: 8, usage: 45, status: 'normal' },
    { name: '内存', current: 54.2, total: 80, usage: 67.8, status: 'warning' },
    { name: 'GPU 显存', current: 6.3, total: 8, usage: 78.8, status: 'warning' },
    { name: '磁盘空间', current: 234, total: 500, usage: 46.8, status: 'normal' },
    { name: '网络带宽', current: 125, total: 1000, usage: 12.5, status: 'normal' }
  ]);

  const [neuralMetrics, setNeuralMetrics] = useState({
    activeNeurons: 8945,
    totalSynapses: 125678,
    firingRate: 23.4,
    synapticPlasticity: 0.67,
    memoryTraces: 1234,
    consciousnessLevel: 0.78
  });

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        updateMetrics();
        setLastUpdate(new Date());
      }, 5000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const updateMetrics = () => {
    setPerformanceMetrics(prev => prev.map(metric => ({
      ...metric,
      value: Math.max(0, metric.value + (Math.random() - 0.5) * 10),
      history: [...metric.history.slice(1), metric.value + (Math.random() - 0.5) * 5]
    })));

    setNeuralMetrics(prev => ({
      ...prev,
      activeNeurons: Math.max(8000, Math.min(10000, prev.activeNeurons + Math.floor((Math.random() - 0.5) * 100))),
      firingRate: Math.max(0, Math.min(100, prev.firingRate + (Math.random() - 0.5) * 5)),
      synapticPlasticity: Math.max(0, Math.min(1, prev.synapticPlasticity + (Math.random() - 0.5) * 0.1))
    }));
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'normal': return 'text-green-600 bg-green-100';
      case 'warning': return 'text-yellow-600 bg-yellow-100';
      case 'critical': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'normal': return <CheckCircle className="h-5 w-5" />;
      case 'warning': return <AlertTriangle className="h-5 w-5" />;
      case 'critical': return <AlertTriangle className="h-5 w-5" />;
      default: return <CheckCircle className="h-5 w-5" />;
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUp className="h-4 w-4 text-green-600" />;
      case 'down': return <TrendingUp className="h-4 w-4 text-red-600 transform rotate-180" />;
      default: return <Activity className="h-4 w-4 text-gray-600" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* 页面标题和控制 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">性能监控</h1>
          <p className="mt-2 text-gray-600">系统资源和神经网络性能实时监控</p>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="text-sm text-gray-500">
            最后更新: {lastUpdate.toLocaleTimeString()}
          </div>
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md ${
              autoRefresh ? 'text-green-700 bg-green-100 border-green-300' : 'text-gray-700 bg-white hover:bg-gray-50'
            }`}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${autoRefresh ? 'animate-spin' : ''}`} />
            自动刷新
          </button>
          <button
            onClick={updateMetrics}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700"
          >
            手动刷新
          </button>
        </div>
      </div>

      {/* 性能指标卡片 */}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {performanceMetrics.map((metric) => (
          <div key={metric.name} className="bg-white overflow-hidden shadow rounded-lg">
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className={`p-3 rounded-md ${getStatusColor(metric.status)}`}>
                    {getStatusIcon(metric.status)}
                  </div>
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">{metric.name}</dt>
                    <dd className="flex items-baseline">
                      <div className="text-2xl font-semibold text-gray-900">
                        {metric.value.toFixed(1)}{metric.unit}
                      </div>
                      <div className="ml-2 flex items-baseline text-sm font-semibold">
                        {getTrendIcon(metric.trend)}
                      </div>
                    </dd>
                  </dl>
                </div>
              </div>
              <div className="mt-4">
                <div className="flex space-x-1 h-8">
                  {metric.history.map((value, index) => (
                    <div
                      key={index}
                      className="bg-blue-500 flex-1 rounded-t"
                      style={{ height: `${Math.max(5, (value / (metric.name.includes('率') ? 100 : metric.value * 2)) * 100)}%` }}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* 系统资源 */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900 flex items-center">
            <Database className="h-5 w-5 mr-2" />
            系统资源使用情况
          </h3>
        </div>
        <div className="p-6">
          <div className="space-y-6">
            {systemResources.map((resource) => (
              <div key={resource.name}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-900">{resource.name}</span>
                  <span className={`text-sm px-2 py-1 rounded-full ${getStatusColor(resource.status)}`}>
                    {resource.usage.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-500 ${
                      resource.status === 'normal' ? 'bg-green-500' :
                      resource.status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${resource.usage}%` }}
                  />
                </div>
                <div className="mt-1 text-xs text-gray-500">
                  {resource.current.toFixed(1)} / {resource.total} {resource.name.includes('核心') ? '核' : 'GB'}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 神经网络性能 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white shadow rounded-lg">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900 flex items-center">
              <Network className="h-5 w-5 mr-2" />
              神经网络指标
            </h3>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">活跃神经元</span>
                <span className="text-lg font-semibold">{neuralMetrics.activeNeurons.toLocaleString()}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">突触总数</span>
                <span className="text-lg font-semibold">{neuralMetrics.totalSynapses.toLocaleString()}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">神经元发放率</span>
                <span className="text-lg font-semibold text-blue-600">{neuralMetrics.firingRate.toFixed(1)} Hz</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">突触可塑性</span>
                <span className="text-lg font-semibold text-green-600">{(neuralMetrics.synapticPlasticity * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">记忆痕迹</span>
                <span className="text-lg font-semibold text-purple-600">{neuralMetrics.memoryTraces.toLocaleString()}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">意识水平</span>
                <span className="text-lg font-semibold text-orange-600">{(neuralMetrics.consciousnessLevel * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white shadow rounded-lg">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900 flex items-center">
              <Clock className="h-5 w-5 mr-2" />
              性能趋势
            </h3>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm text-gray-600 mb-2">
                  <span>CPU 使用率</span>
                  <span>过去1小时</span>
                </div>
                <div className="h-24 flex items-end space-x-1">
                  {Array.from({ length: 24 }, (_, i) => (
                    <div
                      key={i}
                      className="bg-blue-500 rounded-t flex-1"
                      style={{ height: `${30 + Math.random() * 50}%` }}
                    />
                  ))}
                </div>
              </div>
              
              <div>
                <div className="flex justify-between text-sm text-gray-600 mb-2">
                  <span>内存使用率</span>
                  <span>过去1小时</span>
                </div>
                <div className="h-24 flex items-end space-x-1">
                  {Array.from({ length: 24 }, (_, i) => (
                    <div
                      key={i}
                      className="bg-green-500 rounded-t flex-1"
                      style={{ height: `${50 + Math.random() * 30}%` }}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
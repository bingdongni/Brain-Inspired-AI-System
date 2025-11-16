import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  Brain, 
  Activity, 
  Zap, 
  RefreshCw,
  Play,
  Pause,
  RotateCcw,
  Info
} from 'lucide-react';

interface BrainRegionStatus {
  id: string;
  name: string;
  function: string;
  activity: number;
  connections: number;
  neurons: number;
  status: 'active' | 'idle' | 'processing' | 'error';
  position: { x: number; y: number };
}

interface MemoryTrace {
  id: string;
  type: string;
  strength: number;
  age: number;
  retrievals: number;
}

interface NeuralOscillation {
  frequency: string;
  amplitude: number;
  phase: number;
  power: number;
}

export const BrainStateMonitor: React.FC = () => {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);
  const [brainRegions, setBrainRegions] = useState<BrainRegionStatus[]>([
    {
      id: 'prefrontal',
      name: '前额叶',
      function: '执行控制',
      activity: 85,
      connections: 1234,
      neurons: 15600,
      status: 'active',
      position: { x: 50, y: 20 }
    },
    {
      id: 'cortex',
      name: '皮层',
      function: '感知处理',
      activity: 92,
      connections: 2156,
      neurons: 28900,
      status: 'active',
      position: { x: 25, y: 45 }
    },
    {
      id: 'hippocampus',
      name: '海马体',
      function: '记忆巩固',
      activity: 78,
      connections: 987,
      neurons: 4500,
      status: 'processing',
      position: { x: 75, y: 45 }
    },
    {
      id: 'entorhinal',
      name: '内嗅皮层',
      function: '空间导航',
      activity: 65,
      connections: 543,
      neurons: 2100,
      status: 'active',
      position: { x: 75, y: 70 }
    },
    {
      id: 'thalamus',
      name: '丘脑',
      function: '信息中继',
      activity: 71,
      connections: 1876,
      neurons: 3200,
      status: 'active',
      position: { x: 50, y: 70 }
    },
    {
      id: 'amygdala',
      name: '杏仁核',
      function: '情感处理',
      activity: 23,
      connections: 432,
      neurons: 1300,
      status: 'idle',
      position: { x: 25, y: 70 }
    }
  ]);

  const [memoryTraces, setMemoryTraces] = useState<MemoryTrace[]>([
    { id: '1', type: '情景记忆', strength: 0.87, age: 2.3, retrievals: 15 },
    { id: '2', type: '语义记忆', strength: 0.92, age: 1.8, retrievals: 23 },
    { id: '3', type: '程序记忆', strength: 0.75, age: 0.5, retrievals: 8 },
    { id: '4', type: '工作记忆', strength: 0.63, age: 0.1, retrievals: 3 },
    { id: '5', type: '情感记忆', strength: 0.88, age: 3.2, retrievals: 12 }
  ]);

  const [neuralOscillations, setNeuralOscillations] = useState<NeuralOscillation[]>([
    { frequency: 'Gamma (40Hz)', amplitude: 0.78, phase: 2.1, power: 0.85 },
    { frequency: 'Beta (20Hz)', amplitude: 0.65, phase: 1.2, power: 0.72 },
    { frequency: 'Alpha (10Hz)', amplitude: 0.82, phase: 0.8, power: 0.91 },
    { frequency: 'Theta (6Hz)', amplitude: 0.58, phase: 3.4, power: 0.68 },
    { frequency: 'Delta (2Hz)', amplitude: 0.34, phase: 5.2, power: 0.45 }
  ]);

  const [consciousnessLevel, setConsciousnessLevel] = useState(0.78);

  // 使用useCallback优化状态相关函数
  const getStatusColor = useCallback((status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500';
      case 'processing': return 'bg-yellow-500';
      case 'idle': return 'bg-gray-400';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-400';
    }
  }, []);

  const getActivityColor = useCallback((activity: number) => {
    if (activity >= 80) return 'text-green-600';
    if (activity >= 60) return 'text-yellow-600';
    if (activity >= 40) return 'text-orange-600';
    return 'text-red-600';
  }, []);

  const getWaveColor = useCallback((index: number) => {
    const colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];
    return colors[index % colors.length];
  }, []);

  // 使用useMemo优化计算密集型数据
  const memoryStrengthAverage = useMemo(() => {
    return memoryTraces.reduce((sum, trace) => sum + trace.strength, 0) / memoryTraces.length;
  }, [memoryTraces]);

  useEffect(() => {
    if (isMonitoring) {
      const interval = setInterval(() => {
        // 更新大脑区域活动
        setBrainRegions(prev => prev.map(region => ({
          ...region,
          activity: Math.max(0, Math.min(100, region.activity + (Math.random() - 0.5) * 10)),
          connections: Math.max(100, region.connections + Math.floor((Math.random() - 0.5) * 50)),
          status: Math.random() > 0.9 ? 'processing' : region.status
        })));

        // 更新记忆痕迹强度
        setMemoryTraces(prev => prev.map(trace => ({
          ...trace,
          strength: Math.max(0.1, Math.min(1.0, trace.strength + (Math.random() - 0.5) * 0.1)),
          retrievals: trace.retrievals + (Math.random() > 0.7 ? 1 : 0)
        })));

        // 更新神经振荡
        setNeuralOscillations(prev => prev.map(osc => ({
          ...osc,
          amplitude: Math.max(0.1, Math.min(1.0, osc.amplitude + (Math.random() - 0.5) * 0.1)),
          phase: (osc.phase + Math.random() * 0.5) % (2 * Math.PI),
          power: Math.max(0.1, Math.min(1.0, osc.power + (Math.random() - 0.5) * 0.1))
        })));

        // 更新意识水平
        setConsciousnessLevel(prev => Math.max(0.3, Math.min(1.0, prev + (Math.random() - 0.5) * 0.05)));
      }, 1000);

      return () => clearInterval(interval);
    }
  }, [isMonitoring]);

  return (
    <div className="space-y-6">
      {/* 页面标题和控制 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">脑状态监控</h1>
          <p className="mt-2 text-gray-600">实时监控大脑各区域状态和神经活动</p>
        </div>
        
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setIsMonitoring(!isMonitoring)}
            className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white ${
              isMonitoring ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
            }`}
          >
            {isMonitoring ? <Pause className="h-4 w-4 mr-2" /> : <Play className="h-4 w-4 mr-2" />}
            {isMonitoring ? '停止监控' : '开始监控'}
          </button>
          
          <button
            onClick={() => {
              setBrainRegions(prev => prev.map(region => ({
                ...region,
                activity: Math.random() * 100,
                connections: Math.floor(Math.random() * 2000) + 500,
                status: Math.random() > 0.7 ? 'processing' : 'active'
              })));
              setConsciousnessLevel(Math.random());
            }}
            className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            重置
          </button>
        </div>
      </div>

      {/* 整体状态 */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-2">
          <div className="bg-white shadow rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900 flex items-center">
                <Brain className="h-5 w-5 mr-2" />
                大脑状态总览
              </h3>
              <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium ${
                consciousnessLevel >= 0.8 ? 'bg-green-100 text-green-800' :
                consciousnessLevel >= 0.6 ? 'bg-yellow-100 text-yellow-800' :
                'bg-red-100 text-red-800'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  consciousnessLevel >= 0.8 ? 'bg-green-500' :
                  consciousnessLevel >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                }`} />
                意识水平: {(consciousnessLevel * 100).toFixed(1)}%
              </div>
            </div>
            
            <div className="relative h-48">
              <svg className="w-full h-full" viewBox="0 0 400 200">
                {brainRegions.map((region) => (
                  <g key={region.id}>
                    <circle
                      cx={region.position.x * 4}
                      cy={region.position.y}
                      r={Math.max(20, region.activity / 5)}
                      fill={getStatusColor(region.status)}
                      stroke="white"
                      strokeWidth="2"
                      className="cursor-pointer transition-all duration-200 hover:opacity-80"
                      onClick={() => setSelectedRegion(selectedRegion === region.id ? null : region.id)}
                    />
                    <text
                      x={region.position.x * 4}
                      y={region.position.y}
                      textAnchor="middle"
                      dominantBaseline="central"
                      className="fill-white text-xs font-medium pointer-events-none"
                    >
                      {region.name}
                    </text>
                  </g>
                ))}
              </svg>
            </div>
          </div>
        </div>

        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
            <Activity className="h-5 w-5 mr-2" />
            神经振荡
          </h3>
          <div className="space-y-3">
            {neuralOscillations.map((osc, index) => (
              <div key={osc.frequency} className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div 
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: getWaveColor(index) }}
                  />
                  <span className="text-sm text-gray-600">{osc.frequency}</span>
                </div>
                <span className="text-sm font-medium">{osc.power.toFixed(2)}</span>
              </div>
            ))}
          </div>
          
          {/* 振荡波形 */}
          <div className="mt-4 h-20">
            <svg className="w-full h-full" viewBox="0 0 200 80">
              {neuralOscillations.slice(0, 3).map((osc, index) => (
                <path
                  key={index}
                  d={`M 0 ${40 + index * 10} Q 25 ${40 + index * 10 + osc.amplitude * 20}, 50 ${40 + index * 10} T 100 ${40 + index * 10} T 150 ${40 + index * 10} T 200 ${40 + index * 10}`}
                  stroke={getWaveColor(index)}
                  strokeWidth="1"
                  fill="none"
                />
              ))}
            </svg>
          </div>
        </div>

        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
            <Zap className="h-5 w-5 mr-2" />
            记忆概况
          </h3>
          <div className="space-y-3">
            {memoryTraces.slice(0, 4).map((trace) => (
              <div key={trace.id} className="flex items-center justify-between">
                <span className="text-sm text-gray-600">{trace.type}</span>
                <span className="text-sm font-medium">{(trace.strength * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
          
          <div className="mt-4 pt-4 border-t border-gray-200">
            <div className="flex justify-between text-sm text-gray-600">
              <span>总记忆痕迹</span>
              <span className="font-medium">{memoryTraces.length}</span>
            </div>
            <div className="flex justify-between text-sm text-gray-600 mt-1">
              <span>平均强度</span>
              <span className="font-medium">
                {(memoryStrengthAverage * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 详细状态 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white shadow rounded-lg">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">区域详情</h3>
          </div>
          <div className="p-6">
            {selectedRegion ? (
              <div className="space-y-4">
                {(() => {
                  const region = brainRegions.find(r => r.id === selectedRegion);
                  return region ? (
                    <>
                      <div className="flex items-center space-x-4">
                        <div className={`w-4 h-4 rounded-full ${getStatusColor(region.status)}`} />
                        <div>
                          <h4 className="text-xl font-semibold text-gray-900">{region.name}</h4>
                          <p className="text-gray-600">{region.function}</p>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <span className="text-sm text-gray-600">活跃度</span>
                          <div className="mt-1">
                            <div className="flex justify-between text-sm">
                              <span>{region.activity.toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                              <div 
                                className={`h-2 rounded-full ${getActivityColor(region.activity).replace('text-', 'bg-')}`}
                                style={{ width: `${region.activity}%` }}
                              />
                            </div>
                          </div>
                        </div>
                        
                        <div>
                          <span className="text-sm text-gray-600">连接数</span>
                          <p className="text-lg font-semibold text-gray-900">{region.connections.toLocaleString()}</p>
                        </div>
                        
                        <div>
                          <span className="text-sm text-gray-600">神经元数</span>
                          <p className="text-lg font-semibold text-gray-900">{region.neurons.toLocaleString()}</p>
                        </div>
                        
                        <div>
                          <span className="text-sm text-gray-600">状态</span>
                          <p className="text-lg font-semibold capitalize text-gray-900">{region.status}</p>
                        </div>
                      </div>
                    </>
                  ) : null;
                })()}
              </div>
            ) : (
              <div className="text-center text-gray-500 py-8">
                点击大脑区域图中的任意区域查看详情
              </div>
            )}
          </div>
        </div>

        <div className="bg-white shadow rounded-lg">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">记忆痕迹详细</h3>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              {memoryTraces.map((trace) => (
                <div key={trace.id} className="border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-gray-900">{trace.type}</span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      trace.strength >= 0.8 ? 'bg-green-100 text-green-800' :
                      trace.strength >= 0.6 ? 'bg-yellow-100 text-yellow-800' :
                      'bg-red-100 text-red-800'
                    }`}>
                      强度: {(trace.strength * 100).toFixed(0)}%
                    </span>
                  </div>
                  
                  <div className="flex justify-between text-sm text-gray-600">
                    <span>年龄: {trace.age.toFixed(1)} 小时</span>
                    <span>检索次数: {trace.retrievals}</span>
                  </div>
                  
                  <div className="mt-2">
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full ${
                          trace.strength >= 0.8 ? 'bg-green-500' :
                          trace.strength >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${trace.strength * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
import React, { useState, useCallback, useMemo } from 'react';
import { 
  Brain, 
  Eye, 
  Ear, 
  Zap, 
  Database, 
  Settings,
  ArrowRight,
  ArrowDown,
  Info
} from 'lucide-react';

interface BrainRegion {
  id: string;
  name: string;
  function: string;
  position: { x: number; y: number };
  size: number;
  color: string;
  connections: string[];
  status: 'active' | 'idle' | 'processing';
  info: string;
}

export const SystemArchitecture: React.FC = () => {
  const [selectedRegion, setSelectedRegion] = useState<BrainRegion | null>(null);
  const [hoveredRegion, setHoveredRegion] = useState<string | null>(null);

  const brainRegions: BrainRegion[] = [
    {
      id: 'prefrontal',
      name: '前额叶',
      function: '执行控制',
      position: { x: 50, y: 15 },
      size: 80,
      color: '#3B82F6',
      connections: ['cortex', 'hippocampus'],
      status: 'active',
      info: '负责高级认知功能，包括决策制定、工作记忆和注意力控制'
    },
    {
      id: 'cortex',
      name: '皮层',
      function: '感知处理',
      position: { x: 25, y: 40 },
      size: 100,
      color: '#10B981',
      connections: ['prefrontal', 'hippocampus', 'thalamus'],
      status: 'active',
      info: '处理感觉输入，产生感知和认知体验'
    },
    {
      id: 'hippocampus',
      name: '海马体',
      function: '记忆巩固',
      position: { x: 75, y: 40 },
      size: 70,
      color: '#8B5CF6',
      connections: ['prefrontal', 'cortex', 'entorhinal'],
      status: 'processing',
      info: '将短期记忆转换为长期记忆，是记忆形成的关键区域'
    },
    {
      id: 'entorhinal',
      name: '内嗅皮层',
      function: '空间导航',
      position: { x: 75, y: 65 },
      size: 60,
      color: '#F59E0B',
      connections: ['hippocampus', 'thalamus'],
      status: 'active',
      info: '参与空间认知和记忆检索'
    },
    {
      id: 'thalamus',
      name: '丘脑',
      function: '信息中继',
      position: { x: 50, y: 75 },
      size: 50,
      color: '#EF4444',
      connections: ['cortex', 'entorhinal', 'prefrontal'],
      status: 'active',
      info: '大脑的信息中继站，控制意识状态'
    },
    {
      id: 'amygdala',
      name: '杏仁核',
      function: '情感处理',
      position: { x: 25, y: 65 },
      size: 55,
      color: '#EC4899',
      connections: ['cortex', 'hippocampus'],
      status: 'idle',
      info: '处理恐惧、情感记忆和情感学习'
    }
  ];

  const connectionPaths = [
    { from: 'prefrontal', to: 'cortex', type: 'control' },
    { from: 'prefrontal', to: 'hippocampus', type: 'memory' },
    { from: 'cortex', to: 'hippocampus', type: 'input' },
    { from: 'hippocampus', to: 'entorhinal', type: 'output' },
    { from: 'entorhinal', to: 'thalamus', type: 'relay' },
    { from: 'thalamus', to: 'cortex', type: 'feedback' },
    { from: 'amygdala', to: 'hippocampus', type: 'emotional' },
    { from: 'amygdala', to: 'cortex', type: 'emotional' }
  ];

  // 使用useCallback优化状态相关函数
  const getRegionStatus = useCallback((status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500';
      case 'processing': return 'bg-yellow-500';
      case 'idle': return 'bg-gray-400';
      default: return 'bg-gray-400';
    }
  }, []);

  const getConnectionColor = useCallback((type: string) => {
    switch (type) {
      case 'control': return '#3B82F6';
      case 'memory': return '#8B5CF6';
      case 'input': return '#10B981';
      case 'output': return '#F59E0B';
      case 'relay': return '#EF4444';
      case 'feedback': return '#6B7280';
      case 'emotional': return '#EC4899';
      default: return '#6B7280';
    }
  }, []);

  // 使用useMemo优化connectionPaths的渲染
  const renderedConnections = useMemo(() => {
    return connectionPaths.map((connection, index) => {
      const fromRegion = brainRegions.find(r => r.id === connection.from);
      const toRegion = brainRegions.find(r => r.id === connection.to);
      
      if (!fromRegion || !toRegion) return null;

      const x1 = fromRegion.position.x + (fromRegion.size / 400) * 100;
      const y1 = fromRegion.position.y + (fromRegion.size / 400) * 100;
      const x2 = toRegion.position.x + (toRegion.size / 400) * 100;
      const y2 = toRegion.position.y + (toRegion.size / 400) * 100;

      return (
        <g key={index}>
          <defs>
            <marker
              id={`arrowhead-${index}`}
              markerWidth="10"
              markerHeight="7"
              refX="9"
              refY="3.5"
              orient="auto"
            >
              <polygon
                points="0 0, 10 3.5, 0 7"
                fill={getConnectionColor(connection.type)}
                opacity="0.8"
              />
            </marker>
          </defs>
          <line
            x1={`${x1}%`}
            y1={`${y1}%`}
            x2={`${x2}%`}
            y2={`${y2}%`}
            stroke={getConnectionColor(connection.type)}
            strokeWidth="2"
            opacity="0.7"
            markerEnd={`url(#arrowhead-${index})`}
          />
        </g>
      );
    });
  }, [connectionPaths, brainRegions, getConnectionColor]);



  return (
    <div className="space-y-6">
      {/* 页面标题 */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">系统架构</h1>
        <p className="mt-2 text-gray-600">大脑启发AI系统架构图和组件关系</p>
      </div>

      {/* 图例 */}
      <div className="bg-white shadow rounded-lg p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">连接类型</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { type: 'control', label: '控制信号', color: '#3B82F6' },
            { type: 'memory', label: '记忆流', color: '#8B5CF6' },
            { type: 'input', label: '输入流', color: '#10B981' },
            { type: 'output', label: '输出流', color: '#F59E0B' },
            { type: 'relay', label: '中继', color: '#EF4444' },
            { type: 'feedback', label: '反馈', color: '#6B7280' },
            { type: 'emotional', label: '情感', color: '#EC4899' }
          ].map((item) => (
            <div key={item.type} className="flex items-center space-x-2">
              <div 
                className="w-4 h-1 rounded"
                style={{ backgroundColor: item.color }}
              />
              <span className="text-sm text-gray-600">{item.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* 主要架构图 */}
      <div className="bg-white shadow rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">大脑区域架构</h3>
        </div>
        
        <div className="relative h-96 p-6">
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 400 400">
            {renderedConnections}
          </svg>
          
          {brainRegions.map((region) => (
            <div
              key={region.id}
              className={`absolute transform -translate-x-1/2 -translate-y-1/2 cursor-pointer transition-all duration-200 ${
                hoveredRegion === region.id ? 'scale-110 z-10' : 'z-0'
              }`}
              style={{
                left: `${region.position.x}%`,
                top: `${region.position.y}%`,
                width: `${region.size}px`,
                height: `${region.size}px`
              }}
              onMouseEnter={() => setHoveredRegion(region.id)}
              onMouseLeave={() => setHoveredRegion(null)}
              onClick={() => setSelectedRegion(region)}
            >
              <div 
                className={`w-full h-full rounded-full border-4 border-white shadow-lg ${
                  getRegionStatus(region.status)
                }`}
                style={{ backgroundColor: region.color }}
              >
                <div className="flex items-center justify-center h-full text-white font-medium text-sm text-center px-2">
                  {region.name}
                </div>
              </div>
              
              {/* 状态指示器 */}
              <div className={`absolute -top-1 -right-1 w-4 h-4 rounded-full border-2 border-white ${getRegionStatus(region.status)}`} />
            </div>
          ))}
        </div>
      </div>

      {/* 区域详情 */}
      {selectedRegion && (
        <div className="bg-white shadow rounded-lg p-6">
          <div className="flex items-start justify-between">
            <div className="flex items-center space-x-4">
              <div 
                className="w-12 h-12 rounded-full"
                style={{ backgroundColor: selectedRegion.color }}
              />
              <div>
                <h3 className="text-xl font-semibold text-gray-900">{selectedRegion.name}</h3>
                <p className="text-gray-600">{selectedRegion.function}</p>
              </div>
            </div>
            <button
              onClick={() => setSelectedRegion(null)}
              className="text-gray-400 hover:text-gray-600"
            >
              ×
            </button>
          </div>
          
          <div className="mt-4">
            <p className="text-gray-700">{selectedRegion.info}</p>
          </div>
          
          <div className="mt-4">
            <h4 className="font-medium text-gray-900 mb-2">连接区域</h4>
            <div className="flex flex-wrap gap-2">
              {selectedRegion.connections.map((connectionId) => {
                const connectedRegion = brainRegions.find(r => r.id === connectionId);
                return connectedRegion ? (
                  <span
                    key={connectionId}
                    className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800"
                  >
                    {connectedRegion.name}
                  </span>
                ) : null;
              })}
            </div>
          </div>
        </div>
      )}

      {/* 架构说明 */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
        <div className="flex">
          <Info className="h-5 w-5 text-blue-400 mt-0.5" />
          <div className="ml-3">
            <h3 className="text-sm font-medium text-blue-800">架构说明</h3>
            <div className="mt-2 text-sm text-blue-700">
              <ul className="list-disc list-inside space-y-1">
                <li>前额叶负责高级认知控制和决策制定</li>
                <li>皮层处理感觉输入和感知生成</li>
                <li>海马体是记忆巩固的核心区域</li>
                <li>内嗅皮层参与空间认知和记忆检索</li>
                <li>丘脑作为信息中继站，控制意识状态</li>
                <li>杏仁核处理情感和恐惧记忆</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
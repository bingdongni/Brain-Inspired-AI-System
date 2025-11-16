import React from 'react';

interface StatusIndicatorProps {
  status: 'active' | 'idle' | 'processing' | 'error';
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  label?: string;
}

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({ 
  status, 
  size = 'md', 
  showLabel = false, 
  label 
}) => {
  const getStatusColor = () => {
    switch (status) {
      case 'active': return 'bg-green-500';
      case 'processing': return 'bg-yellow-500';
      case 'idle': return 'bg-gray-400';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-400';
    }
  };

  const getSizeClasses = () => {
    switch (size) {
      case 'sm': return 'w-2 h-2';
      case 'lg': return 'w-6 h-6';
      default: return 'w-4 h-4';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'active': return '活跃';
      case 'processing': return '处理中';
      case 'idle': return '空闲';
      case 'error': return '错误';
      default: return '未知';
    }
  };

  return (
    <div className="flex items-center">
      <div className={`${getSizeClasses()} rounded-full ${getStatusColor()}`} />
      {showLabel && (
        <span className="ml-2 text-sm text-gray-600">
          {label || getStatusText()}
        </span>
      )}
    </div>
  );
};
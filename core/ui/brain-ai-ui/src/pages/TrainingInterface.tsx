import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  Play, 
  Pause, 
  Square, 
  Settings, 
  BarChart3,
  Clock,
  TrendingUp,
  AlertCircle,
  CheckCircle
} from 'lucide-react';

interface TrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate: number;
  optimizer: string;
  lossFunction: string;
  validationSplit: number;
}

interface TrainingMetrics {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  trainAccuracy: number;
  valAccuracy: number;
  learningRate: number;
  epochTime: number;
}

export const TrainingInterface: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [config, setConfig] = useState<TrainingConfig>({
    epochs: 100,
    batchSize: 32,
    learningRate: 0.001,
    optimizer: 'Adam',
    lossFunction: 'Cross Entropy',
    validationSplit: 0.2
  });
  
  const [metrics, setMetrics] = useState<TrainingMetrics>({
    epoch: 0,
    trainLoss: 0,
    valLoss: 0,
    trainAccuracy: 0,
    valAccuracy: 0,
    learningRate: 0.001,
    epochTime: 0
  });

  const [trainingHistory, setTrainingHistory] = useState<TrainingMetrics[]>([]);
  const [trainingSpeed, setTrainingSpeed] = useState(1);

  // 使用useCallback优化控制函数
  const startTraining = useCallback(() => {
    setIsTraining(true);
    setCurrentEpoch(0);
    setTrainingHistory([]);
  }, []);

  const pauseTraining = useCallback(() => {
    setIsTraining(false);
  }, []);

  const stopTraining = useCallback(() => {
    setIsTraining(false);
    setCurrentEpoch(0);
  }, []);

  // 使用useMemo优化计算密集型数据
  const progressPercentage = useMemo(() => {
    return (currentEpoch / config.epochs) * 100;
  }, [currentEpoch, config.epochs]);

  const generateMetrics = useCallback((epoch: number): TrainingMetrics => {
    return {
      epoch: epoch + 1,
      trainLoss: Math.max(0.1, 2.0 * Math.exp(-(epoch + 1) * 0.05) + Math.random() * 0.1),
      valLoss: Math.max(0.15, 2.2 * Math.exp(-(epoch + 1) * 0.04) + Math.random() * 0.15),
      trainAccuracy: Math.min(0.95, 0.5 + (epoch + 1) * 0.01 + Math.random() * 0.02),
      valAccuracy: Math.min(0.93, 0.48 + (epoch + 1) * 0.009 + Math.random() * 0.03),
      learningRate: config.learningRate * Math.pow(0.95, epoch / 10),
      epochTime: 2.5 + Math.random() * 2
    };
  }, [config.learningRate]);

  useEffect(() => {
    if (isTraining && currentEpoch < config.epochs) {
      const timer = setTimeout(() => {
        const newMetrics = generateMetrics(currentEpoch);

        setMetrics(newMetrics);
        setTrainingHistory(prev => [...prev, newMetrics]);
        setCurrentEpoch(prev => prev + 1);

        if (currentEpoch + 1 >= config.epochs) {
          setIsTraining(false);
        }
      }, 2000 / trainingSpeed);

      return () => clearTimeout(timer);
    }
  }, [isTraining, currentEpoch, config, trainingSpeed, generateMetrics]);

  // 使用useCallback优化状态相关函数
  const getStatusIcon = useCallback(() => {
    if (isTraining) return <Play className="h-5 w-5 text-green-600" />;
    if (currentEpoch >= config.epochs) return <CheckCircle className="h-5 w-5 text-blue-600" />;
    return <Pause className="h-5 w-5 text-yellow-600" />;
  }, [isTraining, currentEpoch, config.epochs]);

  const getStatusText = useCallback(() => {
    if (isTraining) return '训练中...';
    if (currentEpoch >= config.epochs) return '训练完成';
    return '训练已暂停';
  }, [isTraining, currentEpoch, config.epochs]);

  const getStatusColor = useCallback(() => {
    if (isTraining) return 'text-green-600';
    if (currentEpoch >= config.epochs) return 'text-blue-600';
    return 'text-yellow-600';
  }, [isTraining, currentEpoch, config.epochs]);

  return (
    <div className="space-y-6">
      {/* 页面标题 */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">模型训练</h1>
        <p className="mt-2 text-gray-600">交互式模型训练和参数调整</p>
      </div>

      {/* 训练控制 */}
      <div className="bg-white shadow rounded-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            {getStatusIcon()}
            <span className={`text-lg font-medium ${getStatusColor()}`}>
              {getStatusText()}
            </span>
          </div>
          
          <div className="flex space-x-3">
            <button
              onClick={startTraining}
              disabled={isTraining || currentEpoch >= config.epochs}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
            >
              <Play className="h-4 w-4 mr-2" />
              开始训练
            </button>
            
            <button
              onClick={pauseTraining}
              disabled={!isTraining}
              className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md shadow-sm text-gray-700 bg-white hover:bg-gray-50 disabled:bg-gray-100"
            >
              <Pause className="h-4 w-4 mr-2" />
              暂停
            </button>
            
            <button
              onClick={stopTraining}
              className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md shadow-sm text-gray-700 bg-white hover:bg-gray-50"
            >
              <Square className="h-4 w-4 mr-2" />
              停止
            </button>
          </div>
        </div>

        {/* 进度条 */}
        <div className="mb-4">
          <div className="flex justify-between text-sm text-gray-600 mb-2">
            <span>训练进度</span>
            <span>{currentEpoch} / {config.epochs} epochs</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-blue-600 h-2 rounded-full transition-all duration-500"
              style={{ width: `${progressPercentage}%` }}
            />
          </div>
        </div>

        {/* 实时指标 */}
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{metrics.trainLoss.toFixed(3)}</div>
            <div className="text-sm text-gray-600">训练损失</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{metrics.valLoss.toFixed(3)}</div>
            <div className="text-sm text-gray-600">验证损失</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{(metrics.trainAccuracy * 100).toFixed(1)}%</div>
            <div className="text-sm text-gray-600">训练准确率</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{(metrics.valAccuracy * 100).toFixed(1)}%</div>
            <div className="text-sm text-gray-600">验证准确率</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">{metrics.learningRate.toFixed(6)}</div>
            <div className="text-sm text-gray-600">学习率</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">{metrics.epochTime.toFixed(1)}s</div>
            <div className="text-sm text-gray-600">训练时间</div>
          </div>
        </div>
      </div>

      {/* 参数配置 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
            <Settings className="h-5 w-5 mr-2" />
            训练参数
          </h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">训练轮数</label>
              <input
                type="number"
                value={config.epochs}
                onChange={(e) => setConfig(prev => ({ ...prev, epochs: parseInt(e.target.value) || 100 }))}
                className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                min="1"
                max="1000"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700">批次大小</label>
              <select
                value={config.batchSize}
                onChange={(e) => setConfig(prev => ({ ...prev, batchSize: parseInt(e.target.value) }))}
                className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
              >
                <option value={16}>16</option>
                <option value={32}>32</option>
                <option value={64}>64</option>
                <option value={128}>128</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700">学习率</label>
              <input
                type="number"
                step="0.0001"
                value={config.learningRate}
                onChange={(e) => setConfig(prev => ({ ...prev, learningRate: parseFloat(e.target.value) || 0.001 }))}
                className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                min="0.0001"
                max="0.1"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700">优化器</label>
              <select
                value={config.optimizer}
                onChange={(e) => setConfig(prev => ({ ...prev, optimizer: e.target.value }))}
                className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="Adam">Adam</option>
                <option value="SGD">SGD</option>
                <option value="RMSprop">RMSprop</option>
                <option value="Adagrad">Adagrad</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700">损失函数</label>
              <select
                value={config.lossFunction}
                onChange={(e) => setConfig(prev => ({ ...prev, lossFunction: e.target.value }))}
                className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="Cross Entropy">Cross Entropy</option>
                <option value="MSE">Mean Squared Error</option>
                <option value="MAE">Mean Absolute Error</option>
                <option value="Huber">Huber Loss</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700">验证集比例</label>
              <input
                type="number"
                step="0.1"
                value={config.validationSplit}
                onChange={(e) => setConfig(prev => ({ ...prev, validationSplit: parseFloat(e.target.value) || 0.2 }))}
                className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                min="0.1"
                max="0.5"
              />
            </div>
          </div>
        </div>

        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4 flex items-center">
            <BarChart3 className="h-5 w-5 mr-2" />
            训练历史
          </h3>
          
          {/* 训练速度控制 */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">训练速度</label>
            <input
              type="range"
              min="1"
              max="10"
              value={trainingSpeed}
              onChange={(e) => setTrainingSpeed(parseInt(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-sm text-gray-600 mt-1">
              <span>慢</span>
              <span>{trainingSpeed}x</span>
              <span>快</span>
            </div>
          </div>
          
          {/* 训练图表 */}
          <div className="h-64">
            {trainingHistory.length > 0 ? (
              <div className="space-y-4">
                <div className="flex items-end space-x-1 h-32">
                  {trainingHistory.slice(-20).map((metric, index) => (
                    <div
                      key={index}
                      className="bg-blue-500 rounded-t flex-1"
                      style={{ height: `${Math.max(5, (1 - metric.trainLoss) * 100)}%` }}
                      title={`Epoch ${metric.epoch}: ${metric.trainLoss.toFixed(3)}`}
                    />
                  ))}
                </div>
                <div className="flex items-end space-x-1 h-32">
                  {trainingHistory.slice(-20).map((metric, index) => (
                    <div
                      key={index}
                      className="bg-green-500 rounded-t flex-1"
                      style={{ height: `${Math.max(5, metric.trainAccuracy * 100)}%` }}
                      title={`Epoch ${metric.epoch}: ${(metric.trainAccuracy * 100).toFixed(1)}%`}
                    />
                  ))}
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                开始训练以显示历史数据
              </div>
            )}
          </div>
          
          {/* 图例 */}
          <div className="flex justify-center space-x-4 mt-4 text-sm">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-blue-500 rounded mr-2"></div>
              <span>训练损失</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-green-500 rounded mr-2"></div>
              <span>训练准确率</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
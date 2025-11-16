import React, { useState, useCallback, useMemo } from 'react';
import { 
  BookOpen, 
  Play, 
  CheckCircle, 
  ChevronRight,
  Lightbulb,
  Code,
  Eye,
  Users,
  Star,
  ArrowRight,
  ArrowLeft,
  Clock,
  Target
} from 'lucide-react';

interface Tutorial {
  id: string;
  title: string;
  description: string;
  duration: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  category: string;
  steps: TutorialStep[];
  prerequisites?: string[];
  completed?: boolean;
}

interface TutorialStep {
  id: string;
  title: string;
  content: string;
  code?: string;
  image?: string;
  interactive?: boolean;
}

interface ExampleDemo {
  id: string;
  title: string;
  description: string;
  type: 'visual' | 'interactive' | 'code';
  imageUrl?: string;
  codeExample?: string;
}

export const Tutorials: React.FC = () => {
  const [selectedTutorial, setSelectedTutorial] = useState<Tutorial | null>(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [completedTutorials, setCompletedTutorials] = useState<string[]>([]);

  const tutorials: Tutorial[] = [
    {
      id: '1',
      title: '脑启发AI系统基础',
      description: '了解大脑启发AI系统的基本概念、架构和工作原理',
      duration: '30分钟',
      difficulty: 'beginner',
      category: '基础概念',
      completed: false,
      prerequisites: [],
      steps: [
        {
          id: '1-1',
          title: '什么是脑启发AI',
          content: '脑启发AI是受生物大脑结构和功能启发的机器学习方法。它模拟人脑的神经连接、记忆形成和认知过程，以实现更高效的智能学习。',
          interactive: false
        },
        {
          id: '1-2',
          title: '系统架构概览',
          content: '我们的系统包含6个主要大脑区域：前额叶、皮层、海马体、内嗅皮层、丘脑和杏仁核。每个区域都有特定的功能和连接方式。',
          code: `// 示例代码：创建大脑系统
from brain_ai import BrainSystem

# 初始化大脑系统
brain = BrainSystem()

# 添加大脑区域
brain.add_region('prefrontal', config)
brain.add_region('cortex', config)
brain.add_region('hippocampus', config)`,
          interactive: true
        },
        {
          id: '1-3',
          title: '神经连接与突触',
          content: '大脑区域通过突触连接进行通信。这些连接具有可塑性，可以根据使用频率调整强度，模拟生物大脑的学习过程。',
          interactive: false
        }
      ]
    },
    {
      id: '2',
      title: '记忆系统与检索',
      description: '深入了解海马体的记忆形成机制和不同类型记忆的处理',
      duration: '45分钟',
      difficulty: 'intermediate',
      category: '记忆机制',
      completed: false,
      prerequisites: ['1'],
      steps: [
        {
          id: '2-1',
          title: '记忆类型分类',
          content: '系统支持5种记忆类型：情景记忆、语义记忆、程序记忆、工作记忆和情感记忆。每种记忆都有不同的形成和检索机制。',
          interactive: true
        },
        {
          id: '2-2',
          title: '记忆巩固过程',
          content: '海马体将短期记忆转换为长期记忆，通过与皮层的反复交互实现记忆巩固。',
          code: `# 记忆创建示例
memory_trace = brain.create_memory_trace(
    memory_type=MemoryType.EPISODIC,
    activation_pattern=pattern,
    associated_regions=[BrainRegion.HIPPOCAMPUS, BrainRegion.CORTEX]
)`,
          interactive: false
        },
        {
          id: '2-3',
          title: '模式补全',
          content: '海马体具有模式补全能力，即使输入不完整，也能检索出完整的记忆模式。',
          interactive: true
        }
      ]
    },
    {
      id: '3',
      title: '训练与优化',
      description: '学习如何使用训练框架优化神经网络参数',
      duration: '60分钟',
      difficulty: 'intermediate',
      category: '模型训练',
      completed: false,
      prerequisites: ['1', '2'],
      steps: [
        {
          id: '3-1',
          title: '配置训练参数',
          content: '设置学习率、批次大小、优化器等关键训练参数。合理的参数配置对训练效果至关重要。',
          code: `config = TrainingConfig(
    model=network,
    batch_size=32,
    learning_rate=0.001,
    optimizer=OptimizerType.ADAM,
    loss_function=LossFunction.CROSS_ENTROPY
)`,
          interactive: true
        },
        {
          id: '3-2',
          title: '执行训练循环',
          content: '运行完整的训练流程，监控损失函数和准确率的变化。',
          interactive: false
        },
        {
          id: '3-3',
          title: '监控训练进度',
          content: '实时监控训练指标，识别过拟合和欠拟合问题。',
          interactive: true
        }
      ]
    },
    {
      id: '4',
      title: '高级功能应用',
      description: '探索系统的高级功能，包括持续学习和迁移学习',
      duration: '90分钟',
      difficulty: 'advanced',
      category: '高级应用',
      completed: false,
      prerequisites: ['1', '2', '3'],
      steps: [
        {
          id: '4-1',
          title: '持续学习',
          content: '实现新任务学习而不遗忘旧知识的能力，模拟生物大脑的持续适应特性。',
          interactive: true
        },
        {
          id: '4-2',
          title: '迁移学习',
          content: '利用已学知识加速新任务的学习过程。',
          code: `# 迁移学习配置
transfer_config = create_transfer_learning_config(
    model=target_model,
    pre_trained_model=source_model
)`,
          interactive: false
        },
        {
          id: '4-3',
          title: '元学习应用',
          content: '学会如何学习，快速适应新环境和任务。',
          interactive: true
        }
      ]
    }
  ];

  const exampleDemos: ExampleDemo[] = [
    {
      id: 'demo-1',
      title: '神经活动可视化',
      description: '实时观察神经元激活模式和大脑振荡活动',
      type: 'visual',
      imageUrl: 'https://via.placeholder.com/400x250/3B82F6/FFFFFF?text=神经活动可视化'
    },
    {
      id: 'demo-2',
      title: '记忆形成演示',
      description: '观察记忆痕迹在大脑中的形成和巩固过程',
      type: 'interactive',
      imageUrl: 'https://via.placeholder.com/400x250/10B981/FFFFFF?text=记忆形成演示'
    },
    {
      id: 'demo-3',
      title: '训练过程代码示例',
      description: '完整的模型训练代码实现和参数调优',
      type: 'code',
      codeExample: `import numpy as np
from brain_ai import BrainSystem, TrainingConfig

# 初始化系统
brain = BrainSystem()

# 创建训练配置
config = TrainingConfig(
    model=my_model,
    epochs=100,
    learning_rate=0.001
)

# 开始训练
framework = TrainingFramework(config)
results = framework.train(X_train, y_train)
print(f"最终准确率: {results['test_accuracy']:.3f}")`
    }
  ];

  // 使用useCallback优化函数
  const startTutorial = useCallback((tutorial: Tutorial) => {
    setSelectedTutorial(tutorial);
    setCurrentStep(0);
  }, []);

  const nextStep = useCallback(() => {
    if (selectedTutorial && currentStep < selectedTutorial.steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else if (selectedTutorial) {
      // 完成教程
      setCompletedTutorials(prev => [...prev, selectedTutorial.id]);
      setSelectedTutorial(null);
      setCurrentStep(0);
    }
  }, [selectedTutorial, currentStep]);

  const prevStep = useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  }, [currentStep]);

  const getDifficultyColor = useCallback((difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-800';
      case 'intermediate': return 'bg-yellow-100 text-yellow-800';
      case 'advanced': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  }, []);

  const getDifficultyText = useCallback((difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return '初级';
      case 'intermediate': return '中级';
      case 'advanced': return '高级';
      default: return '未知';
    }
  }, []);

  // 使用useMemo优化教程列表
  const availableTutorials = useMemo(() => {
    return tutorials.filter(tutorial => {
      if (!tutorial.prerequisites || tutorial.prerequisites.length === 0) {
        return true;
      }
      return tutorial.prerequisites.every(prereqId => 
        completedTutorials.includes(prereqId)
      );
    });
  }, [tutorials, completedTutorials]);

  return (
    <div className="space-y-6">
      {/* 页面标题 */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">教程与演示</h1>
        <p className="mt-2 text-gray-600">学习如何使用脑启发AI系统的完整指南</p>
      </div>

      {!selectedTutorial ? (
        <>
          {/* 教程列表 */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {availableTutorials.map((tutorial) => (
              <div key={tutorial.id} className="bg-white shadow rounded-lg overflow-hidden">
                <div className="p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-2">
                        <h3 className="text-lg font-medium text-gray-900">{tutorial.title}</h3>
                        {completedTutorials.includes(tutorial.id) && (
                          <CheckCircle className="h-5 w-5 text-green-500" />
                        )}
                      </div>
                      <p className="text-gray-600 mb-4">{tutorial.description}</p>
                      
                      <div className="flex items-center space-x-4 text-sm text-gray-500 mb-4">
                        <div className="flex items-center">
                          <Clock className="h-4 w-4 mr-1" />
                          {tutorial.duration}
                        </div>
                        <div className="flex items-center">
                          <Target className="h-4 w-4 mr-1" />
                          {getDifficultyText(tutorial.difficulty)}
                        </div>
                        <div className="flex items-center">
                          <BookOpen className="h-4 w-4 mr-1" />
                          {tutorial.steps.length} 步骤
                        </div>
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getDifficultyColor(tutorial.difficulty)}`}>
                          {tutorial.category}
                        </span>
                        
                        <button
                          onClick={() => startTutorial(tutorial)}
                          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
                        >
                          <Play className="h-4 w-4 mr-2" />
                          开始学习
                        </button>
                      </div>
                    </div>
                  </div>
                  
                  {tutorial.prerequisites && tutorial.prerequisites.length > 0 && (
                    <div className="mt-4 pt-4 border-t border-gray-200">
                      <div className="flex items-start space-x-2">
                        <Lightbulb className="h-4 w-4 text-yellow-500 mt-0.5" />
                        <div>
                          <span className="text-sm font-medium text-gray-700">前置要求：</span>
                          <span className="text-sm text-gray-600 ml-1">
                            完成 {tutorial.prerequisites.map(id => {
                              const prereq = availableTutorials.find(t => t.id === id) || tutorials.find(t => t.id === id);
                              return prereq ? prereq.title : id;
                            }).join(', ')}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* 示例演示 */}
          <div className="bg-white shadow rounded-lg">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-medium text-gray-900 flex items-center">
                <Eye className="h-5 w-5 mr-2" />
                示例演示
              </h3>
            </div>
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {exampleDemos.map((demo) => (
                  <div key={demo.id} className="border rounded-lg overflow-hidden">
                    <div className="h-40 bg-gray-100 flex items-center justify-center">
                      {demo.type === 'visual' && <Eye className="h-12 w-12 text-gray-400" />}
                      {demo.type === 'interactive' && <Users className="h-12 w-12 text-gray-400" />}
                      {demo.type === 'code' && <Code className="h-12 w-12 text-gray-400" />}
                    </div>
                    <div className="p-4">
                      <h4 className="font-medium text-gray-900 mb-2">{demo.title}</h4>
                      <p className="text-sm text-gray-600 mb-4">{demo.description}</p>
                      <button className="w-full inline-flex items-center justify-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50">
                        <Play className="h-4 w-4 mr-2" />
                        查看演示
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      ) : (
        /* 教程内容 */
        <div className="max-w-4xl mx-auto">
          <div className="bg-white shadow rounded-lg">
            {/* 教程头部 */}
            <div className="px-6 py-4 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">{selectedTutorial.title}</h2>
                  <p className="text-gray-600">{selectedTutorial.description}</p>
                </div>
                <div className="flex items-center space-x-3">
                  <span className="text-sm text-gray-500">
                    步骤 {currentStep + 1} / {selectedTutorial.steps.length}
                  </span>
                  <button
                    onClick={() => setSelectedTutorial(null)}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    ×
                  </button>
                </div>
              </div>
              
              {/* 进度条 */}
              <div className="mt-4">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${((currentStep + 1) / selectedTutorial.steps.length) * 100}%` }}
                  />
                </div>
              </div>
            </div>

            {/* 教程步骤内容 */}
            <div className="p-6">
              {(() => {
                const step = selectedTutorial.steps[currentStep];
                return (
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-lg font-medium text-gray-900 mb-4">{step.title}</h3>
                      <div className="prose max-w-none">
                        <p className="text-gray-700">{step.content}</p>
                      </div>
                    </div>

                    {step.code && (
                      <div>
                        <h4 className="text-md font-medium text-gray-900 mb-2 flex items-center">
                          <Code className="h-4 w-4 mr-2" />
                          代码示例
                        </h4>
                        <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
                          <pre className="text-sm text-green-400">
                            <code>{step.code}</code>
                          </pre>
                        </div>
                      </div>
                    )}

                    {step.interactive && (
                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <div className="flex items-start">
                          <Lightbulb className="h-5 w-5 text-blue-600 mt-0.5 mr-3" />
                          <div>
                            <h4 className="text-md font-medium text-blue-900 mb-2">互动练习</h4>
                            <p className="text-blue-800">这个步骤包含互动练习，请在右侧面板中操作相关控件。</p>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* 导航按钮 */}
                    <div className="flex justify-between items-center pt-6 border-t border-gray-200">
                      <button
                        onClick={prevStep}
                        disabled={currentStep === 0}
                        className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:bg-gray-100 disabled:cursor-not-allowed"
                      >
                        <ArrowLeft className="h-4 w-4 mr-2" />
                        上一步
                      </button>

                      <div className="flex space-x-2">
                        {selectedTutorial.steps.map((_, index) => (
                          <div
                            key={index}
                            className={`w-3 h-3 rounded-full ${
                              index <= currentStep ? 'bg-blue-600' : 'bg-gray-300'
                            }`}
                          />
                        ))}
                      </div>

                      <button
                        onClick={nextStep}
                        className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
                      >
                        {currentStep === selectedTutorial.steps.length - 1 ? '完成教程' : '下一步'}
                        {currentStep === selectedTutorial.steps.length - 1 ? (
                          <CheckCircle className="h-4 w-4 ml-2" />
                        ) : (
                          <ArrowRight className="h-4 w-4 ml-2" />
                        )}
                      </button>
                    </div>
                  </div>
                );
              })()}
            </div>
          </div>
        </div>
      )}

      {/* 帮助文档 */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">快速参考</h3>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <h4 className="font-medium text-gray-900 mb-3">常用API</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li><code className="bg-gray-100 px-1 rounded">BrainSystem()</code> - 初始化大脑系统</li>
                <li><code className="bg-gray-100 px-1 rounded">add_region()</code> - 添加大脑区域</li>
                <li><code className="bg-gray-100 px-1 rounded">process_input()</code> - 处理输入</li>
                <li><code className="bg-gray-100 px-1 rounded">create_memory_trace()</code> - 创建记忆痕迹</li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-medium text-gray-900 mb-3">配置参数</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li><code className="bg-gray-100 px-1 rounded">learning_rate</code> - 学习率 (0.0001-0.1)</li>
                <li><code className="bg-gray-100 px-1 rounded">batch_size</code> - 批次大小 (16-128)</li>
                <li><code className="bg-gray-100 px-1 rounded">epochs</code> - 训练轮数 (1-1000)</li>
                <li><code className="bg-gray-100 px-1 rounded">optimizer</code> - 优化器类型</li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-medium text-gray-900 mb-3">故障排除</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li>内存不足 → 减少批次大小</li>
                <li>训练缓慢 → 使用GPU加速</li>
                <li>过拟合 → 增加正则化</li>
                <li>收敛困难 → 调整学习率</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
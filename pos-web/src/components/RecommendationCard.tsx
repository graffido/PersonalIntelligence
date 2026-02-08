import React from 'react';
import { motion } from 'framer-motion';
import type { Recommendation } from '@types/index';
import {
  LightBulbIcon,
  LinkIcon,
  BoltIcon,
  BellIcon,
  XMarkIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline';

interface RecommendationCardProps {
  recommendation: Recommendation;
  onDismiss?: (id: string) => void;
  onAction?: (recommendation: Recommendation) => void;
  className?: string;
}

const getTypeIcon = (type: Recommendation['type']) => {
  switch (type) {
    case 'connection':
      return <LinkIcon className="w-5 h-5" />;
    case 'insight':
      return <LightBulbIcon className="w-5 h-5" />;
    case 'action':
      return <BoltIcon className="w-5 h-5" />;
    case 'reminder':
      return <BellIcon className="w-5 h-5" />;
    default:
      return <LightBulbIcon className="w-5 h-5" />;
  }
};

const getTypeColor = (type: Recommendation['type']) => {
  switch (type) {
    case 'connection':
      return 'bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-900/20 dark:text-blue-300';
    case 'insight':
      return 'bg-yellow-50 text-yellow-700 border-yellow-200 dark:bg-yellow-900/20 dark:text-yellow-300';
    case 'action':
      return 'bg-green-50 text-green-700 border-green-200 dark:bg-green-900/20 dark:text-green-300';
    case 'reminder':
      return 'bg-purple-50 text-purple-700 border-purple-200 dark:bg-purple-900/20 dark:text-purple-300';
    default:
      return 'bg-gray-50 text-gray-700 border-gray-200';
  }
};

const getTypeLabel = (type: Recommendation['type']) => {
  switch (type) {
    case 'connection':
      return '关联发现';
    case 'insight':
      return '洞察';
    case 'action':
      return '建议行动';
    case 'reminder':
      return '提醒';
    default:
      return '推荐';
  }
};

export const RecommendationCard: React.FC<RecommendationCardProps> = ({
  recommendation,
  onDismiss,
  onAction,
  className = '',
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, x: -100 }}
      className={`relative bg-white dark:bg-dark-800 rounded-xl border border-gray-200 dark:border-dark-700 shadow-sm hover:shadow-md transition-shadow ${className}`}
    >
      {/* 头部 */}
      <div className={`flex items-center gap-3 px-4 py-3 rounded-t-xl ${getTypeColor(recommendation.type)}`}>
        <div className="p-1.5 bg-white/50 dark:bg-black/20 rounded-lg">
          {getTypeIcon(recommendation.type)}
        </div>
        <div className="flex-1">
          <span className="text-xs font-medium uppercase tracking-wide opacity-75">
            {getTypeLabel(recommendation.type)}
          </span>
          <p className="font-semibold">{recommendation.title}</p>
        </div>
        
        <button
          onClick={() => onDismiss?.(recommendation.id)}
          className="p-1 rounded-lg hover:bg-black/10 dark:hover:bg-white/10 transition-colors"
        >
          <XMarkIcon className="w-5 h-5" />
        </button>
      </div>
      
      {/* 内容 */}
      <div className="p-4">
        <p className="text-gray-600 dark:text-gray-300 mb-3">{recommendation.description}</p>
        
        {/* 相关实体 */}
        {recommendation.related_entities.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            <span className="text-sm text-gray-500">相关:</span>
            {recommendation.related_entities.map((entityId) => (
              <span
                key={entityId}
                className="px-2 py-0.5 bg-gray-100 dark:bg-dark-700 text-xs rounded-full"
              >
                {entityId.slice(0, 8)}...
              </span>
            ))}
          </div>
        )}
        
        {/* 置信度 */}
        <div className="flex items-center gap-2 mb-3">
          <span className="text-xs text-gray-500">置信度</span>
          <div className="flex-1 h-1.5 bg-gray-200 dark:bg-dark-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-primary-500 rounded-full transition-all"
              style={{ width: `${recommendation.confidence * 100}%` }}
            />
          </div>
          <span className="text-xs font-medium">
            {Math.round(recommendation.confidence * 100)}%
          </span>
        </div>
        
        {/* 操作按钮 */}
        <button
          onClick={() => onAction?.(recommendation)}
          className="flex items-center gap-1 text-sm text-primary-600 hover:text-primary-700 font-medium"
        >
          查看详情
          <ChevronRightIcon className="w-4 h-4" />
        </button>
      </div>
    </motion.div>
  );
};

// 推荐列表组件
interface RecommendationsListProps {
  recommendations: Recommendation[];
  onDismiss: (id: string) => void;
  onAction: (recommendation: Recommendation) => void;
  onDismissAll: () => void;
  className?: string;
  maxItems?: number;
}

export const RecommendationsList: React.FC<RecommendationsListProps> = ({
  recommendations,
  onDismiss,
  onAction,
  onDismissAll,
  className = '',
  maxItems = 5,
}) => {
  const displayedRecommendations = recommendations.slice(0, maxItems);
  const hasMore = recommendations.length > maxItems;
  
  if (recommendations.length === 0) {
    return (
      <div className={`text-center py-8 text-gray-500 ${className}`}>
        <LightBulbIcon className="w-12 h-12 mx-auto mb-3 opacity-50" />
        <p>暂无推荐</p>
        <p className="text-sm">系统会根据你的记录生成个性化推荐</p>
      </div>
    );
  }
  
  return (
    <div className={`space-y-4 ${className}`}>
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-gray-900 dark:text-gray-100">
          智能推荐
          <span className="ml-2 px-2 py-0.5 bg-primary-100 text-primary-700 text-xs rounded-full">
            {recommendations.length}
          </span>
        </h3>
        
        <button
          onClick={onDismissAll}
          className="text-sm text-gray-500 hover:text-gray-700"
        >
          全部忽略
        </button>
      </div>
      
      <div className="space-y-3">
        {displayedRecommendations.map((rec) => (
          <RecommendationCard
            key={rec.id}
            recommendation={rec}
            onDismiss={onDismiss}
            onAction={onAction}
          />
        ))}
      </div>
      
      {hasMore && (
        <button className="w-full py-2 text-sm text-gray-500 hover:text-gray-700 border border-dashed border-gray-300 dark:border-dark-600 rounded-lg">
          查看全部 {recommendations.length} 条推荐
        </button>
      )}
    </div>
  );
};

export default RecommendationCard;

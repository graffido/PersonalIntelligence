import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { format, parseISO, isToday, isYesterday } from 'date-fns';
import { zhCN } from 'date-fns/locale';
import { motion } from 'framer-motion';
import type { TimelineEvent, Memory } from '@types/index';
import { CalendarIcon, MapPinIcon, TagIcon } from '@heroicons/react/24/outline';

interface TimelineViewProps {
  events: TimelineEvent[];
  memories: Memory[];
  onEventClick?: (event: TimelineEvent) => void;
  onMemoryClick?: (memory: Memory) => void;
  className?: string;
}

export const TimelineView: React.FC<TimelineViewProps> = ({
  events,
  memories,
  onEventClick,
  onMemoryClick,
  className = '',
}) => {
  const parentRef = useRef<HTMLDivElement>(null);
  const [groupedEvents, setGroupedEvents] = useState<Map<string, TimelineEvent[]>>(new Map());
  
  // 合并并按日期分组事件
  useEffect(() => {
    const allEvents = [
      ...events,
      ...memories.map((m) => ({
        id: m.id,
        timestamp: m.timestamp,
        title: m.content.slice(0, 50) + (m.content.length > 50 ? '...' : ''),
        description: m.content,
        type: m.type,
        data: m,
      })),
    ].sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
    
    const groups = new Map<string, TimelineEvent[]>();
    allEvents.forEach((event) => {
      const date = format(parseISO(event.timestamp), 'yyyy-MM-dd');
      if (!groups.has(date)) {
        groups.set(date, []);
      }
      groups.get(date)!.push(event);
    });
    
    setGroupedEvents(groups);
  }, [events, memories]);
  
  const dates = Array.from(groupedEvents.keys());
  
  const rowVirtualizer = useVirtualizer({
    count: dates.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 200,
    overscan: 5,
  });
  
  const formatDateLabel = (dateStr: string) => {
    const date = parseISO(dateStr);
    if (isToday(date)) return '今天';
    if (isYesterday(date)) return '昨天';
    return format(date, 'M月d日 EEEE', { locale: zhCN });
  };
  
  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'note': return '📝';
      case 'event': return '📅';
      case 'task': return '✅';
      case 'idea': return '💡';
      default: return '📌';
    }
  };
  
  const getTypeColor = (type: string) => {
    switch (type) {
      case 'note': return 'bg-blue-500';
      case 'event': return 'bg-green-500';
      case 'task': return 'bg-yellow-500';
      case 'idea': return 'bg-purple-500';
      default: return 'bg-gray-500';
    }
  };
  
  return (
    <div ref={parentRef} className={`h-full overflow-auto ${className}`}>
      <div
        style={{
          height: `${rowVirtualizer.getTotalSize()}px`,
          width: '100%',
          position: 'relative',
        }}
      >
        {rowVirtualizer.getVirtualItems().map((virtualRow) => {
          const date = dates[virtualRow.index];
          const dayEvents = groupedEvents.get(date) || [];
          
          return (
            <div
              key={date}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: `${virtualRow.size}px`,
                transform: `translateY(${virtualRow.start}px)`,
              }}
              className="px-4 py-2"
            >
              {/* 日期标题 */}
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 rounded-full bg-primary-100 dark:bg-primary-900 flex items-center justify-center">
                  <span className="text-lg">{format(parseISO(date), 'd')}</span>
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                    {formatDateLabel(date)}
                  </h3>
                  <span className="text-sm text-gray-500">
                    {dayEvents.length} 条记录
                  </span>
                </div>
              </div>
              
              {/* 时间线 */}
              <div className="relative pl-5">
                <div className="absolute left-[9px] top-0 bottom-0 w-0.5 bg-gray-200 dark:bg-dark-700" />
                
                {dayEvents.map((event, idx) => (
                  <motion.div
                    key={event.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.05 }}
                    onClick={() => event.data ? onMemoryClick?.(event.data as Memory) : onEventClick?.(event)}
                    className="relative mb-4 cursor-pointer group"
                  >
                    {/* 时间点 */}
                    <div
                      className={`absolute left-0 top-1.5 w-5 h-5 rounded-full border-2 border-white dark:border-dark-800 ${getTypeColor(event.type)}`}
                    />
                    
                    {/* 内容卡片 */}
                    <div className="ml-8 p-3 bg-white dark:bg-dark-800 rounded-lg border border-gray-200 dark:border-dark-700 hover:shadow-md transition-shadow">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span>{getTypeIcon(event.type)}</span>
                            <span className="text-xs text-gray-500">
                              {format(parseISO(event.timestamp), 'HH:mm')}
                            </span>
                          </div>
                          
                          <p className="text-gray-900 dark:text-gray-100">{event.title}</p>
                          
                          {event.description && (
                            <p className="text-sm text-gray-500 mt-1 line-clamp-2">{event.description}</p>
                          )}
                          
                          {/* 记忆详情 */}
                          {event.data && (
                            <div className="flex flex-wrap gap-2 mt-2">
                              {(event.data as Memory).location && (
                                <span className="flex items-center gap-1 text-xs text-gray-500">
                                  <MapPinIcon className="w-3 h-3" />
                                  {(event.data as Memory).location!.name || '有位置'}
                                </span>
                              )}
                              {(event.data as Memory).tags.map((tag) => (
                                <span
                                  key={tag}
                                  className="px-2 py-0.5 bg-gray-100 dark:bg-dark-700 text-xs rounded-full"
                                >
                                  #{tag}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default TimelineView;

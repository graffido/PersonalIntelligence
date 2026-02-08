import React, { useState, useRef, useCallback } from 'react';
import { MicrophoneIcon, PaperAirplaneIcon, PhotoIcon, MapPinIcon, ClockIcon } from '@heroicons/react/24/outline';
import { useVoiceInput, useGeolocation } from '@hooks/index';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';

interface UnifiedInputProps {
  onSubmit: (data: {
    content: string;
    type: 'note' | 'event' | 'task' | 'idea';
    location?: { lat: number; lng: number };
    attachments?: File[];
  }) => Promise<void>;
  placeholder?: string;
  className?: string;
}

export const UnifiedInput: React.FC<UnifiedInputProps> = ({
  onSubmit,
  placeholder = '记录你的想法、事件或任务...',
  className = '',
}) => {
  const [content, setContent] = useState('');
  const [selectedType, setSelectedType] = useState<'note' | 'event' | 'task' | 'idea'>('note');
  const [isExpanded, setIsExpanded] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [attachments, setAttachments] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const { isListening, transcript, interimTranscript, startListening, stopListening, reset, error } = useVoiceInput();
  const { location } = useGeolocation();
  
  // 合并语音转录到输入
  React.useEffect(() => {
    if (transcript) {
      setContent((prev) => prev + transcript);
      reset();
    }
  }, [transcript, reset]);
  
  const handleSubmit = useCallback(async () => {
    if (!content.trim() && attachments.length === 0) return;
    
    setIsSubmitting(true);
    try {
      await onSubmit({
        content: content.trim(),
        type: selectedType,
        location: location ? {
          lat: location.coords.latitude,
          lng: location.coords.longitude,
        } : undefined,
        attachments: attachments.length > 0 ? attachments : undefined,
      });
      
      setContent('');
      setAttachments([]);
      setIsExpanded(false);
      toast.success('已保存');
    } catch (error) {
      toast.error('保存失败，请重试');
    } finally {
      setIsSubmitting(false);
    }
  }, [content, selectedType, location, attachments, onSubmit]);
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      handleSubmit();
    }
  };
  
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    setAttachments((prev) => [...prev, ...files]);
  };
  
  const typeOptions = [
    { value: 'note', label: '笔记', icon: '📝', color: 'bg-blue-100 text-blue-700' },
    { value: 'event', label: '事件', icon: '📅', color: 'bg-green-100 text-green-700' },
    { value: 'task', label: '任务', icon: '✅', color: 'bg-yellow-100 text-yellow-700' },
    { value: 'idea', label: '想法', icon: '💡', color: 'bg-purple-100 text-purple-700' },
  ] as const;
  
  return (
    <motion.div
      initial={false}
      animate={{ height: isExpanded ? 'auto' : 'auto' }}
      className={`bg-white dark:bg-dark-800 rounded-xl shadow-lg border border-gray-200 dark:border-dark-700 ${className}`}
    >
      {/* 类型选择器 */}
      <div className="flex items-center gap-1 p-2 border-b border-gray-100 dark:border-dark-700">
        {typeOptions.map((option) => (
          <button
            key={option.value}
            onClick={() => setSelectedType(option.value)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
              selectedType === option.value
                ? option.color
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-dark-700'
            }`}
          >
            <span>{option.icon}</span>
            <span>{option.label}</span>
          </button>
        ))}
      </div>
      
      {/* 输入区域 */}
      <div className="p-3">
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          onFocus={() => setIsExpanded(true)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          rows={isExpanded ? 4 : 2}
          className="w-full resize-none border-0 bg-transparent text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:ring-0 p-0"
        />
        
        {/* 语音转录预览 */}
        <AnimatePresence>
          {interimTranscript && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="text-gray-500 dark:text-gray-400 text-sm mt-2 italic"
            >
              {interimTranscript}...
            </motion.div>
          )}
        </AnimatePresence>
        
        {/* 附件预览 */}
        {attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-3">
            {attachments.map((file, idx) => (
              <div
                key={idx}
                className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 dark:bg-dark-700 rounded-lg text-sm"
              >
                <PhotoIcon className="w-4 h-4 text-gray-500" />
                <span className="truncate max-w-[150px]">{file.name}</span>
                <button
                  onClick={() => setAttachments((prev) => prev.filter((_, i) => i !== idx))}
                  className="text-gray-400 hover:text-red-500"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}
        
        {/* 工具栏 */}
        <div className="flex items-center justify-between mt-3">
          <div className="flex items-center gap-1">
            {/* 语音按钮 */}
            <button
              onClick={isListening ? stopListening : startListening}
              className={`p-2 rounded-lg transition-all ${
                isListening
                  ? 'bg-red-100 text-red-600 animate-pulse'
                  : 'text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-dark-700'
              }`}
              title={isListening ? '停止录音' : '语音输入'}
            >
              <MicrophoneIcon className="w-5 h-5" />
            </button>
            
            {/* 附件按钮 */}
            <button
              onClick={() => fileInputRef.current?.click()}
              className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-dark-700 transition-all"
              title="添加附件"
            >
              <PhotoIcon className="w-5 h-5" />
            </button>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/*,audio/*"
              onChange={handleFileSelect}
              className="hidden"
            />
            
            {/* 位置按钮 */}
            {location && (
              <button
                className="p-2 rounded-lg text-green-600 hover:bg-green-50 dark:hover:bg-green-900/20 transition-all"
                title="已获取位置"
              >
                <MapPinIcon className="w-5 h-5" />
              </button>
            )}
            
            {/* 快捷输入 */}
            <div className="hidden sm:flex items-center gap-1 ml-2 border-l border-gray-200 dark:border-dark-700 pl-2">
              <QuickInputButton
                label="今天"
                onClick={() => setContent((prev) => prev + '今天 ')}
              />
              <QuickInputButton
                label="明天"
                onClick={() => setContent((prev) => prev + '明天 ')}
              />
              <QuickInputButton
                label="重要"
                onClick={() => setContent((prev) => prev + '【重要】')}
              />
            </div>
          </div>
          
          {/* 提交按钮 */}
          <button
            onClick={handleSubmit}
            disabled={(!content.trim() && attachments.length === 0) || isSubmitting}
            className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {isSubmitting ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>保存中...</span>
              </>
            ) : (
              <>
                <PaperAirplaneIcon className="w-4 h-4" />
                <span>保存</span>
              </>
            )}
          </button>
        </div>
      </div>
    </motion.div>
  );
};

// 快捷输入按钮组件
const QuickInputButton: React.FC<{ label: string; onClick: () => void }> = ({ label, onClick }) => (
  <button
    onClick={onClick}
    className="px-2 py-1 text-xs text-gray-500 bg-gray-100 dark:bg-dark-700 dark:text-gray-400 rounded hover:bg-gray-200 dark:hover:bg-dark-600 transition-all"
  >
    {label}
  </button>
);

export default UnifiedInput;

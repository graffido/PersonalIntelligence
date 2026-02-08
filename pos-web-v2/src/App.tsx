import React, { useEffect, useState } from 'react';
import {
  UnifiedInput,
  TimelineView,
  MapView,
  KnowledgeGraphView,
  RecommendationsList,
  SettingsPanel,
} from '@components/index';
import { useAppStore, useSyncStore } from '@stores/index';
import { api } from '@services/api';
import { useNetworkStatus, useTheme } from '@hooks/index';
import { Toaster } from 'react-hot-toast';
import {
  Bars3Icon,
  MagnifyingGlassIcon,
  BellIcon,
  Cog6ToothIcon,
  MapIcon,
  CalendarIcon,
  ShareIcon,
  ListBulletIcon,
  CloudIcon,
  CloudSlashIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';
import type { KnowledgeGraph, TimelineEvent } from '@types/index';

// 侧边栏组件
const Sidebar: React.FC = () => {
  const { isSidebarOpen, toggleSidebar, activeView, setActiveView } = useAppStore();
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  
  const navItems = [
    { id: 'timeline', label: '时间轴', icon: CalendarIcon },
    { id: 'map', label: '地图', icon: MapIcon },
    { id: 'graph', label: '知识图谱', icon: ShareIcon },
    { id: 'list', label: '列表', icon: ListBulletIcon },
  ] as const;
  
  return (
    <>
      <aside
        className={`fixed left-0 top-0 h-full bg-white dark:bg-dark-800 border-r border-gray-200 dark:border-dark-700 transition-all duration-300 z-40 ${
          isSidebarOpen ? 'w-64' : 'w-0 overflow-hidden'
        }`}
      >
        <div className="p-4">
          <div className="flex items-center gap-3 mb-8">
            <div className="w-10 h-10 bg-primary-600 rounded-xl flex items-center justify-center">
              <span className="text-white font-bold text-lg">POS</span>
            </div>
            <div>
              <h1 className="font-semibold text-gray-900 dark:text-gray-100">个人本体</h1>
              <p className="text-xs text-gray-500">记忆管理系统</p>
            </div>
          </div>
          
          <nav className="space-y-1">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => setActiveView(item.id as any)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
                  activeView === item.id
                    ? 'bg-primary-50 text-primary-700 dark:bg-primary-900/20 dark:text-primary-300'
                    : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-dark-700'
                }`}
              >
                <item.icon className="w-5 h-5" />
                {item.label}
              </button>
            ))}
          </nav>
          
          <div className="absolute bottom-4 left-4 right-4">
            <button
              onClick={() => setIsSettingsOpen(true)}
              className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-dark-700 transition-all"
            >
              <Cog6ToothIcon className="w-5 h-5" />
              设置
            </button>
          </div>
        </div>
      </aside>
      
      <SettingsPanel isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} />
    </>
  );
};

// 顶部导航栏
const Header: React.FC = () => {
  const { toggleSidebar, searchQuery, setSearchQuery } = useAppStore();
  const { state: syncState } = useSyncStore();
  
  return (
    <header className="h-16 bg-white dark:bg-dark-800 border-b border-gray-200 dark:border-dark-700 flex items-center justify-between px-4 sticky top-0 z-30">
      <div className="flex items-center gap-4">
        <button
          onClick={toggleSidebar}
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-700 transition-colors"
        >
          <Bars3Icon className="w-6 h-6 text-gray-600 dark:text-gray-400" />
        </button>
        
        <div className="relative">
          <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="搜索记忆、实体..."
            className="pl-10 pr-4 py-2 w-64 bg-gray-100 dark:bg-dark-700 rounded-lg border-0 focus:ring-2 focus:ring-primary-500"
          />
        </div>
      </div>
      
      <div className="flex items-center gap-3">
        {/* 同步状态 */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-100 dark:bg-dark-700">
          {syncState.isOnline ? (
            syncState.isSyncing ? (
              <>
                <ArrowPathIcon className="w-4 h-4 text-blue-500 animate-spin" />
                <span className="text-xs text-gray-600 dark:text-gray-400">同步中...</span>
              </>
            ) : (
              <>
                <CloudIcon className="w-4 h-4 text-green-500" />
                <span className="text-xs text-gray-600 dark:text-gray-400">已同步</span>
              </>
            )
          ) : (
            <>
              <CloudSlashIcon className="w-4 h-4 text-orange-500" />
              <span className="text-xs text-gray-600 dark:text-gray-400">离线</span>
            </>
          )}
        </div>
        
        <button className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-700 relative">
          <BellIcon className="w-6 h-6 text-gray-600 dark:text-gray-400" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
        </button>
      </div>
    </header>
  );
};

// 主内容区
const MainContent: React.FC = () => {
  const { activeView, memories, entities, recommendations, setRecommendations } = useAppStore();
  const [graphData, setGraphData] = useState<KnowledgeGraph | null>(null);
  const [timelineEvents, setTimelineEvents] = useState<TimelineEvent[]>([]);
  const [heatmapData, setHeatmapData] = useState([]);
  
  // 加载数据
  useEffect(() => {
    const loadData = async () => {
      try {
        // 加载知识图谱
        const graphRes = await api.graph.get();
        setGraphData(graphRes.data);
        
        // 加载时间轴
        const timelineRes = await api.timeline.get();
        setTimelineEvents(timelineRes.data);
        
        // 加载热力图数据
        const heatmapRes = await api.heatmap.get();
        setHeatmapData(heatmapRes.data);
        
        // 加载推荐
        const recRes = await api.recommendations.list();
        setRecommendations(recRes.data);
      } catch (error) {
        console.error('Failed to load data:', error);
      }
    };
    
    loadData();
  }, [setRecommendations]);
  
  // 处理新建记忆
  const handleSubmit = async (data: any) => {
    try {
      await api.memories.create(data);
      // 刷新数据
      const res = await api.memories.list();
      // 更新store
    } catch (error) {
      throw error;
    }
  };
  
  return (
    <div className="flex-1 overflow-auto p-6">
      <div className="max-w-6xl mx-auto">
        {/* 输入区 */}
        <div className="mb-6">
          <UnifiedInput onSubmit={handleSubmit} />
        </div>
        
        {/* 主内容 */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 左侧主要内容 */}
          <div className="lg:col-span-2">
            {activeView === 'timeline' && (
              <div className="bg-white dark:bg-dark-800 rounded-xl border border-gray-200 dark:border-dark-700 h-[600px]">
                <TimelineView
                  events={timelineEvents}
                  memories={memories}
                  className="h-full"
                />
              </div>
            )}
            
            {activeView === 'map' && (
              <div className="bg-white dark:bg-dark-800 rounded-xl border border-gray-200 dark:border-dark-700 h-[600px] overflow-hidden">
                <MapView
                  memories={memories}
                  heatmapData={heatmapData}
                  className="h-full"
                />
              </div>
            )}
            
            {activeView === 'graph' && graphData && (
              <div className="bg-white dark:bg-dark-800 rounded-xl border border-gray-200 dark:border-dark-700 h-[600px]">
                <KnowledgeGraphView
                  graph={graphData}
                  className="h-full"
                />
              </div>
            )}
            
            {activeView === 'list' && (
              <div className="bg-white dark:bg-dark-800 rounded-xl border border-gray-200 dark:border-dark-700 p-4">
                <h3 className="font-semibold mb-4">记忆列表</h3>
                <div className="space-y-2">
                  {memories.map((memory) => (
                    <div
                      key={memory.id}
                      className="p-3 bg-gray-50 dark:bg-dark-700 rounded-lg"
                    >
                      <p className="font-medium">{memory.content.slice(0, 100)}</p>
                      <div className="flex items-center gap-2 mt-2 text-sm text-gray-500">
                        <span>{memory.type}</span>
                        <span>·</span>
                        <span>{new Date(memory.timestamp).toLocaleDateString()}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          {/* 右侧推荐区 */}
          <div className="lg:col-span-1">
            <RecommendationsList
              recommendations={recommendations}
              onDismiss={(id) => {
                setRecommendations(recommendations.filter((r) => r.id !== id));
              }}
              onAction={(rec) => console.log('Action:', rec)}
              onDismissAll={() => setRecommendations([])}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

// 主应用组件
const App: React.FC = () => {
  const { isSidebarOpen } = useAppStore();
  const isOnline = useNetworkStatus();
  const { theme } = useTheme();
  
  // 更新网络状态
  useEffect(() => {
    useSyncStore.getState().setOnline(isOnline);
  }, [isOnline]);
  
  return (
    <div className={`min-h-screen bg-gray-50 dark:bg-dark-950 ${theme}`}>
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 3000,
          style: {
            background: '#363636',
            color: '#fff',
          },
        }}
      />
      
      <Sidebar />
      
      <div
        className={`transition-all duration-300 ${
          isSidebarOpen ? 'ml-64' : 'ml-0'
        }`}
      >
        <Header />
        <MainContent />
      </div>
    </div>
  );
};

export default App;

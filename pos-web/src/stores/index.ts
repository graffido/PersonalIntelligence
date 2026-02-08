import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { 
  Entity, Memory, Recommendation, UserSettings, 
  SyncState, TimelineEvent, KnowledgeGraph 
} from '@types/index';

interface AppState {
  // 用户设置
  settings: UserSettings;
  updateSettings: (settings: Partial<UserSettings>) => void;
  toggleTheme: () => void;
  
  // 数据状态
  entities: Entity[];
  memories: Memory[];
  recommendations: Recommendation[];
  timeline: TimelineEvent[];
  knowledgeGraph: KnowledgeGraph | null;
  
  // 操作
  setEntities: (entities: Entity[]) => void;
  addEntity: (entity: Entity) => void;
  updateEntity: (id: string, data: Partial<Entity>) => void;
  deleteEntity: (id: string) => void;
  
  setMemories: (memories: Memory[]) => void;
  addMemory: (memory: Memory) => void;
  updateMemory: (id: string, data: Partial<Memory>) => void;
  deleteMemory: (id: string) => void;
  
  setRecommendations: (recommendations: Recommendation[]) => void;
  setTimeline: (timeline: TimelineEvent[]) => void;
  setKnowledgeGraph: (graph: KnowledgeGraph) => void;
  
  // UI状态
  selectedEntityId: string | null;
  selectedMemoryId: string | null;
  activeView: 'timeline' | 'map' | 'graph' | 'list';
  isSidebarOpen: boolean;
  searchQuery: string;
  
  setSelectedEntity: (id: string | null) => void;
  setSelectedMemory: (id: string | null) => void;
  setActiveView: (view: 'timeline' | 'map' | 'graph' | 'list') => void;
  toggleSidebar: () => void;
  setSearchQuery: (query: string) => void;
}

const defaultSettings: UserSettings = {
  theme: 'system',
  language: 'zh-CN',
  privacy_mode: false,
  offline_mode: false,
  auto_sync: true,
  voice_input: true,
  notifications: true,
};

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      // 设置
      settings: defaultSettings,
      updateSettings: (newSettings) => 
        set((state) => ({ 
          settings: { ...state.settings, ...newSettings } 
        })),
      toggleTheme: () => 
        set((state) => ({
          settings: {
            ...state.settings,
            theme: state.settings.theme === 'dark' ? 'light' : 'dark',
          },
        })),
      
      // 数据
      entities: [],
      memories: [],
      recommendations: [],
      timeline: [],
      knowledgeGraph: null,
      
      // Entity操作
      setEntities: (entities) => set({ entities }),
      addEntity: (entity) => 
        set((state) => ({ entities: [...state.entities, entity] })),
      updateEntity: (id, data) =>
        set((state) => ({
          entities: state.entities.map((e) =>
            e.id === id ? { ...e, ...data, updated_at: new Date().toISOString() } : e
          ),
        })),
      deleteEntity: (id) =>
        set((state) => ({
          entities: state.entities.filter((e) => e.id !== id),
        })),
      
      // Memory操作
      setMemories: (memories) => set({ memories }),
      addMemory: (memory) =>
        set((state) => ({ memories: [memory, ...state.memories] })),
      updateMemory: (id, data) =>
        set((state) => ({
          memories: state.memories.map((m) =>
            m.id === id ? { ...m, ...data } : m
          ),
        })),
      deleteMemory: (id) =>
        set((state) => ({
          memories: state.memories.filter((m) => m.id !== id),
        })),
      
      // 其他数据
      setRecommendations: (recommendations) => set({ recommendations }),
      setTimeline: (timeline) => set({ timeline }),
      setKnowledgeGraph: (graph) => set({ knowledgeGraph: graph }),
      
      // UI状态
      selectedEntityId: null,
      selectedMemoryId: null,
      activeView: 'timeline',
      isSidebarOpen: true,
      searchQuery: '',
      
      setSelectedEntity: (id) => set({ selectedEntityId: id }),
      setSelectedMemory: (id) => set({ selectedMemoryId: id }),
      setActiveView: (view) => set({ activeView: view }),
      toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),
      setSearchQuery: (query) => set({ searchQuery: query }),
    }),
    {
      name: 'pos-app-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({ settings: state.settings }),
    }
  )
);

// 同步状态管理
interface SyncStore {
  state: SyncState;
  setOnline: (online: boolean) => void;
  setSyncing: (syncing: boolean) => void;
  updateLastSync: () => void;
  setPendingChanges: (count: number) => void;
  incrementPending: () => void;
  decrementPending: () => void;
}

export const useSyncStore = create<SyncStore>()((set) => ({
  state: {
    isOnline: navigator.onLine,
    isSyncing: false,
    pendingChanges: 0,
    conflicts: 0,
  },
  setOnline: (online) => set((state) => ({ state: { ...state.state, isOnline: online } })),
  setSyncing: (syncing) => set((state) => ({ state: { ...state.state, isSyncing: syncing } })),
  updateLastSync: () => set((state) => ({ 
    state: { ...state.state, lastSyncAt: new Date().toISOString() } 
  })),
  setPendingChanges: (count) => set((state) => ({ 
    state: { ...state.state, pendingChanges: count } 
  })),
  incrementPending: () => set((state) => ({ 
    state: { ...state.state, pendingChanges: state.state.pendingChanges + 1 } 
  })),
  decrementPending: () => set((state) => ({ 
    state: { ...state.state, pendingChanges: Math.max(0, state.state.pendingChanges - 1) } 
  })),
}));

// 离线队列管理
interface OfflineQueue {
  queue: Array<{ id: string; type: string; data: any; timestamp: string }>;
  addToQueue: (item: Omit<OfflineQueue['queue'][0], 'id' | 'timestamp'>) => void;
  removeFromQueue: (id: string) => void;
  clearQueue: () => void;
  getQueue: () => OfflineQueue['queue'];
}

export const useOfflineQueue = create<OfflineQueue>()(
  persist(
    (set, get) => ({
      queue: [],
      addToQueue: (item) =>
        set((state) => ({
          queue: [
            ...state.queue,
            { ...item, id: crypto.randomUUID(), timestamp: new Date().toISOString() },
          ],
        })),
      removeFromQueue: (id) =>
        set((state) => ({
          queue: state.queue.filter((item) => item.id !== id),
        })),
      clearQueue: () => set({ queue: [] }),
      getQueue: () => get().queue,
    }),
    {
      name: 'pos-offline-queue',
      storage: createJSONStorage(() => localStorage),
    }
  )
);

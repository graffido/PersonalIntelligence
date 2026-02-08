import axios, { AxiosError, AxiosInstance, InternalAxiosRequestConfig } from 'axios';
import { useSyncStore, useOfflineQueue } from '@stores/index';

// 创建axios实例
const apiClient: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = localStorage.getItem('pos_token');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean };
    
    // 网络错误处理
    if (!error.response) {
      // 添加到离线队列
      const { addToQueue } = useOfflineQueue.getState();
      addToQueue({
        type: 'request',
        data: {
          method: originalRequest.method,
          url: originalRequest.url,
          data: originalRequest.data,
        },
      });
      
      useSyncStore.getState().setOnline(false);
      return Promise.reject(new Error('网络连接失败，已添加到离线队列'));
    }
    
    // 401 未授权
    if (error.response.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      // 刷新token逻辑
      try {
        const refreshToken = localStorage.getItem('pos_refresh_token');
        const response = await axios.post('/api/auth/refresh', { refreshToken });
        const { token } = response.data;
        localStorage.setItem('pos_token', token);
        
        if (originalRequest.headers) {
          originalRequest.headers.Authorization = `Bearer ${token}`;
        }
        return apiClient(originalRequest);
      } catch (refreshError) {
        // 刷新失败，登出
        localStorage.removeItem('pos_token');
        localStorage.removeItem('pos_refresh_token');
        window.location.href = '/login';
        return Promise.reject(refreshError);
      }
    }
    
    return Promise.reject(error);
  }
);

// API方法
export const api = {
  // 实体
  entities: {
    list: (params?: { type?: string; search?: string }) =>
      apiClient.get('/entities', { params }),
    get: (id: string) => apiClient.get(`/entities/${id}`),
    create: (data: any) => apiClient.post('/entities', data),
    update: (id: string, data: any) => apiClient.put(`/entities/${id}`, data),
    delete: (id: string) => apiClient.delete(`/entities/${id}`),
  },
  
  // 记忆
  memories: {
    list: (params?: { type?: string; start?: string; end?: string }) =>
      apiClient.get('/memories', { params }),
    get: (id: string) => apiClient.get(`/memories/${id}`),
    create: (data: any) => apiClient.post('/memories', data),
    update: (id: string, data: any) => apiClient.put(`/memories/${id}`, data),
    delete: (id: string) => apiClient.delete(`/memories/${id}`),
    search: (query: string) => apiClient.get('/memories/search', { params: { q: query } }),
  },
  
  // 知识图谱
  graph: {
    get: () => apiClient.get('/graph'),
    getEntityRelations: (entityId: string) =>
      apiClient.get(`/graph/entity/${entityId}`),
  },
  
  // 推荐
  recommendations: {
    list: () => apiClient.get('/recommendations'),
    dismiss: (id: string) => apiClient.post(`/recommendations/${id}/dismiss`),
  },
  
  // 时间轴
  timeline: {
    get: (params?: { start?: string; end?: string }) =>
      apiClient.get('/timeline', { params }),
  },
  
  // 热力图
  heatmap: {
    get: (params?: { start?: string; end?: string; type?: string }) =>
      apiClient.get('/heatmap', { params }),
  },
  
  // 搜索
  search: {
    global: (query: string) => apiClient.get('/search', { params: { q: query } }),
  },
  
  // 健康检查
  health: () => apiClient.get('/health'),
};

export default apiClient;

export interface Entity {
  id: string;
  name: string;
  type: string;
  description?: string;
  properties: Record<string, any>;
  created_at: string;
  updated_at: string;
}

export interface Relation {
  id: string;
  source_id: string;
  target_id: string;
  relation_type: string;
  properties?: Record<string, any>;
  created_at: string;
}

export interface Memory {
  id: string;
  content: string;
  type: 'note' | 'event' | 'task' | 'idea';
  timestamp: string;
  location?: GeoLocation;
  entities: string[];
  tags: string[];
  sentiment?: Sentiment;
  importance: number;
  attachments?: Attachment[];
}

export interface GeoLocation {
  latitude: number;
  longitude: number;
  name?: string;
  address?: string;
}

export interface Sentiment {
  polarity: number;
  subjectivity: number;
  label: 'positive' | 'negative' | 'neutral';
}

export interface Attachment {
  id: string;
  type: 'image' | 'audio' | 'file';
  url: string;
  name: string;
  size: number;
}

export interface KnowledgeGraph {
  nodes: GraphNode[];
  links: GraphLink[];
}

export interface GraphNode {
  id: string;
  name: string;
  type: string;
  group: number;
  value?: number;
  x?: number;
  y?: number;
}

export interface GraphLink {
  source: string;
  target: string;
  type: string;
  value?: number;
}

export interface HeatmapPoint {
  lat: number;
  lng: number;
  intensity: number;
  count: number;
  category?: string;
}

export interface Recommendation {
  id: string;
  type: 'connection' | 'insight' | 'action' | 'reminder';
  title: string;
  description: string;
  confidence: number;
  related_entities: string[];
  created_at: string;
}

export interface UserSettings {
  theme: 'light' | 'dark' | 'system';
  language: string;
  privacy_mode: boolean;
  offline_mode: boolean;
  auto_sync: boolean;
  voice_input: boolean;
  notifications: boolean;
  default_location?: GeoLocation;
}

export interface TimelineEvent {
  id: string;
  timestamp: string;
  title: string;
  description?: string;
  type: string;
  icon?: string;
  color?: string;
  data?: any;
}

export interface VoiceInputState {
  isListening: boolean;
  transcript: string;
  interimTranscript: string;
  confidence: number;
  error?: string;
}

export interface SyncState {
  isOnline: boolean;
  isSyncing: boolean;
  lastSyncAt?: string;
  pendingChanges: number;
  conflicts: number;
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, any>;
}

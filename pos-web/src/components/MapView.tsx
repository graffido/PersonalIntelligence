import React, { useEffect, useState, useMemo } from 'react';
import { MapContainer, TileLayer, useMap, CircleMarker, Popup, useMapEvents } from 'react-leaflet';
import { format } from 'date-fns';
import type { HeatmapPoint, Memory } from '@types/index';
import 'leaflet/dist/leaflet.css';
import { MapPinIcon, ClockIcon, TagIcon } from '@heroicons/react/24/outline';

// 修复Leaflet图标问题
import L from 'leaflet';
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface MapViewProps {
  memories: Memory[];
  heatmapData?: HeatmapPoint[];
  onMemoryClick?: (memory: Memory) => void;
  className?: string;
  center?: [number, number];
  zoom?: number;
}

// 热力图组件
const HeatmapLayer: React.FC<{ points: HeatmapPoint[] }> = ({ points }) => {
  const map = useMap();
  
  useEffect(() => {
    // 动态导入 leaflet.heat
    import('leaflet.heat').then(() => {
      const heatLayer = (L as any).heatLayer(
        points.map((p) => [p.lat, p.lng, p.intensity]),
        {
          radius: 25,
          blur: 15,
          maxZoom: 10,
          max: 1.0,
          gradient: {
            0.4: 'blue',
            0.6: 'cyan',
            0.7: 'lime',
            0.8: 'yellow',
            1.0: 'red',
          },
        }
      );
      
      heatLayer.addTo(map);
      
      return () => {
        map.removeLayer(heatLayer);
      };
    });
  }, [map, points]);
  
  return null;
};

// 点击地图获取坐标
const MapClickHandler: React.FC<{ onClick: (lat: number, lng: number) => void }> = ({ onClick }) => {
  useMapEvents({
    click: (e) => {
      onClick(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
};

export const MapView: React.FC<MapViewProps> = ({
  memories,
  heatmapData,
  onMemoryClick,
  className = '',
  center = [39.9042, 116.4074], // 默认北京
  zoom = 12,
}) => {
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [showMarkers, setShowMarkers] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  
  // 提取有位置的记忆
  const memoriesWithLocation = useMemo(() => {
    return memories.filter((m) => m.location);
  }, [memories]);
  
  // 按类型分组
  const categories = useMemo(() => {
    const cats = new Set(memoriesWithLocation.map((m) => m.type));
    return Array.from(cats);
  }, [memoriesWithLocation]);
  
  // 过滤后的记忆
  const filteredMemories = useMemo(() => {
    if (!selectedCategory) return memoriesWithLocation;
    return memoriesWithLocation.filter((m) => m.type === selectedCategory);
  }, [memoriesWithLocation, selectedCategory]);
  
  const getTypeColor = (type: string) => {
    switch (type) {
      case 'note': return '#3b82f6';
      case 'event': return '#22c55e';
      case 'task': return '#eab308';
      case 'idea': return '#a855f7';
      default: return '#6b7280';
    }
  };
  
  const handleMapClick = (lat: number, lng: number) => {
    console.log('Map clicked:', lat, lng);
  };
  
  return (
    <div className={`relative ${className}`}>
      {/* 控制面板 */}
      <div className="absolute top-4 left-4 z-[1000] bg-white dark:bg-dark-800 rounded-lg shadow-lg p-3 space-y-2">
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={showHeatmap}
              onChange={(e) => setShowHeatmap(e.target.checked)}
              className="rounded border-gray-300"
            />
            显示热力图
          </label>
        </div>
        
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={showMarkers}
              onChange={(e) => setShowMarkers(e.target.checked)}
              className="rounded border-gray-300"
            />
            显示标记
          </label>
        </div>
        
        <hr className="border-gray-200 dark:border-dark-700" />
        
        <div className="space-y-1">
          <p className="text-xs text-gray-500">过滤类型</p>
          <button
            onClick={() => setSelectedCategory(null)}
            className={`block w-full text-left px-2 py-1 rounded text-sm ${
              selectedCategory === null ? 'bg-primary-100 text-primary-700' : 'hover:bg-gray-100 dark:hover:bg-dark-700'
            }`}
          >
            全部
          </button>
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`block w-full text-left px-2 py-1 rounded text-sm ${
                selectedCategory === cat ? 'bg-primary-100 text-primary-700' : 'hover:bg-gray-100 dark:hover:bg-dark-700'
              }`}
            >
              <span
                className="inline-block w-2 h-2 rounded-full mr-2"
                style={{ backgroundColor: getTypeColor(cat) }}
              />
              {cat}
            </button>
          ))}
        </div>
      </div>
      
      {/* 统计信息 */}
      <div className="absolute top-4 right-4 z-[1000] bg-white dark:bg-dark-800 rounded-lg shadow-lg p-3">
        <div className="text-sm">
          <p className="text-gray-500">位置记录</p>
          <p className="text-2xl font-bold">{memoriesWithLocation.length}</p>
        </div>
      </div>
      
      {/* 地图 */}
      <MapContainer
        center={center}
        zoom={zoom}
        style={{ height: '100%', width: '100%' }}
        className="rounded-lg"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        
        {showHeatmap && heatmapData && heatmapData.length > 0 && (
          <HeatmapLayer points={heatmapData} />
        )}
        
        {showMarkers && filteredMemories.map((memory) => (
          memory.location && (
            <CircleMarker
              key={memory.id}
              center={[memory.location.latitude, memory.location.longitude]}
              radius={8}
              fillColor={getTypeColor(memory.type)}
              color="#fff"
              weight={2}
              opacity={1}
              fillOpacity={0.8}
              eventHandlers={{
                click: () => onMemoryClick?.(memory),
              }}
            >
              <Popup>
                <div className="p-2 max-w-xs">
                  <p className="font-medium mb-2">{memory.content.slice(0, 100)}</p>
                  <div className="text-xs text-gray-500 space-y-1">
                    <p className="flex items-center gap-1">
                      <ClockIcon className="w-3 h-3" />
                      {format(new Date(memory.timestamp), 'yyyy-MM-dd HH:mm')}
                    </p>
                    {memory.location.name && (
                      <p className="flex items-center gap-1">
                        <MapPinIcon className="w-3 h-3" />
                        {memory.location.name}
                      </p>
                    )}
                    {memory.tags.length > 0 && (
                      <p className="flex items-center gap-1 flex-wrap">
                        <TagIcon className="w-3 h-3" />
                        {memory.tags.map((t) => `#${t}`).join(' ')}
                      </p>
                    )}
                  </div>
                </div>
              </Popup>
            </CircleMarker>
          )
        ))}
        
        <MapClickHandler onClick={handleMapClick} />
      </MapContainer>
    </div>
  );
};

export default MapView;

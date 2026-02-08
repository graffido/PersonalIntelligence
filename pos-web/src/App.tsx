import React, { useState } from 'react'
import { 
  Database, 
  Search, 
  MapPin, 
  Share2, 
  Settings,
  Menu,
  X
} from 'lucide-react'
import { IngestPanel } from './components/IngestPanel'
import { QueryPanel } from './components/QueryPanel'
import { SpatiotemporalPanel } from './components/SpatiotemporalPanel'
import { GraphPanel } from './components/GraphPanel'
import { StatsPanel } from './components/StatsPanel'

type TabType = 'ingest' | 'query' | 'spatiotemporal' | 'graph' | 'stats'

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('query')
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const tabs = [
    { id: 'ingest' as TabType, label: '数据导入', icon: Database },
    { id: 'query' as TabType, label: '知识查询', icon: Search },
    { id: 'spatiotemporal' as TabType, label: '时空查询', icon: MapPin },
    { id: 'graph' as TabType, label: '知识图谱', icon: Share2 },
    { id: 'stats' as TabType, label: '统计', icon: Settings },
  ]

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar */}
      <aside 
        className={`bg-white shadow-lg transition-all duration-300 ${
          sidebarOpen ? 'w-64' : 'w-16'
        }`}
      >
        <div className="p-4 border-b">
          <div className="flex items-center justify-between">
            {sidebarOpen && (
              <div>
                <h1 className="text-lg font-bold text-gray-800">POS系统</h1>
                <p className="text-xs text-gray-500">个人本体记忆</p>
              </div>
            )}
            <button 
              onClick={() =u003e setSidebarOpen(!sidebarOpen)}
              className="p-2 hover:bg-gray-100 rounded"
            >
              {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </div>

        <nav className="p-2 space-y-1">
          {tabs.map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg transition-colors ${
                  activeTab === tab.id
                    ? 'bg-blue-50 text-blue-600'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                <Icon size={20} />
                {sidebarOpen && <span>{tab.label}</span>}
              </button>
            )
          })}
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-6 overflow-auto">
        <div className="max-w-6xl mx-auto">
          {activeTab === 'ingest' && <IngestPanel />}
          {activeTab === 'query' && <QueryPanel />}
          {activeTab === 'spatiotemporal' && <SpatiotemporalPanel />}
          {activeTab === 'graph' && <GraphPanel />}
          {activeTab === 'stats' && <StatsPanel />}
        </div>
      </main>
    </div>
  )
}

export default App

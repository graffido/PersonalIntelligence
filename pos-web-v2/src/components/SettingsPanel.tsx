import React, { useState } from 'react';
import { Dialog, Switch, Tab } from '@headlessui/react';
import { useAppStore } from '@stores/index';
import type { UserSettings } from '@types/index';
import {
  XMarkIcon,
  MoonIcon,
  SunIcon,
  ComputerDesktopIcon,
  ShieldCheckIcon,
  CloudIcon,
  BellIcon,
  GlobeAltIcon,
  MicrophoneIcon,
} from '@heroicons/react/24/outline';

interface SettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

const tabs = [
  { name: '外观', icon: SunIcon },
  { name: '隐私', icon: ShieldCheckIcon },
  { name: '同步', icon: CloudIcon },
  { name: '通知', icon: BellIcon },
];

export const SettingsPanel: React.FC<SettingsPanelProps> = ({ isOpen, onClose }) => {
  const { settings, updateSettings } = useAppStore();
  const [selectedTab, setSelectedTab] = useState(0);
  
  const handleThemeChange = (theme: UserSettings['theme']) => {
    updateSettings({ theme });
  };
  
  return (
    <Dialog open={isOpen} onClose={onClose} className="relative z-50">
      {/* 背景遮罩 */}
      <div className="fixed inset-0 bg-black/30 backdrop-blur-sm" aria-hidden="true" />
      
      {/* 对话框 */}
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <Dialog.Panel className="w-full max-w-2xl bg-white dark:bg-dark-800 rounded-2xl shadow-2xl overflow-hidden">
          {/* 头部 */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-dark-700">
            <Dialog.Title className="text-xl font-semibold text-gray-900 dark:text-gray-100">
              设置
            </Dialog.Title>
            
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-700 transition-colors"
            >
              <XMarkIcon className="w-6 h-6 text-gray-500" />
            </button>
          </div>
          
          {/* 标签页 */}
          <div className="flex">
            <Tab.Group selectedIndex={selectedTab} onChange={setSelectedTab}>
              <div className="w-48 border-r border-gray-200 dark:border-dark-700">
                <Tab.List className="flex flex-col p-2 space-y-1">
                  {tabs.map((tab) => (
                    <Tab
                      key={tab.name}
                      className={({ selected }) =>
                        `flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all text-left ${
                          selected
                            ? 'bg-primary-50 text-primary-700 dark:bg-primary-900/20 dark:text-primary-300'
                            : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-dark-700'
                        }`
                      }
                    >
                      <tab.icon className="w-5 h-5" />
                      {tab.name}
                    </Tab>
                  ))}
                </Tab.List>
              </div>
              
              <div className="flex-1 p-6">
                <Tab.Panels>
                  {/* 外观设置 */}
                  <Tab.Panel className="space-y-6">
                    <div>
                      <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">主题</h4>
                      <div className="grid grid-cols-3 gap-4">
                        <button
                          onClick={() => handleThemeChange('light')}
                          className={`p-4 rounded-xl border-2 transition-all ${
                            settings.theme === 'light'
                              ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                              : 'border-gray-200 dark:border-dark-700 hover:border-gray-300'
                          }`}
                        >
                          <SunIcon className="w-8 h-8 mx-auto mb-2 text-yellow-500" />
                          <p className="text-sm font-medium">浅色</p>
                        </button>
                        
                        <button
                          onClick={() => handleThemeChange('dark')}
                          className={`p-4 rounded-xl border-2 transition-all ${
                            settings.theme === 'dark'
                              ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                              : 'border-gray-200 dark:border-dark-700 hover:border-gray-300'
                          }`}
                        >
                          <MoonIcon className="w-8 h-8 mx-auto mb-2 text-indigo-500" />
                          <p className="text-sm font-medium">深色</p>
                        </button>
                        
                        <button
                          onClick={() => handleThemeChange('system')}
                          className={`p-4 rounded-xl border-2 transition-all ${
                            settings.theme === 'system'
                              ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                              : 'border-gray-200 dark:border-dark-700 hover:border-gray-300'
                          }`}
                        >
                          <ComputerDesktopIcon className="w-8 h-8 mx-auto mb-2 text-gray-500" />
                          <p className="text-sm font-medium">跟随系统</p>
                        </button>
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">语言</h4>
                      <select
                        value={settings.language}
                        onChange={(e) => updateSettings({ language: e.target.value })}
                        className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-dark-600 bg-white dark:bg-dark-700"
                      >
                        <option value="zh-CN">简体中文</option>
                        <option value="zh-TW">繁體中文</option>
                        <option value="en">English</option>
                        <option value="ja">日本語</option>
                      </select>
                    </div>
                  </Tab.Panel>
                  
                  {/* 隐私设置 */}
                  <Tab.Panel className="space-y-6">
                    <div className="space-y-4">
                      <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-dark-700 rounded-lg">
                        <div className="flex items-center gap-3">
                          <ShieldCheckIcon className="w-6 h-6 text-green-500" />
                          <div>
                            <p className="font-medium">隐私模式</p>
                            <p className="text-sm text-gray-500">数据仅存储在本地，不上传到云端</p>
                          </div>
                        </div>
                        <Switch
                          checked={settings.privacy_mode}
                          onChange={(checked) => updateSettings({ privacy_mode: checked })}
                          className={`${settings.privacy_mode ? 'bg-green-500' : 'bg-gray-300'}
                            relative inline-flex h-6 w-11 items-center rounded-full transition-colors`
                          }
                        >
                          <span
                            className={`${settings.privacy_mode ? 'translate-x-6' : 'translate-x-1'}
                              inline-block h-4 w-4 transform rounded-full bg-white transition-transform`
                            }
                          />
                        </Switch>
                      </div>
                      
                      <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-dark-700 rounded-lg">
                        <div className="flex items-center gap-3">
                          <MicrophoneIcon className="w-6 h-6 text-blue-500" />
                          <div>
                            <p className="font-medium">语音输入</p>
                            <p className="text-sm text-gray-500">启用语音转文字功能</p>
                          </div>
                        </div>
                        <Switch
                          checked={settings.voice_input}
                          onChange={(checked) => updateSettings({ voice_input: checked })}
                          className={`${settings.voice_input ? 'bg-blue-500' : 'bg-gray-300'}
                            relative inline-flex h-6 w-11 items-center rounded-full transition-colors`
                          }
                        >
                          <span
                            className={`${settings.voice_input ? 'translate-x-6' : 'translate-x-1'}
                              inline-block h-4 w-4 transform rounded-full bg-white transition-transform`
                            }
                          />
                        </Switch>
                      </div>
                    </div>
                  </Tab.Panel>
                  
                  {/* 同步设置 */}
                  <Tab.Panel className="space-y-6">
                    <div className="space-y-4">
                      <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-dark-700 rounded-lg">
                        <div className="flex items-center gap-3">
                          <CloudIcon className="w-6 h-6 text-blue-500" />
                          <div>
                            <p className="font-medium">自动同步</p>
                            <p className="text-sm text-gray-500">自动将数据同步到云端</p>
                          </div>
                        </div>
                        <Switch
                          checked={settings.auto_sync}
                          onChange={(checked) => updateSettings({ auto_sync: checked })}
                          disabled={settings.privacy_mode}
                          className={`${settings.auto_sync ? 'bg-blue-500' : 'bg-gray-300'}
                            relative inline-flex h-6 w-11 items-center rounded-full transition-colors
                            disabled:opacity-50 disabled:cursor-not-allowed`
                          }
                        >
                          <span
                            className={`${settings.auto_sync ? 'translate-x-6' : 'translate-x-1'}
                              inline-block h-4 w-4 transform rounded-full bg-white transition-transform`
                            }
                          />
                        </Switch>
                      </div>
                      
                      <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-dark-700 rounded-lg">
                        <div className="flex items-center gap-3">
                          <GlobeAltIcon className="w-6 h-6 text-purple-500" />
                          <div>
                            <p className="font-medium">离线模式</p>
                            <p className="text-sm text-gray-500">优先使用本地缓存</p>
                          </div>
                        </div>
                        <Switch
                          checked={settings.offline_mode}
                          onChange={(checked) => updateSettings({ offline_mode: checked })}
                          className={`${settings.offline_mode ? 'bg-purple-500' : 'bg-gray-300'}
                            relative inline-flex h-6 w-11 items-center rounded-full transition-colors`
                          }
                        >
                          <span
                            className={`${settings.offline_mode ? 'translate-x-6' : 'translate-x-1'}
                              inline-block h-4 w-4 transform rounded-full bg-white transition-transform`
                            }
                          />
                        </Switch>
                      </div>
                    </div>
                  </Tab.Panel>
                  
                  {/* 通知设置 */}
                  <Tab.Panel className="space-y-6">
                    <div className="space-y-4">
                      <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-dark-700 rounded-lg">
                        <div className="flex items-center gap-3">
                          <BellIcon className="w-6 h-6 text-orange-500" />
                          <div>
                            <p className="font-medium">启用通知</p>
                            <p className="text-sm text-gray-500">接收智能提醒和推荐</p>
                          </div>
                        </div>
                        <Switch
                          checked={settings.notifications}
                          onChange={(checked) => updateSettings({ notifications: checked })}
                          className={`${settings.notifications ? 'bg-orange-500' : 'bg-gray-300'}
                            relative inline-flex h-6 w-11 items-center rounded-full transition-colors`
                          }
                        >
                          <span
                            className={`${settings.notifications ? 'translate-x-6' : 'translate-x-1'}
                              inline-block h-4 w-4 transform rounded-full bg-white transition-transform`
                            }
                          />
                        </Switch>
                      </div>
                    </div>
                  </Tab.Panel>
                </Tab.Panels>
              </div>
            </Tab.Group>
          </div>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
};

export default SettingsPanel;

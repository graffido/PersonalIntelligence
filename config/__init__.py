"""
配置管理模块 - 支持YAML配置和环境变量
"""
import os
import yaml
from pathlib import Path
from typing import Any, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False


@dataclass
class DatabaseConfig:
    url: str = "sqlite:///data/pos.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30


@dataclass
class PrivacyConfig:
    privacy_mode: bool = False
    encrypt_sensitive: bool = True
    local_only: bool = False


@dataclass
class AppConfig:
    debug: bool = False
    environment: str = "development"
    server: ServerConfig = field(default_factory=ServerConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    
    # 存储原始配置字典
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._config: Optional[AppConfig] = None
        self._env_prefix = "POS_"
    
    def load(self, env: str = None) -> AppConfig:
        """加载配置"""
        # 确定环境
        if env is None:
            env = os.environ.get("POS_ENV", "development")
        
        # 加载默认配置
        config_dict = self._load_yaml("config.yaml")
        
        # 加载环境特定配置
        env_config_path = f"config.{env}.yaml"
        if (self.config_dir / env_config_path).exists():
            env_config = self._load_yaml(env_config_path)
            config_dict = self._merge_dicts(config_dict, env_config)
        
        # 从环境变量覆盖
        config_dict = self._apply_env_vars(config_dict)
        
        # 解析为配置对象
        self._config = self._parse_config(config_dict)
        self._config._raw = config_dict
        
        return self._config
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """加载YAML文件"""
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def _merge_dicts(self, base: Dict, override: Dict) -> Dict:
        """合并字典"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用环境变量"""
        # 遍历环境变量
        for key, value in os.environ.items():
            if key.startswith(self._env_prefix):
                # POS_DATABASE_URL -> database.url
                config_key = key[len(self._env_prefix):].lower().replace('_', '.')
                self._set_nested_value(config, config_key, self._parse_value(value))
        
        return config
    
    def _set_nested_value(self, config: Dict, key: str, value: Any):
        """设置嵌套值"""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _parse_value(self, value: str) -> Any:
        """解析值类型"""
        # 布尔值
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # 整数
        try:
            return int(value)
        except ValueError:
            pass
        
        # 浮点数
        try:
            return float(value)
        except ValueError:
            pass
        
        # 字符串
        return value
    
    def _parse_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """解析配置字典为配置对象"""
        return AppConfig(
            debug=config_dict.get('debug', False),
            environment=config_dict.get('environment', 'development'),
            server=ServerConfig(**config_dict.get('server', {})),
            database=DatabaseConfig(**config_dict.get('database', {})),
            llm=LLMConfig(**config_dict.get('llm', {})),
            privacy=PrivacyConfig(**config_dict.get('privacy', {}))
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值（支持点号路径）"""
        if self._config is None:
            self.load()
        
        keys = key.split('.')
        value = self._config._raw
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def reload(self) -> AppConfig:
        """重新加载配置"""
        self._config = None
        return self.load()


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


def get_config() -> AppConfig:
    """获取全局配置"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    if _config_manager._config is None:
        _config_manager.load()
    
    return _config_manager._config


def get_config_value(key: str, default: Any = None) -> Any:
    """获取配置值"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    return _config_manager.get(key, default)


# 示例用法
if __name__ == "__main__":
    # 设置测试环境变量
    os.environ["POS_DATABASE_URL"] = "postgresql://user:pass@localhost/pos"
    os.environ["POS_DEBUG"] = "true"
    
    # 加载配置
    config = get_config()
    
    print(f"环境: {config.environment}")
    print(f"调试模式: {config.debug}")
    print(f"服务器端口: {config.server.port}")
    print(f"数据库URL: {config.database.url}")
    print(f"LLM模型: {config.llm.model}")
    
    # 获取原始配置值
    print(f"日志级别: {get_config_value('logging.level', 'INFO')}")

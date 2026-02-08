"""
数据安全和隐私模块 - 加密、本地优先、隐私模式
"""
import os
import base64
import hashlib
import json
from typing import Optional, Any, Dict
from pathlib import Path
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

logger = logging.getLogger(__name__)


@dataclass
class PrivacyConfig:
    """隐私配置"""
    privacy_mode: bool = False           # 隐私模式（不上云）
    local_only: bool = False             # 仅本地存储
    encrypt_sensitive: bool = True       # 加密敏感信息
    encryption_key_path: str = ".keys/encryption.key"
    salt_path: str = ".keys/salt"
    sensitive_fields: tuple = (          # 敏感字段列表
        "password",
        "token",
        "secret",
        "credit_card",
        "ssn",
        "phone",
        "email",
        "location",
        "address"
    )


class DataEncryption:
    """数据加密管理器"""
    
    def __init__(self, config: PrivacyConfig = None):
        self.config = config or PrivacyConfig()
        self._key: Optional[bytes] = None
        self._fernet: Optional[Fernet] = None
    
    def _get_or_create_key(self) -> bytes:
        """获取或创建加密密钥"""
        if self._key is not None:
            return self._key
        
        key_path = Path(self.config.encryption_key_path)
        salt_path = Path(self.config.salt_path)
        
        # 确保目录存在
        key_path.parent.mkdir(parents=True, exist_ok=True)
        
        if key_path.exists():
            # 读取现有密钥
            with open(key_path, 'rb') as f:
                self._key = base64.urlsafe_b64decode(f.read())
        else:
            # 生成新密钥
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=480000,
            )
            
            # 从环境变量获取主密码或使用默认
            master_password = os.environ.get('POS_MASTER_PASSWORD', 'default-password-change-in-production')
            key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))
            
            # 保存密钥和盐
            with open(key_path, 'wb') as f:
                f.write(key)
            with open(salt_path, 'wb') as f:
                f.write(salt)
            
            logger.info("生成新的加密密钥")
            self._key = base64.urlsafe_b64decode(key)
        
        self._fernet = Fernet(base64.urlsafe_b64encode(self._key))
        return self._key
    
    def encrypt(self, data: str) -> str:
        """加密字符串"""
        if not self.config.encrypt_sensitive:
            return data
        
        fernet = Fernet(base64.urlsafe_b64encode(self._get_or_create_key()))
        encrypted = fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """解密字符串"""
        if not self.config.encrypt_sensitive:
            return encrypted_data
        
        fernet = Fernet(base64.urlsafe_b64encode(self._get_or_create_key()))
        encrypted = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = fernet.decrypt(encrypted)
        return decrypted.decode()
    
    def encrypt_dict(self, data: dict, fields: tuple = None) -> dict:
        """加密字典中的敏感字段"""
        if not self.config.encrypt_sensitive:
            return data
        
        fields = fields or self.config.sensitive_fields
        result = {}
        
        for key, value in data.items():
            if any(field in key.lower() for field in fields):
                if isinstance(value, str):
                    result[key] = {
                        "__encrypted__": True,
                        "value": self.encrypt(value)
                    }
                else:
                    result[key] = value
            elif isinstance(value, dict):
                result[key] = self.encrypt_dict(value, fields)
            elif isinstance(value, list):
                result[key] = [
                    self.encrypt_dict(item, fields) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value
        
        return result
    
    def decrypt_dict(self, data: dict) -> dict:
        """解密字典中的加密字段"""
        if not self.config.encrypt_sensitive:
            return data
        
        result = {}
        
        for key, value in data.items():
            if isinstance(value, dict) and value.get("__encrypted__"):
                result[key] = self.decrypt(value["value"])
            elif isinstance(value, dict):
                result[key] = self.decrypt_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.decrypt_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value
        
        return result


class PrivacyManager:
    """隐私管理器"""
    
    def __init__(self, config: PrivacyConfig = None):
        self.config = config or PrivacyConfig()
        self.encryption = DataEncryption(self.config)
    
    def is_privacy_mode(self) -> bool:
        """检查是否处于隐私模式"""
        return self.config.privacy_mode
    
    def can_upload_to_cloud(self) -> bool:
        """检查是否可以上传到云端"""
        return not self.config.privacy_mode and not self.config.local_only
    
    def sanitize_for_cloud(self, data: dict) -> dict:
        """清理数据以便上传到云端（移除敏感信息）"""
        if self.config.privacy_mode:
            raise PrivacyError("隐私模式已启用，禁止上传数据到云端")
        
        # 移除敏感字段
        sanitized = {}
        for key, value in data.items():
            if any(field in key.lower() for field in self.config.sensitive_fields):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_for_cloud(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self.sanitize_for_cloud(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def anonymize(self, data: str) -> str:
        """匿名化处理"""
        # 简单的哈希匿名化
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def mask_sensitive(self, text: str, mask_char: str = '*') -> str:
        """遮盖敏感信息"""
        # 邮箱遮盖
        import re
        
        # 遮盖邮箱
        def mask_email(match):
            email = match.group(0)
            if '@' in email:
                local, domain = email.split('@')
                masked_local = local[0] + mask_char * (len(local) - 2) + local[-1] if len(local) > 2 else mask_char * len(local)
                return f"{masked_local}@{domain}"
            return mask_char * len(email)
        
        # 遮盖手机号
        def mask_phone(match):
            phone = match.group(0)
            if len(phone) >= 7:
                return phone[:3] + mask_char * (len(phone) - 6) + phone[-3:]
            return mask_char * len(phone)
        
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', mask_email, text)
        text = re.sub(r'\b1[3-9]\d{9}\b', mask_phone, text)
        
        return text
    
    def encrypt_sensitive_data(self, data: Any) -> Any:
        """加密敏感数据"""
        if not self.config.encrypt_sensitive:
            return data
        
        if isinstance(data, dict):
            return self.encryption.encrypt_dict(data)
        elif isinstance(data, str):
            return self.encryption.encrypt(data)
        return data
    
    def decrypt_sensitive_data(self, data: Any) -> Any:
        """解密敏感数据"""
        if not self.config.encrypt_sensitive:
            return data
        
        if isinstance(data, dict):
            return self.encryption.decrypt_dict(data)
        elif isinstance(data, str):
            try:
                return self.encryption.decrypt(data)
            except:
                return data
        return data


class PrivacyError(Exception):
    """隐私错误"""
    pass


class LocalFirstStorage:
    """本地优先存储"""
    
    def __init__(self, base_path: str = "data/local"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.privacy_manager = PrivacyManager()
    
    def _get_file_path(self, key: str) -> Path:
        """获取文件路径"""
        # 使用哈希避免文件名问题
        filename = hashlib.sha256(key.encode()).hexdigest()[:16] + ".json"
        return self.base_path / filename
    
    def save(self, key: str, data: Any, encrypt: bool = False) -> None:
        """保存数据到本地"""
        file_path = self._get_file_path(key)
        
        if encrypt:
            data = self.privacy_manager.encrypt_sensitive_data(data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                "key": key,
                "encrypted": encrypt,
                "data": data
            }, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"数据已保存到本地: {key}")
    
    def load(self, key: str, decrypt: bool = False) -> Optional[Any]:
        """从本地加载数据"""
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            stored = json.load(f)
        
        data = stored["data"]
        
        if decrypt or stored.get("encrypted"):
            data = self.privacy_manager.decrypt_sensitive_data(data)
        
        return data
    
    def delete(self, key: str) -> bool:
        """删除本地数据"""
        file_path = self._get_file_path(key)
        
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    def list_keys(self) -> list:
        """列出所有存储的键"""
        keys = []
        for file_path in self.base_path.glob("*.json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                keys.append(data.get("key", file_path.stem))
        return keys
    
    def clear(self) -> None:
        """清除所有本地数据"""
        for file_path in self.base_path.glob("*.json"):
            file_path.unlink()
        logger.info("本地存储已清除")


# 数据分类
class DataClassification:
    """数据分类"""
    
    PUBLIC = "public"           # 公开数据
    INTERNAL = "internal"       # 内部数据
    CONFIDENTIAL = "confidential"  # 机密数据
    RESTRICTED = "restricted"   # 受限数据


class DataClassifier:
    """数据分类器"""
    
    def __init__(self):
        self.classification_rules = {
            DataClassification.RESTRICTED: [
                r'\b\d{18}\b',  # 身份证号
                r'\b\d{16,19}\b',  # 银行卡号
                r'password|secret|token|key',
            ],
            DataClassification.CONFIDENTIAL: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # 邮箱
                r'\b1[3-9]\d{9}\b',  # 手机号
                r'location|address|coordinate',
            ],
            DataClassification.INTERNAL: [
                r'entity|relation|memory|note',
            ]
        }
    
    def classify(self, data: dict) -> str:
        """分类数据"""
        import re
        
        data_str = json.dumps(data, ensure_ascii=False).lower()
        
        for classification, patterns in self.classification_rules.items():
            for pattern in patterns:
                if re.search(pattern, data_str, re.IGNORECASE):
                    return classification
        
        return DataClassification.PUBLIC


# 全局隐私管理器实例
def get_privacy_manager() -> PrivacyManager:
    """获取全局隐私管理器"""
    return PrivacyManager()


# 示例用法
if __name__ == "__main__":
    # 创建隐私管理器
    config = PrivacyConfig(
        privacy_mode=False,
        encrypt_sensitive=True
    )
    
    pm = PrivacyManager(config)
    
    # 测试加密
    sensitive_data = {
        "username": "user123",
        "password": "mysecretpassword",
        "email": "user@example.com",
        "notes": "一些普通笔记"
    }
    
    encrypted = pm.encrypt_sensitive_data(sensitive_data)
    print("加密后:", json.dumps(encrypted, indent=2))
    
    decrypted = pm.decrypt_sensitive_data(encrypted)
    print("解密后:", json.dumps(decrypted, indent=2))
    
    # 测试本地存储
    storage = LocalFirstStorage()
    storage.save("test_key", sensitive_data, encrypt=True)
    loaded = storage.load("test_key", decrypt=True)
    print("加载的数据:", json.dumps(loaded, indent=2))
    
    # 测试数据分类
    classifier = DataClassifier()
    print("数据分类:", classifier.classify(sensitive_data))
    
    # 测试隐私模式
    print("可以上传云端:", pm.can_upload_to_cloud())
    
    sanitized = pm.sanitize_for_cloud(sensitive_data)
    print("云端数据:", json.dumps(sanitized, indent=2))

"""
数据导入功能模块
支持日历、邮件、社交媒体、照片元数据等多种数据源的导入
"""

import os
import re
import json
import email
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any, Iterator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ============== 数据模型 ==============

@dataclass
class Event:
    """通用事件模型"""
    id: str
    title: str
    start_time: datetime
    end_time: Optional[datetime] = None
    location: Optional[str] = None
    description: Optional[str] = None
    source: str = ""  # 来源：calendar, email, social, photo
    participants: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.participants is None:
            self.participants = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Contact:
    """联系人模型"""
    id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    source: str = ""
    interaction_count: int = 0
    last_interaction: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class PhotoMetadata:
    """照片元数据模型"""
    id: str
    file_path: str
    taken_time: Optional[datetime] = None
    location: Optional[Tuple[float, float]] = None  # (latitude, longitude)
    location_name: Optional[str] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    people: List[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.people is None:
            self.people = []
        if self.tags is None:
            self.tags = []


# ============== 日历导入 (.ics) ==============

class CalendarImporter:
    """ICS日历文件导入器"""
    
    def __init__(self):
        self.events: List[Event] = []
    
    def parse_datetime(self, dt_string: str) -> Optional[datetime]:
        """解析ICS日期时间格式"""
        if not dt_string:
            return None
        
        # 处理UTC格式 (20240115T120000Z)
        if 'Z' in dt_string:
            try:
                dt_string = dt_string.replace('Z', '')
                return datetime.strptime(dt_string, '%Y%m%dT%H%M%S')
            except ValueError:
                pass
        
        # 处理本地时间格式 (20240115T120000)
        try:
            if 'T' in dt_string:
                return datetime.strptime(dt_string, '%Y%m%dT%H%M%S')
            else:
                # 仅日期格式 (20240115)
                return datetime.strptime(dt_string, '%Y%m%d')
        except ValueError:
            pass
        
        # 尝试ISO格式
        try:
            return datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
        except ValueError:
            pass
        
        logger.warning(f"无法解析日期时间: {dt_string}")
        return None
    
    def parse_ics(self, ics_content: str) -> List[Event]:
        """解析ICS文件内容"""
        events = []
        
        # 按VEVENT分割
        event_blocks = re.findall(
            r'BEGIN:VEVENT(.*?)END:VEVENT',
            ics_content,
            re.DOTALL
        )
        
        for block in event_blocks:
            event_data = {}
            
            # 提取UID
            uid_match = re.search(r'UID:(.*?)(?:\r?\n|$)', block)
            uid = uid_match.group(1).strip() if uid_match else f"event_{len(events)}"
            
            # 提取标题
            summary_match = re.search(r'SUMMARY(?:;[^:]*)?:(.*?)(?:\r?\n|$)', block)
            title = summary_match.group(1).strip() if summary_match else "无标题"
            
            # 提取开始时间
            dtstart_match = re.search(r'DTSTART(?:;[^:]*)?:(.*?)(?:\r?\n|$)', block)
            start_time = self.parse_datetime(dtstart_match.group(1).strip()) if dtstart_match else None
            
            # 提取结束时间
            dtend_match = re.search(r'DTEND(?:;[^:]*)?:(.*?)(?:\r?\n|$)', block)
            end_time = self.parse_datetime(dtend_match.group(1).strip()) if dtend_match else None
            
            # 提取地点
            location_match = re.search(r'LOCATION(?:;[^:]*)?:(.*?)(?:\r?\n|$)', block)
            location = location_match.group(1).strip() if location_match else None
            
            # 提取描述
            description_match = re.search(r'DESCRIPTION(?:;[^:]*)?:(.*?)(?:END:|\r?\n[A-Z])', block, re.DOTALL)
            description = description_match.group(1).strip() if description_match else None
            # 处理ICS中的转义字符
            if description:
                description = description.replace('\\n', '\n').replace('\\,', ',')
            
            # 提取参与者
            attendees = re.findall(r'ATTENDEE[^:]*:(.*?)(?:\r?\n|$)', block)
            participants = []
            for attendee in attendees:
                # 提取邮箱
                email_match = re.search(r'mailto:(.*?)(?:\r?\n|$)', attendee)
                if email_match:
                    participants.append(email_match.group(1))
            
            if start_time:
                event = Event(
                    id=uid,
                    title=title,
                    start_time=start_time,
                    end_time=end_time,
                    location=location,
                    description=description,
                    source="calendar",
                    participants=participants,
                    metadata={
                        "raw_ics": block[:500]  # 保留原始数据的前500字符
                    }
                )
                events.append(event)
        
        self.events.extend(events)
        logger.info(f"从ICS解析了 {len(events)} 个事件")
        return events
    
    def import_file(self, file_path: str) -> List[Event]:
        """从文件导入日历"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return self.parse_ics(content)


# ============== 邮件导入 ==============

class EmailImporter:
    """邮件导入器 (.eml格式和Gmail API)"""
    
    def __init__(self):
        self.events: List[Event] = []
        self.contacts: Dict[str, Contact] = {}
    
    def parse_eml(self, eml_content: bytes) -> Optional[Event]:
        """解析.eml文件内容"""
        try:
            msg = email.message_from_bytes(eml_content)
            
            # 提取基本信息
            subject = msg.get('Subject', '')
            from_addr = msg.get('From', '')
            to_addr = msg.get('To', '')
            date_str = msg.get('Date', '')
            message_id = msg.get('Message-ID', '')
            
            # 解析日期
            received_time = None
            if date_str:
                try:
                    received_time = email.utils.parsedate_to_datetime(date_str)
                except Exception:
                    pass
            
            # 提取正文
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        try:
                            body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                            break
                        except:
                            pass
                    elif content_type == "text/html":
                        try:
                            html = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                            # 简单HTML到文本转换
                            body = re.sub(r'<[^>]+>', ' ', html)
                            body = re.sub(r'\s+', ' ', body).strip()
                        except:
                            pass
            else:
                try:
                    body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                except:
                    body = str(msg.get_payload())
            
            # 提取参与者邮箱
            participants = []
            for addr_field in [from_addr, to_addr, msg.get('Cc', ''), msg.get('Bcc', '')]:
                emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', addr_field)
                participants.extend(emails)
            participants = list(set(participants))
            
            # 更新联系人
            for email_addr in participants:
                if email_addr not in self.contacts:
                    self.contacts[email_addr] = Contact(
                        id=email_addr,
                        name=email_addr,
                        email=email_addr,
                        source="email"
                    )
                self.contacts[email_addr].interaction_count += 1
                if received_time:
                    self.contacts[email_addr].last_interaction = received_time
            
            event = Event(
                id=message_id or f"email_{hash(eml_content)}",
                title=subject or "无主题",
                start_time=received_time or datetime.now(),
                end_time=None,
                location=None,
                description=body[:2000] if body else None,  # 限制描述长度
                source="email",
                participants=participants,
                metadata={
                    "from": from_addr,
                    "to": to_addr,
                    "email_size": len(eml_content)
                }
            )
            
            return event
            
        except Exception as e:
            logger.error(f"解析邮件失败: {e}")
            return None
    
    def import_eml_file(self, file_path: str) -> Optional[Event]:
        """从.eml文件导入"""
        with open(file_path, 'rb') as f:
            content = f.read()
        event = self.parse_eml(content)
        if event:
            self.events.append(event)
        return event
    
    def import_eml_directory(self, directory: str) -> List[Event]:
        """从目录批量导入.eml文件"""
        events = []
        path = Path(directory)
        for eml_file in path.glob("**/*.eml"):
            event = self.import_eml_file(str(eml_file))
            if event:
                events.append(event)
        logger.info(f"从目录 {directory} 导入了 {len(events)} 封邮件")
        return events
    
    def get_contacts(self) -> List[Contact]:
        """获取提取的联系人列表"""
        return list(self.contacts.values())


# ============== 社交媒体导入 ==============

class SocialMediaImporter:
    """社交媒体数据导入器（微信/微博导出文件）"""
    
    def __init__(self):
        self.events: List[Event] = []
        self.contacts: Dict[str, Contact] = {}
    
    def parse_wechat_export(self, content: str) -> List[Event]:
        """
        解析微信聊天记录导出文件
        支持常见格式：CSV, JSON, 或特定格式的文本
        """
        events = []
        
        # 尝试JSON格式
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    event = self._parse_wechat_json_item(item)
                    if event:
                        events.append(event)
                return events
        except json.JSONDecodeError:
            pass
        
        # 尝试CSV格式解析
        lines = content.strip().split('\n')
        if len(lines) > 1 and ',' in lines[0]:
            # 假设是CSV格式
            for line in lines[1:]:  # 跳过标题行
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        # 假设格式：时间,发送者,内容
                        time_str = parts[0].strip()
                        sender = parts[1].strip()
                        message_content = ','.join(parts[2:]).strip()
                        
                        # 尝试解析时间
                        msg_time = self._parse_wechat_time(time_str)
                        
                        event = Event(
                            id=f"wechat_{hash(line)}",
                            title=f"微信消息: {sender}",
                            start_time=msg_time or datetime.now(),
                            description=message_content,
                            source="wechat",
                            participants=[sender],
                            metadata={
                                "sender": sender,
                                "message_type": "text"
                            }
                        )
                        events.append(event)
                        
                        # 更新联系人
                        if sender not in self.contacts:
                            self.contacts[sender] = Contact(
                                id=sender,
                                name=sender,
                                source="wechat",
                                interaction_count=1,
                                last_interaction=msg_time
                            )
                        else:
                            self.contacts[sender].interaction_count += 1
                            
                    except Exception as e:
                        logger.warning(f"解析微信消息行失败: {e}")
        
        # 尝试文本格式（微信自带的聊天记录导出格式）
        # 格式示例: "2024-01-15 10:30:45 张三: 消息内容"
        if not events:
            pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+([^:]+):\s*(.+)'
            matches = re.findall(pattern, content)
            
            for time_str, sender, message_content in matches:
                msg_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                
                event = Event(
                    id=f"wechat_{hash(time_str + sender + message_content)}",
                    title=f"微信消息: {sender}",
                    start_time=msg_time,
                    description=message_content,
                    source="wechat",
                    participants=[sender],
                    metadata={
                        "sender": sender,
                        "message_type": "text"
                    }
                )
                events.append(event)
                
                # 更新联系人
                if sender not in self.contacts:
                    self.contacts[sender] = Contact(
                        id=sender,
                        name=sender,
                        source="wechat",
                        interaction_count=1,
                        last_interaction=msg_time
                    )
                else:
                    self.contacts[sender].interaction_count += 1
        
        self.events.extend(events)
        logger.info(f"从微信导出解析了 {len(events)} 条消息")
        return events
    
    def _parse_wechat_json_item(self, item: Dict) -> Optional[Event]:
        """解析微信JSON格式的单条消息"""
        try:
            msg_time = datetime.fromtimestamp(item.get('time', 0))
            sender = item.get('sender', '未知')
            content = item.get('content', '')
            
            return Event(
                id=f"wechat_{item.get('msg_id', hash(str(item)))}",
                title=f"微信消息: {sender}",
                start_time=msg_time,
                description=content,
                source="wechat",
                participants=[sender],
                metadata=item
            )
        except Exception as e:
            logger.warning(f"解析微信JSON项失败: {e}")
            return None
    
    def _parse_wechat_time(self, time_str: str) -> Optional[datetime]:
        """解析微信时间字符串"""
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%Y年%m月%d日 %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%H:%M:%S'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        
        # 如果只有时间，假设是今天
        try:
            time_part = datetime.strptime(time_str, '%H:%M')
            now = datetime.now()
            return now.replace(hour=time_part.hour, minute=time_part.minute, second=0)
        except ValueError:
            pass
        
        return None
    
    def parse_weibo_export(self, content: str) -> List[Event]:
        """解析微博导出文件"""
        events = []
        
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    event = self._parse_weibo_item(item)
                    if event:
                        events.append(event)
            elif isinstance(data, dict) and 'cards' in data:
                # 微博卡片格式
                for card in data['cards']:
                    event = self._parse_weibo_card(card)
                    if event:
                        events.append(event)
        except json.JSONDecodeError:
            # 尝试文本格式
            logger.warning("微博导出文件不是标准JSON格式")
        
        self.events.extend(events)
        logger.info(f"从微博导出解析了 {len(events)} 条内容")
        return events
    
    def _parse_weibo_item(self, item: Dict) -> Optional[Event]:
        """解析微博单条数据"""
        try:
            created_at = item.get('created_at', '')
            # 微博时间格式: "Mon Jan 15 10:30:45 +0800 2024"
            msg_time = self._parse_weibo_time(created_at)
            
            text = item.get('text', '')
            # 去除HTML标签
            text = re.sub(r'<[^>]+>', '', text)
            
            return Event(
                id=f"weibo_{item.get('id', hash(str(item)))}",
                title=text[:50] + "..." if len(text) > 50 else text,
                start_time=msg_time or datetime.now(),
                description=text,
                source="weibo",
                participants=[item.get('user', {}).get('screen_name', '未知用户')],
                metadata={
                    "reposts_count": item.get('reposts_count', 0),
                    "comments_count": item.get('comments_count', 0),
                    "attitudes_count": item.get('attitudes_count', 0)
                }
            )
        except Exception as e:
            logger.warning(f"解析微博项失败: {e}")
            return None
    
    def _parse_weibo_card(self, card: Dict) -> Optional[Event]:
        """解析微博卡片"""
        mblog = card.get('mblog', {})
        return self._parse_weibo_item(mblog)
    
    def _parse_weibo_time(self, time_str: str) -> Optional[datetime]:
        """解析微博时间字符串"""
        try:
            # 微博格式: "Mon Jan 15 10:30:45 +0800 2024"
            return datetime.strptime(time_str, '%a %b %d %H:%M:%S %z %Y')
        except ValueError:
            pass
        
        try:
            # 另一种格式
            return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            pass
        
        return None
    
    def import_file(self, file_path: str, platform: str = "auto") -> List[Event]:
        """从文件导入社交媒体数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if platform == "auto":
            # 自动检测平台
            if "wechat" in file_path.lower() or "微信" in file_path:
                platform = "wechat"
            elif "weibo" in file_path.lower() or "微博" in file_path:
                platform = "weibo"
        
        if platform == "wechat":
            return self.parse_wechat_export(content)
        elif platform == "weibo":
            return self.parse_weibo_export(content)
        else:
            # 尝试两种格式
            events = self.parse_wechat_export(content)
            if not events:
                events = self.parse_weibo_export(content)
            return events
    
    def get_contacts(self) -> List[Contact]:
        """获取提取的联系人列表"""
        return list(self.contacts.values())


# ============== 照片元数据提取 ==============

class PhotoMetadataImporter:
    """照片元数据提取器 (EXIF)"""
    
    def __init__(self):
        self.photos: List[PhotoMetadata] = []
    
    def extract_exif(self, image_path: str) -> Optional[PhotoMetadata]:
        """从图片文件提取EXIF元数据"""
        try:
            # 尝试导入PIL
            try:
                from PIL import Image
                from PIL.ExifTags import TAGS, GPSTAGS
                has_pil = True
            except ImportError:
                has_pil = False
                logger.warning("PIL未安装，使用基础文件信息")
            
            photo_id = f"photo_{hash(image_path)}"
            taken_time = None
            location = None
            location_name = None
            camera_make = None
            camera_model = None
            
            if has_pil:
                with Image.open(image_path) as img:
                    exif = img._getexif()
                    
                    if exif:
                        exif_data = {}
                        for tag_id, value in exif.items():
                            tag = TAGS.get(tag_id, tag_id)
                            exif_data[tag] = value
                        
                        # 提取拍摄时间
                        date_time = exif_data.get('DateTime') or exif_data.get('DateTimeOriginal')
                        if date_time:
                            try:
                                taken_time = datetime.strptime(date_time, '%Y:%m:%d %H:%M:%S')
                            except ValueError:
                                pass
                        
                        # 提取GPS信息
                        gps_info = exif_data.get('GPSInfo')
                        if gps_info:
                            location = self._parse_gps_info(gps_info)
                        
                        # 提取相机信息
                        camera_make = exif_data.get('Make')
                        camera_model = exif_data.get('Model')
            
            # 如果PIL不可用或没有EXIF，使用文件修改时间
            if not taken_time:
                stat = os.stat(image_path)
                taken_time = datetime.fromtimestamp(stat.st_mtime)
            
            # 尝试反向地理编码获取地点名称（如果有GPS）
            if location and not location_name:
                location_name = self._reverse_geocode(location)
            
            photo = PhotoMetadata(
                id=photo_id,
                file_path=image_path,
                taken_time=taken_time,
                location=location,
                location_name=location_name,
                camera_make=camera_make,
                camera_model=camera_model,
                people=[],
                tags=[]
            )
            
            return photo
            
        except Exception as e:
            logger.error(f"提取照片元数据失败 {image_path}: {e}")
            return None
    
    def _parse_gps_info(self, gps_info) -> Optional[Tuple[float, float]]:
        """解析GPS信息为经纬度"""
        try:
            # PIL的GPSTAGS
            from PIL.ExifTags import GPSTAGS
            
            gps_data = {}
            for key in gps_info.keys():
                decode = GPSTAGS.get(key, key)
                gps_data[decode] = gps_info[key]
            
            def convert_to_degrees(value):
                """将GPS坐标转换为度数"""
                d = float(value[0])
                m = float(value[1])
                s = float(value[2])
                return d + (m / 60.0) + (s / 3600.0)
            
            lat = None
            lon = None
            
            if 'GPSLatitude' in gps_data and 'GPSLatitudeRef' in gps_data:
                lat = convert_to_degrees(gps_data['GPSLatitude'])
                if gps_data['GPSLatitudeRef'] != 'N':
                    lat = -lat
            
            if 'GPSLongitude' in gps_data and 'GPSLongitudeRef' in gps_data:
                lon = convert_to_degrees(gps_data['GPSLongitude'])
                if gps_data['GPSLongitudeRef'] != 'E':
                    lon = -lon
            
            if lat is not None and lon is not None:
                return (lat, lon)
            
        except Exception as e:
            logger.warning(f"解析GPS信息失败: {e}")
        
        return None
    
    def _reverse_geocode(self, location: Tuple[float, float]) -> Optional[str]:
        """反向地理编码（简化版本）"""
        # 实际项目中可以调用地理编码API
        # 这里返回一个简单的占位符
        return f"({location[0]:.4f}, {location[1]:.4f})"
    
    def import_directory(self, directory: str, recursive: bool = True) -> List[PhotoMetadata]:
        """从目录批量导入照片"""
        photos = []
        path = Path(directory)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        
        pattern = "**/*" if recursive else "*"
        for img_file in path.glob(pattern):
            if img_file.suffix.lower() in image_extensions:
                photo = self.extract_exif(str(img_file))
                if photo:
                    photos.append(photo)
        
        self.photos.extend(photos)
        logger.info(f"从目录 {directory} 导入了 {len(photos)} 张照片")
        return photos
    
    def photos_to_events(self) -> List[Event]:
        """将照片转换为事件"""
        events = []
        for photo in self.photos:
            event = Event(
                id=photo.id,
                title=f"照片: {Path(photo.file_path).name}",
                start_time=photo.taken_time or datetime.now(),
                end_time=None,
                location=photo.location_name,
                description=f"拍摄于 {photo.location_name or '未知位置'}",
                source="photo",
                participants=photo.people,
                metadata={
                    "file_path": photo.file_path,
                    "camera": f"{photo.camera_make or ''} {photo.camera_model or ''}".strip(),
                    "gps": photo.location
                }
            )
            events.append(event)
        return events


# ============== 数据存储 ==============

class DataStore:
    """统一数据存储（SQLite）"""
    
    def __init__(self, db_path: str = "data_imported.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 事件表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    location TEXT,
                    description TEXT,
                    source TEXT,
                    participants TEXT,  -- JSON array
                    metadata TEXT       -- JSON object
                )
            ''')
            
            # 联系人表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS contacts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT,
                    phone TEXT,
                    source TEXT,
                    interaction_count INTEGER DEFAULT 0,
                    last_interaction TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            # 照片表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS photos (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    taken_time TIMESTAMP,
                    latitude REAL,
                    longitude REAL,
                    location_name TEXT,
                    camera_make TEXT,
                    camera_model TEXT,
                    people TEXT,  -- JSON array
                    tags TEXT     -- JSON array
                )
            ''')
            
            conn.commit()
    
    def save_events(self, events: List[Event]):
        """保存事件到数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for event in events:
                cursor.execute('''
                    INSERT OR REPLACE INTO events 
                    (id, title, start_time, end_time, location, description, source, participants, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.id,
                    event.title,
                    event.start_time,
                    event.end_time,
                    event.location,
                    event.description,
                    event.source,
                    json.dumps(event.participants),
                    json.dumps(event.metadata)
                ))
            conn.commit()
    
    def save_contacts(self, contacts: List[Contact]):
        """保存联系人到数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for contact in contacts:
                cursor.execute('''
                    INSERT OR REPLACE INTO contacts 
                    (id, name, email, phone, source, interaction_count, last_interaction, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    contact.id,
                    contact.name,
                    contact.email,
                    contact.phone,
                    contact.source,
                    contact.interaction_count,
                    contact.last_interaction,
                    json.dumps(contact.metadata)
                ))
            conn.commit()
    
    def save_photos(self, photos: List[PhotoMetadata]):
        """保存照片到数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for photo in photos:
                cursor.execute('''
                    INSERT OR REPLACE INTO photos 
                    (id, file_path, taken_time, latitude, longitude, location_name, 
                     camera_make, camera_model, people, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    photo.id,
                    photo.file_path,
                    photo.taken_time,
                    photo.location[0] if photo.location else None,
                    photo.location[1] if photo.location else None,
                    photo.location_name,
                    photo.camera_make,
                    photo.camera_model,
                    json.dumps(photo.people),
                    json.dumps(photo.tags)
                ))
            conn.commit()
    
    def get_events(self, source: Optional[str] = None, 
                   start: Optional[datetime] = None,
                   end: Optional[datetime] = None) -> List[Event]:
        """查询事件"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM events WHERE 1=1"
            params = []
            
            if source:
                query += " AND source = ?"
                params.append(source)
            if start:
                query += " AND start_time >= ?"
                params.append(start)
            if end:
                query += " AND start_time <= ?"
                params.append(end)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            events = []
            for row in rows:
                events.append(Event(
                    id=row['id'],
                    title=row['title'],
                    start_time=row['start_time'],
                    end_time=row['end_time'],
                    location=row['location'],
                    description=row['description'],
                    source=row['source'],
                    participants=json.loads(row['participants']) if row['participants'] else [],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                ))
            
            return events


# ============== 统一导入接口 ==============

class DataImportManager:
    """统一数据导入管理器"""
    
    def __init__(self, db_path: str = "data_imported.db"):
        self.calendar_importer = CalendarImporter()
        self.email_importer = EmailImporter()
        self.social_importer = SocialMediaImporter()
        self.photo_importer = PhotoMetadataImporter()
        self.store = DataStore(db_path)
    
    def import_calendar(self, file_path: str) -> List[Event]:
        """导入日历"""
        events = self.calendar_importer.import_file(file_path)
        self.store.save_events(events)
        return events
    
    def import_email(self, file_path: str) -> Optional[Event]:
        """导入单封邮件"""
        event = self.email_importer.import_eml_file(file_path)
        if event:
            self.store.save_events([event])
            self.store.save_contacts(self.email_importer.get_contacts())
        return event
    
    def import_email_directory(self, directory: str) -> List[Event]:
        """批量导入邮件目录"""
        events = self.email_importer.import_eml_directory(directory)
        self.store.save_events(events)
        self.store.save_contacts(self.email_importer.get_contacts())
        return events
    
    def import_social_media(self, file_path: str, platform: str = "auto") -> List[Event]:
        """导入社交媒体数据"""
        events = self.social_importer.import_file(file_path, platform)
        self.store.save_events(events)
        self.store.save_contacts(self.social_importer.get_contacts())
        return events
    
    def import_photos(self, directory: str, recursive: bool = True) -> List[PhotoMetadata]:
        """导入照片"""
        photos = self.photo_importer.import_directory(directory, recursive)
        self.store.save_photos(photos)
        # 同时保存为事件
        events = self.photo_importer.photos_to_events()
        self.store.save_events(events)
        return photos
    
    def get_all_events(self) -> List[Event]:
        """获取所有事件"""
        return self.store.get_events()
    
    def get_events_by_source(self, source: str) -> List[Event]:
        """按来源获取事件"""
        return self.store.get_events(source=source)
    
    def get_timeline(self, start: Optional[datetime] = None, 
                     end: Optional[datetime] = None) -> List[Event]:
        """获取时间线"""
        return self.store.get_events(start=start, end=end)


# ============== 使用示例 ==============

def example_usage():
    """使用示例"""
    # 初始化管理器
    manager = DataImportManager()
    
    # 示例1: 导入日历文件
    # events = manager.import_calendar("calendar.ics")
    # print(f"导入 {len(events)} 个日历事件")
    
    # 示例2: 导入邮件目录
    # email_events = manager.import_email_directory("./emails")
    # print(f"导入 {len(email_events)} 封邮件")
    
    # 示例3: 导入社交媒体
    # social_events = manager.import_social_media("wechat_export.txt", platform="wechat")
    # print(f"导入 {len(social_events)} 条社交消息")
    
    # 示例4: 导入照片
    # photos = manager.import_photos("./photos", recursive=True)
    # print(f"导入 {len(photos)} 张照片")
    
    # 示例5: 获取时间线
    # from datetime import datetime, timedelta
    # last_week = datetime.now() - timedelta(days=7)
    # timeline = manager.get_timeline(start=last_week)
    # for event in timeline[:10]:
    #     print(f"[{event.source}] {event.start_time}: {event.title}")
    
    print("数据导入模块初始化完成")


if __name__ == "__main__":
    example_usage()

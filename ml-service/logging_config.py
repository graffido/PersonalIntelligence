"""
结构化日志配置 - JSON格式、日志轮转、错误上报
"""
import json
import logging
import logging.handlers
import sys
import traceback
from datetime import datetime
from typing import Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import functools

# 可选的Sentry集成
try:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False


class JSONFormatter(logging.Formatter):
    """JSON格式日志格式化器"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # 添加额外字段
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # 添加上下文信息
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "session_id"):
            log_data["session_id"] = record.session_id
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


@dataclass
class LogConfig:
    """日志配置"""
    level: str = "INFO"
    log_dir: str = "logs"
    app_name: str = "pos-service"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 10
    enable_console: bool = True
    enable_file: bool = True
    json_format: bool = True
    
    # Sentry配置
    sentry_dsn: Optional[str] = None
    sentry_environment: str = "development"
    sentry_traces_sample_rate: float = 0.1


class ContextFilter(logging.Filter):
    """添加上下文信息的日志过滤器"""
    
    def __init__(self):
        super().__init__()
        self.context = {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        # 添加上下文信息到记录
        for key, value in self.context.items():
            setattr(record, key, value)
        return True
    
    def set_context(self, **kwargs):
        """设置上下文"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """清除上下文"""
        self.context.clear()


# 全局上下文过滤器
context_filter = ContextFilter()


def setup_logging(config: LogConfig = None) -> logging.Logger:
    """设置日志系统"""
    config = config or LogConfig()
    
    # 创建logger
    logger = logging.getLogger(config.app_name)
    logger.setLevel(getattr(logging, config.level.upper()))
    logger.handlers = []  # 清除现有处理器
    logger.addFilter(context_filter)
    
    formatter = JSONFormatter() if config.json_format else logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    if config.enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器（带轮转）
    if config.enable_file:
        log_path = Path(config.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # 应用日志
        app_handler = logging.handlers.RotatingFileHandler(
            log_path / f"{config.app_name}.log",
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        app_handler.setFormatter(formatter)
        logger.addHandler(app_handler)
        
        # 错误日志（单独文件）
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / f"{config.app_name}.error.log",
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
    
    # Sentry集成
    if SENTRY_AVAILABLE and config.sentry_dsn:
        sentry_logging = LoggingIntegration(
            level=logging.INFO,
            event_level=logging.ERROR
        )
        
        sentry_sdk.init(
            dsn=config.sentry_dsn,
            environment=config.sentry_environment,
            traces_sample_rate=config.sentry_traces_sample_rate,
            integrations=[sentry_logging]
        )
        
        logger.info("Sentry集成已启用")
    
    return logger


# 结构化日志记录函数
def log_structured(
    logger: logging.Logger,
    level: str,
    message: str,
    extra: dict = None,
    **kwargs
):
    """记录结构化日志"""
    extra_data = extra or {}
    extra_data.update(kwargs)
    
    record = logger.makeRecord(
        logger.name,
        getattr(logging, level.upper()),
        "",
        0,
        message,
        (),
        None
    )
    record.extra = extra_data
    
    logger.handle(record)


class LoggerMixin:
    """日志混入类"""
    
    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(self.__class__.__module__)
        return self._logger
    
    def log_info(self, message: str, **kwargs):
        """记录信息日志"""
        log_structured(self.logger, "INFO", message, **kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """记录警告日志"""
        log_structured(self.logger, "WARNING", message, **kwargs)
    
    def log_error(self, message: str, **kwargs):
        """记录错误日志"""
        log_structured(self.logger, "ERROR", message, **kwargs)
    
    def log_debug(self, message: str, **kwargs):
        """记录调试日志"""
        log_structured(self.logger, "DEBUG", message, **kwargs)


def log_execution_time(logger: logging.Logger = None, level: str = "INFO"):
    """记录函数执行时间的装饰器"""
    def decorator(func):
        log = logger or logging.getLogger(func.__module__)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = datetime.utcnow()
            try:
                result = await func(*args, **kwargs)
                duration = (datetime.utcnow() - start).total_seconds()
                log_structured(
                    log, level,
                    f"函数 {func.__name__} 执行完成",
                    duration_seconds=duration,
                    function=func.__name__
                )
                return result
            except Exception as e:
                duration = (datetime.utcnow() - start).total_seconds()
                log_structured(
                    log, "ERROR",
                    f"函数 {func.__name__} 执行失败",
                    duration_seconds=duration,
                    function=func.__name__,
                    error=str(e)
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = datetime.utcnow()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.utcnow() - start).total_seconds()
                log_structured(
                    log, level,
                    f"函数 {func.__name__} 执行完成",
                    duration_seconds=duration,
                    function=func.__name__
                )
                return result
            except Exception as e:
                duration = (datetime.utcnow() - start).total_seconds()
                log_structured(
                    log, "ERROR",
                    f"函数 {func.__name__} 执行失败",
                    duration_seconds=duration,
                    function=func.__name__,
                    error=str(e)
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# 请求日志中间件（FastAPI/Starlette）
import asyncio

class RequestLoggingMiddleware:
    """请求日志中间件"""
    
    def __init__(self, app, logger: logging.Logger = None):
        self.app = app
        self.logger = logger or logging.getLogger("pos.request")
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = datetime.utcnow()
        request_id = scope.get("headers", {}).get(b"x-request-id", [b"unknown"])[0].decode()
        
        # 设置上下文
        context_filter.set_context(
            request_id=request_id,
            method=scope["method"],
            path=scope["path"]
        )
        
        try:
            await self.app(scope, receive, send)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            log_structured(
                self.logger, "INFO",
                f"{scope['method']} {scope['path']} 完成",
                duration_seconds=duration,
                status_code=200  # 实际应从响应中获取
            )
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            log_structured(
                self.logger, "ERROR",
                f"{scope['method']} {scope['path']} 失败",
                duration_seconds=duration,
                error=str(e)
            )
            raise
        finally:
            context_filter.clear_context()


# 错误上报函数
def report_error(
    error: Exception,
    context: dict = None,
    level: str = "error"
):
    """手动上报错误"""
    if SENTRY_AVAILABLE and sentry_sdk.Hub.current.client:
        with sentry_sdk.push_scope() as scope:
            if context:
                for key, value in context.items():
                    scope.set_extra(key, value)
            
            if level == "fatal":
                sentry_sdk.capture_exception(error, level="fatal")
            elif level == "warning":
                sentry_sdk.capture_exception(error, level="warning")
            else:
                sentry_sdk.capture_exception(error)
    
    # 同时记录到日志
    logger = logging.getLogger("pos.errors")
    log_structured(
        logger, level.upper(),
        f"错误上报: {str(error)}",
        error_type=error.__class__.__name__,
        error_message=str(error),
        **(context or {})
    )


# 示例用法
if __name__ == "__main__":
    # 配置并设置日志
    config = LogConfig(
        level="DEBUG",
        json_format=True,
        enable_file=True
    )
    
    logger = setup_logging(config)
    
    # 测试不同级别的日志
    logger.debug("调试信息")
    logger.info("普通信息")
    logger.warning("警告信息")
    
    # 测试结构化日志
    log_structured(
        logger, "INFO",
        "用户登录",
        user_id="12345",
        ip_address="192.168.1.1",
        user_agent="Mozilla/5.0"
    )
    
    # 测试异常日志
    try:
        1 / 0
    except Exception:
        logger.exception("发生错误")

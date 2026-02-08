"""
系统健壮性模块 - 重试机制、熔断器、降级策略、健康检查
"""
import asyncio
import functools
import time
import random
from enum import Enum
from typing import Callable, TypeVar, Optional, Any, List
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"      # 正常状态，允许请求通过
    OPEN = "open"          # 熔断状态，拒绝请求
    HALF_OPEN = "half_open"  # 半开状态，允许试探性请求


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5           # 失败阈值
    recovery_timeout: float = 30.0       # 恢复超时时间（秒）
    half_open_max_calls: int = 3         # 半开状态最大试探请求数
    success_threshold: int = 2           # 半开状态成功阈值


@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3                # 最大重试次数
    base_delay: float = 1.0              # 基础延迟（秒）
    max_delay: float = 60.0              # 最大延迟（秒）
    exponential_base: float = 2.0        # 指数基数
    jitter: bool = True                  # 是否添加随机抖动
    retryable_exceptions: tuple = (Exception,)  # 可重试的异常


@dataclass
class HealthStatus:
    """健康状态"""
    status: str  # "healthy", "degraded", "unhealthy"
    checks: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class CircuitBreaker:
    """熔断器实现"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """执行被保护的函数"""
        async with self._lock:
            await self._update_state()
            
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(f"熔断器 '{self.name}' 已打开")
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerOpenError(f"熔断器 '{self.name}' 半开状态请求数已达上限")
                self.half_open_calls += 1
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise
    
    async def _update_state(self):
        """更新熔断器状态"""
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and \
               (time.time() - self.last_failure_time) >= self.config.recovery_timeout:
                logger.info(f"熔断器 '{self.name}' 进入半开状态")
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                self.success_count = 0
    
    async def _on_success(self):
        """成功回调"""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    logger.info(f"熔断器 '{self.name}' 关闭")
                    self._reset()
            else:
                self.failure_count = max(0, self.failure_count - 1)
    
    async def _on_failure(self):
        """失败回调"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                logger.warning(f"熔断器 '{self.name}' 半开状态失败，重新打开")
                self.state = CircuitState.OPEN
            elif self.failure_count >= self.config.failure_threshold:
                logger.warning(f"熔断器 '{self.name}' 打开，失败次数: {self.failure_count}")
                self.state = CircuitState.OPEN
    
    def _reset(self):
        """重置熔断器"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.last_failure_time = None


class CircuitBreakerOpenError(Exception):
    """熔断器打开异常"""
    pass


class RetryPolicy:
    """重试策略"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
    
    def _calculate_delay(self, attempt: int) -> float:
        """计算延迟时间（指数退避 + 抖动）"""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        if self.config.jitter:
            # 添加 ±20% 的抖动
            jitter = delay * 0.2 * (2 * random.random() - 1)
            delay += jitter
        
        return delay
    
    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """执行带重试的函数"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except self.config.retryable_exceptions as e:
                last_exception = e
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"函数执行失败（尝试 {attempt + 1}/{self.config.max_attempts}），"
                        f"{delay:.2f}秒后重试: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"函数执行失败，已达最大重试次数: {e}")
        
        raise last_exception


def with_retry(config: RetryConfig = None):
    """重试装饰器"""
    policy = RetryPolicy(config)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await policy.execute(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 同步函数不支持自动重试
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


class FallbackManager:
    """降级策略管理器"""
    
    def __init__(self):
        self.fallbacks: dict[str, Callable] = {}
        self.cache: dict[str, Any] = {}
        self.cache_ttl: dict[str, float] = {}
    
    def register(self, name: str, fallback_func: Callable):
        """注册降级函数"""
        self.fallbacks[name] = fallback_func
    
    async def execute_with_fallback(
        self,
        name: str,
        primary_func: Callable[..., T],
        *args,
        use_cache: bool = True,
        cache_ttl: float = 300,
        **kwargs
    ) -> T:
        """执行带降级的函数"""
        # 尝试执行主函数
        try:
            result = await primary_func(*args, **kwargs)
            
            # 缓存结果
            if use_cache:
                self.cache[name] = result
                self.cache_ttl[name] = time.time() + cache_ttl
            
            return result
        except Exception as e:
            logger.warning(f"主函数 '{name}' 执行失败: {e}，尝试降级")
            
            # 尝试使用缓存
            if use_cache and name in self.cache:
                if time.time() < self.cache_ttl.get(name, 0):
                    logger.info(f"使用缓存结果作为降级: {name}")
                    return self.cache[name]
            
            # 尝试执行降级函数
            if name in self.fallbacks:
                fallback_func = self.fallbacks[name]
                logger.info(f"执行降级函数: {name}")
                
                if asyncio.iscoroutinefunction(fallback_func):
                    return await fallback_func(*args, **kwargs)
                else:
                    return fallback_func(*args, **kwargs)
            
            # 无降级可用，抛出异常
            raise FallbackError(f"主函数失败且无可用降级: {e}") from e


class FallbackError(Exception):
    """降级错误"""
    pass


class HealthChecker:
    """健康检查器"""
    
    def __init__(self):
        self.checks: dict[str, Callable[[], Any]] = {}
        self.status_history: List[HealthStatus] = []
        self.max_history = 100
    
    def register(self, name: str, check_func: Callable[[], Any]):
        """注册健康检查"""
        self.checks[name] = check_func
    
    async def check(self) -> HealthStatus:
        """执行所有健康检查"""
        results = {}
        healthy_count = 0
        degraded_count = 0
        
        for name, check_func in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                results[name] = {
                    "status": "healthy" if result else "unhealthy",
                    "result": result if isinstance(result, dict) else {"value": result}
                }
                
                if result:
                    healthy_count += 1
                else:
                    degraded_count += 1
                    
            except Exception as e:
                results[name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                degraded_count += 1
        
        # 确定整体状态
        total = len(self.checks)
        if healthy_count == total:
            status = "healthy"
        elif degraded_count < total / 2:
            status = "degraded"
        else:
            status = "unhealthy"
        
        health_status = HealthStatus(
            status=status,
            checks=results
        )
        
        # 保存历史
        self.status_history.append(health_status)
        if len(self.status_history) > self.max_history:
            self.status_history.pop(0)
        
        return health_status
    
    def get_history(self, limit: int = 10) -> List[HealthStatus]:
        """获取健康检查历史"""
        return self.status_history[-limit:]


# 全局实例
circuit_breakers: dict[str, CircuitBreaker] = {}
fallback_manager = FallbackManager()
health_checker = HealthChecker()


def get_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """获取或创建熔断器"""
    if name not in circuit_breakers:
        circuit_breakers[name] = CircuitBreaker(name, config)
    return circuit_breakers[name]


async def with_circuit_breaker(
    name: str,
    func: Callable[..., T],
    config: CircuitBreakerConfig = None,
    *args,
    **kwargs
) -> T:
    """使用熔断器执行函数"""
    breaker = get_circuit_breaker(name, config)
    return await breaker.call(func, *args, **kwargs)


# 常用降级策略
class FallbackStrategies:
    """预定义的降级策略"""
    
    @staticmethod
    def empty_list(*args, **kwargs):
        """返回空列表"""
        return []
    
    @staticmethod
    def empty_dict(*args, **kwargs):
        """返回空字典"""
        return {}
    
    @staticmethod
    def none_value(*args, **kwargs):
        """返回 None"""
        return None
    
    @staticmethod
    def cached_result(cache_key: str):
        """返回缓存结果"""
        def wrapper(*args, **kwargs):
            return fallback_manager.cache.get(cache_key)
        return wrapper
    
    @staticmethod
    def default_value(value: Any):
        """返回默认值"""
        def wrapper(*args, **kwargs):
            return value
        return wrapper
    
    @staticmethod
    def llm_simple_response(*args, **kwargs):
        """LLM不可用时的简单响应"""
        return {
            "content": "AI服务暂时不可用，请稍后再试。",
            "fallback": True,
            "timestamp": time.time()
        }


# 示例用法
if __name__ == "__main__":
    async def main():
        # 测试重试
        @with_retry(RetryConfig(max_attempts=3, base_delay=0.5))
        async def unstable_function():
            if random.random() < 0.7:
                raise Exception("随机失败")
            return "成功!"
        
        try:
            result = await unstable_function()
            print(f"结果: {result}")
        except Exception as e:
            print(f"最终失败: {e}")
        
        # 测试熔断器
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2))
        
        async def failing_function():
            raise Exception("总是失败")
        
        for i in range(5):
            try:
                await breaker.call(failing_function)
            except Exception as e:
                print(f"尝试 {i+1}: {e}")
    
    asyncio.run(main())

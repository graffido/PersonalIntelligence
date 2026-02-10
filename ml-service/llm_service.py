"""
LLM服务 - 集成多种大语言模型
支持OpenAI/Claude/Kimi API和本地模型（llama.cpp/ollama）
智能路由：简单任务本地，复杂任务API
"""
import os
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, AsyncGenerator, Generator
from dataclasses import dataclass, field
from enum import Enum
import hashlib


class ModelType(Enum):
    """模型类型"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    KIMI = "kimi"              # Moonshot AI / Kimi
    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    LOCAL = "local"


class TaskComplexity(Enum):
    """任务复杂度"""
    SIMPLE = "simple"        # 简单任务：问候、简短回答
    MODERATE = "moderate"    # 中等任务：一般问答
    COMPLEX = "complex"      # 复杂任务：分析、推理、长文本


@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    model: str
    model_type: ModelType
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "model": self.model,
            "model_type": self.model_type.value,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata
        }


@dataclass
class Message:
    """对话消息"""
    role: str  # system, user, assistant
    content: str
    name: Optional[str] = None
    
    def to_dict(self) -> Dict:
        msg = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        return msg
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        return cls(
            role=data["role"],
            content=data["content"],
            name=data.get("name")
        )


class PromptCache:
    """Prompt缓存"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict] = {}
    
    def _get_key(self, messages: List[Message], model: str, **kwargs) -> str:
        """生成缓存键"""
        content = json.dumps({
            "messages": [m.to_dict() for m in messages],
            "model": model,
            **kwargs
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, messages: List[Message], model: str, **kwargs) -> Optional[LLMResponse]:
        """获取缓存"""
        key = self._get_key(messages, model, **kwargs)
        
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                return entry["response"]
            else:
                del self.cache[key]
        
        return None
    
    def set(self, messages: List[Message], model: str, response: LLMResponse, **kwargs):
        """设置缓存"""
        key = self._get_key(messages, model, **kwargs)
        
        if len(self.cache) >= self.max_size:
            # 移除最旧的
            oldest = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest]
        
        self.cache[key] = {
            "response": response,
            "timestamp": time.time()
        }
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()


class BaseLLMProvider:
    """LLM Provider基类"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config.get("model", "")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 1024)
    
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """聊天接口"""
        raise NotImplementedError
    
    def stream_chat(self, messages: List[Message], **kwargs) -> Generator[str, None, None]:
        """流式聊天接口"""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """检查是否可用"""
        raise NotImplementedError


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API Provider"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except ImportError:
                print("OpenAI package not installed")
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.model_name),
            messages=[m.to_dict() for m in messages],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            **{k: v for k, v in kwargs.items() if k not in ["model", "temperature", "max_tokens"]}
        )
        
        latency = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            model_type=ModelType.OPENAI,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            latency_ms=latency,
            finish_reason=response.choices[0].finish_reason
        )
    
    def stream_chat(self, messages: List[Message], **kwargs) -> Generator[str, None, None]:
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        stream = self.client.chat.completions.create(
            model=kwargs.get("model", self.model_name),
            messages=[m.to_dict() for m in messages],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
            **{k: v for k, v in kwargs.items() if k not in ["model", "temperature", "max_tokens"]}
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class KimiProvider(BaseLLMProvider):
    """
    Kimi (Moonshot AI) Provider
    支持Kimi Code / kimi-k2.5 等模型
    使用OpenAI兼容API格式
    """
    
    # Kimi 模型列表
    AVAILABLE_MODELS = [
        "kimi-k2.5",
        "kimi-latest", 
        "moonshot-v1-8k",
        "moonshot-v1-32k",
        "moonshot-v1-128k"
    ]
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
        self.base_url = config.get("base_url", "https://api.moonshot.cn/v1")
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key, 
                    base_url=self.base_url
                )
            except ImportError:
                print("OpenAI package not installed (required for Kimi API)")
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        if not self.client:
            raise RuntimeError("Kimi client not initialized. Please set KIMI_API_KEY environment variable.")
        
        start_time = time.time()
        
        model = kwargs.get("model", self.model_name)
        # 默认使用 kimi-k2.5
        if not model or model == "default":
            model = "kimi-k2.5"
        
        # Kimi 支持的工具调用参数
        extra_kwargs = {}
        if "tools" in kwargs:
            extra_kwargs["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs:
            extra_kwargs["tool_choice"] = kwargs["tool_choice"]
        if "response_format" in kwargs:
            extra_kwargs["response_format"] = kwargs["response_format"]
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[m.to_dict() for m in messages],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            **extra_kwargs
        )
        
        latency = (time.time() - start_time) * 1000
        
        # 处理可能的 tool_calls
        message = response.choices[0].message
        content = message.content or ""
        metadata = {}
        if hasattr(message, 'tool_calls') and message.tool_calls:
            metadata["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in message.tool_calls
            ]
        
        return LLMResponse(
            content=content,
            model=response.model,
            model_type=ModelType.KIMI,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            latency_ms=latency,
            finish_reason=response.choices[0].finish_reason,
            metadata=metadata
        )
    
    def stream_chat(self, messages: List[Message], **kwargs) -> Generator[str, None, None]:
        if not self.client:
            raise RuntimeError("Kimi client not initialized")
        
        model = kwargs.get("model", self.model_name)
        if not model or model == "default":
            model = "kimi-k2.5"
        
        stream = self.client.chat.completions.create(
            model=model,
            messages=[m.to_dict() for m in messages],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude Provider"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = config.get("base_url")
        self.client = None
        
        if self.api_key:
            try:
                from anthropic import Anthropic
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self.client = Anthropic(**kwargs)
            except ImportError:
                print("Anthropic package not installed")
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        if not self.client:
            raise RuntimeError("Anthropic client not initialized")
        
        start_time = time.time()
        
        # 分离system message
        system_msg = None
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})
        
        request_kwargs = {
            "model": kwargs.get("model", self.model_name),
            "messages": chat_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        
        if system_msg:
            request_kwargs["system"] = system_msg
        
        response = self.client.messages.create(**request_kwargs)
        
        latency = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            model_type=ModelType.ANTHROPIC,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            },
            latency_ms=latency,
            finish_reason=response.stop_reason
        )
    
    def stream_chat(self, messages: List[Message], **kwargs) -> Generator[str, None, None]:
        if not self.client:
            raise RuntimeError("Anthropic client not initialized")
        
        system_msg = None
        chat_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})
        
        request_kwargs = {
            "model": kwargs.get("model", self.model_name),
            "messages": chat_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": True
        }
        
        if system_msg:
            request_kwargs["system"] = system_msg
        
        with self.client.messages.stream(**request_kwargs) as stream:
            for text in stream.text_stream:
                yield text


class OllamaProvider(BaseLLMProvider):
    """Ollama本地模型Provider"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.client = None
        
        try:
            import ollama
            self.client = ollama
        except ImportError:
            print("Ollama package not installed")
    
    def is_available(self) -> bool:
        if not self.client:
            return False
        try:
            # 检查Ollama服务是否运行
            import urllib.request
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """列出可用模型"""
        if not self.is_available():
            return []
        
        try:
            models = self.client.list()
            return [m["model"] for m in models.get("models", [])]
        except:
            return []
    
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        if not self.client:
            raise RuntimeError("Ollama client not initialized")
        
        start_time = time.time()
        
        response = self.client.chat(
            model=kwargs.get("model", self.model_name),
            messages=[m.to_dict() for m in messages],
            options={
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        )
        
        latency = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response["message"]["content"],
            model=kwargs.get("model", self.model_name),
            model_type=ModelType.OLLAMA,
            usage={
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0)
            },
            latency_ms=latency,
            metadata={"done": response.get("done", False)}
        )
    
    def stream_chat(self, messages: List[Message], **kwargs) -> Generator[str, None, None]:
        if not self.client:
            raise RuntimeError("Ollama client not initialized")
        
        stream = self.client.chat(
            model=kwargs.get("model", self.model_name),
            messages=[m.to_dict() for m in messages],
            stream=True,
            options={
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        )
        
        for chunk in stream:
            if chunk["message"].get("content"):
                yield chunk["message"]["content"]


class LlamaCppProvider(BaseLLMProvider):
    """llama.cpp Provider - 直接加载GGUF模型"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_path = config.get("model_path", "")
        self.n_ctx = config.get("n_ctx", 4096)
        self.n_gpu_layers = config.get("n_gpu_layers", 0)
        self.model = None
        
        if self.model_path and Path(self.model_path).exists():
            try:
                from llama_cpp import Llama
                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False
                )
            except ImportError:
                print("llama-cpp-python not installed")
    
    def is_available(self) -> bool:
        return self.model is not None
    
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        if not self.model:
            raise RuntimeError("Llama model not loaded")
        
        start_time = time.time()
        
        response = self.model.create_chat_completion(
            messages=[m.to_dict() for m in messages],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        
        latency = (time.time() - start_time) * 1000
        
        message = response["choices"][0]["message"]
        usage = response.get("usage", {})
        
        return LLMResponse(
            content=message.get("content", ""),
            model=Path(self.model_path).name,
            model_type=ModelType.LLAMA_CPP,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            },
            latency_ms=latency,
            finish_reason=response["choices"][0].get("finish_reason")
        )
    
    def stream_chat(self, messages: List[Message], **kwargs) -> Generator[str, None, None]:
        if not self.model:
            raise RuntimeError("Llama model not loaded")
        
        stream = self.model.create_chat_completion(
            messages=[m.to_dict() for m in messages],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True
        )
        
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                yield delta["content"]


class ComplexityAnalyzer:
    """任务复杂度分析器"""
    
    def __init__(self):
        # 复杂度指示词
        self.complex_indicators = [
            "分析", "比较", "评估", "解释", "总结", "推理", "论证",
            "analyze", "compare", "evaluate", "explain", "summarize", "reasoning",
            "为什么", "how", "why", "what if", "如果"
        ]
        
        self.simple_patterns = [
            r'^你好|^hi|^hello|^hey',
            r'^谢谢|^thanks|^thank you',
            r'^再见|^bye|^goodbye',
            r'^是的|^no|^没错|^正确',
        ]
    
    def analyze(self, text: str) -> TaskComplexity:
        """分析任务复杂度"""
        text_lower = text.lower()
        
        # 检查简单模式
        for pattern in self.simple_patterns:
            if re.search(pattern, text_lower):
                return TaskComplexity.SIMPLE
        
        # 检查长度
        if len(text) < 20:
            return TaskComplexity.SIMPLE
        
        # 检查复杂指示词
        complex_score = sum(1 for indicator in self.complex_indicators if indicator in text_lower)
        
        # 检查多个问题
        question_count = text.count("?") + text.count("？") + text.count(",")
        
        if complex_score >= 2 or question_count >= 2 or len(text) > 200:
            return TaskComplexity.COMPLEX
        elif complex_score >= 1 or len(text) > 100:
            return TaskComplexity.MODERATE
        
        return TaskComplexity.SIMPLE


class LLMService:
    """
    LLM服务主类
    智能路由：简单任务本地，复杂任务API
    支持 Kimi (Moonshot AI) 作为默认推荐模型
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 初始化providers
        self.providers: Dict[ModelType, BaseLLMProvider] = {}
        
        # OpenAI
        if self.config.get("openai", {}).get("enabled", True):
            openai_config = self.config.get("openai", {})
            self.providers[ModelType.OPENAI] = OpenAIProvider(openai_config)
        
        # Kimi (Moonshot AI) - 新增
        if self.config.get("kimi", {}).get("enabled", True):
            kimi_config = self.config.get("kimi", {})
            self.providers[ModelType.KIMI] = KimiProvider(kimi_config)
        
        # Anthropic
        if self.config.get("anthropic", {}).get("enabled", True):
            anthropic_config = self.config.get("anthropic", {})
            self.providers[ModelType.ANTHROPIC] = AnthropicProvider(anthropic_config)
        
        # Ollama
        if self.config.get("ollama", {}).get("enabled", True):
            ollama_config = self.config.get("ollama", {})
            self.providers[ModelType.OLLAMA] = OllamaProvider(ollama_config)
        
        # Llama.cpp
        if self.config.get("llama_cpp", {}).get("enabled", False):
            llama_config = self.config.get("llama_cpp", {})
            self.providers[ModelType.LLAMA_CPP] = LlamaCppProvider(llama_config)
        
        # 路由配置 - 默认优先使用 Kimi
        self.routing_config = self.config.get("routing", {
            "simple": ModelType.OLLAMA,
            "moderate": ModelType.KIMI,      # 默认使用 Kimi
            "complex": ModelType.KIMI        # 复杂任务也使用 Kimi
        })
        
        # 默认模型 - 优先 Kimi
        self.default_model = self.config.get("default_model", "kimi-k2.5")
        self.default_provider = ModelType(self.config.get("default_provider", "kimi"))
        
        # 缓存
        self.cache = PromptCache(
            max_size=self.config.get("cache_size", 1000),
            ttl_seconds=self.config.get("cache_ttl", 3600)
        ) if self.config.get("use_cache", True) else None
        
        # 复杂度分析器
        self.complexity_analyzer = ComplexityAnalyzer()
        
        # 使用统计
        self.usage_stats = {
            "total_requests": 0,
            "by_provider": {},
            "by_complexity": {}
        }
    
    def get_available_providers(self) -> List[Dict]:
        """获取可用的providers"""
        available = []
        for model_type, provider in self.providers.items():
            if provider.is_available():
                available.append({
                    "type": model_type.value,
                    "model": provider.model_name,
                    "available": True
                })
        return available
    
    def route(self, messages: List[Message], complexity: Optional[TaskComplexity] = None) -> ModelType:
        """
        智能路由到合适的provider
        
        Args:
            messages: 消息列表
            complexity: 指定的复杂度，自动检测
        
        Returns:
            选择的provider类型
        """
        if complexity is None:
            # 分析最后一条用户消息的复杂度
            user_messages = [m for m in messages if m.role == "user"]
            if user_messages:
                complexity = self.complexity_analyzer.analyze(user_messages[-1].content)
            else:
                complexity = TaskComplexity.MODERATE
        
        # 根据复杂度路由
        provider_type = self.routing_config.get(complexity.value, self.default_provider)
        
        # 检查provider是否可用
        if provider_type in self.providers and self.providers[provider_type].is_available():
            return provider_type
        
        # 回退到其他可用provider (优先Kimi, 其次Ollama, 最后OpenAI)
        for ptype in [ModelType.KIMI, ModelType.OLLAMA, ModelType.OPENAI, ModelType.ANTHROPIC]:
            if ptype in self.providers and self.providers[ptype].is_available():
                return ptype
        
        raise RuntimeError("No LLM provider available")
    
    def chat(
        self,
        messages: Union[str, List[Message]],
        provider: Optional[ModelType] = None,
        complexity: Optional[TaskComplexity] = None,
        use_cache: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        聊天接口
        
        Args:
            messages: 消息或消息列表
            provider: 指定provider
            complexity: 任务复杂度
            use_cache: 是否使用缓存
            **kwargs: 其他参数
        
        Returns:
            LLM响应
        """
        # 标准化消息
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        
        # 路由选择
        if provider is None:
            provider = self.route(messages, complexity)
        
        provider_instance = self.providers.get(provider)
        if not provider_instance or not provider_instance.is_available():
            raise RuntimeError(f"Provider {provider.value} not available")
        
        # 检查缓存
        if use_cache and self.cache:
            cached = self.cache.get(messages, provider_instance.model_name, **kwargs)
            if cached:
                cached.metadata["cached"] = True
                return cached
        
        # 调用provider
        response = provider_instance.chat(messages, **kwargs)
        response.model_type = provider
        
        # 更新统计
        self.usage_stats["total_requests"] += 1
        self.usage_stats["by_provider"][provider.value] = \
            self.usage_stats["by_provider"].get(provider.value, 0) + 1
        
        # 缓存响应
        if use_cache and self.cache:
            self.cache.set(messages, provider_instance.model_name, response, **kwargs)
        
        return response
    
    def stream_chat(
        self,
        messages: Union[str, List[Message]],
        provider: Optional[ModelType] = None,
        complexity: Optional[TaskComplexity] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """流式聊天"""
        # 标准化消息
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        
        # 路由选择
        if provider is None:
            provider = self.route(messages, complexity)
        
        provider_instance = self.providers.get(provider)
        if not provider_instance or not provider_instance.is_available():
            raise RuntimeError(f"Provider {provider.value} not available")
        
        # 更新统计
        self.usage_stats["total_requests"] += 1
        self.usage_stats["by_provider"][provider.value] = \
            self.usage_stats["by_provider"].get(provider.value, 0) + 1
        
        # 流式响应
        yield from provider_instance.stream_chat(messages, **kwargs)
    
    def simple_chat(self, message: str, **kwargs) -> str:
        """简化版聊天接口"""
        response = self.chat(message, **kwargs)
        return response.content
    
    def get_stats(self) -> Dict:
        """获取使用统计"""
        return self.usage_stats.copy()
    
    def clear_cache(self):
        """清空缓存"""
        if self.cache:
            self.cache.clear()


# 便捷函数
def create_llm_service(config_path: Optional[str] = None) -> LLMService:
    """
    创建LLM服务实例
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        LLMService实例
    """
    config = {}
    
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f).get("llm", {})
    
    return LLMService(config)


# 全局实例
_llm_service: Optional[LLMService] = None

def get_llm_service(config_path: Optional[str] = None) -> LLMService:
    """获取全局LLM服务实例"""
    global _llm_service
    if _llm_service is None:
        _llm_service = create_llm_service(config_path)
    return _llm_service


def quick_chat(message: str, provider: Optional[str] = None, **kwargs) -> str:
    """
    快速聊天接口 - 无需创建服务实例
    
    Args:
        message: 用户消息
        provider: 指定provider (kimi, openai, anthropic, ollama)
        **kwargs: 其他参数
    
    Returns:
        回复文本
    """
    service = get_llm_service()
    
    provider_type = None
    if provider:
        try:
            provider_type = ModelType(provider.lower())
        except ValueError:
            pass
    
    return service.simple_chat(message, provider=provider_type, **kwargs)


if __name__ == "__main__":
    # 测试代码
    print("Testing LLM Service...")
    
    # 测试Kimi
    kimi_config = {
        "enabled": True,
        "model": "kimi-k2.5",
        "api_key": os.getenv("KIMI_API_KEY", ""),
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    kimi = KimiProvider(kimi_config)
    if kimi.is_available():
        print("✓ Kimi provider is available")
        response = kimi.chat([Message(role="user", content="你好，请介绍一下自己")])
        print(f"Response: {response.content[:100]}...")
    else:
        print("✗ Kimi provider not available (set KIMI_API_KEY)")

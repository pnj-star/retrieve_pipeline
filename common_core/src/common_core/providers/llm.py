"""OpenAI 兼容对话和本地 Embedding 适配器。包含对话类（OpenAICompatibleLLM）与本地向量化类（LocalEmbedder），都采用懒加载的方式初始化重负荷库。"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import Any, AsyncIterator

from ..config import LLMConfig

logger = logging.getLogger(__name__)


class OpenAICompatibleLLM:
    """针对所有兼容 OpenAI 协议的对话接口的轻量级异步封装层，提供普通/JSON/流式生成，并可选上报 token 指标。    """
    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        metrics: Any = None,  # 可观测性/指标对象
    ) -> None:
        """初始化对话客户端的配置与可观测悬钩。
参数:
    config: 对话模型配置，包含 base_url/api_key/model/temperature 等。
    metrics: 可观测性对象（可为 None），用于上报 token 消耗。
        """
        self.config = config or LLMConfig()
        self.metrics = metrics
        self._client: Any = None

    def client(self) -> Any:
        """懒加获取异步客户端，并按配置初始化 base_url/api_key/超时。
返回:
    openai.AsyncOpenAI 实例。
        """
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout_seconds,
            )
        return self._client

    def _record_usage(self, usage: Any) -> None:
        """若有可观测性对象与用例信息，将 prompt/completion 的 token 数量上报。
参数:
    usage: 对话响应的 usage 对象，可为 None。
        """
        if self.metrics is None or usage is None:
            return
        try:
            self.metrics.record_tokens(  # todo 还没有自定义 record_tokens 方法
                "prompt",
                self.config.model,
                int(getattr(usage, "prompt_tokens", 0) or 0),
            )
            self.metrics.record_tokens(
                "completion",
                self.config.model,
                int(getattr(usage, "completion_tokens", 0) or 0),
            )
        except Exception:
            logger.debug("token metrics record failed (ignored)", exc_info=True)

    async def chat(  # todo chat 感觉多余了
        self,
        messages: list[dict[str, str]],
        *,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """执行一次普通对话生成并返回文本。
参数:
    messages: 对话历史，各项为 {"role": ..., "content": ...} 字典。
    system_prompt: 可选的系统提示词，会插在 messages 之前。
    temperature: 元温；None 时用配置默认值。
    max_tokens: 最大生成 token 数；None 时用配置默认值。
    response_format: 可选的响应格式约定。
返回:
    生成的文本内容。
        """
        full_messages: list[dict[str, str]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": full_messages,
            "temperature": (
                temperature
                if temperature is not None
                else self.config.temperature
            ),
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = await self.client().chat.completions.create(**kwargs)
        self._record_usage(getattr(response, "usage", None))
        return response.choices[0].message.content or ""

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        system_prompt: str = "",
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """检索并解析一个 JSON 对象响应。外层会自动使用 response_format=json_object 并容错利用 fenced code block 。

        参数:
            messages: 对话历史，各项为 {"role": ..., "content": ...} 字典。
            system_prompt: 可选的系统提示词，会插在 messages 之前。
            temperature: 采样温度；None 时用配置默认值。

        返回:
            解析后的字典；解析失败时返回空字典 {}。
        """
        raw = await self.chat(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 3:
                raw = "\n".join(lines[1:-1])
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """以流式方式生成对话内容，逐块产生文本分段。

        参数:
            messages: 对话历史，各项为 {"role": ..., "content": ...} 字典。
            system_prompt: 可选的系统提示词，会插在 messages 之前。
            temperature: 采样温度；None 时用配置默认值。
            max_tokens: 最大生成 token 数；None 时用配置默认值。

        返回:
            一个异步迭代器，每次 yield 一段文本。
        """
        full_messages: list[dict[str, str]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        stream = await self.client().chat.completions.create(
            model=self.config.model,
            messages=full_messages,
            temperature=(
                temperature
                if temperature is not None
                else self.config.temperature
            ),
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if (
                chunk.choices
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content
            ):
                yield chunk.choices[0].delta.content


class LocalEmbedder:
    """持懒加载的 SentenceTransformer 向量化器，并提供异步封装。    """
    def __init__(
        self,
        model_name: str = "",
        *,
        config: LLMConfig | None = None,
        device: str | None = None,
        offline: bool = False,
    ) -> None:
        """初始化向量化器。
参数:
    model_name: 模型名；空时从 config.embedding_model 读取。
    config: 可选配置，用于取得 embedding_model。
    device: 运行设备；None 时自动探测 cuda/cpu。
    offline: 为 True 时仅使用本地缓存的模型。
        """
        self.model_name = model_name or (config.embedding_model if config else "")
        self.device = device
        self.offline = offline
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        """懒加载并缓存模型实例，线程安全仅加载一次。
返回:
    加载好的 SentenceTransformer 模型。
异常:
    ValueError: 没有配置模型名时抛出。
        """
        if not self.model_name:
            raise ValueError("Embedding model name is not configured.")
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from sentence_transformers import SentenceTransformer
            import torch

            device = self.device or os.getenv("EMBEDDING_DEVICE") or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.info(
                "Loading SentenceTransformer (%s) on %s; may take a while on first use...",
                self.model_name,
                device,
            )
            self._model = SentenceTransformer(
                self.model_name,
                local_files_only=self.offline
                or os.environ.get("HF_HUB_OFFLINE", "0") == "1",
                device=device,
            )
        return self._model

    def _embed_sync(self, text: str) -> list[float]:
        """同步单条向量化，并归一化向量。
返回:
    子快浮点数组。
        """
        model = self._load()
        return model.encode(text, normalize_embeddings=True).tolist()

    def _embed_many_sync(self, texts: list[str]) -> list[list[float]]:
        """同步批量向量化。
返回:
    向量列表。
        """
        model = self._load()
        return model.encode(texts, normalize_embeddings=True).tolist()

    async def embed(self, text: str) -> list[float]:
        """异步单条向量化，在辅助线程执行。
返回:
    子快浮点数组。
        """
        return await asyncio.to_thread(self._embed_sync, text)

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """异步批量向量化，在辅助线程执行。
返回:
    向量列表。
        """
        return await asyncio.to_thread(self._embed_many_sync, texts)

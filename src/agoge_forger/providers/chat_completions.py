from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Generator, List, Optional

import httpx
from pydantic import BaseModel, Field, field_validator


class ChatCompletionsConfig(BaseModel):
    provider: str = "chat_completions"
    base_url: str = "http://localhost:8000/v1"
    model: str = ""
    timeout_s: float = 120.0
    stream: bool = True
    max_tokens: int = 512
    temperature: float = 0.7
    api_key: str = ""

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"


@dataclass
class InferenceResult:
    provider: str
    base_url: str
    model: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    prompt_hash: str = ""
    response_text: str = ""
    reasoning_text: str = ""
    finish_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    raw_response_path: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


class ChatCompletionsClient:
    def __init__(self, config: ChatCompletionsConfig, run_name: str = "run"):
        self.config = config
        self.run_name = run_name
        self._raw_dir = os.path.join("runs", run_name, "raw")
        os.makedirs(self._raw_dir, exist_ok=True)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        stream: Optional[bool] = None,
    ) -> Dict[str, Any]:
        is_streaming = stream if stream is not None else self.config.stream
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": is_streaming,
        }
        if is_streaming:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _save_raw(self, request_id: str, data: Any) -> str:
        path = os.path.join(self._raw_dir, f"{request_id}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def _parse_usage(self, usage: Optional[Dict[str, Any]]) -> Dict[str, int]:
        if not usage or not isinstance(usage, dict):
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        return {
            "input_tokens": usage.get("prompt_tokens") or 0,
            "output_tokens": usage.get("completion_tokens") or 0,
            "total_tokens": usage.get("total_tokens") or 0,
        }

    def _parse_choice(self, choice: Dict[str, Any]) -> Dict[str, str]:
        message = choice.get("message") or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        finish = choice.get("finish_reason") or ""
        return {"response_text": content, "reasoning_text": reasoning, "finish_reason": finish}

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: Optional[bool] = None,
    ) -> InferenceResult:
        use_stream = stream if stream is not None else self.config.stream

        prompt_text = json.dumps(messages, sort_keys=True)
        prompt_hash = _hash_prompt(prompt_text)
        request_id = uuid.uuid4().hex[:12]

        result = InferenceResult(
            provider=self.config.provider,
            base_url=self.config.base_url,
            model=self.config.model,
            request_id=request_id,
            prompt_hash=prompt_hash,
        )

        payload = self._build_payload(messages, stream=use_stream)

        t_start = time.monotonic()
        try:
            if use_stream:
                self._chat_streaming(payload, result, t_start)
            else:
                self._chat_non_streaming(payload, result)
        except httpx.HTTPStatusError as exc:
            result.error = f"HTTP {exc.response.status_code}"
        except httpx.ConnectError:
            result.error = "Connection refused"
        except httpx.TimeoutException:
            result.error = "Request timed out"
        except Exception as exc:
            result.error = type(exc).__name__
        result.latency_ms = (time.monotonic() - t_start) * 1000

        return result

    def _chat_non_streaming(
        self, payload: Dict[str, Any], result: InferenceResult
    ) -> None:
        t0 = time.monotonic()
        resp = httpx.post(
            self.config.chat_url,
            headers=self._headers(),
            json=payload,
            timeout=self.config.timeout_s,
        )
        resp.raise_for_status()
        result.time_to_first_token_ms = (time.monotonic() - t0) * 1000

        body = resp.json()
        raw_path = self._save_raw(result.request_id, body)
        result.raw_response_path = raw_path

        usage = self._parse_usage(body.get("usage"))
        result.input_tokens = usage["input_tokens"]
        result.output_tokens = usage["output_tokens"]
        result.total_tokens = usage["total_tokens"]

        choices = body.get("choices", [])
        if choices:
            parsed = self._parse_choice(choices[0])
            result.response_text = parsed["response_text"]
            result.reasoning_text = parsed["reasoning_text"]
            result.finish_reason = parsed["finish_reason"]

    def _chat_streaming(
        self, payload: Dict[str, Any], result: InferenceResult, t_start: float
    ) -> None:
        collected_content: List[str] = []
        collected_reasoning: List[str] = []
        finish_reason = ""
        first_token = True
        raw_chunks: List[Dict[str, Any]] = []

        with httpx.stream(
            "POST",
            self.config.chat_url,
            headers=self._headers(),
            json=payload,
            timeout=self.config.timeout_s,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].lstrip()
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                raw_chunks.append(chunk)

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                reasoning = delta.get("reasoning_content", "") or delta.get("reasoning", "")
                if first_token and (content or reasoning):
                    result.time_to_first_token_ms = (time.monotonic() - t_start) * 1000
                    first_token = False
                if content:
                    collected_content.append(content)
                if reasoning:
                    collected_reasoning.append(reasoning)
                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr

        result.response_text = "".join(collected_content)
        result.reasoning_text = "".join(collected_reasoning)
        result.finish_reason = finish_reason

        if raw_chunks:
            usage = self._parse_usage(raw_chunks[-1].get("usage"))
            result.input_tokens = usage["input_tokens"]
            result.output_tokens = usage["output_tokens"]
            result.total_tokens = usage["total_tokens"]

        raw_path = self._save_raw(result.request_id, {"chunks": raw_chunks})
        result.raw_response_path = raw_path

    def chat_simple(
        self,
        prompt: str,
        system: str = "",
        stream: Optional[bool] = None,
    ) -> InferenceResult:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, stream=stream)

import json
import os
from unittest.mock import patch, MagicMock

import httpx
import pytest

from agoge_forger.providers.chat_completions import (
    ChatCompletionsClient,
    ChatCompletionsConfig,
    InferenceResult,
    _hash_prompt,
)


class TestChatCompletionsConfig:
    def test_defaults(self):
        cfg = ChatCompletionsConfig(model="test-model")
        assert cfg.provider == "chat_completions"
        assert cfg.base_url == "http://localhost:8000/v1"
        assert cfg.timeout_s == 120.0
        assert cfg.stream is True
        assert cfg.model == "test-model"

    def test_strip_trailing_slash(self):
        cfg = ChatCompletionsConfig(base_url="http://host:8000/v1/")
        assert cfg.base_url == "http://host:8000/v1"

    def test_chat_url(self):
        cfg = ChatCompletionsConfig(base_url="http://host:8000/v1", model="m")
        assert cfg.chat_url == "http://host:8000/v1/chat/completions"

    def test_custom_values(self):
        cfg = ChatCompletionsConfig(
            base_url="http://remote:9999/v1",
            model="big-model",
            timeout_s=30.0,
            stream=False,
            max_tokens=1024,
            temperature=0.2,
            api_key="sk-test",
        )
        assert cfg.base_url == "http://remote:9999/v1"
        assert cfg.model == "big-model"
        assert cfg.timeout_s == 30.0
        assert cfg.stream is False
        assert cfg.api_key == "sk-test"


class TestInferenceResult:
    def test_defaults(self):
        r = InferenceResult(provider="chat_completions", base_url="http://x", model="m")
        assert r.provider == "chat_completions"
        assert r.request_id
        assert r.response_text == ""
        assert r.error == ""
        assert r.latency_ms == 0.0
        assert r.input_tokens == 0

    def test_to_dict(self):
        r = InferenceResult(provider="p", base_url="b", model="m", response_text="hi")
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["provider"] == "p"
        assert d["response_text"] == "hi"

    def test_to_json(self):
        r = InferenceResult(provider="p", base_url="b", model="m")
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["provider"] == "p"

    def test_unique_request_ids(self):
        r1 = InferenceResult(provider="p", base_url="b", model="m")
        r2 = InferenceResult(provider="p", base_url="b", model="m")
        assert r1.request_id != r2.request_id


class TestHashPrompt:
    def test_deterministic(self):
        h1 = _hash_prompt("hello")
        h2 = _hash_prompt("hello")
        assert h1 == h2
        assert len(h1) == 16

    def test_different_prompts(self):
        h1 = _hash_prompt("hello")
        h2 = _hash_prompt("world")
        assert h1 != h2


def _mock_response(body: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    resp.raise_for_status.return_value = None
    return resp


NON_STREAM_BODY = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello from vLLM!",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    },
}

NON_STREAM_REASONING_BODY = {
    "id": "chatcmpl-456",
    "object": "chat.completion",
    "model": "reasoning-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The answer is 42.",
                "reasoning_content": "Let me think about this...",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 15,
        "total_tokens": 35,
    },
}


class TestChatCompletionsClientNonStreaming:
    @patch("agoge_forger.providers.chat_completions.httpx.post")
    def test_basic_chat(self, mock_post, tmp_path):
        mock_post.return_value = _mock_response(NON_STREAM_BODY)

        cfg = ChatCompletionsConfig(model="test-model", stream=False)
        client = ChatCompletionsClient(cfg, run_name=str(tmp_path / "runs" / "test"))
        result = client.chat([{"role": "user", "content": "Hello"}])

        assert result.response_text == "Hello from vLLM!"
        assert result.finish_reason == "stop"
        assert result.error == ""
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.total_tokens == 15
        assert result.latency_ms > 0
        assert result.raw_response_path
        assert os.path.exists(result.raw_response_path)

    @patch("agoge_forger.providers.chat_completions.httpx.post")
    def test_reasoning_content(self, mock_post, tmp_path):
        mock_post.return_value = _mock_response(NON_STREAM_REASONING_BODY)

        cfg = ChatCompletionsConfig(model="reasoning-model", stream=False)
        client = ChatCompletionsClient(cfg, run_name=str(tmp_path / "runs" / "test"))
        result = client.chat([{"role": "user", "content": "Think"}])

        assert result.response_text == "The answer is 42."
        assert result.reasoning_text == "Let me think about this..."
        assert result.input_tokens == 20

    @patch("agoge_forger.providers.chat_completions.httpx.post")
    def test_raw_response_saved(self, mock_post, tmp_path):
        mock_post.return_value = _mock_response(NON_STREAM_BODY)

        cfg = ChatCompletionsConfig(model="test-model", stream=False)
        client = ChatCompletionsClient(cfg, run_name=str(tmp_path / "runs" / "test"))
        result = client.chat([{"role": "user", "content": "Hi"}])

        with open(result.raw_response_path) as f:
            raw = json.load(f)
        assert raw["id"] == "chatcmpl-123"
        assert raw["choices"][0]["message"]["content"] == "Hello from vLLM!"

    @patch("agoge_forger.providers.chat_completions.httpx.post")
    def test_error_handling(self, mock_post, tmp_path):
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        cfg = ChatCompletionsConfig(model="test-model", stream=False)
        client = ChatCompletionsClient(cfg, run_name=str(tmp_path / "runs" / "test"))
        result = client.chat([{"role": "user", "content": "Hi"}])

        assert result.error == "Connection refused"
        assert result.response_text == ""

    @patch("agoge_forger.providers.chat_completions.httpx.post")
    def test_chat_simple(self, mock_post, tmp_path):
        mock_post.return_value = _mock_response(NON_STREAM_BODY)

        cfg = ChatCompletionsConfig(model="test-model", stream=False)
        client = ChatCompletionsClient(cfg, run_name=str(tmp_path / "runs" / "test"))
        result = client.chat_simple("Hello", system="You are helpful.")

        assert result.response_text == "Hello from vLLM!"

        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"

    @patch("agoge_forger.providers.chat_completions.httpx.post")
    def test_api_key_in_headers(self, mock_post, tmp_path):
        mock_post.return_value = _mock_response(NON_STREAM_BODY)

        cfg = ChatCompletionsConfig(model="test-model", stream=False, api_key="sk-secret")
        client = ChatCompletionsClient(cfg, run_name=str(tmp_path / "runs" / "test"))
        client.chat([{"role": "user", "content": "Hi"}])

        call_args = mock_post.call_args
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
        assert headers["Authorization"] == "Bearer sk-secret"

    @patch("agoge_forger.providers.chat_completions.httpx.post")
    def test_no_api_key_no_auth_header(self, mock_post, tmp_path):
        mock_post.return_value = _mock_response(NON_STREAM_BODY)

        cfg = ChatCompletionsConfig(model="test-model", stream=False)
        client = ChatCompletionsClient(cfg, run_name=str(tmp_path / "runs" / "test"))
        client.chat([{"role": "user", "content": "Hi"}])

        call_args = mock_post.call_args
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
        assert "Authorization" not in headers

    @patch("agoge_forger.providers.chat_completions.httpx.post")
    def test_prompt_hash_populated(self, mock_post, tmp_path):
        mock_post.return_value = _mock_response(NON_STREAM_BODY)

        cfg = ChatCompletionsConfig(model="test-model", stream=False)
        client = ChatCompletionsClient(cfg, run_name=str(tmp_path / "runs" / "test"))
        result = client.chat([{"role": "user", "content": "Hello"}])

        assert result.prompt_hash
        assert len(result.prompt_hash) == 16

    @patch("agoge_forger.providers.chat_completions.httpx.post")
    def test_no_credentials_in_error(self, mock_post, tmp_path):
        mock_post.side_effect = Exception("Failed with key=sk-secret123")

        cfg = ChatCompletionsConfig(model="test-model", stream=False)
        client = ChatCompletionsClient(cfg, run_name=str(tmp_path / "runs" / "test"))
        result = client.chat([{"role": "user", "content": "Hi"}])

        result_json = result.to_json()
        assert "sk-secret123" not in result_json

    @patch("agoge_forger.providers.chat_completions.httpx.post")
    def test_usage_missing(self, mock_post, tmp_path):
        body = {
            "id": "chatcmpl-789",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }
        mock_post.return_value = _mock_response(body)

        cfg = ChatCompletionsConfig(model="test-model", stream=False)
        client = ChatCompletionsClient(cfg, run_name=str(tmp_path / "runs" / "test"))
        result = client.chat([{"role": "user", "content": "Hi"}])

        assert result.input_tokens == 0
        assert result.output_tokens == 0


STREAM_CHUNKS = [
    {"id": "chatcmpl-s1", "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
    {"id": "chatcmpl-s1", "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
    {"id": "chatcmpl-s1", "choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
    {"id": "chatcmpl-s1", "choices": [{"delta": {}, "finish_reason": "stop"}],
     "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}},
]


class TestChatCompletionsClientStreaming:
    @patch("agoge_forger.providers.chat_completions.httpx.stream")
    def test_streaming_chat(self, mock_stream_fn, tmp_path):
        lines = []
        for chunk in STREAM_CHUNKS:
            lines.append(f"data: {json.dumps(chunk)}")
        lines.append("data: [DONE]")

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_lines.return_value = iter(lines)
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_resp)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_stream_fn.return_value = mock_ctx

        cfg = ChatCompletionsConfig(model="test-model", stream=True)
        client = ChatCompletionsClient(cfg, run_name=str(tmp_path / "runs" / "test"))
        result = client.chat([{"role": "user", "content": "Hi"}])

        assert result.response_text == "Hello world"
        assert result.finish_reason == "stop"
        assert result.input_tokens == 8
        assert result.output_tokens == 3
        assert result.total_tokens == 11
        assert result.raw_response_path

    @patch("agoge_forger.providers.chat_completions.httpx.stream")
    def test_streaming_with_reasoning(self, mock_stream_fn, tmp_path):
        chunks = [
            {"id": "s", "choices": [{"delta": {"reasoning_content": "Thinking"}, "finish_reason": None}]},
            {"id": "s", "choices": [{"delta": {"content": "Answer"}, "finish_reason": None}]},
            {"id": "s", "choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        lines = [f"data: {json.dumps(c)}" for c in chunks]
        lines.append("data: [DONE]")

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_lines.return_value = iter(lines)
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_resp)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_stream_fn.return_value = mock_ctx

        cfg = ChatCompletionsConfig(model="test-model", stream=True)
        client = ChatCompletionsClient(cfg, run_name=str(tmp_path / "runs" / "test"))
        result = client.chat([{"role": "user", "content": "Think"}])

        assert result.reasoning_text == "Thinking"
        assert result.response_text == "Answer"

    @patch("agoge_forger.providers.chat_completions.httpx.stream")
    def test_streaming_error(self, mock_stream_fn, tmp_path):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("Stream failed")
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_resp)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_stream_fn.return_value = mock_ctx

        cfg = ChatCompletionsConfig(model="test-model", stream=True)
        client = ChatCompletionsClient(cfg, run_name=str(tmp_path / "runs" / "test"))
        result = client.chat([{"role": "user", "content": "Hi"}])

        assert result.error == "Exception"

    @patch("agoge_forger.providers.chat_completions.httpx.stream")
    def test_streaming_ignores_malformed(self, mock_stream_fn, tmp_path):
        chunks = [
            {"id": "s", "choices": [{"delta": {"content": "ok"}, "finish_reason": None}]},
            {"id": "s", "choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        lines = [f"data: {json.dumps(c)}" for c in chunks]
        lines.insert(1, "not a data line")
        lines.insert(2, "data: {bad json")
        lines.append("data: [DONE]")

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_lines.return_value = iter(lines)
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_resp)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_stream_fn.return_value = mock_ctx

        cfg = ChatCompletionsConfig(model="test-model", stream=True)
        client = ChatCompletionsClient(cfg, run_name=str(tmp_path / "runs" / "test"))
        result = client.chat([{"role": "user", "content": "Hi"}])

        assert result.response_text == "ok"
        assert result.error == ""


class TestCanTargetLocalVLLM:
    @patch("agoge_forger.providers.chat_completions.httpx.post")
    def test_custom_base_url(self, mock_post, tmp_path):
        mock_post.return_value = _mock_response(NON_STREAM_BODY)

        cfg = ChatCompletionsConfig(
            base_url="http://my-vllm:8000/v1",
            model="my-model",
            stream=False,
        )
        client = ChatCompletionsClient(cfg, run_name=str(tmp_path / "runs" / "test"))
        result = client.chat([{"role": "user", "content": "Hi"}])

        call_args = mock_post.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        assert "my-vllm:8000" in url
        assert result.error == ""

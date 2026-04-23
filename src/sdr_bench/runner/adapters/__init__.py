"""Adapter registry for model providers."""

from __future__ import annotations

from sdr_bench.runner.adapters.anthropic import AnthropicAdapter
from sdr_bench.runner.adapters.base import ModelAdapter
from sdr_bench.runner.adapters.openai import OpenAIAdapter
from sdr_bench.runner.adapters.openai_compatible import OpenAICompatibleAdapter


def load_adapter(spec: str) -> ModelAdapter:
    provider, separator, remainder = spec.partition(":")
    if not separator or not remainder:
        raise ValueError(
            "Model spec must be in the form 'provider:model' or "
            "'openai_compatible:model@base_url'."
        )

    if provider == "anthropic":
        return AnthropicAdapter(model=remainder)
    if provider == "openai":
        return OpenAIAdapter(model=remainder)
    if provider == "openai_compatible":
        model_name, at_sign, base_url = remainder.rpartition("@")
        if at_sign:
            return OpenAICompatibleAdapter(model=model_name, base_url=base_url)
        return OpenAICompatibleAdapter(model=remainder)

    raise ValueError(f"Unsupported adapter provider: {provider}")

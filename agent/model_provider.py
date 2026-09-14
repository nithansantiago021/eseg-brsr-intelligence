"""
model_provider.py — Builds the Strands `Model` the Compliance Sentinel agent runs on.
"""

from __future__ import annotations

import os

from config.settings import cfg, GROQ_API_KEY


def build_model():
    provider = os.getenv("AGENT_MODEL_PROVIDER", "bedrock").strip().lower()

    if provider == "bedrock":
        from strands.models.bedrock import BedrockModel
        # Verify the exact model ID is enabled for your account/region in the Bedrock console before relying on this default
        model_id = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-west-2"))
        return BedrockModel(model_id=model_id, region_name=region)

    if provider in ("groq", "openai"):
        from strands.models.openai import OpenAIModel
        api_key = os.getenv("AGENT_OPENAI_API_KEY", GROQ_API_KEY)
        base_url = os.getenv("AGENT_OPENAI_BASE_URL", cfg.llm.base_url)
        model_id = os.getenv("AGENT_OPENAI_MODEL_ID", cfg.llm.model_name)
        if not api_key:
            raise RuntimeError(
                "AGENT_MODEL_PROVIDER=groq/openai requires GROQ_API_KEY "
                "(or AGENT_OPENAI_API_KEY) to be set."
            )
        return OpenAIModel(
            client_args={
                "api_key": api_key,
                "base_url": base_url,
                "max_retries": int(os.getenv("AGENT_OPENAI_MAX_RETRIES", "6")),
                "timeout": float(os.getenv("AGENT_OPENAI_TIMEOUT", "60")),
            },
            model_id=model_id,
            params={"extra_body": {"reasoning_format": "hidden"}},
        )

    raise ValueError(f"Unknown AGENT_MODEL_PROVIDER: {provider!r} (expected 'bedrock' or 'groq')")

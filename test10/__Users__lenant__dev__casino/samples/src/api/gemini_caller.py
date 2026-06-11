import logging
import os
from typing import Optional, Type, TypeVar, Tuple, cast

from google import genai
from google.genai import types
from pydantic import BaseModel

from src.constants import ReasoningEffort
from src.entities.entities import LlmUsage

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class GeminiCaller:
    """Class for making calls to Google Gemini API using the google-genai SDK with structured outputs."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        self._client = genai.Client(api_key=self.api_key)

    def _is_gemini_3(self, model: str) -> bool:
        """Check if model is Gemini 3 (uses thinking_level instead of thinking_budget)."""
        return "gemini-3" in model

    async def call_structured(
        self,
        system_message: str,
        user_message: str,
        schema: Type[T],
        model: str,
        temperature: float = 0,
        max_tokens: int = 8192,
        reasoning_effort: Optional[ReasoningEffort] = None,
    ) -> Tuple[T, LlmUsage]:
        """
        Call Google Gemini API with structured output parsing.

        Args:
            system_message: System instruction
            user_message: User query
            schema: Pydantic model for structured output
            model: Model ID
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            reasoning_effort: Optional reasoning effort level (controls thinking_budget)

        Returns a tuple of (parsed_response, usage).
        """
        try:
            # Combine system and user message
            prompt = f"{user_message}" if not system_message else f"{system_message}\n\n{user_message}"

            # Build config
            config_kwargs = {
                "response_mime_type": "application/json",
                "response_schema": schema,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            # Add thinking config if reasoning effort specified
            if reasoning_effort is not None:
                if self._is_gemini_3(model):
                    # Gemini 3 uses thinking_level enum (LOW/HIGH)
                    thinking_level_str = reasoning_effort.to_gemini_thinking_level()
                    thinking_level = types.ThinkingLevel(thinking_level_str)
                    config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_level=thinking_level
                    )
                else:
                    # Gemini 2.5 uses thinking_budget
                    thinking_budget = reasoning_effort.to_gemini_thinking_budget()
                    config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=thinking_budget
                    )

            response = await self._client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception:
            logger.exception("Gemini API call failed")
            raise

        parsed = response.parsed
        if parsed is None:
            logger.error("Structured output missing parsed payload in Gemini API response")
            raise ValueError("Structured output missing parsed payload")

        usage = self._extract_llm_usage(response)
        return cast(T, parsed), usage

    def _extract_llm_usage(self, response) -> LlmUsage:
        """Extract LlmUsage from Gemini response object."""
        usage_md = getattr(response, "usage_metadata", None)
        if usage_md is None:
            return LlmUsage(input_tokens=0, cached_input_tokens=0, output_tokens=0)

        input_tokens = getattr(usage_md, "prompt_token_count", 0) or 0
        cached_tokens = getattr(usage_md, "cached_content_token_count", 0) or 0
        output_tokens = getattr(usage_md, "candidates_token_count", 0) or 0
        thoughts_tokens = getattr(usage_md, "thoughts_token_count", 0) or 0

        return LlmUsage(
            input_tokens=input_tokens,
            cached_input_tokens=cached_tokens,
            output_tokens=output_tokens + thoughts_tokens,
        )

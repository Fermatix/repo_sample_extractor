import asyncio
from typing import List, Optional, Type, TypeVar, Tuple

from pydantic import BaseModel

from src.constants import ReasoningEffort
from src.entities.entities import LlmUsage
from src.api.openai_caller import OpenAICaller
from src.api.gemini_caller import GeminiCaller

T = TypeVar("T", bound=BaseModel)


class LlmMessage:
    """Class representing a message to be sent to an LLM"""

    def __init__(self, system_message: str, user_message: str):
        """
        Initialize a new LLM message.

        Args:
            system_message: The system instruction message
            user_message: The user query message
        """
        self.system_message = system_message
        self.user_message = user_message


class LLMCaller:
    """Main class for calling language models with structured output support."""

    def __init__(self, openai_caller: OpenAICaller, gemini_caller: GeminiCaller):
        self.openai_caller = openai_caller
        self.gemini_caller = gemini_caller

    def _is_openai_model(self, model: str) -> bool:
        return model.startswith("gpt-")

    def _is_gemini_model(self, model: str) -> bool:
        return model.startswith("gemini-")

    async def call_structured(
        self,
        system_message: str,
        user_message: str,
        schema: Type[T],
        model: str,
        temperature: float = 0,
        max_tokens: int = 8192,
        reasoning_effort: Optional[ReasoningEffort] = ReasoningEffort.LOW,
    ) -> Tuple[T, LlmUsage]:
        """
        Call LLM with structured output parsing.

        Args:
            system_message: System instruction
            user_message: User query
            schema: Pydantic model for structured output
            model: Model ID
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            reasoning_effort: Optional reasoning effort level

        Returns a tuple of (parsed_response, usage).
        Raises ValueError for unsupported models.
        """
        if self._is_openai_model(model):
            return await self.openai_caller.call_structured(
                system_message, user_message, schema, model, temperature, max_tokens, reasoning_effort
            )
        elif self._is_gemini_model(model):
            return await self.gemini_caller.call_structured(
                system_message, user_message, schema, model, temperature, max_tokens, reasoning_effort
            )
        else:
            raise ValueError(
                f"Unsupported model: {model}. Use a model name that starts with 'gpt-' or 'gemini-'"
            )

    async def call_batch_structured(
        self,
        messages: List[LlmMessage],
        schema: Type[T],
        model: str,
        temperature: float = 0,
        max_tokens: int = 8192,
        reasoning_effort: Optional[ReasoningEffort] = ReasoningEffort.LOW,
    ) -> List[Tuple[T, LlmUsage]]:
        """
        Call LLM with structured output parsing for multiple messages in parallel.

        Returns a list of (parsed_response, usage) tuples.
        """
        tasks = [
            self.call_structured(
                message.system_message,
                message.user_message,
                schema,
                model,
                temperature,
                max_tokens,
                reasoning_effort,
            )
            for message in messages
        ]
        return await asyncio.gather(*tasks)

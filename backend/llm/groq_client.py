import os
import logging
from typing import Optional, List, Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from backend.config import settings

logger = logging.getLogger(__name__)


class GroqLLMClient:
    """
    Manages Groq LLM client connectivity with fallback handling for local development/testing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.api_key = api_key or settings.groq_api_key
        self.model_name = model_name or settings.groq_model
        self.temperature = temperature if temperature is not None else settings.groq_temperature
        self.max_tokens = max_tokens or settings.groq_max_tokens
        self._llm = None

    def get_chat_model(self) -> BaseChatModel:
        """Initializes and returns the LangChain ChatGroq instance."""
        if not self.api_key:
            logger.warning(
                "GROQ_API_KEY is not configured in .env. LLM calls will return a notice."
            )
            return self._create_mock_model()

        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                groq_api_key=self.api_key,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatGroq client: {str(e)}")
            return self._create_mock_model()

    def invoke_with_fallback(self, messages: List[Any], config: Optional[Dict[str, Any]] = None) -> str:
        """
        Invokes the model with automatic fallback to alternate available models on Groq.
        """
        if not self.api_key:
            return (
                "⚠️ **Groq API Key Not Configured**\n\n"
                "Retrieved medical evidence successfully, but an active `GROQ_API_KEY` is required "
                "in `.env` to generate grounded LLM answers."
            )

        candidate_models = [
            self.model_name,
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b",
            "groq/compound",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ]
        
        # Deduplicate while preserving order
        models_to_try = list(dict.fromkeys(candidate_models))
        last_error = None

        from langchain_groq import ChatGroq

        for model in models_to_try:
            try:
                llm = ChatGroq(
                    groq_api_key=self.api_key,
                    model_name=model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                response = llm.invoke(messages, config=config)
                raw_text = response.content if hasattr(response, "content") else str(response)
                
                # Strip reasoning thinking tags if present
                import re
                cleaned_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
                return cleaned_text if cleaned_text else raw_text
            except Exception as e:
                err_msg = str(e)
                last_error = e
                logger.warning(f"Model '{model}' failed ({err_msg[:80]}). Trying next candidate...")
                continue

        raise last_error or RuntimeError("All Groq model candidates failed.")

    def _create_mock_model(self) -> BaseChatModel:
        """Fallback mock chat model when no Groq API key is present."""
        from langchain_community.chat_models.fake import FakeListChatModel
        default_response = (
            "⚠️ **Groq API Key Not Configured**\n\n"
            "Retrieved medical evidence successfully, but an active `GROQ_API_KEY` is required "
            "in `.env` to generate full grounded LLM answers.\n\n"
            "Please configure `GROQ_API_KEY=gsk_...` in your `.env` file."
        )
        return FakeListChatModel(responses=[default_response])

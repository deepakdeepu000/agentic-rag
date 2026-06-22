"""
llm_factory.py - Provider-agnostic LLM construction for coordinator, router, and summarizer.
"""
from typing import Literal

from langchain_ollama import ChatOllama

from core.config import Settings


Purpose = Literal["coordinator", "router", "summarizer"]


def _resolve_model_name(settings: Settings, purpose: Purpose) -> str:
    if purpose == "router" and settings.router_model:
        return settings.router_model
    if purpose == "summarizer":
        return settings.ollama_summarizer_model
    return settings.ollama_model


def get_chat_model(settings: Settings, purpose: Purpose):
    """
    Return a configured chat model for the given purpose.

    Supported providers: ollama, openai, anthropic.
    """
    provider = settings.llm_provider.strip().lower()
    temperature = settings.coordinator_temperature
    if purpose == "router":
        temperature = settings.router_temperature
    elif purpose == "summarizer":
        temperature = settings.summarizer_temperature

    if provider == "ollama":
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=_resolve_model_name(settings, purpose),
            temperature=temperature,
        )

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is required for OPENAI providers. "
                "Install with: pip install langchain-openai"
            ) from exc
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=temperature,
        )

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise ImportError(
                "langchain-google-genai is required for GEMINI providers. "
                "Install with: pip install langchain-google-genai"
            ) from exc

        return ChatGoogleGenerativeAI(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            temperature=temperature,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. "
        "Use: ollama, openai, or gemini."
    )

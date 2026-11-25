"""Chat model factory. Each provider is imported only when it is used."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from .config import Settings

PROVIDERS = ("extractive", "anthropic", "openai", "ollama")


class LLMError(RuntimeError):
    """The model could not be used. The message is safe to show to the user."""


def get_llm(settings: Settings, api_key: str | None = None) -> BaseChatModel | None:
    """Return a chat model, or None for the no-LLM 'extractive' mode."""
    provider, model = settings.provider, settings.resolved_model
    key = {"api_key": api_key} if api_key else {}
    if provider == "extractive":
        return None
    try:
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            # No temperature: current Claude models reject sampling parameters.
            # max_tokens leaves room for the model's built-in thinking.
            return ChatAnthropic(model=model, max_tokens=settings.max_tokens, **key)
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=model, temperature=0, **key)
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=model,
                base_url=settings.ollama_url,
                temperature=0,
                seed=7,
                num_ctx=4096,
                num_predict=400,
            )
    except Exception as exc:  # a missing key raises provider-specific errors here
        raise LLMError(f"Could not start the {provider} model. Is the API key set?") from exc
    raise LLMError(f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}.")

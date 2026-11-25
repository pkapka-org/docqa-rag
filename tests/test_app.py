"""Drives the real Streamlit app, with the real embedding model, in no-LLM mode."""

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "extractive")
    return AppTest.from_file("../app.py", default_timeout=120).run()


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_upload_ask_cite_refuse_remove(app):
    assert not app.exception
    assert app.chat_input[0].disabled

    button(app, "Load the sample documents").click().run()
    assert not app.exception and not app.chat_input[0].disabled

    app.chat_input[0].set_value("What is the daily meal limit when travelling?").run()
    assert any("55 euros" in m.value for m in app.markdown)
    assert any("employee_handbook.pdf, page 4" in e.label for e in app.expander)

    app.chat_input[0].set_value("What is the capital of Australia?").run()
    assert any("could not find" in i.value for i in app.info)

    app.button(key="remove-employee_handbook.pdf").click().run()
    assert not app.exception
    assert not any("employee_handbook.pdf" in m.value for m in app.sidebar.markdown)


def test_missing_key_shows_an_error_not_a_crash(app, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    button(app, "Load the sample documents").click().run()
    app.sidebar.selectbox[0].set_value("openai").run()
    app.chat_input[0].set_value("What is the daily meal limit?").run()
    assert not app.exception
    assert any("API key" in e.value for e in app.error)

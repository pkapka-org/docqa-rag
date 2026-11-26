# User guide

## Install

Python 3.12 or newer.

```bash
git clone https://github.com/pkapka-org/docqa-rag && cd docqa-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the embedding model (about 70 MB).

## The web app

```bash
streamlit run app.py
```

Open the address it prints (normally http://localhost:8501).

1. **Add documents** in the sidebar: PDF, TXT or Markdown, up to 20 MB each. Or click
   **Load the sample documents** to try the invented company's handbook, product manual and policy.
2. **Choose a model** under *Answering*. *No LLM* works with no setup.
3. **Ask** in the box at the bottom. Each answer ends with numbers like `[1]`; open the source below it to
   read the exact passage, with its file and page or section.
4. **Remove** a file with the ✕ next to its name.

When the documents do not answer the question, the app says
*"I could not find the answer in the uploaded documents."*, and the grey line under it says why.

Documents live in memory for your browser session only, and other visitors cannot see them. Reloading the
page starts empty.

## Choosing a model

| Choice | Needs | Notes |
|---|---|---|
| No LLM | nothing | Quotes the closest sentences. Free and fast, but cannot combine facts. |
| Ollama | [Ollama](https://ollama.com) running on the same machine | `ollama pull qwen2.5:1.5b` (1 GB) or `qwen2.5:3b` (1.9 GB). Private; slow without a GPU. |
| Claude | an Anthropic API key | Default model `claude-haiku-5-5`. |
| OpenAI | an OpenAI API key | Default model `gpt-4o-mini`. Also works with any server that speaks the OpenAI API: set `OPENAI_BASE_URL`. |

Keys typed into the sidebar are kept for that session only. With Claude or OpenAI, the question and the
retrieved passages are sent to that company; the documents themselves are not.

## Command line

The command line keeps its index on disk, in `.chroma/`.

```bash
python -m docqa add handbook.pdf policy.txt       # index files (re-adding a changed file replaces it)
python -m docqa list                              # show indexed files
python -m docqa ask "How many days of annual leave do I get?"
python -m docqa remove handbook.pdf
```

## Settings

Put these in the environment or in a `.env` file (copy `.env.example`). All are optional.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `extractive` | `extractive`, `ollama`, `anthropic` or `openai` |
| `LLM_MODEL` | per provider | model name |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | none | keys for the hosted models |
| `OPENAI_BASE_URL` | OpenAI | another OpenAI-compatible server |
| `OLLAMA_URL` | `http://localhost:11434` | where Ollama runs |
| `TOP_K` | `5` | passages given to the model |
| `MIN_RELEVANCE` | `0.60` | similarity the best passage needs before the model is asked at all |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | `900`, `150` | passage size in characters |
| `MAX_FILE_MB` | `20` | largest file accepted |
| `DB_PATH` | `.chroma` | index folder for the command line |

Changing `CHUNK_SIZE` or the embedding model needs a fresh index: delete `.chroma/`.

## Publishing a demo

On [share.streamlit.io](https://share.streamlit.io), choose this repository and `app.py`. Do not add an API
key to a public demo, or every visitor will spend it: visitors can use *No LLM* or paste their own key.
Ollama is not available there.

## Problems

| Message | Cause |
|---|---|
| *no readable text found. It may be a scan* | the PDF is an image; OCR is not supported |
| *the PDF is password protected* | remove the password first |
| *Could not start the … model. Is the API key set?* | missing or wrong key |
| *Best match scored 0.52, below 0.60* | nothing in the documents is close to the question; try other words |
| *No retrieved passage supports the model's answer* | the model answered, but its answer could not be traced to a passage, so it was hidden |
| Ollama answers take a minute | normal on a CPU; use a smaller model, or fewer passages with `TOP_K=3` |

# Case study: Document Q&A with checked citations

**Problem.** Staff and customers ask questions that a handbook, manual or policy already answers.
Search returns pages to read, and a plain chatbot may invent an answer. The goal was a tool that answers
from the documents only, shows where each answer came from, and admits when it does not know.

**What it does.** The user uploads PDF, text or Markdown files and asks in plain language. The system finds
the most relevant passages, has a language model answer from them, and checks each sentence against the
passage it cites before showing it.

**Built with.** Python, LangChain, Chroma, FastEmbed (bge-small, on CPU), Streamlit and pypdf. Answers come
from Claude, OpenAI, Ollama, or a no-LLM mode. Tests and linting run in GitHub Actions.

**Examples** (sample documents, qwen2.5:1.5b on a 4-core machine with no GPU)

| Question | Answer | Source |
|---|---|---|
| How much paid leave does the secondary carer of a new baby get? | 8 weeks | employee_handbook.pdf, page 3 |
| What do standard and express delivery each cost? | Standard delivery costs 4.90 euros. Express delivery costs 12.50 euros. | shipping_and_returns.txt, chunk 1 |
| What does the Rill not filter out? | The Rill does not filter out viruses, dissolved salt, or heavy metals. | rill_purifier_manual.md, What it removes |
| What is the dress code at the office? | I could not find the answer in the uploaded documents. | none |

**The interesting part.** The first version accepted any citation whose number existed. Evaluation showed the
small model citing the wrong page, or nothing at all. The fix checks that a passage contains the sentence's
words and all its numbers, then moves, adds or removes citations. Together with a prompt change, it took
correct answers on the tuning questions from 1/6 to 6/6.

**Results.** On 15 held-out questions with a 1.5-billion-parameter model, the right passage was retrieved
11/11 times, the answer was correct 8/11 times (7/11 on an earlier run), and 4/4 unanswerable questions were
refused. One wrong answer carried a plausible citation; the README lists this and the other known limits.

**Next.** Score Claude and GPT on the same questions, check meaning as well as words, support follow-up
questions, add OCR.

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

**The interesting part.** Evaluation drove the design, in three rounds:

1. The first version accepted any citation whose number existed. The small model cited the wrong page, or
   nothing. A check that the cited passage contains the sentence's words and numbers, plus a prompt fix, took
   correct answers on the first tuning questions from 1/6 to 6/6.
2. A 3B model first looked better than the 1.5B one (9/11 against 8/11). More questions showed the opposite,
   so the smaller model stayed the default.
3. Small models answered "not found" when the honest answer was "no". Four prompt variants were measured and
   none helped, so the finding is documented instead of hidden behind a tuned score.

**Results.** On 20 questions written after all tuning and scored once, the 1.5-billion-parameter model
answered 10 of 14 answerable questions correctly, refused all 6 unanswerable ones, and showed no wrong answer:
every miss was a "not found". Method and limits: [EVALUATION.md](EVALUATION.md).

**Next.** Score Claude and GPT on the same questions, check meaning as well as words, support follow-up
questions, add OCR.

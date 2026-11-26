# Evaluation

How well the system answers questions about the three sample documents, how that was measured, and what
the measurements changed.

## Method

**Questions.** Three sets, all about the invented company in [sample_docs](../sample_docs):

| Set | Questions | Used for |
|---|---|---|
| [dev.json](../evaluation/dev.json) | 26 (17 answerable, 9 not) | choosing every setting and prompt |
| [test.json](../evaluation/test.json) | 15 (11 answerable, 4 not) | the first held-out set. Its answers were read while fixing problems, so it is no longer independent; kept for history. |
| [test2.json](../evaluation/test2.json) | 20 (14 answerable, 6 not) | written after the problems were found and **scored once**, after all settings were fixed. These are the headline numbers. |

Answerable questions include ones that need two facts combined, a yes/no answer, or a small calculation.
The unanswerable ones are on the documents' topics (*Which colours is the Rill sold in?*) or off them
(*Who wrote Hamlet?*).

**Measures**, per question ([evaluation/run.py](../evaluation/run.py)):

- *retrieved*: the passage holding the answer is among the 5 retrieved;
- *correct*: the answer is shown and contains the required facts (word lists per question; a grader that
  checks words, not meaning);
- *cited*: the answer cites the passage holding the answer;
- *refused*: an unanswerable question gets *not found*;
- *wrong answers shown*: an answer shown that is wrong, or shown for an unanswerable question. This is the
  error that matters most, because a user may act on it.

**Setup.** One 4-core virtual machine, no GPU. Ollama with temperature 0 and a fixed seed. Final settings:
5 passages, relevance gate 0.60, the prompt in [pipeline.py](../docqa/pipeline.py).

## Results on the held-out set (test2, scored once)

| Mode | Retrieved | Correct | Cited | Refused | Wrong answers shown | Median time |
|---|---|---|---|---|---|---|
| No LLM | 13/14 | 9/14 | 11/14 | 3/6 | 6 | 0.1 s |
| Ollama qwen2.5:1.5b | 13/14 | **10/14** | 10/14 | **6/6** | **0** | 5.8 s |
| Ollama qwen2.5:3b | 13/14 | 6/14 | 6/14 | 6/6 | 0 | 10.6 s |

Per-question answers: [evaluation/results](../evaluation/results).

What the misses were:

- **Every miss by the two models is a *not found*, never a wrong answer.** The system fails safe: it
  sometimes says it does not know when the documents do answer.
- **Retrieval missed once:** *How long may I work from **abroad**?* The handbook says *working from
  **another country***, and the embedding model did not connect the two.
- **The relevance gate blocked 2 answerable questions** whose best passages scored 0.598 and 0.59, just under
  0.60. It was set on the dev questions, where no answerable question scored below 0.617. The gate is a
  trade-off and costs some answers.
- **The 3B model refused more than the 1.5B model** on both dev (9/17 against 12/17) and test2. An earlier
  run on the old test set suggested the opposite (9/11 against 8/11); with more questions that turned out to
  be noise.
- **No LLM** shows a quote for almost every question, so half of the unanswerable questions got an
  unrelated quote.

## Results on the tuning set (dev, final settings)

| Mode | Retrieved | Correct | Cited | Refused |
|---|---|---|---|---|
| No LLM | 17/17 | 13/17 | 16/17 | 3/9 |
| qwen2.5:1.5b | 17/17 | 12/17 | 12/17 | 9/9 |
| qwen2.5:3b | 17/17 | 9/17 | 10/17 | 9/9 |

## What the evaluation changed

1. **Citations are checked after the model answers.** In the first version the 1.5B model gave no citation
   in five of six answers, and its one citation, `[2]`, was copied from an example in the prompt and pointed
   at the wrong page. Removing the example and adding the [citation check](DESIGN.md#the-citation-check)
   took correct answers on the first 6 answerable dev questions from 1/6 to 6/6.
2. **A grading rule was fixed.** One question was marked correct because an unrelated sentence contained
   "cannot". The stricter rule lowered the no-LLM score on the old test set from 8/11 to 7/11.
3. **The gate moved from 0.65 to 0.60.** At 0.65 it blocked an answerable dev question before any model saw
   it. At 0.60 every unanswerable dev question was still refused.
4. **Prompt changes were tried and rejected.** On the 17 answerable dev questions:

   | Prompt | qwen2.5:1.5b | qwen2.5:3b |
   |---|---|---|
   | final prompt | 12/17 | 9/17 (gate 0.60) |
   | question moved to the end | 9/17 | not run |
   | "look for the same meaning; NOT_FOUND only when nothing helps" | 11/17 | 6/17 |
   | "no can be an answer; simple arithmetic is allowed" | 12/17 | 9/17, plus a wrong answer |
   | 3 passages instead of 5 | 11/17 | 10/17 (gate 0.65) |

   The first and fourth rows ran with the gate at 0.60, the others at 0.65; the final prompt scored 12/17
   with the 1.5B model at both. None beat the final prompt, so it was kept. Small models answer *not found* to questions whose answer
   is "no" (*Can I return an opened cartridge?*) or needs a step (*leave after seven years* = 26 + 3).
   Prompting did not fix that; a larger model is the remedy.

## Limits of this evaluation

- 61 questions about 3 short documents, written by the same author as the documents. It shows how the parts
  behave. It does not predict accuracy on someone else's documents.
- The documents make only 15 passages, so retrieval is easy.
- The grader checks for words, so a correct answer worded differently can be marked wrong, and the reverse.
- Claude and OpenAI were not scored: no API keys were available. Their setup is unit-tested.
- Earlier runs varied by one question between identical runs.

## Reproduce

```bash
python -m evaluation.run --set test2 --provider extractive
ollama pull qwen2.5:1.5b
python -m evaluation.run --set test2 --provider ollama --model qwen2.5:1.5b
python -m evaluation.run --set test2 --provider anthropic        # needs ANTHROPIC_API_KEY
```

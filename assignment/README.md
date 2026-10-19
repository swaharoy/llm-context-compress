# COMP 580 — Assignment 1: Bloom Filters for LLM Context Compression

In this assignment, you will explore whether **Bloom Filters** can be used to reduce the amount of context processed by a Large Language Model (LLM) while maintaining question-answering accuracy.

The goal is simple:

> **Use Bloom Filters to reduce LLM input token usage while answering as many questions correctly as possible.**

## Repository Structure

The repository contains two directories:

```text
Assignment1_COMP_580/
├── contexts/
├── questions/
└── answers/
```

* `contexts/` contains the long-context documents.
* `questions/` contains the corresponding questions for each context.
* `answers/` contains the corresponding answers to the questions for each context.

## Contexts

There are **8 contexts** in total, consisting of `v1` and `v2` versions.

For the assignment, **use the `v1` contexts**. These are approximately **15K–50K tokens** long.

Each context has approximately **20 questions**, for a total of approximately **80 questions**.

The `v2` contexts are significantly longer. You are welcome to experiment with them, but they are **optional** and may require more capable or expensive LLMs to process effectively.

## Task

First, establish a **full-context baseline** by answering each question and comparing the answers with the gold answers in `answers/` using the entire corresponding context:

```text
LLM(Full Context + Question)
```

Next, design your own method for using **Bloom Filters to compress the context**. Then query the LLM using the reduced context:

```text
LLM(Compressed Context + Question)
```

You are encouraged to be creative in how you use Bloom Filters for context reduction.

## Experiments

Vary the parameters of your Bloom Filter and study how they affect:

* **Bloom Filter memory usage:** The total memory required to store the Bloom
  Filter(s) used to compress the context, reported in bits, bytes, KB, or MB.

* **False-positive rate:** The fraction of membership queries for elements
  that are not present in the indexed set but are incorrectly reported as
  present by the Bloom Filter.

* **Amount of context retained:** The fraction of the original context that
  remains after applying your Bloom-Filter-based context selection method.
  Report this as either the number of retained tokens or as a percentage of
  the original context.

* **LLM input token usage:** The total number of input tokens sent to the LLM
  when answering the questions. This should include the retained context,
  question, and any additional prompt/instructions used by your method.

* **Question-answering accuracy:** The fraction of questions answered
  correctly by the LLM using the retained context. Compare this against the
  accuracy obtained when using the full, uncompressed context.

Your experiments should demonstrate the tradeoff between **memory, token efficiency, and accuracy**.

## LLM

Use **Gemini-3.5-Flash-Lite**.

Use the **same model and prompting strategy** for all experiments so that your results are directly comparable.

## Reproducibility

Your submitted code must reproduce all reported experiments.

**The TAs will rigorously test your implementation and reported results.**

Fix random seeds whenever applicable and ensure that your experiments can be executed without manual intervention.

## Submission

Submit:

* Source code (ideally, 2 python files should suffice for this assignment), a requirements.txt file, a README containing the exact execution command.
* A short report describing your approach, experiments, results, and generated plots

Your report should clearly compare the **full-context baseline** against your **Bloom-Filter-based approach**, including both question-answering accuracy and input token usage.

Good luck!

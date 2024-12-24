# Prodify: Product Assistant in Coding

**(C) IURII TRUKHIN, yuri@trukhin.com, 2024**  
_An experimental codebase assistant that learns and indexes your entire repository to answer questions — and soon, it might even write code for you!_

---

## Introduction

**Prodify** was created because _I, as a product manager, shouldn’t need to know how to write code_. Yet, I wanted a tool that understands the entire codebase of my application and lets me ask any question about it. Now, Prodify:

- Indexes your project’s source files.
- Provides an interactive UI to explore code contexts and ask AI-driven questions.
- Will (hopefully soon) **write code** on my behalf.  

This repository is a “**just for fun**” personal experiment to see how close we can get to a code-savvy product assistant. If you find it interesting, try it out and contribute improvements!

---

## Project Structure

- **`prepare.sh`**  
  A simple shell script that creates a Python `venv`, upgrades pip, and installs all required dependencies for both **`indexer.py`** and **`ask.py`**.

- **`indexer.py`**  
  A script to **index** the entire repository’s code. It stores vector embeddings in a local `.chromadb` folder, uniquely named with your project path + date/time + GUID.

- **`ask.py`**  
  A script to **ask** questions about your code in a custom prompt_toolkit-based interface. You can:
  1. Choose which indexed codebase to query.
  2. Ask free-form questions about your code.
  3. Delete unneeded indexes if desired.

The scripts rely on **OpenAI** for embeddings and conversation (or fallback to local community embeddings if needed).

---

## How It Works

- **Embed the code**:  
  *indexer.py* uses OpenAI embeddings by default (or falls back to community embeddings if needed). It reads through your project files, splits them into manageable chunks, and stores their embeddings in Chroma, located in the `.chromadb/` directory.

- **Question answering**:  
  *ask.py* provides a console-based interface via [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/). You select an index, ask a question (e.g., `Q:`), and the system retrieves the most relevant code chunks. Then it invokes OpenAI with the context plus your query to produce an AI-driven answer.

- **Potential code writing**:  
  While Prodify currently focuses on code analysis and Q&A, the ultimate aim is to help generate or refactor code snippets automatically, bridging the gap between product management insights and coding tasks.

---

## Just for Fun

Everything in **Prodify** has been built and shared **just for fun**—an experimental project exploring how a non-coding product manager can still manage and query large codebases. If you find it intriguing, feel free to clone, modify, and contribute your own improvements. Enjoy!  

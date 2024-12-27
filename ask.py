# ------------------------------------------------------------------------------
# Prodify: Product assistant in coding
# (C) IURII TRUKHIN, yuri@trukhin.com, 2024
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
# ------------------------------------------------------------------------------

"""
We have released a new major version of our SDK, and we recommend upgrading promptly.

It's a total rewrite of the library, so many things have changed, but we've made upgrading easy with a
code migration script and detailed docs below. It was extensively beta tested prior to release.

( ... Migration details ... )
"""

import os
import sys
import re
import getpass
import shutil
import queue
import threading
import time

import openai
from openai import OpenAIError, RateLimitError

try:
    from prompt_toolkit import prompt
    from prompt_toolkit.application import Application
    from prompt_toolkit.application.current import get_app
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, VSplit, Window, WindowAlign
    from prompt_toolkit.widgets import Dialog, Button, Label, RadioList
    from prompt_toolkit.styles import Style
    from prompt_toolkit.key_binding import KeyBindings
except ImportError:
    print("(C) IURII TRUKHIN, yuri@trukhin.com, 2024")
    print("prompt_toolkit is missing. Please install it: pip install prompt_toolkit")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm is missing. Please install it: pip install tqdm")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.markdown import Markdown
except ImportError:
    print("rich is missing. Please install it: pip install rich")
    sys.exit(1)

try:
    import tiktoken
except ImportError:
    print("tiktoken is missing. Please install it: pip install tiktoken")
    sys.exit(1)

# Attempt to import langchain + chroma. Otherwise fallback.
try:
    from langchain_openai.embeddings import OpenAIEmbeddings
    from langchain_chroma import Chroma
except ImportError:
    print("Falling back to langchain_community.")
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma

console = Console()

NUM_WORKERS = 4
QUEUE_SIZE = 100

# For o1-mini, we allow up to 16384 tokens
MAX_TOKENS = 16384

MAX_RETRIES = 5
INITIAL_DELAY = 1.0
K = 100

global_lock = threading.Lock()

##############################################################################
# Conversation + Document History
##############################################################################

# We keep the entire conversation in memory. Each user query + assistant answer is appended.
conversation = [
    {
        "role": "user",
        "content": (
            "You are a code assistant using the maximum context possible (16384 tokens). "
            "You remember all previous user queries and your answers. "
            "Answer as helpfully as possible."
        )
    }
]

# We also keep a list of document contexts from previous requests.
# Each entry is a string of concatenated doc context from a single request.
doc_history = []


def prune_conversation_if_needed(conversation_list, enc, model="o1-mini"):
    """
    If the conversation is too large (potentially exceeding the token limit),
    gradually remove the oldest user/assistant messages (but not the first user message).
    This is a simplified logic: we remove entire messages from the start.
    """
    while True:
        total_tokens = 0
        for msg in conversation_list:
            total_tokens += len(enc.encode(msg["content"]))
        if total_tokens <= MAX_TOKENS:
            break

        # Remove the earliest user/assistant message except the very first user message
        to_remove = None
        for i, msg in enumerate(conversation_list):
            if msg["role"] in ("user", "assistant") and i > 0:
                to_remove = i
                break
        if to_remove is None:
            break
        conversation_list.pop(to_remove)


##############################################################################
# Utility functions
##############################################################################

def parse_date_time(index_name: str):
    pattern = r"^(?P<name>.+)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<guid>[0-9a-fA-F]+)$"
    match = re.match(pattern, index_name)
    if not match:
        return index_name, ""
    proj = match.group("name") or "project"
    d = match.group("date")
    t = match.group("time")

    yyyy = d[0:4]
    mm = d[4:6]
    dd = d[6:8]
    hh = t[0:2]
    mn = t[2:4]
    return proj, f"{yyyy}-{mm}-{dd} {hh}:{mn}"


def parse_file_update_instructions(answer: str):
    pattern = re.compile(
        r"\[FILE_UPDATE\]\s*filename:\s*(.+?)\s*code:\s*(.+?)\[\/FILE_UPDATE\]",
        flags=re.DOTALL
    )
    matches = pattern.findall(answer)
    results = []
    for fname, code in matches:
        fname = fname.strip()
        code = code.strip()
        results.append((fname, code))
    return results


def update_file_contents(file_path: str, new_content: str):
    parent_dir = os.path.dirname(file_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)


##############################################################################
# AskWorker
##############################################################################

class AskWorker(threading.Thread):
    """
    Worker thread:
      - If user says "continue"/"продолжи", then we reuse *all previous doc contexts* (doc_history).
      - Otherwise, we retrieve new docs and store them in doc_history as well.
      - Combine doc_history with conversation, build prompt, call openai.chat.completions.create.
      - Save the user+assistant messages in the conversation.
    """
    def __init__(self, task_queue, retriever, enc):
        super().__init__()
        self.task_queue = task_queue
        self.retriever = retriever
        self.enc = enc
        self.daemon = True

    def run(self):
        while True:
            item = self.task_queue.get()
            if item is None:
                self.task_queue.task_done()
                break

            query, idx = item
            self.process_query(query, idx)
            self.task_queue.task_done()

    def process_query(self, query, idx):
        with global_lock:
            console.print(f"\n[bold](Processing question #{idx})[/bold] Q: {query}", style="dim")

        with tqdm(total=2, desc=f"Processing question #{idx}", unit="step") as pbar:
            # Check if user just wants to continue
            skip_retrieval = query.strip().lower() in ["continue", "продолжи"]

            if skip_retrieval:
                # Reuse all doc contexts from doc_history
                combined_docs_text = "\n".join(doc_history)
                pbar.update(1)
            else:
                # Retrieve new docs
                docs = []
                try:
                    docs = self.retriever.invoke(query)
                except Exception as e:
                    with global_lock:
                        console.print(f"Error retrieving docs: {e}", style="bold red")
                    return
                pbar.update(1)

                # Build doc context from these docs
                doc_context = []
                total_doc_tokens = 0
                for i, doc in enumerate(docs, start=1):
                    source = doc.metadata.get("source", "unknown")
                    piece = f"--- document {i} source: {source} ---\n{doc.page_content}\n\n"
                    piece_tokens = len(self.enc.encode(piece))
                    if total_doc_tokens + piece_tokens > MAX_TOKENS:
                        break
                    doc_context.append(piece)
                    total_doc_tokens += piece_tokens
                new_docs_text = "".join(doc_context)

                # Append to doc_history
                if new_docs_text.strip():
                    doc_history.append(new_docs_text)

                # Combine all doc_history
                combined_docs_text = "\n".join(doc_history)

            # Build temporary conversation
            with global_lock:
                temp_convo = list(conversation)

                # Add doc context (all from doc_history if any)
                if combined_docs_text.strip():
                    temp_convo.append({
                        "role": "user",
                        "content": f"Relevant code context (all previous docs):\n{combined_docs_text}"
                    })

                # The user's new query
                temp_convo.append({"role": "user", "content": query})

                prune_conversation_if_needed(temp_convo, self.enc, model="o1-mini")

            # Call openai
            answer = None
            delay = INITIAL_DELAY
            for attempt in range(MAX_RETRIES):
                try:
                    resp = openai.chat.completions.create(
                        model="o1-mini",
                        messages=temp_convo,
                    )
                    answer = resp.choices[0].message.content
                    break
                except RateLimitError as e:
                    if attempt < MAX_RETRIES - 1:
                        console.print(
                            f"Rate limit. Waiting {delay:.1f}s (attempt {attempt+1}/{MAX_RETRIES})...",
                            style="bold red"
                        )
                        time.sleep(delay)
                        delay *= 2
                    else:
                        with global_lock:
                            console.print(f"RateLimitError: {e}", style="bold red")
                        return
                except OpenAIError as e:
                    with global_lock:
                        console.print(f"OpenAIError: {e}", style="bold red")
                    return
                except Exception as e:
                    with global_lock:
                        console.print(f"Unexpected error: {e}", style="bold red")
                    return
            pbar.update(1)

        with global_lock:
            if answer:
                console.print("[dim]\n=== Answer ===[/dim]", style="dim")
                console.print(Markdown(answer))

                # Check for [FILE_UPDATE]
                updates = parse_file_update_instructions(answer)
                for fname, new_code in updates:
                    console.print(
                        f"\n[bold yellow]AI suggests updating file:[/bold yellow] {fname}",
                        style="bold yellow"
                    )
                    console.print("Proposed new content:\n", style="dim")
                    console.print(Markdown(f"```\n{new_code}\n```"))
                    confirm = input("Apply this update? [y/N] ").strip().lower()
                    if confirm == 'y':
                        update_file_contents(fname, new_code)
                        console.print(f"File {fname} has been updated.\n", style="bold green")

                # Store user+assistant messages in global conversation
                conversation.append({"role": "user", "content": query})
                conversation.append({"role": "assistant", "content": answer})
                prune_conversation_if_needed(conversation, self.enc, model="o1-mini")
            else:
                console.print("No answer was returned. Something went wrong.", style="bold red")


##############################################################################
# A simple radio-list dialog
##############################################################################

def radio_with_three_buttons_dialog(title: str, text: str, values, style=None):
    from prompt_toolkit.widgets import Dialog, Button, Label, RadioList
    from prompt_toolkit.layout.containers import HSplit
    from prompt_toolkit.styles import Style
    from prompt_toolkit.application import Application
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.application.current import get_app

    radio = RadioList(values=values)
    result_index = [None]
    result_action = [None]

    def on_use():
        result_index[0] = radio.current_value
        result_action[0] = "use"
        get_app().exit()

    def on_delete():
        result_index[0] = radio.current_value
        result_action[0] = "delete"
        get_app().exit()

    def on_exit():
        result_index[0] = radio.current_value
        result_action[0] = "exit"
        get_app().exit()

    btn_use = Button(text="Use", handler=on_use)
    btn_use.style = "fg:green"

    btn_del = Button(text="Delete", handler=on_delete)
    btn_del.style = "fg:red"

    btn_exit = Button(text="Exit", handler=on_exit)

    body = HSplit([
        Label(text=title, dont_extend_height=True),
        Label(text=text, dont_extend_height=True),
        radio,
    ])

    dialog = Dialog(
        body=body,
        buttons=[btn_use, btn_del, btn_exit],
        with_background=False
    )

    kb = KeyBindings()

    @kb.add("escape")
    def _(event):
        result_index[0] = None
        result_action[0] = None
        event.app.exit()

    layout = Layout(dialog)
    application = Application(
        layout=layout,
        key_bindings=kb,
        style=style or Style(),
        full_screen=False
    )
    application.run()

    return result_index[0], result_action[0]


##############################################################################
# Main entry point
##############################################################################

def main():
    console.print("Prodify: Product assistant in coding", style="bold")
    console.print("(C) IURII TRUKHIN, yuri@trukhin.com, 2024\n", style="bold")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        console.print("No OPENAI_API_KEY found in the environment.", style="bold red")
        key = getpass.getpass("Please enter your OpenAI API key (sk-...): ").strip()
        if not key:
            console.print("No API key provided. Exiting.", style="bold red")
            sys.exit(1)
        os.environ["OPENAI_API_KEY"] = key

    openai.api_key = os.environ["OPENAI_API_KEY"]

    base_dir = os.path.join(os.path.expanduser("~"), ".chromadb")
    os.makedirs(base_dir, exist_ok=True)

    if not os.path.isdir(base_dir):
        console.print("No indexes found in ~/.chromadb. Exiting.", style="bold red")
        sys.exit(0)

    while True:
        subfolders = []
        for entry in os.scandir(base_dir):
            if entry.is_dir():
                subfolders.append(entry.name)
        if not subfolders:
            console.print("No indexes found. Exiting.", style="bold red")
            break

        subfolders.sort()
        radio_values = []
        for folder_name in subfolders:
            proj, dt_str = parse_date_time(folder_name)
            label = f"{proj} ({dt_str})" if dt_str else folder_name
            radio_values.append((folder_name, label))

        index_choice, action = radio_with_three_buttons_dialog(
            title="Choose an index from ~/.chromadb",
            text=(
                "Use ↑/↓ to move, Enter to select an item.\n"
                "Then click [Use], [Delete], or [Exit].\n"
                "Press Esc to cancel."
            ),
            values=radio_values,
            style=Style.from_dict({
                "dialog":       "bg:#ffffff #000000",
                "dialog.body":  "bg:#ffffff #000000",
                "dialog.shadow":"bg:#cccccc",
            }),
        )

        if index_choice is None or action is None:
            console.print("\nNo selection was made. Exiting.\n", style="bold yellow")
            break

        chosen_path = os.path.join(base_dir, index_choice)
        console.print(f"You selected: {chosen_path}\n", style="bold")

        if action == "use":
            console.print("Loading index for Q&A...", style="dim")
            try:
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-large",
                    openai_api_key=openai.api_key,
                )
                db = Chroma(
                    collection_name=index_choice,
                    embedding_function=embeddings,
                    persist_directory=chosen_path
                )
            except Exception as e:
                console.print(f"Error loading {chosen_path}: {e}", style="bold red")
                continue

            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": K}
            )
            enc = tiktoken.get_encoding("cl100k_base")

            task_queue = queue.Queue(maxsize=QUEUE_SIZE)
            workers = []
            for _ in range(NUM_WORKERS):
                w = AskWorker(task_queue, retriever, enc)
                w.start()
                workers.append(w)

            console.print(
                "\n[bold dim]You can now ask questions about the project codebase.[/bold dim]\n"
                " - Type your question and press Enter.\n"
                " - Type 'continue' or 'продолжи' to reuse all previous doc contexts.\n"
                " - Press Ctrl+C to exit.\n",
                style="dim"
            )

            question_counter = 0
            try:
                while True:
                    user_query = input("Q: ")
                    if not user_query.strip():
                        console.print("[gray]Empty question. Try again or Ctrl+C to exit.[/gray]")
                        continue
                    question_counter += 1
                    task_queue.put((user_query, question_counter))
            except KeyboardInterrupt:
                console.print("\nInterrupted by user.\n", style="bold yellow")

            # signal the workers to end
            for _ in range(NUM_WORKERS):
                task_queue.put(None)
            task_queue.join()
            for w in workers:
                w.join()

            console.print("Done with Q&A.\n", style="dim")

        elif action == "delete":
            console.print(f"Deleting index folder: {chosen_path}\n", style="bold red")
            try:
                shutil.rmtree(chosen_path)
                console.print(f"Deleted: {chosen_path}\n", style="bold red")
            except Exception as e:
                console.print(f"Error deleting {chosen_path}: {e}", style="bold red")

        elif action == "exit":
            console.print("Exiting program.\n", style="bold yellow")
            sys.exit(0)

    console.print("\nAll done!\n", style="dim")


if __name__ == "__main__":
    main()

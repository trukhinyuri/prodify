# ------------------------------------------------------------------------------
# Prodify: Product assistant in coding
# (C) IURII TRUKHIN, yuri@trukhin.com, 2024
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
# ------------------------------------------------------------------------------

import os
import sys
import time
import queue
import threading
import getpass
import shutil
import re

import openai

# prompt_toolkit layout-based imports
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
    print("(C) IURII TRUKHIN, yuri@trukhin.com, 2024\nprompt_toolkit is missing. Please install it:")
    print("   pip install prompt_toolkit")
    sys.exit(1)

# tqdm for progress
try:
    from tqdm import tqdm
except ImportError:
    print("(C) IURII TRUKHIN, yuri@trukhin.com, 2024\ntqdm is missing. Please install it:")
    print("   pip install tqdm")
    sys.exit(1)

# rich for console formatting
try:
    from rich.console import Console
    from rich.markdown import Markdown
except ImportError:
    print("(C) IURII TRUKHIN, yuri@trukhin.com, 2024\nrich is missing. Please install it:")
    print("   pip install rich")
    sys.exit(1)

# openai error
from openai import APIError

# tiktoken for token counting
try:
    import tiktoken
except ImportError:
    print("(C) IURII TRUKHIN, yuri@trukhin.com, 2024\ntiktoken is missing. Please install it:")
    print("   pip install tiktoken")
    sys.exit(1)

# Attempt to import langchain + chroma. Otherwise fallback.
try:
    from langchain_openai.embeddings import OpenAIEmbeddings
    from langchain_chroma import Chroma
except ImportError:
    print("(C) IURII TRUKHIN, yuri@trukhin.com, 2024\nFalling back to langchain_community.")
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma


# ------------------------------------------------------------------------------
# "Prodify" single-screen ask.py that displays a RadioList of .chromadb indexes
# and three buttons:
#   1) "Use"   - in green color
#   2) "Delete" - in red color
#   3) "Exit"   - ends the program
# (C) IURII TRUKHIN, yuri@trukhin.com, 2024
# ------------------------------------------------------------------------------
console = Console()

NUM_WORKERS = 4
QUEUE_SIZE = 100
MAX_TOKENS = 128_000
MAX_RETRIES = 5
INITIAL_DELAY = 1.0
K = 100

global_lock = threading.Lock()


def parse_date_time(index_name: str):
    """
    Attempt to parse something like "projectName_YYYYmmdd_HHMMSS_guid"
    and return (projectName, "YYYY-mm-dd HH:MM") or fallback
    """
    pattern = r"^(?P<name>.+)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<guid>[0-9a-fA-F]+)$"
    match = re.match(pattern, index_name)
    if not match:
        return index_name, ""
    proj = match.group("name") or "project"
    d = match.group("date")
    t = match.group("time")

    yyyy = d[0:4]
    mm   = d[4:6]
    dd   = d[6:8]

    hh   = t[0:2]
    mn   = t[2:4]
    return proj, f"{yyyy}-{mm}-{dd} {hh}:{mn}"


class AskWorker(threading.Thread):
    """
    Thread for processing user queries (Q:) one at a time:
      1) Retrieve relevant docs
      2) Build context prompt
      3) Send to OpenAI with optional rate-limit retries
      4) Print the response with code highlighting
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
        # Show the question is being processed, do not clear screen
        with global_lock:
            console.print(
                f"[bold]\n(Processing question #{idx})[/bold] Q: {query}",
                style="dim"
            )

        with tqdm(total=2, desc=f"Processing question #{idx}", unit="step") as pbar:
            docs = []
            try:
                docs = self.retriever.invoke(query)
            except Exception as e:
                with global_lock:
                    console.print(f"Error retrieving docs: {e}", style="bold red")
                return
            pbar.update(1)

            system_prefix = "You are a code assistant. Use the provided context:\n\nContext:\n"
            suffix = f"\nQuestion: {query}"
            base_msg = f"{system_prefix}<CONTEXT_PLACEHOLDER>{suffix}"
            base_tokens = len(self.enc.encode(base_msg))

            context_parts = []
            current_tokens = base_tokens
            for i, doc in enumerate(docs, start=1):
                source = doc.metadata.get("source", "unknown")
                piece = f"--- document {i} source: {source} ---\n{doc.page_content}\n\n"
                piece_tokens = len(self.enc.encode(piece))
                if current_tokens + piece_tokens > MAX_TOKENS:
                    break
                context_parts.append(piece)
                current_tokens += piece_tokens

            user_prompt = f"{system_prefix}{''.join(context_parts)}{suffix}"

            delay = INITIAL_DELAY
            answer = None
            for attempt in range(MAX_RETRIES):
                try:
                    resp = openai.chat.completions.create(
                        model="o1-preview",
                        messages=[{"role": "user", "content": user_prompt}]
                    )
                    answer = resp.choices[0].message.content
                    break
                except APIError as e:
                    if e.status_code == 429 and attempt < MAX_RETRIES - 1:
                        with global_lock:
                            console.print(
                                f"Rate limit (429), waiting {delay:.1f}s (attempt {attempt+1}/{MAX_RETRIES})...",
                                style="bold red"
                            )
                        time.sleep(delay)
                        delay *= 2
                    else:
                        with global_lock:
                            console.print(f"OpenAI APIError: {e}", style="bold red")
                        return
                except Exception as e:
                    with global_lock:
                        console.print(f"Unexpected error: {e}", style="bold red")
                    return

            pbar.update(1)

        # Print the answer below the question (still in main scroll buffer)
        with global_lock:
            if answer:
                console.print("[dim]\n=== Ответ ===[/dim]", style="dim")
                console.print(Markdown(answer))
            else:
                console.print("No answer (something went wrong).", style="bold red")


def radio_with_three_buttons_dialog(title: str, text: str, values, style=None):
    """
    A custom single-screen dialog that shows:
      - A RadioList for picking an index
      - Three buttons: [Use] (green), [Delete] (red), [Exit] (gray -> quits)
    Returns (selected_index, action):
      action is "use", "delete", or "exit".
    If user presses Esc => (None, None)
    """
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

    # Make "Use" green, "Delete" red, "Exit" normal/gray
    btn_use = Button(text="Use", handler=on_use)
    btn_use.style = "fg:green"

    btn_del = Button(text="Delete", handler=on_delete)
    btn_del.style = "fg:red"

    btn_exit = Button(text="Exit", handler=on_exit)
    # no special color

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
        full_screen=False,  # so we don't clear the entire screen buffer
    )

    application.run()

    return (result_index[0], result_action[0])


def main():
    # Do not clear screen, so user can scroll up to see the banner.
    console.print("Prodify: Product assistant in coding", style="bold")
    console.print("(C) IURII TRUKHIN, yuri@trukhin.com, 2024\n", style="bold")

    # Check or request OPENAI_API_KEY
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        console.print("No OPENAI_API_KEY found in environment.", style="bold red")
        key = getpass.getpass("Please enter your OpenAI API key (sk-...): ").strip()
        if not key:
            console.print("No API key provided. Exiting.", style="bold red")
            sys.exit(1)
        os.environ["OPENAI_API_KEY"] = key

    openai.api_key = os.environ.get("OPENAI_API_KEY")

    base_dir = ".chromadb"
    if not os.path.isdir(base_dir):
        console.print("No indexes found in .chromadb.", style="bold red")
        sys.exit(0)

    # Main loop
    while True:
        # Gather subfolders
        subfolders = []
        for entry in os.scandir(base_dir):
            if entry.is_dir():
                subfolders.append(entry.name)
        if not subfolders:
            console.print("No indexes found. Exiting.", style="bold red")
            break

        subfolders.sort()

        # Build the radio list
        radio_values = []
        for folder_name in subfolders:
            proj, dt_str = parse_date_time(folder_name)
            label = f"{proj} ({dt_str})" if dt_str else folder_name
            radio_values.append((folder_name, label))

        index_choice, action = radio_with_three_buttons_dialog(
            title="Choose an index from .chromadb",
            text=(
                "Use ↑/↓ to move; Enter to select an item.\n"
                "Then choose [Use], [Delete], or [Exit].\n"
                "Press Esc to cancel."
            ),
            values=radio_values,
            style=Style.from_dict({
                "dialog":            "bg:#ffffff #000000",
                "dialog.body":       "bg:#ffffff #000000",
                "dialog.shadow":     "bg:#cccccc",
            }),
        )

        if index_choice is None or action is None:
            console.print("\nNo selection made. Exiting.\n", style="bold yellow")
            break

        chosen_path = os.path.join(base_dir, index_choice)
        console.print(f"You selected: {chosen_path}\n", style="bold")

        if action == "use":
            console.print("loading index for Q&A...", style="dim")
            try:
                embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
                db = Chroma(
                    collection_name=index_choice,
                    embedding_function=embeddings,
                    persist_directory=chosen_path
                )
            except Exception as e:
                console.print(f"Error loading {chosen_path}: {e}", style="bold red")
                continue

            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": K})
            enc = tiktoken.get_encoding("cl100k_base")

            task_queue = queue.Queue(maxsize=QUEUE_SIZE)
            workers = []
            for _ in range(NUM_WORKERS):
                w = AskWorker(task_queue, retriever, enc)
                w.start()
                workers.append(w)

            console.print("\nAsk a question about the entire project code base. (Press Ctrl+C to exit):\n", style="dim")
            question_counter = 0
            try:
                while True:
                    query = input("Q: ")
                    if not query.strip():
                        console.print("[gray]Empty question. Try again or Ctrl+C to exit.[/gray]")
                        continue
                    question_counter += 1
                    task_queue.put((query, question_counter))
            except KeyboardInterrupt:
                console.print("\nInterrupted by user.\n", style="bold yellow")

            for _ in range(NUM_WORKERS):
                task_queue.put(None)
            task_queue.join()
            for w in workers:
                w.join()

            console.print("Done with Q&A.\n", style="dim")

        elif action == "delete":
            console.print(f"Deleting: {chosen_path}\n", style="bold red")
            try:
                shutil.rmtree(chosen_path)
                console.print(f"Deleted index folder: {chosen_path}\n", style="bold red")
            except Exception as e:
                console.print(f"Error deleting {chosen_path}: {e}", style="bold red")

        elif action == "exit":
            console.print("Exiting program.\n", style="bold yellow")
            sys.exit(0)

    console.print("\nAll done!\n", style="dim")


if __name__ == "__main__":
    main()


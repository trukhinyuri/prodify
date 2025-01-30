#!/usr/bin/env python3
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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
# ------------------------------------------------------------------------------

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
    from prompt_toolkit.layout.containers import HSplit, VSplit, Window
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

# We'll keep a high token limit, e.g. 16384
MAX_TOKENS = 16384
MAX_RETRIES = 5
INITIAL_DELAY = 1.0

# Number of chunks from Chroma
K = 100

# Global conversation context
conversation = [
    {
        "role": "user",
        "content": (
            "You are a coding assistant with a large context window. Keep track of previous code fragments and questions. "
            "Provide detailed and thorough responses."
        )
    }
]

# We store retrieved docs here; used if user types "continue"
doc_history = []

global_lock = threading.Lock()


##############################################################################
# Prune conversation
##############################################################################
def prune_conversation_if_needed(conversation_list, enc):
    """
    If total tokens in the conversation exceed MAX_TOKENS,
    remove oldest user/assistant messages (but not the very first user message).
    """
    while True:
        total_tokens = 0
        for msg in conversation_list:
            total_tokens += len(enc.encode(msg["content"]))
        if total_tokens <= MAX_TOKENS:
            break

        to_remove = None
        for i, msg in enumerate(conversation_list):
            if msg["role"] in ("user", "assistant") and i > 0:
                to_remove = i
                break
        if to_remove is None:
            break
        conversation_list.pop(to_remove)


##############################################################################
# File update instructions
##############################################################################
def parse_file_update_instructions(answer: str):
    """
    Look for blocks like:
      [FILE_UPDATE] filename: <...> code: <...> [/FILE_UPDATE]
    Return list of (filename, code).
    """
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
# Worker for Q&A
##############################################################################
class AskWorker(threading.Thread):
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
            console.print(f"\n[bold](Q#{idx})[/bold] Your question: {query}", style="dim")

        with tqdm(total=2, desc=f"Q#{idx}", unit="step") as pbar:
            skip_retrieval = query.strip().lower() == "continue"

            if skip_retrieval:
                # Reuse doc_history
                combined_docs_text = "\n".join(doc_history)
                pbar.update(1)
            else:
                # Retrieve new docs from Chroma
                docs = []
                try:
                    docs = self.retriever.invoke(query)
                except Exception as e:
                    with global_lock:
                        console.print(f"Error retrieving from Chroma: {e}", style="bold red")
                    return
                pbar.update(1)

                doc_context = []
                total_doc_tokens = 0
                for i, doc in enumerate(docs, start=1):
                    source = doc.metadata.get("source", "unknown")
                    piece = f"--- document {i} | source: {source} ---\n{doc.page_content}\n\n"
                    piece_tokens = len(self.enc.encode(piece))
                    if total_doc_tokens + piece_tokens > MAX_TOKENS:
                        break
                    doc_context.append(piece)
                    total_doc_tokens += piece_tokens

                new_docs_text = "".join(doc_context)
                if new_docs_text.strip():
                    doc_history.append(new_docs_text)

                combined_docs_text = "\n".join(doc_history)

            with global_lock:
                temp_convo = list(conversation)
                if combined_docs_text.strip():
                    temp_convo.append({
                        "role": "user",
                        "content": f"Here are the retrieved code fragments:\n{combined_docs_text}"
                    })

                temp_convo.append({"role": "user", "content": query})
                prune_conversation_if_needed(temp_convo, self.enc)

            # Use openai.chat.completions.create (>=1.0.0)
            answer = None
            delay = INITIAL_DELAY
            for attempt in range(MAX_RETRIES):
                try:
                    resp = openai.chat.completions.create(
                        model="o1-preview",  # The strongest model available if your key has access
                        messages=temp_convo,
                    )
                    answer = resp.choices[0].message.content
                    break
                except RateLimitError:
                    if attempt < MAX_RETRIES - 1:
                        console.print(
                            f"Rate limit encountered. Sleeping {delay:.1f}s (attempt {attempt+1}/{MAX_RETRIES})...",
                            style="bold red"
                        )
                        time.sleep(delay)
                        delay *= 2
                    else:
                        with global_lock:
                            console.print("RateLimitError: max retries exceeded.", style="bold red")
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

                # Check for file update blocks
                updates = parse_file_update_instructions(answer)
                for fname, new_code in updates:
                    console.print(f"\n[bold yellow]AI suggests updating file:[/bold yellow] {fname}")
                    console.print("Proposed content:\n", style="dim")
                    console.print(Markdown(f"```\n{new_code}\n```"))
                    confirm = input("Apply this change? [y/N] ").strip().lower()
                    if confirm == 'y':
                        update_file_contents(fname, new_code)
                        console.print(f"File {fname} updated.\n", style="bold green")

                # Update global conversation
                conversation.append({"role": "user", "content": query})
                conversation.append({"role": "assistant", "content": answer})
                prune_conversation_if_needed(conversation, self.enc)
            else:
                console.print("No answer returned. Something went wrong.", style="bold red")


##############################################################################
# Dialog for choosing folder in ~/.chromadb
##############################################################################
def radio_with_three_buttons_dialog(title: str, text: str, values, style=None):
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
    btn_del = Button(text="Delete", handler=on_delete)
    btn_exit = Button(text="Exit", handler=on_exit)

    body = HSplit([
        Label(text=title, dont_extend_height=True),
        Label(text=text, dont_extend_height=True),
        radio,
    ])

    dialog = Dialog(
        body=body,
        buttons=[btn_use, btn_del, btn_exit],
        with_background=True,
    )

    kb = KeyBindings()

    @kb.add("up")
    def move_up(event):
        if event.app.layout.has_focus(radio):
            radio.selected_index = (radio.selected_index - 1) % len(radio.values)

    @kb.add("down")
    def move_down(event):
        if event.app.layout.has_focus(radio):
            radio.selected_index = (radio.selected_index + 1) % len(radio.values)

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
        full_screen=True
    )
    application.run()

    return result_index[0], result_action[0]


##############################################################################
# MAIN
##############################################################################
def main():
    console.print("Prodify: Product assistant in coding", style="bold")
    console.print("(C) IURII TRUKHIN, yuri@trukhin.com, 2024\n", style="bold")

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        console.print("Environment variable OPENAI_API_KEY not found.", style="bold red")
        key = getpass.getpass("Please enter your OpenAI API key (sk-...): ").strip()
        if not key:
            console.print("No API key. Exiting.", style="bold red")
            sys.exit(1)
        os.environ["OPENAI_API_KEY"] = key

    openai.api_key = os.environ["OPENAI_API_KEY"]

    base_dir = os.path.join(os.path.expanduser("~"), ".chromadb")
    os.makedirs(base_dir, exist_ok=True)

    # Find subfolders in ~/.chromadb
    subfolders = [entry.name for entry in os.scandir(base_dir) if entry.is_dir()]
    if not subfolders:
        console.print("No indexes found in ~/.chromadb. Exiting.", style="bold red")
        sys.exit(0)

    while True:
        subfolders.sort()
        radio_values = [(folder_name, folder_name) for folder_name in subfolders]

        index_choice, action = radio_with_three_buttons_dialog(
            title="Select an index from ~/.chromadb",
            text=(
                "Use up/down to move, Enter to focus buttons.\n"
                "Choose 'Use' to open the index, 'Delete' to remove it, or 'Exit' to quit."
            ),
            values=radio_values,
            style=Style.from_dict({
                "dialog":        "bg:#ffffff #000000",
                "dialog.body":   "bg:#ffffff #000000",
                "dialog.shadow": "bg:#aaaaaa",
            }),
        )

        if index_choice is None or action is None:
            console.print("\nNo selection made. Exiting.\n", style="bold yellow")
            break

        chosen_path = os.path.join(base_dir, index_choice)
        console.print(f"You selected: {chosen_path}\n", style="bold")

        if action == "use":
            console.print("Loading index for Q&A...", style="dim")
            try:
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-ada-002",
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

            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": K})
            enc = tiktoken.get_encoding("cl100k_base")

            task_queue = queue.Queue(maxsize=QUEUE_SIZE)
            workers = []
            for _ in range(NUM_WORKERS):
                w = AskWorker(task_queue, retriever, enc)
                w.start()
                workers.append(w)

            console.print(
                "\nAsk questions about your code below.\n"
                " - Type your question and press Enter.\n"
                " - Type 'continue' to reuse previous doc context.\n"
                " - Press Ctrl+C to exit.\n",
                style="dim"
            )

            question_counter = 0
            try:
                while True:
                    user_query = input("Q: ")
                    if not user_query.strip():
                        console.print("[gray]Empty query. Please try again or Ctrl+C to exit.[/gray]")
                        continue
                    question_counter += 1
                    task_queue.put((user_query, question_counter))
            except KeyboardInterrupt:
                console.print("\nQ&A session ended (Ctrl+C).\n", style="bold yellow")

            # Signal end
            for _ in range(NUM_WORKERS):
                task_queue.put(None)
            task_queue.join()
            for w in workers:
                w.join()

            console.print("Done with Q&A session.\n", style="dim")

        elif action == "delete":
            console.print(f"Deleting index folder: {chosen_path}\n", style="bold red")
            try:
                shutil.rmtree(chosen_path)
                console.print(f"Deleted: {chosen_path}\n", style="bold red")
                subfolders.remove(index_choice)
            except Exception as e:
                console.print(f"Error deleting {chosen_path}: {e}", style="bold red")

        elif action == "exit":
            console.print("Exiting the program.\n", style="bold yellow")
            sys.exit(0)

    console.print("\nAll done!\n", style="dim")


if __name__ == "__main__":
    main()

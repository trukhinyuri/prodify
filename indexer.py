# ------------------------------------------------------------------------------
# Prodify: Product assistant in coding
# (C) TRUKHIN IURII, 2024 yuri@trukhin.com
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
import threading
import queue
import uuid
import shutil  # for removing incomplete folders if interrupted
import getpass  # for prompting API key

try:
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import PathCompleter
except ImportError:
    print("Prodify: Product assistant in coding")
    print("(C) TRUKHIN IURII, 2024 yuri@trukhin.com")
    print("prompt_toolkit is missing. Please install it: pip install prompt_toolkit")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Prodify: Product assistant in coding")
    print("(C) TRUKHIN IURII, 2024 yuri@trukhin.com")
    print("tqdm is missing. Please install it: pip install tqdm")
    sys.exit(1)

try:
    from rich.console import Console
except ImportError:
    print("Prodify: Product assistant in coding")
    print("(C) TRUKHIN IURII, 2024 yuri@trukhin.com")
    print("rich is missing. Please install it: pip install rich")
    sys.exit(1)

# Attempting to import RateLimitError
try:
    from openai.error import RateLimitError
except ImportError:
    # Fallback for older openai versions
    class RateLimitError(Exception):
        pass

# Attempting to import LangChain items
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai.embeddings import OpenAIEmbeddings
    from langchain_chroma import Chroma
except ImportError:
    print("Prodify: Product assistant in coding")
    print("(C) TRUKHIN IURII, 2024 yuri@trukhin.com")
    print("Falling back to langchain_community... Please consider installing langchain_openai and langchain_chroma.")
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma

# ------------------------------------------------------------------------------
# Prodify: Product assistant in coding
# (C) TRUKHIN IURII, 2024 yuri@trukhin.com
# This script indexes a project's codebase into a new Chroma vector store.
# The resulting index subfolder is stored under .chromadb/<ProjectName_DateTime_GUID>.
# If Ctrl+C is pressed before completion, the partial folder is removed.
# ------------------------------------------------------------------------------

console = Console()

# ------------------------------------------------------------------------------
# GLOBAL SETTINGS
# ------------------------------------------------------------------------------
NUM_WORKERS = 8
QUEUE_SIZE = 1000
BATCH_SIZE = 100
MAX_RETRIES = 5
INITIAL_DELAY = 1.0
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

global_lock = threading.Lock()

def is_probably_text_file(filepath, max_bytes=4096, text_ratio=0.8):
    """
    Checks if a file is likely to contain text based on its initial bytes.
    If the ratio of printable characters is >= text_ratio, treat it as text.
    """
    try:
        with open(filepath, "rb") as f:
            data = f.read(max_bytes)
    except Exception:
        return False

    if not data:
        return True

    text_chars = b"\n\r\t\b"
    printable = sum(1 for byte in data if (32 <= byte <= 126) or (byte in text_chars))
    ratio = printable / len(data)
    return ratio >= text_ratio


def file_extension_filter():
    """
    Returns two sets for quick extension-based checks:
    known_ext (extensions with dots),
    known_no_dot (file names like 'Dockerfile' that have no extension).
    """
    file_extensions = [
        ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx",
        ".m", ".mm", ".swift",
        ".java", ".scala", ".kt", ".kts",
        ".cs",
        ".go",
        ".py",
        ".rb",
        ".php",
        ".js", ".jsx", ".ts", ".tsx",
        ".vue",
        ".sh", ".bash", ".zsh", ".ksh",
        ".lua",
        ".rs",
        ".erl", ".hrl", ".ex", ".exs",
        ".pl", ".pm",
        ".r",
        ".jl",
        ".dart",
        ".f", ".f90", ".for",
        ".coffee",
        ".nim",
        ".clj", ".cljs",
        ".vb", ".vbs", ".bas",
        ".html", ".htm", ".css", ".scss", ".sass",
        ".xml", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
        ".md", ".markdown", ".rst", ".txt",
        ".sql", ".csv", ".tsv",
        ".ps1", ".bat",
        ".ipynb",
        "Dockerfile", ".dockerignore",
        "Makefile", "CMakeLists.txt", ".mk",
    ]
    known_ext = {ext.lower() for ext in file_extensions if ext.startswith(".")}
    known_no_dot = {ext for ext in file_extensions if not ext.startswith(".")}
    return known_ext, known_no_dot


def is_code_file(file_path, known_extensions, known_no_dot):
    """
    Checks if a file path should be considered code or text.
    If it has a known extension/name, we assume code/text. Otherwise, do a heuristic check.
    """
    if os.path.isdir(file_path):
        return False
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    base_name = os.path.basename(file_path)
    if ext in known_extensions or base_name in known_no_dot:
        return True
    return is_probably_text_file(file_path)


class Worker(threading.Thread):
    """
    A worker thread that processes file paths from a queue, reads file contents,
    splits them into chunks, and adds them to Chroma in batches under a global lock.
    """
    def __init__(self, task_queue, db, splitter, known_ext, known_no_dot, pbar):
        super().__init__()
        self.task_queue = task_queue
        self.db = db
        self.splitter = splitter
        self.known_ext = known_ext
        self.known_no_dot = known_no_dot
        self.pbar = pbar
        self.batch_chunks = []
        self.batch_size = BATCH_SIZE
        self.total_chunks = 0

    def run(self):
        while True:
            file_path = self.task_queue.get()
            if file_path is None:
                # Flush any remaining chunks
                self._flush_batch()
                self.task_queue.task_done()
                break

            self.process_file(file_path)
            self.task_queue.task_done()
            self.pbar.update(1)

    def process_file(self, file_path):
        if not is_code_file(file_path, self.known_ext, self.known_no_dot):
            return

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            console.print("[bold]Prodify: Product assistant in coding[/bold]", style="grey50")
            console.print(f"(C) TRUKHIN IURII, 2024 yuri@trukhin.com\nError reading {file_path}: {e}", style="red")
            return

        chunks = self.splitter.split_text(text)
        for chunk in chunks:
            self.batch_chunks.append((chunk, {"source": file_path}))
            if len(self.batch_chunks) >= self.batch_size:
                self._flush_batch()

    def _flush_batch(self):
        if not self.batch_chunks:
            return

        to_add_texts = [c[0] for c in self.batch_chunks]
        to_add_metas = [c[1] for c in self.batch_chunks]

        delay = INITIAL_DELAY
        for attempt in range(MAX_RETRIES):
            try:
                with global_lock:
                    self.db.add_texts(to_add_texts, metadatas=to_add_metas)
                break
            except RateLimitError:
                if attempt < MAX_RETRIES - 1:
                    console.print("[bold]Prodify: Product assistant in coding[/bold]", style="grey50")
                    console.print(
                        f"(C) TRUKHIN IURII, 2024 yuri@trukhin.com\n"
                        f"Rate limit reached, retrying in {delay:.1f}s "
                        f"(attempt {attempt+1}/{MAX_RETRIES})...",
                        style="yellow"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    console.print("[bold]Prodify: Product assistant in coding[/bold]", style="grey50")
                    console.print(
                        f"(C) TRUKHIN IURII, 2024 yuri@trukhin.com\n"
                        "RateLimitError: max retries exceeded, skipping this batch.",
                        style="red"
                    )
                    break

        self.total_chunks += len(self.batch_chunks)
        self.batch_chunks.clear()


def main():
    console.clear()
    console.print("[bold]Prodify: Product assistant in coding[/bold]", style="green")
    console.print("(C) TRUKHIN IURII, 2024 yuri@trukhin.com", style="dim")

    # Check or request OPENAI_API_KEY
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        console.print("No OPENAI_API_KEY found in environment.", style="bold red")
        key = getpass.getpass("Please enter your OpenAI API key (sk-...): ").strip()
        if not key:
            console.print("No API key provided. Exiting.", style="bold red")
            sys.exit(1)
        os.environ["OPENAI_API_KEY"] = key

    indexing_in_progress = True
    persist_dir = None

    try:
        # Optionally read openai.api_key
        from openai import api_key
    except ImportError:
        pass

    try:
        path_completer = PathCompleter(expanduser=True)
        project_path_input = prompt(
            "Enter the path to your project directory: ",
            completer=path_completer
        ).strip()

        project_path = os.path.abspath(os.path.expanduser(project_path_input))
        project_name = os.path.basename(project_path) or "unnamed_project"

        date_str = time.strftime("%Y%m%d_%H%M%S")
        guid_str = str(uuid.uuid4())[:8]
        index_name = f"{project_name}_{date_str}_{guid_str}"

        persist_dir = os.path.join(".chromadb", index_name)
        os.makedirs(persist_dir, exist_ok=True)

        console.print(
            f"Selected project path: [light_sky_blue1]{project_path}[/light_sky_blue1]\n"
            f"Creating new index folder: [spring_green2]{persist_dir}[/spring_green2]\n",
            style="bold"
        )

        embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
        db = Chroma(
            collection_name=index_name,
            persist_directory=persist_dir,
            embedding_function=embeddings
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        known_ext, known_no_dot = file_extension_filter()

        console.print("Counting files...", style="bold")
        file_count = 0
        for _, _, files in os.walk(project_path):
            file_count += len(files)
        console.print(f"Found [light_yellow]{file_count}[/light_yellow] total files.\n", style="bold")

        task_queue = queue.Queue(maxsize=QUEUE_SIZE)

        console.print("Starting worker threads...\n", style="bold")

        pbar = tqdm(total=file_count, desc="Processing files", unit="file")
        workers = []
        for _ in range(NUM_WORKERS):
            w = Worker(task_queue, db, splitter, known_ext, known_no_dot, pbar)
            w.start()
            workers.append(w)

        # Enqueue file paths
        for root, _, files in os.walk(project_path):
            for filename in files:
                full_path = os.path.join(root, filename)
                task_queue.put(full_path)

        # Signal end
        for _ in range(NUM_WORKERS):
            task_queue.put(None)

        task_queue.join()

        for w in workers:
            w.join()

        pbar.close()

        console.print(
            f"Indexing complete!\n"
            f"New index: [khaki1]{index_name}[/khaki1]\n"
            f"Stored in: [spring_green2]{persist_dir}[/spring_green2]",
            style="bold"
        )
        indexing_in_progress = False

    except KeyboardInterrupt:
        console.print("\nIndexing interrupted by user.", style="bold red")
        if indexing_in_progress and persist_dir and os.path.isdir(persist_dir):
            console.print(f"Removing incomplete index folder: {persist_dir}", style="bold red")
            shutil.rmtree(persist_dir, ignore_errors=True)
        sys.exit(1)

    except Exception as e:
        console.print(f"\nAn error occurred: {e}", style="bold red")
        if indexing_in_progress and persist_dir and os.path.isdir(persist_dir):
            console.print(f"Removing incomplete index folder: {persist_dir}", style="bold red")
            shutil.rmtree(persist_dir, ignore_errors=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


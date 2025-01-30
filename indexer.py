#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Prodify: Product assistant in coding
# (C) TRUKHIN IURII, 2024 yuri@trukhin.com
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
import time
import threading
import queue
import uuid
import shutil
import getpass
import fnmatch
import hashlib

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

# Attempt to import RateLimitError from openai
try:
    from openai.error import RateLimitError
except ImportError:
    class RateLimitError(Exception):
        """Fallback if openai.error.RateLimitError is not available."""
        pass

# Attempt to import LangChain items
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai.embeddings import OpenAIEmbeddings
    from langchain_chroma import Chroma
except ImportError:
    print("Prodify: Product assistant in coding")
    print("(C) TRUKHIN IURII, 2024 yuri@trukhin.com")
    print("Falling back to langchain_community. Consider installing langchain_openai and langchain_chroma.")
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma

console = Console()

# ------------------------------------------------------------------------------
# GLOBAL SETTINGS
# ------------------------------------------------------------------------------
NUM_WORKERS = 4
QUEUE_SIZE = 1000
BATCH_SIZE = 200
MAX_RETRIES = 5
INITIAL_DELAY = 2.0

CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200
MAX_FILE_SIZE_MB = 10

global_lock = threading.Lock()

# ------------------------------------------------------------------------------
# IGNORE LOGIC
# ------------------------------------------------------------------------------
def load_indexer_ignore_patterns(project_path: str):
    """
    Load patterns from the .indexerIgnore file if it exists in the project root.
    Lines starting with '!' are exceptions (do not ignore),
    and lines without '!' are patterns to ignore.
    """
    ignore_file_path = os.path.join(project_path, ".indexerIgnore")
    if not os.path.isfile(ignore_file_path):
        return [], []

    ignore_patterns = []
    ignore_exceptions = []

    with open(ignore_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('!'):
                pattern = line[1:].strip()
                if pattern:
                    ignore_exceptions.append(pattern)
            else:
                ignore_patterns.append(line)

    return ignore_patterns, ignore_exceptions


def is_ignored(rel_path: str, ignore_patterns, ignore_exceptions, debug=False) -> bool:
    """
    Determine if a file or directory (rel_path) should be ignored
    based on the .indexerIgnore patterns.
    """
    # First, check exceptions (if it matches an exception, do not ignore).
    for patt in ignore_exceptions:
        if fnmatch.fnmatch(rel_path, patt):
            if debug:
                print(f"EXCEPTION match: {patt} => do NOT ignore {rel_path}")
            return False

    # If it matches any ignore pattern, ignore it.
    for patt in ignore_patterns:
        if fnmatch.fnmatch(rel_path, patt):
            if debug:
                print(f"IGNORE match: {patt} => ignoring {rel_path}")
            return True

    return False

# ------------------------------------------------------------------------------
# FILE TYPE DETECTION
# ------------------------------------------------------------------------------
def is_probably_text_file(filepath, max_bytes=4096, text_ratio=0.8):
    """
    Basic heuristic to determine if a file is text:
    Read up to max_bytes and check the fraction of printable characters.
    If ratio exceeds text_ratio, treat it as a text file.
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
    List of commonly used file extensions for code and config files.
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
    Check if a file is potentially source code or textual content.
    Skips large files, and also does a text heuristic.
    """
    if os.path.isdir(file_path):
        return False

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    base_name = os.path.basename(file_path)
    if ext in known_extensions or base_name in known_no_dot:
        return True

    return is_probably_text_file(file_path)

# ------------------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------------------
def sha1_of_text(text: str) -> str:
    """
    Compute the SHA-1 hash of the given text (UTF-8, ignoring errors).
    """
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def get_existing_chunks_for_file(db, file_path: str):
    """
    Retrieve all chunks that belong to this file (metadata.source == file_path)
    and return a dict {chunk_index: (chunkhash, doc_id)} for comparison during updates.
    """
    results = db.get(where={"source": file_path})
    chunk_map = {}
    for doc_id, metadata in zip(results["ids"], results["metadatas"]):
        if not metadata:
            continue
        idx = metadata.get("chunk_index", None)
        chash = metadata.get("chunkhash", None)
        if idx is None or chash is None:
            continue
        chunk_map[int(idx)] = (chash, doc_id)
    return chunk_map


def delete_single_chunk(db, doc_id):
    """
    Delete a single document (chunk) from Chroma by doc_id.
    """
    db.delete(ids=[doc_id])


# ------------------------------------------------------------------------------
# Worker
# ------------------------------------------------------------------------------
class Worker(threading.Thread):
    """
    A worker thread that processes file chunks for partial updates:
      - If chunk is unchanged, skip.
      - If changed, re-embed and update.
      - If there are extra chunks in the old version, remove them.
    """
    def __init__(self, task_queue, db, pbar):
        super().__init__()
        self.task_queue = task_queue
        self.db = db
        self.pbar = pbar
        self.batch_chunks = []
        self.batch_size = BATCH_SIZE
        self.total_chunks = 0

    def run(self):
        while True:
            item = self.task_queue.get()
            if item is None:
                self._flush_batch()
                self.task_queue.task_done()
                break

            file_path, splitted_chunks, old_chunk_map = item
            self.process_file_chunks(file_path, splitted_chunks, old_chunk_map)
            self.task_queue.task_done()
            self.pbar.update(1)

    def process_file_chunks(self, file_path, splitted_chunks, old_chunk_map):
        new_indexes = set()

        for chunk_index, chunk_text in splitted_chunks:
            new_indexes.add(chunk_index)
            new_hash = sha1_of_text(chunk_text)

            # If old chunk exists and hash is the same, skip re-embedding
            if chunk_index in old_chunk_map:
                old_hash, old_doc_id = old_chunk_map[chunk_index]
                if old_hash == new_hash:
                    continue
                else:
                    # Hash changed => remove the old chunk
                    with global_lock:
                        delete_single_chunk(self.db, old_doc_id)

            # Add the new chunk to the batch
            metadata = {
                "source": file_path,
                "chunk_index": chunk_index,
                "chunkhash": new_hash
            }
            self.batch_chunks.append((chunk_text, metadata))
            if len(self.batch_chunks) >= self.batch_size:
                self._flush_batch()

        # Remove chunks that no longer exist in the new text
        old_indexes = set(old_chunk_map.keys())
        removed_indexes = old_indexes - new_indexes
        for idx in removed_indexes:
            _, old_doc_id = old_chunk_map[idx]
            with global_lock:
                delete_single_chunk(self.db, old_doc_id)

    def _flush_batch(self):
        if not self.batch_chunks:
            return

        texts = [c[0] for c in self.batch_chunks]
        metas = [c[1] for c in self.batch_chunks]

        delay = INITIAL_DELAY
        for attempt in range(MAX_RETRIES):
            try:
                with global_lock:
                    self.db.add_texts(texts, metadatas=metas)
                break
            except RateLimitError:
                if attempt < MAX_RETRIES - 1:
                    console.print("[bold]Prodify: Product assistant in coding[/bold]", style="grey50")
                    console.print(
                        f"Rate limit reached. Retrying in {delay:.1f}s "
                        f"(attempt {attempt+1}/{MAX_RETRIES})...",
                        style="yellow"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    console.print("[bold]Prodify: Product assistant in coding[/bold]", style="grey50")
                    console.print(
                        "RateLimitError: max retries exceeded, skipping this batch.",
                        style="red"
                    )
                    break

        self.total_chunks += len(self.batch_chunks)
        self.batch_chunks.clear()


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    console.clear()
    console.print("[bold]Prodify: Product assistant in coding[/bold]", style="green")
    console.print("(C) TRUKHIN IURII, 2024 yuri@trukhin.com", style="dim")

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
        console.print(
            "[bold]This script indexes your project into a Chroma DB with partial updates.[/bold]\n"
            "Unchanged chunks are skipped, while new or modified chunks are re-embedded.\n"
            "After indexing, you can run [italic]ask.py[/italic] to query your code.\n",
            style="cyan"
        )

        path_completer = PathCompleter(expanduser=True)
        project_path_input = prompt(
            "Enter the path to your project directory: ",
            completer=path_completer
        ).strip()

        project_path = os.path.abspath(os.path.expanduser(project_path_input))
        project_name = os.path.basename(project_path) or "unnamed_project"

        ignore_patterns, ignore_exceptions = load_indexer_ignore_patterns(project_path)
        print("=== .indexerIgnore loaded patterns ===")
        print("  ignore_patterns:", ignore_patterns)
        print("  ignore_exceptions:", ignore_exceptions)
        print("======================================")

        # Use ~/.chromadb/PROJECT_NAME to store the index
        base_dir = os.path.join(os.path.expanduser("~"), ".chromadb")
        os.makedirs(base_dir, exist_ok=True)

        index_name = project_name
        persist_dir = os.path.join(base_dir, index_name)
        os.makedirs(persist_dir, exist_ok=True)

        console.print(
            f"Selected project: [light_sky_blue1]{project_path}[/light_sky_blue1]\n"
            f"Index folder: [spring_green2]{persist_dir}[/spring_green2]\n",
            style="bold"
        )

        # Create/load the Chroma DB
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=os.environ["OPENAI_API_KEY"],
        )
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

        console.print("Collecting files for partial indexing...", style="bold")
        file_paths = []
        total_files_to_process = 0

        for root, dirs, files in os.walk(project_path):
            relative_root = os.path.relpath(root, project_path).replace("\\", "/")
            if relative_root == ".":
                relative_root = ""

            # Filter out ignored subdirectories
            for d in list(dirs):
                d_path = os.path.join(relative_root, d).replace("\\", "/")
                if is_ignored(d_path + "/", ignore_patterns, ignore_exceptions):
                    console.print(f"Skipping directory: {d_path + '/'}", style="dim")
                    dirs.remove(d)

            # Filter out ignored files
            for filename in files:
                full_path = os.path.join(root, filename)
                rel_file_path = os.path.join(relative_root, filename).replace("\\", "/")
                if is_ignored(rel_file_path, ignore_patterns, ignore_exceptions):
                    continue

                if not is_code_file(full_path, known_ext, known_no_dot):
                    continue

                file_paths.append(full_path)
                total_files_to_process += 1

        if total_files_to_process == 0:
            console.print("No suitable files found for indexing. Exiting.", style="bold red")
            indexing_in_progress = False
            return

        console.print(
            f"Found [light_yellow]{total_files_to_process}[/light_yellow] files to process.\n",
            style="bold"
        )

        # Prepare tasks for each file (split into chunks, fetch old chunks)
        tasks = []
        console.print("Preparing chunk lists for partial updates...", style="bold")
        for fpath in tqdm(file_paths, desc="Preparing", unit="file"):
            old_chunk_map = get_existing_chunks_for_file(db, fpath)

            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception as e:
                console.print(f"Error reading {fpath}: {e}", style="red")
                continue

            splitted = []
            chunks = splitter.split_text(text)
            for i, chunk_text in enumerate(chunks):
                splitted.append((i, chunk_text))

            tasks.append((fpath, splitted, old_chunk_map))

        console.print("Starting file indexing in multiple threads...\n", style="bold")

        task_queue = queue.Queue(maxsize=QUEUE_SIZE)
        pbar = tqdm(total=len(tasks), desc="Indexing", unit="file")
        workers = []

        for _ in range(NUM_WORKERS):
            w = Worker(task_queue, db, pbar)
            w.start()
            workers.append(w)

        for item in tasks:
            task_queue.put(item)

        # End signal
        for _ in range(NUM_WORKERS):
            task_queue.put(None)

        task_queue.join()
        for w in workers:
            w.join()
        pbar.close()

        console.print(
            f"Partial indexing complete!\n"
            f"Index name: [khaki1]{index_name}[/khaki1]\n"
            f"Location: [spring_green2]{persist_dir}[/spring_green2]\n",
            style="bold"
        )
        console.print("You can now run [italic]ask.py[/italic] to query your code.\n", style="dim")
        indexing_in_progress = False

    except KeyboardInterrupt:
        console.print("\nIndexing interrupted by user.", style="bold red")
        sys.exit(1)

    except Exception as e:
        console.print(f"\nAn error occurred: {e}", style="bold red")
        sys.exit(1)


if __name__ == "__main__":
    main()

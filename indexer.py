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
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
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
# GLOBAL SETTINGS: optimized for large repos
# ------------------------------------------------------------------------------

# Lower concurrency if hitting rate-limit or want to slow cost burn:
NUM_WORKERS = 4
QUEUE_SIZE = 1000

# Larger batch size reduces the overhead but watch out for memory usage:
BATCH_SIZE = 200

# Try fewer retries; or keep the same if you do see a lot of rate limits:
MAX_RETRIES = 5
INITIAL_DELAY = 2.0  # start with 2 seconds if 429

# Larger chunk size => fewer embedding calls => cheaper and faster overall
# (but each chunk is bigger and possibly less granular).
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200

# For very large files, skip them entirely:
MAX_FILE_SIZE_MB = 10  # skip files > 10 MB

global_lock = threading.Lock()


def load_indexer_ignore_patterns(project_path: str):
    """
    Loads patterns from .indexerIgnore (if present).
    Returns (ignore_patterns, ignore_exceptions).
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
                # comment or empty
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
    Checks if rel_path should be ignored, using ignore_patterns and ignore_exceptions.
    """
    # Exceptions override everything
    for patt in ignore_exceptions:
        if fnmatch.fnmatch(rel_path, patt):
            if debug:
                print(f"EXCEPTION match: {patt} => do NOT ignore {rel_path}")
            return False

    # If any ignore pattern matches => ignore
    for patt in ignore_patterns:
        if fnmatch.fnmatch(rel_path, patt):
            if debug:
                print(f"IGNORE match: {patt} => ignoring {rel_path}")
            return True

    return False


def is_probably_text_file(filepath, max_bytes=4096, text_ratio=0.8):
    """
    Reads up to max_bytes from a file to decide if it's text or binary.
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
    Returns (known_ext, known_no_dot) for quick extension checks.
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
    Decides if the file is code/text or not. If large or not recognized as text => skip.
    """
    if os.path.isdir(file_path):
        return False

    # If file is bigger than MAX_FILE_SIZE_MB => skip
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        # skip big files
        return False

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    base_name = os.path.basename(file_path)
    if ext in known_extensions or base_name in known_no_dot:
        return True

    return is_probably_text_file(file_path)


class Worker(threading.Thread):
    """
    Worker thread that reads from a queue of file paths, splits them, and adds them to the DB.
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
            console.print(f"Error reading {file_path}: {e}", style="red")
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
            "[bold]This utility indexes your project into a Chroma DB.[/bold]\n"
            "It is optimized for large repositories to reduce cost/time.\n"
            "After indexing, you can run ask.py to query the code.\n",
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

        # If you want a cheaper or self-hosted embeddings approach,
        # replace OpenAIEmbeddings with a local or smaller model.
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

        console.print("Counting files to index...", style="bold")
        file_count = 0
        file_paths = []

        # Collect all files, skipping .indexerIgnore patterns or large binary files
        for root, dirs, files in os.walk(project_path):
            relative_root = os.path.relpath(root, project_path).replace("\\", "/")
            if relative_root == ".":
                relative_root = ""

            # If you want to skip known large folders (like .idea, .git, build, out),
            # you can remove them here or rely on .indexerIgnore
            for d in list(dirs):
                d_path = os.path.join(relative_root, d).replace("\\", "/")
                if is_ignored(d_path + "/", ignore_patterns, ignore_exceptions):
                    console.print(f"Skipping directory: {d_path + '/'}", style="dim")
                    dirs.remove(d)

            for filename in files:
                full_path = os.path.join(root, filename)
                rel_file_path = os.path.join(relative_root, filename).replace("\\", "/")
                if is_ignored(rel_file_path, ignore_patterns, ignore_exceptions):
                    # console.print(f"Skipping file: {rel_file_path}", style="dim")
                    continue

                file_paths.append(full_path)
                file_count += 1

        console.print(
            f"Found [light_yellow]{file_count}[/light_yellow] files to index.\n",
            style="bold"
        )

        task_queue = queue.Queue(maxsize=QUEUE_SIZE)

        console.print("Starting worker threads for indexing...\n", style="bold")

        pbar = tqdm(total=file_count, desc="Indexing files", unit="file")
        workers = []
        for _ in range(NUM_WORKERS):
            w = Worker(task_queue, db, splitter, known_ext, known_no_dot, pbar)
            w.start()
            workers.append(w)

        # Enqueue file paths
        for fpath in file_paths:
            task_queue.put(fpath)

        # Signal the end
        for _ in range(NUM_WORKERS):
            task_queue.put(None)

        task_queue.join()
        for w in workers:
            w.join()
        pbar.close()

        console.print(
            f"Indexing complete!\n"
            f"New index: [khaki1]{index_name}[/khaki1]\n"
            f"Stored in: [spring_green2]{persist_dir}[/spring_green2]\n",
            style="bold"
        )
        console.print("You can now run [italic]ask.py[/italic] for Q&A on this index.\n", style="dim")
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

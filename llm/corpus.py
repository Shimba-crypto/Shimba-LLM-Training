"""
corpus.py — Corpus loading for .txt / .json / .jsonl training data.

All loaders return plain strings, so the tokenizer and trainer never care
which format the text came from. Documents from every file are joined with
DOC_SEPARATOR before training.

Supported inputs
----------------
.txt   Raw text, used as-is.
.json  A JSON array of documents, or a single JSON object. Each document is
       usually an object with a text field (see `extract_doc_text`); plain
       strings in the array are used directly.
.jsonl One JSON value per line. Lines that are not valid JSON are kept as
       plain text lines instead of failing the run.

`--text-field` (comma-separated) names the object field(s) to read first;
when omitted or when none of the named fields is present, the loader falls
back to auto-detection.
"""

import fnmatch
import glob
import json
import os

DOC_SEPARATOR = "\n\n" + "=" * 60 + "\n\n"

SUPPORTED_EXTENSIONS = (".txt", ".json", ".jsonl")
DEFAULT_PATTERN = "*.txt,*.json,*.jsonl"

_MESSAGE_LIST_KEYS = ("messages", "conversation", "conversations", "turns")
_CONTENT_KEYS = ("content", "text", "value", "utterance")
_ROLE_KEYS = ("role", "from", "speaker", "author")
_OUTPUT_KEYS = ("completion", "output", "answer", "response")


def split_patterns(pattern: str) -> list:
    """Split a comma-separated glob pattern into a clean list."""
    return [p.strip() for p in (pattern or "").split(",") if p.strip()]


def matches_any(filename: str, patterns: list) -> bool:
    """True when the filename matches any of the glob patterns."""
    lowered = filename.lower()
    return any(fnmatch.fnmatch(lowered, p.lower()) for p in patterns)


def collect_files(data_path: str, pattern: str, recurse: bool) -> list:
    """Sorted list of files under a directory matching the pattern(s)."""
    patterns = split_patterns(pattern)
    if recurse:
        matches = []
        for root, dirs, files in os.walk(data_path):
            dirs.sort()
            for fname in sorted(files):
                if matches_any(fname, patterns):
                    matches.append(os.path.join(root, fname))
        return matches
    found = []
    for pat in patterns:
        found.extend(glob.glob(os.path.join(data_path, pat)))
    return sorted(set(found))


def detect_format(path: str, data_format: str = "auto") -> str:
    """Resolve 'auto' to txt/json/jsonl from the file extension."""
    if data_format != "auto":
        return data_format
    ext = os.path.splitext(path)[1].lower()
    if ext in (".json", ".jsonl"):
        return ext[1:]
    return "txt"


def parse_text_field(text_field) -> list:
    """Normalise --text-field into a list of field names."""
    if not text_field:
        return []
    if isinstance(text_field, str):
        return [f.strip() for f in text_field.split(",") if f.strip()]
    return list(text_field)


def load_file_docs(path: str, text_field=None, data_format: str = "auto") -> list:
    """
    Read one file and return a list of document strings.

    Empty documents are dropped. Raises FileNotFoundError for a missing path
    and ValueError for a .json file that is not valid JSON.
    """
    fmt = detect_format(path, data_format)
    if fmt == "json":
        return _load_json_docs(path, text_field)
    if fmt == "jsonl":
        return _load_jsonl_docs(path, text_field)
    text = _read_text_file(path)
    return [text] if text.strip() else []


def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def _read_json_file(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return json.load(f)


def _load_json_docs(path: str, text_field=None) -> list:
    try:
        payload = _read_json_file(path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{path}': {exc}")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Cannot decode '{path}': {exc}")

    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        single = extract_doc_text(payload, text_field)
        if single:
            return [single]
        # Wrapper object such as {"data": [...]} — use the first list value
        # that yields documents.
        for value in payload.values():
            if isinstance(value, list):
                docs = _docs_from_items(value, text_field)
                if docs:
                    return docs
        raise ValueError(
            f"No training text found in '{path}'. Expected a 'text'/'content' "
            f"field, a messages list, or a list of documents."
        )
    else:
        raise ValueError(
            f"Top-level JSON in '{path}' must be an array or an object."
        )
    return _docs_from_items(items, text_field)


def _load_jsonl_docs(path: str, text_field=None) -> list:
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            lines = f.read().splitlines()

    docs = []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Not JSON — keep the raw line as plain text.
            docs.append(line)
            continue
        if isinstance(obj, str):
            if obj.strip():
                docs.append(obj)
            continue
        if isinstance(obj, dict):
            text = extract_doc_text(obj, text_field)
            if text:
                docs.append(text)
            continue
        raise ValueError(
            f"Unsupported JSON value on line {lineno} of '{path}'. "
            f"Expected an object or a string."
        )
    return docs


def _docs_from_items(items: list, text_field=None) -> list:
    docs = []
    for item in items:
        if isinstance(item, str):
            if item.strip():
                docs.append(item)
        elif isinstance(item, dict):
            text = extract_doc_text(item, text_field)
            if text:
                docs.append(text)
        # Other JSON scalars carry no training text and are skipped.
    return docs


def extract_doc_text(obj: dict, text_field=None):
    """
    Extract one training document from a JSON object.

    Returns None when the object holds no usable text. Field priority:
      1. --text-field name(s), in order
      2. "text", then "content"
      3. instruction / prompt / question templates
      4. message / conversation lists
      5. remaining string values joined together
    """
    fields = parse_text_field(text_field)
    if fields and any(name in obj for name in fields):
        # A named field is present, so honour it strictly: the first
        # non-blank named value wins, otherwise the document is skipped.
        for name in fields:
            value = obj.get(name)
            if isinstance(value, str) and value.strip():
                return value
        return None

    for key in ("text", "content"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value

    templated = _template_doc(obj)
    if templated:
        return templated

    for key in _MESSAGE_LIST_KEYS:
        value = obj.get(key)
        if isinstance(value, list) and value:
            text = _format_messages(value)
            if text:
                return text

    strings = [v for k, v in sorted(obj.items())
               if isinstance(v, str) and v.strip()]
    if strings:
        return "\n".join(strings)
    return None


def _template_doc(obj: dict):
    """Instruction / prompt / question templates. Returns None if no match."""
    if isinstance(obj.get("instruction"), str) and obj["instruction"].strip():
        parts = [f"Instruction: {obj['instruction'].strip()}"]
        for key in ("input", "context"):
            if isinstance(obj.get(key), str) and obj[key].strip():
                parts.append(f"Input: {obj[key].strip()}")
        for key in _OUTPUT_KEYS:
            if isinstance(obj.get(key), str) and obj[key].strip():
                parts.append(f"Response: {obj[key].strip()}")
                break
        return "\n".join(parts)

    if isinstance(obj.get("prompt"), str) and obj["prompt"].strip():
        parts = [obj["prompt"]]
        for key in _OUTPUT_KEYS:
            if isinstance(obj.get(key), str) and obj[key].strip():
                parts.append(obj[key])
                break
        text = "\n".join(parts)
        return text if text.strip() else None

    for qkey, akey in (("question", "answer"), ("input", "output")):
        q, a = obj.get(qkey), obj.get(akey)
        if isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip():
            return f"{q.strip()}\n{a.strip()}"

    return None


def _format_messages(value: list):
    """Render a message/turn list as 'Role: content' lines."""
    lines = []
    for turn in value:
        if isinstance(turn, str):
            if turn.strip():
                lines.append(turn.strip())
        elif isinstance(turn, dict):
            content = None
            for key in _CONTENT_KEYS:
                if isinstance(turn.get(key), str) and turn[key].strip():
                    content = turn[key].strip()
                    break
            if content is None:
                continue
            role = None
            for key in _ROLE_KEYS:
                if isinstance(turn.get(key), str) and turn[key].strip():
                    role = turn[key].strip().capitalize()
                    break
            lines.append(f"{role}: {content}" if role else content)
    return "\n".join(lines) if lines else None


def collect_docs(data_path: str, pattern: str = DEFAULT_PATTERN,
                 recurse: bool = True, text_field=None,
                 data_format: str = "auto") -> list:
    """
    Collect document strings from a file or a directory of files.

    Exits via SystemExit when the path does not exist or yields no documents,
    matching the previous behaviour of the per-script loaders.
    """
    import sys

    if os.path.isfile(data_path):
        try:
            docs = load_file_docs(data_path, text_field, data_format)
        except ValueError as exc:
            print(f"[error] {exc}")
            sys.exit(1)
        if not docs:
            print(f"[error] No training documents found in '{data_path}'")
            sys.exit(1)
        print(f"[corpus] {len(docs)} document(s) from '{data_path}'")
        return docs

    if os.path.isdir(data_path):
        files = collect_files(data_path, pattern, recurse)
        if not files:
            print(f"[error] No files matching '{pattern}' found in '{data_path}'")
            sys.exit(1)
        print(f"[corpus] found {len(files)} file(s) in '{data_path}'")
        docs = []
        skipped_files = 0
        for i, fpath in enumerate(files, 1):
            try:
                file_docs = load_file_docs(fpath, text_field, data_format)
            except ValueError as exc:
                print(f"[error] {exc}")
                sys.exit(1)
            if file_docs:
                docs.extend(file_docs)
            else:
                skipped_files += 1
            if i % 20 == 0 or i == len(files):
                print(f"  loaded {i}/{len(files)} files  ({len(docs)} docs so far)")
        if not docs:
            print(f"[error] No training documents found in '{data_path}'")
            sys.exit(1)
        if skipped_files:
            print(f"[corpus] skipped {skipped_files} file(s) with no documents")
        print(f"[corpus] {len(docs)} document(s) across {len(files)} file(s)")
        return docs

    print(f"[error] --data path does not exist: '{data_path}'")
    sys.exit(1)


def load_corpus_text(data_path: str, pattern: str = DEFAULT_PATTERN,
                     recurse: bool = True, text_field=None,
                     data_format: str = "auto") -> str:
    """Load and join every document into one training string."""
    docs = collect_docs(data_path, pattern, recurse, text_field, data_format)
    combined = DOC_SEPARATOR.join(docs)
    print(f"[corpus] total: {len(combined):,} characters across {len(docs)} document(s)")
    return combined

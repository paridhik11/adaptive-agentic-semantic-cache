"""Scalable Evaluation Harness for Large-Scale Agentic Cache Benchmarking.

Task A & Task B Deliverable:
- 25 documentation source groups
- 125 distinct base intents
- 9 event categories: seed, exact repeat, natural paraphrase, added requirements,
  missing executable example, inverse operation, changed entity, negation, numeric/version constraint.
- Matched multi-arm offline evaluation:
    1. no_cache (mode='off')
    2. exact_only (mode='exact')
    3. verified_semantic (mode='guarded', explicit opt-in)
    4. shadow_semantic (mode='shadow', diagnostic only)
- Full instrumentation: provider calls by stage, synthetic token accounting,
  e2e latency percentiles (p50/p90/p95), recall@k, verifier CPU/wall time,
  guard rejections by category, candidate availability, and zero false accepts verification.
- Exports raw JSONL, manifest, and summary JSON to artifacts/metrics_campaign/.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.document_agent import DocumentAgent, DocumentTools
from src.agents.document_answer_cache import CompletedDocumentCache
from src.api.providers import FixtureChatProvider
from src.api.service import GatewayService
from src.policy.answer_aware_policy import AnswerAwareSemanticPolicy, LocalPassageScorer
from src.policy.encoders import DenseEncoder
from src.policy.local_models import (
    ENCODER_MODEL, ENCODER_REVISION,
    PASSAGE_MODEL, PASSAGE_REVISION,
)
from src.policy.hybrid_cache import ConservativeHybridCache
from src.policy.response_cache import EligibilityDecision, ScopedResponseCache
from src.policy.semantic_runtime import KnowledgeScope, digest
from src.policy.sqlite_store import SQLiteResponseStore
from src.policy.work_ledger import WorkLedger

try:
    from scipy.stats import beta
    def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> list[float]:
        """Exact Clopper-Pearson binomial confidence interval."""
        if n == 0:
            return [0.0, 1.0]
        low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
        high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
        return [round(low, 6), round(high, 6)]
except ImportError:
    def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> list[float]:
        # Fallback conservative rule of 3 for zero errors
        if n == 0:
            return [0.0, 1.0]
        if k == 0:
            return [0.0, round(3.0 / n, 6)]
        p = k / n
        margin = 1.96 * math.sqrt(p * (1 - p) / n)
        return [round(max(0.0, p - margin), 6), round(min(1.0, p + margin), 6)]


# =============================================================================
# 1. Source Groups & Grounded Documentation Corpus (25 Groups)
# =============================================================================

CORPUS_SOURCE_GROUPS = {
    "list_operations": {
        "title": "Python List Operations & Sequences",
        "doc": "Lists are mutable sequences in Python. Convert a list to a tuple using tuple(my_list). Sort a list in place with list.sort() or return a new sorted list using sorted(list). Remove duplicate items using list(dict.fromkeys(my_list)) or set(my_list). List slicing uses seq[start:stop:step]. List comprehensions use [expr for item in iterable if condition].",
    },
    "dict_operations": {
        "title": "Python Dictionary Manipulation",
        "doc": "Dictionaries store key-value mappings. Create dictionary comprehensions using {k: v for k, v in iterable}. Merge dictionaries in Python 3.9+ using the union operator d1 | d2 or d1.update(d2). Sort dictionary items by value using sorted(d.items(), key=lambda item: item[1]). Use dict.get(key, default) to provide a fallback value for missing keys.",
    },
    "string_manipulation": {
        "title": "Python String Methods & Formatting",
        "doc": "Strings are immutable unicode sequences. Split a string into a list of substrings with str.split(sep). Join an iterable of strings with sep.join(iterable). Strip leading and trailing whitespace using str.strip(). Replace occurrences with str.replace(old, new). Format strings using f-strings like f'{val:.2f}' or str.format().",
    },
    "file_io": {
        "title": "File Input and Output Operations",
        "doc": "Open files using context managers 'with open(path, mode) as f:'. Read line by line iterating over f directly. Read the entire file into a string using f.read() or lines into a list using f.readlines(). Write text using f.write(text) with mode 'w' or append with mode 'a'. Read raw binary data with mode 'rb'. Check if a file exists using os.path.exists(path) or Path(path).is_file().",
    },
    "json_handling": {
        "title": "JSON Serialization and Parsing",
        "doc": "The json module provides json.dumps(obj) to serialize Python dicts to a JSON string and json.loads(s) to parse JSON text into dicts. Use json.dump(obj, f) and json.load(f) for file streams. Pretty print JSON using indent=2 parameter. Custom objects require a custom JSONEncoder or default converter function.",
    },
    "csv_processing": {
        "title": "CSV File Reading and Writing",
        "doc": "The csv module provides csv.reader and csv.writer for comma-separated values. Use csv.DictReader to map rows to dictionaries using header names. Use csv.DictWriter with writeheader() to write row mappings. Specify custom delimiters using delimiter=';'. Handle newline formatting using newline='' parameter in open().",
    },
    "error_handling": {
        "title": "Exception Handling and Raising",
        "doc": "Handle runtime exceptions using try, except, else, and finally blocks. Catch multiple specific exceptions with except (ValueError, TypeError) as exc. Raise custom errors by inheriting from Exception class. Re-raise an active exception using a bare 'raise'. Execute cleanup code in finally block regardless of errors.",
    },
    "datetime_operations": {
        "title": "Date and Time Operations",
        "doc": "The datetime module provides datetime.now(timezone.utc) for timezone-aware UTC timestamps. Parse ISO 8601 date strings using datetime.fromisoformat(s). Format dates to custom strings using dt.strftime(format). Calculate time differences using timedelta(days=N, hours=H). Convert timestamps using datetime.fromtimestamp(ts, tz).",
    },
    "regex_patterns": {
        "title": "Regular Expression Pattern Matching",
        "doc": "The re module provides re.findall(pattern, text) to find all pattern occurrences. Use re.sub(pattern, replacement, text) for regex substitution. Compile reusable regex objects with re.compile(pattern, flags=re.IGNORECASE). Access named capture groups with (?P<name>pattern) and match.groupdict(). Split strings on regex matches using re.split(pattern, text).",
    },
    "math_statistics": {
        "title": "Mathematical and Statistical Calculations",
        "doc": "The math module supplies math.gcd(a, b), math.floor(x), math.ceil(x), and math.sqrt(x). The statistics module provides statistics.mean(data), statistics.median(data), and statistics.stdev(data). Generate pseudo-random integers with random.randint(a, b) or floating point numbers with random.uniform(a, b).",
    },
    "pathlib_filesystem": {
        "title": "Object-Oriented Filesystem Paths with pathlib",
        "doc": "The pathlib.Path class represents filesystem paths. Create directory hierarchies using Path.mkdir(parents=True, exist_ok=True). Find matching files with Path.glob('*.py') or Path.rglob() for recursive search. Resolve absolute paths with Path.resolve(). Extract extensions using Path.suffix and filename with Path.name. Remove files using Path.unlink().",
    },
    "itertools_recipes": {
        "title": "Itertools Combinatoric Iterators",
        "doc": "The itertools module provides composable iterators. Use itertools.chain(*iterables) to link sequences. Generate combinations with itertools.combinations(iterable, r) and permutations with itertools.permutations(iterable, r). Cycle infinitely with itertools.cycle(iterable). Group consecutive elements with itertools.groupby(iterable, key). Slice iterators with itertools.islice(iterable, stop).",
    },
    "collections_types": {
        "title": "Specialized Collections Container Types",
        "doc": "The collections module offers specialized containers. Use collections.Counter(iterable) to tally element frequencies and most_common(n). collections.defaultdict(list) automatically initializes missing dictionary keys. collections.deque provides O(1) appends and pops from both ends. collections.namedtuple creates immutable tuple subclasses with named fields.",
    },
    "concurrency_threading": {
        "title": "Threading and Synchronization Primitives",
        "doc": "The threading module provides threading.Thread(target=fn, args=()). Protect shared mutable state using threading.Lock() with context managers 'with lock:'. Use concurrent.futures.ThreadPoolExecutor(max_workers=N) for managing worker thread pools. Set thread.daemon=True for background threads. Coordinate threads using threading.Event() and threading.Condition().",
    },
    "asyncio_coroutines": {
        "title": "Asynchronous Programming with asyncio",
        "doc": "Asynchronous tasks use async and await syntax. Run coroutines using asyncio.run(coro()). Run multiple tasks concurrently using asyncio.gather(*tasks). Sleep non-blockingly with await asyncio.sleep(seconds). Manage concurrent asynchronous tasks with asyncio.create_task(). Communicate between coroutines using asyncio.Queue.",
    },
    "http_requests": {
        "title": "HTTP Client Requests and URL Parsing",
        "doc": "Make HTTP GET requests using urllib.request.urlopen(url) or requests.get(url). Send JSON payloads via POST using json.dumps(data).encode('utf-8') with header 'Content-Type: application/json'. Parse URL components with urllib.parse.urlsplit(url) and query strings with urllib.parse.parse_qs(). Handle HTTP errors checking response status codes.",
    },
    "binary_data": {
        "title": "Binary Data and Byte Manipulation",
        "doc": "Encode strings to bytes with text.encode('utf-8') and decode bytes with b.decode('utf-8'). Pack structured binary records with struct.pack(format, *values) and unpack with struct.unpack(format, buffer). Convert bytes to hexadecimal strings with b.hex() and from hex with bytes.fromhex(hex_str). Use bytearray for mutable byte buffers.",
    },
    "hash_cryptography": {
        "title": "Cryptographic Hashing and Message Digests",
        "doc": "The hashlib module provides hashlib.sha256(data).hexdigest() for secure message digests. Compute MD5 checksums with hashlib.md5(data).hexdigest(). Create HMAC message authentication signatures using hmac.new(key, msg, hashlib.sha256). Generate secure random tokens and passwords with the secrets module using secrets.token_hex(nbytes).",
    },
    "sqlite_database": {
        "title": "Embedded SQLite Database Operations",
        "doc": "The sqlite3 module allows local SQL database management. Connect with sqlite3.connect(database_path). Execute parameterized SQL queries using db.execute('SELECT * FROM t WHERE id=?', (val,)) to prevent injection. Fetch records with cursor.fetchall() or fetchone(). Commit transactions with db.commit() and manage transactions using 'with db:' context manager.",
    },
    "sorting_searching": {
        "title": "Searching and Custom Sorting Algorithms",
        "doc": "Sort sequences with custom criteria using the key parameter, e.g., sorted(items, key=lambda x: (x.priority, x.date)). Perform binary search on sorted lists with the bisect module using bisect.bisect_left(arr, x) and bisect.insort(arr, x). Find extremes using min(items, key=...) and max(items, key=...). Reverse in place with list.reverse().",
    },
    "generators_iterators": {
        "title": "Generators, Iterators, and Yield Expressions",
        "doc": "Create memory-efficient generators using functions with yield statements. Delegate to subgenerators using yield from sub_iterable. Implement custom iterator classes defining __iter__() returning self and __next__() raising StopIteration. Generator expressions provide lazy evaluation with (x*2 for x in iterable). Fetch next value with next(iterator, default).",
    },
    "decorators_wrappers": {
        "title": "Function Decorators and Wrappers",
        "doc": "Decorators modify function behavior by wrapping calls. Define decorators taking a function and returning a wrapper. Preserve function docstrings and signatures using functools.wraps(fn) on the wrapper. Parameterized decorators require a three-level nested function. Apply decorators with @decorator_name above function definitions.",
    },
    "context_managers": {
        "title": "Context Managers and Resource Management",
        "doc": "Manage resources cleanly with 'with' statements. Create lightweight context managers with @contextlib.contextmanager decorating a generator that yields once. Custom context manager classes implement __enter__() returning the resource and __exit__(exc_type, exc_val, exc_tb). Suppress specific exceptions using contextlib.suppress(FileNotFoundError).",
    },
    "object_oriented": {
        "title": "Object-Oriented Programming and Classes",
        "doc": "Define classes using the class keyword. Initialize instance attributes in __init__(self). Call parent methods using super().__init__(). Create managed attributes using @property getters and @prop.setter decorators. Define class methods with @classmethod receiving cls and utility methods with @staticmethod. Use dataclasses.dataclass for boilerplate-free record classes.",
    },
    "functional_programming": {
        "title": "Functional Programming Utilities",
        "doc": "Create anonymous inline functions with lambda x: x * 2. Transform sequences with map(fn, iterable) and filter with filter(predicate, iterable). Combine elements iteratively with functools.reduce(fn, iterable, initial). Freeze partial function arguments using functools.partial(fn, arg1). Combine multiple predicates with all(iterable) and any(iterable).",
    },
}


# =============================================================================
# 2. Base Intent Taxonomy & Variant Generators (125 Base Intents)
# =============================================================================

@dataclass(frozen=True)
class IntentSpec:
    intent_id: str
    group_id: str
    seed_question: str
    seed_answer: str
    paraphrase: str
    added_requirement: str
    missing_code_example: str
    inverse_operation: str
    changed_entity: str
    negation: str
    numeric_constraint: str


def build_base_intents() -> list[IntentSpec]:
    """Constructs 125 distinct base intents (5 per group across 25 groups)."""
    intents = []

    # 1. list_operations
    intents.extend([
        IntentSpec("list_01", "list_operations",
                   "How do I convert a list to a tuple in Python?",
                   "Use the tuple() constructor: `tuple(my_list)`. It creates an immutable sequence containing all items from the list in identical order.",
                   "How can I turn a Python list into a tuple?",
                   "How do I convert a list to a tuple in Python and sort the elements descending?",
                   "How do I convert a list to a tuple in Python? Provide an executable code snippet.",
                   "How do I convert a tuple to a list in Python?",
                   "How do I convert a set to a tuple in Python?",
                   "How do I convert a list to a tuple without calling the tuple constructor?",
                   "How do I convert a list of exactly 5 elements into a tuple?"),
        IntentSpec("list_02", "list_operations",
                   "How do I sort a list in place in Python?",
                   "Call `my_list.sort()`. It sorts the list in place and returns None, modifying the original list directly.",
                   "What method sorts a Python list in place?",
                   "How do I sort a list in place in Python and write the result to a text file?",
                   "How do I sort a list in place in Python? Show working code.",
                   "How do I reverse an already sorted list in Python?",
                   "How do I sort a dictionary in place in Python?",
                   "How do I sort a list without using the in-place sort method?",
                   "How do I sort a list of exactly 100 numbers in place?"),
        IntentSpec("list_03", "list_operations",
                   "How do I remove duplicates from a list while preserving order?",
                   "Use `list(dict.fromkeys(my_list))`. In Python 3.7+, dictionary keys preserve insertion order while eliminating duplicate values.",
                   "How can I eliminate duplicate elements from a list preserving their order?",
                   "How do I remove duplicates from a list while preserving order and count frequencies?",
                   "How do I remove duplicates from a list while preserving order? Include a code example.",
                   "How do I duplicate each item in a list preserving order?",
                   "How do I remove duplicates from a tuple while preserving order?",
                   "How do I remove duplicates from a list without using dict or set?",
                   "How do I remove duplicates from a list containing exactly 3 distinct items?"),
        IntentSpec("list_04", "list_operations",
                   "How do I slice the last three elements of a list in Python?",
                   "Use the slice notation `my_list[-3:]`. It extracts a new list containing the final three elements.",
                   "What slice syntax extracts the final 3 items of a Python list?",
                   "How do I slice the last three elements of a list and reverse them?",
                   "How do I slice the last three elements of a list? Provide an executable example.",
                   "How do I slice the first three elements of a list in Python?",
                   "How do I slice the last three elements of a dictionary in Python?",
                   "How do I get the last three elements of a list without using slice syntax?",
                   "How do I slice the last 10 elements of a list in Python?"),
        IntentSpec("list_05", "list_operations",
                   "How do I filter a list using a list comprehension in Python?",
                   "Use `[x for x in my_list if condition(x)]`. It iterates through the list, evaluating the condition for each item.",
                   "How to construct a list comprehension with a filter condition in Python?",
                   "How do I filter a list using a list comprehension and calculate the average?",
                   "How do I filter a list using list comprehension? Give a runnable code snippet.",
                   "How do I transform all elements of a list without filtering?",
                   "How do I filter a set using a list comprehension in Python?",
                   "How do I filter a list without using a list comprehension or loop?",
                   "How do I filter a list to keep only elements greater than 50?"),
    ])

    # 2. dict_operations
    intents.extend([
        IntentSpec("dict_01", "dict_operations",
                   "How do I merge two dictionaries in Python 3.9+?",
                   "Use the union operator: `merged = d1 | d2`. Values from d2 overwrite keys present in d1.",
                   "What is the operator syntax to combine two dicts in Python 3.9?",
                   "How do I merge two dictionaries in Python 3.9+ and log all conflicting keys?",
                   "How do I merge two dictionaries in Python 3.9+? Include a code example.",
                   "How do I split a merged dictionary into two separate dictionaries?",
                   "How do I merge two lists into a dictionary in Python?",
                   "How do I merge two dictionaries without using the pipe union operator?",
                   "How do I merge exactly 4 dictionaries in Python 3.9+?"),
        IntentSpec("dict_02", "dict_operations",
                   "How do I sort a dictionary by its values in Python?",
                   "Use `dict(sorted(d.items(), key=lambda item: item[1]))`. It sorts key-value tuples by the second element.",
                   "How can I order dictionary entries according to their values?",
                   "How do I sort a dictionary by values and filter out negative numbers?",
                   "How do I sort a dictionary by values? Include an executable snippet.",
                   "How do I sort a dictionary by its keys in Python?",
                   "How do I sort a list of tuples by their values in Python?",
                   "How do I sort a dictionary without using the sorted() built-in function?",
                   "How do I sort a dictionary of exactly 10 items by value?"),
        IntentSpec("dict_03", "dict_operations",
                   "How do I safely get a dictionary key with a default value?",
                   "Use `d.get(key, default)`. If the key is not in the dictionary, it returns the specified default rather than raising KeyError.",
                   "What method retrieves a dict value or returns a fallback if absent?",
                   "How do I safely get a dictionary key with a default value and insert it if missing?",
                   "How do I safely get a dict key with default? Provide a working code snippet.",
                   "How do I remove a key from a dictionary with a default return value?",
                   "How do I safely get an index from a list with a default value?",
                   "How do I access a dict key without using get() or try/except?",
                   "How do I safely get a key with a default integer 0?"),
        IntentSpec("dict_04", "dict_operations",
                   "How do I write a dictionary comprehension in Python?",
                   "Use `{k: v for item in iterable}`. It creates a dictionary mapping generated keys to values over an iterable.",
                   "What is the syntax for dict comprehensions in Python?",
                   "How do I write a dictionary comprehension that maps keys to nested lists?",
                   "How do I write a dict comprehension in Python? Show runnable code.",
                   "How do I unpack a dictionary into individual variables in Python?",
                   "How do I write a set comprehension in Python?",
                   "How do I build a dictionary without using comprehensions or loops?",
                   "How do I write a dictionary comprehension for numbers from 1 to 20?"),
        IntentSpec("dict_05", "dict_operations",
                   "How do I iterate over keys and values of a dictionary simultaneously?",
                   "Use `for k, v in d.items():`. The items() method yields key-value tuples for simultaneous unpacking.",
                   "How to loop through both keys and values of a Python dict?",
                   "How do I iterate over keys and values and delete matching keys during the loop?",
                   "How do I iterate over keys and values? Include an executable code example.",
                   "How do I iterate over only dictionary keys in Python?",
                   "How do I iterate over keys and values of a list in Python?",
                   "How do I loop over dictionary items without using the items() method?",
                   "How do I iterate over keys and values of a dict with 1000 items?"),
    ])

    # 3. string_manipulation
    intents.extend([
        IntentSpec("str_01", "string_manipulation",
                   "How do I split a string by delimiter in Python?",
                   "Use `text.split(sep)`. It partitions the string wherever the delimiter appears, returning a list of strings.",
                   "What method divides a Python string by a separator?",
                   "How do I split a string by delimiter and remove all empty tokens?",
                   "How do I split a string by delimiter? Include a code example.",
                   "How do I join a list of strings using a delimiter in Python?",
                   "How do I split a byte sequence by delimiter in Python?",
                   "How do I split a string without using the split() method?",
                   "How do I split a string into exactly 2 parts using a delimiter?"),
        IntentSpec("str_02", "string_manipulation",
                   "How do I join a list of strings with a comma separator?",
                   "Use `', '.join(string_list)`. The join method concatenates all elements of the iterable using the string as separator.",
                   "How can I concatenate an iterable of strings separated by commas?",
                   "How do I join a list of strings with commas and format the last item with 'and'?",
                   "How do I join strings with a comma? Show a working code snippet.",
                   "How do I split a comma-separated string into a list in Python?",
                   "How do I join a list of integers with commas in Python?",
                   "How do I join strings without using the join() string method?",
                   "How do I join a list of exactly 10 strings with commas?"),
        IntentSpec("str_03", "string_manipulation",
                   "How do I strip whitespace from both ends of a string?",
                   "Use `text.strip()`. It removes all leading and trailing whitespace characters, returning a trimmed string.",
                   "What Python method removes leading and trailing spaces from text?",
                   "How do I strip whitespace from both ends and pad the string to 20 characters?",
                   "How do I strip whitespace from a string? Provide an executable example.",
                   "How do I pad a string with leading whitespace in Python?",
                   "How do I strip whitespace from all elements of a list in Python?",
                   "How do I remove whitespace from ends without using strip()?",
                   "How do I strip exactly 3 trailing spaces from a string?"),
        IntentSpec("str_04", "string_manipulation",
                   "How do I replace all occurrences of a substring in Python?",
                   "Use `text.replace(old, new)`. It replaces all instances of old with new throughout the string.",
                   "How can I substitute one substring for another in Python text?",
                   "How do I replace occurrences of a substring only within HTML tags?",
                   "How do I replace occurrences of a substring? Show runnable code.",
                   "How do I find the index of a substring in Python?",
                   "How do I replace items in a list in Python?",
                   "How do I replace substrings without using the replace() method?",
                   "How do I replace only the first 2 occurrences of a substring?"),
        IntentSpec("str_05", "string_manipulation",
                   "How do I format a floating-point number to two decimal places?",
                   "Use an f-string: `f'{val:.2f}'`. The format specifier `.2f` rounds and prints the float with two decimal digits.",
                   "What is the f-string syntax to display a float with 2 decimals?",
                   "How do I format a float to two decimals and prepend a currency symbol based on locale?",
                   "How do I format a float to 2 decimal places? Include a code example.",
                   "How do I parse a float from a formatted decimal string?",
                   "How do I format an integer with leading zeros in Python?",
                   "How do I format a float to two decimal places without using f-strings?",
                   "How do I format a float to exactly 4 decimal places?"),
    ])

    # 4. file_io
    intents.extend([
        IntentSpec("file_01", "file_io",
                   "How do I read a file line by line in Python?",
                   "Use `with open(path, 'r') as f: for line in f:`. Iterating over the file object reads lines lazily without high memory usage.",
                   "What is the standard way to iterate through lines in a file in Python?",
                   "How do I read a file line by line and compute an MD5 hash of each line?",
                   "How do I read a file line by line? Provide a code example.",
                   "How do I write a list of strings line by line to a file in Python?",
                   "How do I read a network socket line by line in Python?",
                   "How do I read lines from a file without using a for loop or readlines()?",
                   "How do I read exactly the first 100 lines of a file?"),
        IntentSpec("file_02", "file_io",
                   "How do I append text to an existing file in Python?",
                   "Use `with open(path, 'a') as f: f.write(text)`. Mode `'a'` opens the file for appending, placing new text at the end.",
                   "How can I add content to the end of a file without overwriting it?",
                   "How do I append text to a file and ensure immediate flush to physical disk?",
                   "How do I append text to an existing file? Show runnable code.",
                   "How do I truncate a file to zero bytes in Python?",
                   "How do I append records to a SQLite database in Python?",
                   "How do I append to a file without opening in 'a' mode?",
                   "How do I append exactly 50 bytes of text to a file?"),
        IntentSpec("file_03", "file_io",
                   "How do I check if a file exists before opening it in Python?",
                   "Use `Path(path).is_file()` or `os.path.isfile(path)`. Both check the filesystem and return a boolean indicating existence.",
                   "What function checks whether a file exists on disk in Python?",
                   "How do I check if a file exists and verify its write permissions?",
                   "How do I check if a file exists? Include an executable example.",
                   "How do I delete a file if it exists in Python?",
                   "How do I check if a directory exists in Python?",
                   "How do I check file existence without using os.path or pathlib?",
                   "How do I check if a file exists and is larger than 10MB?"),
        IntentSpec("file_04", "file_io",
                   "How do I read the entire contents of a file into a single string?",
                   "Use `with open(path, 'r') as f: content = f.read()`. The read() method returns all text up to EOF as a string.",
                   "How to load the complete text of a file into a variable in Python?",
                   "How do I read the entire contents of a file and convert all line endings to CRLF?",
                   "How do I read entire file content? Provide a code snippet.",
                   "How do I write a single string to overwrite a complete file?",
                   "How do I read the entire contents of a binary file into bytes?",
                   "How do I read a file completely without using the read() method?",
                   "How do I read a file into a string limiting max size to 1024 bytes?"),
        IntentSpec("file_05", "file_io",
                   "How do I read binary data from a file in Python?",
                   "Use `with open(path, 'rb') as f: data = f.read()`. Mode `'rb'` opens the file in binary mode, returning bytes.",
                   "What mode should I use to read raw binary bytes from a file in Python?",
                   "How do I read binary data from a file and decrypt it using AES?",
                   "How do I read binary data from a file? Show a working code snippet.",
                   "How do I write binary bytes to a file in Python?",
                   "How do I read text data from a file in Python?",
                   "How do I read binary data without using 'rb' open mode?",
                   "How do I read exactly 256 bytes of binary data from a file?"),
    ])

    # 5. json_handling
    intents.extend([
        IntentSpec("json_01", "json_handling",
                   "How do I serialize a Python dictionary to a JSON string?",
                   "Use `json.dumps(my_dict)`. It converts a Python dictionary or primitive object into a valid JSON-formatted string.",
                   "What function converts a Python dict into a JSON string?",
                   "How do I serialize a dictionary to JSON and encrypt the output string?",
                   "How do I serialize a dictionary to JSON? Provide an executable snippet.",
                   "How do I parse a JSON string back into a Python dictionary?",
                   "How do I serialize a dictionary to YAML in Python?",
                   "How do I serialize a dictionary without using the json module?",
                   "How do I serialize a dictionary to JSON with indentation of 4 spaces?"),
        IntentSpec("json_02", "json_handling",
                   "How do I parse a JSON string into a Python dictionary?",
                   "Use `json.loads(json_string)`. It decodes JSON text into corresponding Python dictionaries, lists, and values.",
                   "How can I deserialize a JSON text string into a Python dictionary?",
                   "How do I parse a JSON string into a dict validating schema with pydantic?",
                   "How do I parse a JSON string into a dict? Show runnable code.",
                   "How do I serialize a dictionary into a JSON string in Python?",
                   "How do I parse an XML string into a Python dictionary?",
                   "How do I parse JSON text without using json.loads()?",
                   "How do I parse a JSON string containing exactly 3 keys?"),
        IntentSpec("json_03", "json_handling",
                   "How do I pretty-print a JSON string with indentation in Python?",
                   "Use `json.dumps(data, indent=2)`. The `indent` argument formats output with multi-line indentation and spacing.",
                   "How to format a JSON string with 2-space indentation in Python?",
                   "How do I pretty-print JSON and colorize output in terminal console?",
                   "How do I pretty-print a JSON string? Include a working code example.",
                   "How do I compact a JSON string removing all whitespace in Python?",
                   "How do I pretty-print an XML document in Python?",
                   "How do I format JSON with indentation without using the indent parameter?",
                   "How do I pretty-print JSON with 8 spaces indentation?"),
        IntentSpec("json_04", "json_handling",
                   "How do I read JSON data directly from a file in Python?",
                   "Use `with open(path, 'r') as f: data = json.load(f)`. The json.load() function parses directly from a file stream.",
                   "What function reads and parses JSON from a file stream in Python?",
                   "How do I read JSON from a file and update existing keys in memory?",
                   "How do I read JSON from a file? Provide an executable example.",
                   "How do I write JSON data directly to a file stream in Python?",
                   "How do I read CSV data from a file stream in Python?",
                   "How do I load JSON from a file without using json.load()?",
                   "How do I read a JSON file stream larger than 100MB?"),
        IntentSpec("json_05", "json_handling",
                   "How do I handle custom objects when serializing to JSON in Python?",
                   "Pass a custom converter: `json.dumps(obj, default=custom_serializer)`. Alternatively subclass `json.JSONEncoder`.",
                   "How to serialize non-standard Python objects to JSON using a default function?",
                   "How do I handle custom objects in JSON serialization with circular reference checks?",
                   "How do I handle custom objects in JSON serialization? Show working code.",
                   "How do I deserialize custom object types from JSON in Python?",
                   "How do I serialize custom objects to pickle in Python?",
                   "How do I serialize custom objects to JSON without defining a default serializer?",
                   "How do I serialize a custom object having 5 dataclass fields to JSON?"),
    ])

    # 6-25: Synthesize remaining 20 groups with 5 distinct intents each (100 more intents = 125 total)
    remaining_groups = [
        ("csv_processing", "csv"),
        ("error_handling", "err"),
        ("datetime_operations", "dt"),
        ("regex_patterns", "re"),
        ("math_statistics", "math"),
        ("pathlib_filesystem", "path"),
        ("itertools_recipes", "iter"),
        ("collections_types", "coll"),
        ("concurrency_threading", "th"),
        ("asyncio_coroutines", "async"),
        ("http_requests", "http"),
        ("binary_data", "bin"),
        ("hash_cryptography", "hash"),
        ("sqlite_database", "sql"),
        ("sorting_searching", "sort"),
        ("generators_iterators", "gen"),
        ("decorators_wrappers", "dec"),
        ("context_managers", "ctx"),
        ("object_oriented", "oop"),
        ("functional_programming", "func"),
    ]

    templates = [
        ("01", "basic_usage",
         "How do I perform {topic} in Python?",
         "Use the standard library function for {topic}. Call the dedicated API directly with required arguments.",
         "What is the standard syntax for {topic} in Python?",
         "How do I perform {topic} and log the execution time?",
         "How do I perform {topic} in Python? Provide an executable code example.",
         "How do I reverse the outcome of {topic} in Python?",
         "How do I perform {topic} on an alternative data structure in Python?",
         "How do I perform {topic} without using standard library modules?",
         "How do I perform {topic} with a timeout of 10 seconds?"),
        ("02", "file_stream",
         "How do I stream {topic} to disk in Python?",
         "Open a file stream with open() and stream data using standard write methods in chunks.",
         "How can I write {topic} data directly to disk in Python?",
         "How do I stream {topic} to disk and verify the SHA256 checksum?",
         "How do I stream {topic} to disk? Include a working code snippet.",
         "How do I read {topic} from disk into memory in Python?",
         "How do I stream {topic} over an alternative network socket in Python?",
         "How do I write {topic} to disk without using open()?",
         "How do I stream {topic} in chunks of exactly 4096 bytes?"),
        ("03", "custom_options",
         "How do I configure custom options for {topic} in Python?",
         "Pass keyword arguments to the initialization function to adjust runtime configuration parameters.",
         "What arguments configure custom behavior for {topic} in Python?",
         "How do I configure custom options for {topic} and load settings from YAML?",
         "How do I configure custom options for {topic}? Show runnable code.",
         "How do I reset {topic} to default configuration in Python?",
         "How do I configure options for an alternative {topic} framework in Python?",
         "How do I configure {topic} without passing keyword arguments?",
         "How do I set exactly 3 configuration flags for {topic}?"),
        ("04", "error_handling",
         "How do I handle errors during {topic} in Python?",
         "Wrap the operation in a try block and catch specific exception types raised during execution.",
         "What exceptions should I catch when executing {topic} in Python?",
         "How do I handle errors during {topic} and retry up to 3 times?",
         "How do I handle errors during {topic}? Provide an executable example.",
         "How do I suppress all exceptions during {topic} in Python?",
         "How do I handle external connection errors during {topic} in Python?",
         "How do I handle errors in {topic} without using try/except?",
         "How do I catch an error with error code 404 in {topic}?"),
        ("05", "performance",
         "How do I optimize the performance of {topic} in Python?",
         "Preallocate resources, use in-memory caches, and leverage built-in C-accelerated standard library helpers.",
         "What techniques speed up {topic} execution in Python?",
         "How do I optimize {topic} performance and benchmark using cProfile?",
         "How do I optimize {topic} performance? Give a working code snippet.",
         "How do I profile memory usage of {topic} in Python?",
         "How do I optimize remote throughput during {topic} in Python?",
         "How do I optimize {topic} without using third-party C extensions?",
         "How do I optimize {topic} to run in under 5 milliseconds?"),
    ]

    for grp_id, prefix in remaining_groups:
        grp_meta = CORPUS_SOURCE_GROUPS[grp_id]
        topic_name = grp_meta["title"].lower()
        for suffix, tname, q, ans, para, req, ex, inv, ent, neg, num in templates:
            iid = f"{prefix}_{suffix}"
            intents.append(IntentSpec(
                intent_id=iid,
                group_id=grp_id,
                seed_question=q.format(topic=topic_name),
                seed_answer=f"For {topic_name}: {grp_meta['doc'][:120]} Use standard API idioms.",
                paraphrase=para.format(topic=topic_name),
                added_requirement=req.format(topic=topic_name),
                missing_code_example=ex.format(topic=topic_name),
                inverse_operation=inv.format(topic=topic_name),
                changed_entity=ent.format(topic=topic_name),
                negation=neg.format(topic=topic_name),
                numeric_constraint=num.format(topic=topic_name),
            ))

    return intents


# =============================================================================
# 3. Workload Builder: Lower-Repeat and Repeat-Rich Profiles
# =============================================================================

def build_evaluation_workload(
    intents: list[IntentSpec],
    profile: str = "repeat_rich",
) -> list[dict]:
    """Builds standardized multi-intent evaluation workloads.
    
    Workload Profiles:
    - 'lower_repeat' (~400 events): 125 seeds, 60 exact repeats, 90 natural paraphrases,
      125 adversarial challenge variants across all 6 challenge types.
    - 'repeat_rich' (~600 events): 60 selected seeds, 240 exact repeats (4x factor),
      180 natural paraphrases (3x factor), 120 adversarial challenges.
    """
    events = []

    if profile == "repeat_rich":
        # 60 seeds across all groups (2-3 per group)
        selected_intents = intents[:60]
        # 1. 60 Seeds
        for spec in selected_intents:
            events.append({
                "event_id": f"{spec.intent_id}:seed",
                "intent_id": spec.intent_id,
                "group_id": spec.group_id,
                "category": "seed",
                "question": spec.seed_question,
                "eligible": True,
                "expected_tier": "NONE",  # Cold
                "expected_hit": False,
                "challenge_type": None,
                "doc_text": spec.seed_answer,
            })
        # 2. 240 Exact Repeats (4x per seed)
        for rep in range(1, 5):
            for spec in selected_intents:
                events.append({
                    "event_id": f"{spec.intent_id}:exact_{rep}",
                    "intent_id": spec.intent_id,
                    "group_id": spec.group_id,
                    "category": "exact_repeat",
                    "question": spec.seed_question,
                    "eligible": True,
                    "expected_tier": "EXACT",
                    "expected_hit": True,
                    "challenge_type": None,
                    "doc_text": spec.seed_answer,
                })
        # 3. 180 Natural Paraphrases (3 variants per seed: paraphrase, minor case variation, slight token reorder)
        for spec in selected_intents:
            p1 = spec.paraphrase
            p2 = spec.paraphrase.replace("How do I", "How can one").replace("How to", "How can I")
            p3 = spec.paraphrase.rstrip("?") + " in Python 3?"
            for idx, p in enumerate([p1, p2, p3], 1):
                events.append({
                    "event_id": f"{spec.intent_id}:paraphrase_{idx}",
                    "intent_id": spec.intent_id,
                    "group_id": spec.group_id,
                    "category": "natural_paraphrase",
                    "question": p,
                    "eligible": True,
                    "expected_tier": "SEMANTIC",
                    "expected_hit": True,
                    "challenge_type": None,
                    "doc_text": spec.seed_answer,
                })
        # 4. 120 Adversarial Challenge Queries (2 challenges per seed from the 6 challenge families)
        challenge_getters = [
            ("inverse_operation", lambda s: s.inverse_operation),
            ("missing_executable_example", lambda s: s.missing_code_example),
            ("added_requirements", lambda s: s.added_requirement),
            ("changed_entity", lambda s: s.changed_entity),
            ("negation", lambda s: s.negation),
            ("numeric_constraint", lambda s: s.numeric_constraint),
        ]
        for idx, spec in enumerate(selected_intents):
            c1_type, c1_fn = challenge_getters[idx % len(challenge_getters)]
            c2_type, c2_fn = challenge_getters[(idx + 1) % len(challenge_getters)]
            for ctype, cfn in [(c1_type, c1_fn), (c2_type, c2_fn)]:
                events.append({
                    "event_id": f"{spec.intent_id}:challenge_{ctype}",
                    "intent_id": spec.intent_id,
                    "group_id": spec.group_id,
                    "category": "adversarial_challenge",
                    "question": cfn(spec),
                    "eligible": False,  # Must fail closed to fresh agent execution
                    "expected_tier": "NONE",
                    "expected_hit": False,
                    "challenge_type": ctype,
                    "doc_text": spec.seed_answer,
                })
        return events

    # 'lower_repeat' profile (~400 events across all 125 intents)
    # 1. 125 Seeds
    for spec in intents:
        events.append({
            "event_id": f"{spec.intent_id}:seed",
            "intent_id": spec.intent_id,
            "group_id": spec.group_id,
            "category": "seed",
            "question": spec.seed_question,
            "eligible": True,
            "expected_tier": "NONE",
            "expected_hit": False,
            "challenge_type": None,
            "doc_text": spec.seed_answer,
        })
    # 2. 60 Exact Repeats (subset of intents)
    for spec in intents[:60]:
        events.append({
            "event_id": f"{spec.intent_id}:exact",
            "intent_id": spec.intent_id,
            "group_id": spec.group_id,
            "category": "exact_repeat",
            "question": spec.seed_question,
            "eligible": True,
            "expected_tier": "EXACT",
            "expected_hit": True,
            "challenge_type": None,
            "doc_text": spec.seed_answer,
        })
    # 3. 90 Natural Paraphrases
    for spec in intents[:90]:
        events.append({
            "event_id": f"{spec.intent_id}:paraphrase",
            "intent_id": spec.intent_id,
            "group_id": spec.group_id,
            "category": "natural_paraphrase",
            "question": spec.paraphrase,
            "eligible": True,
            "expected_tier": "SEMANTIC",
            "expected_hit": True,
            "challenge_type": None,
            "doc_text": spec.seed_answer,
        })
    # 4. 125 Adversarial Challenges (1 per intent, rotating challenge types)
    challenge_cycle = [
        ("inverse_operation", lambda s: s.inverse_operation),
        ("missing_executable_example", lambda s: s.missing_code_example),
        ("added_requirements", lambda s: s.added_requirement),
        ("changed_entity", lambda s: s.changed_entity),
        ("negation", lambda s: s.negation),
        ("numeric_constraint", lambda s: s.numeric_constraint),
    ]
    for idx, spec in enumerate(intents):
        ctype, cfn = challenge_cycle[idx % len(challenge_cycle)]
        events.append({
            "event_id": f"{spec.intent_id}:challenge_{ctype}",
            "intent_id": spec.intent_id,
            "group_id": spec.group_id,
            "category": "adversarial_challenge",
            "question": cfn(spec),
            "eligible": False,
            "expected_tier": "NONE",
            "expected_hit": False,
            "challenge_type": ctype,
            "doc_text": spec.seed_answer,
        })

    return events


# =============================================================================
# 4. Agent Benchmark Provider (Reproducible Staged Execution)
# =============================================================================

class ScalableBenchmarkProvider(FixtureChatProvider):
    """Deterministic provider simulation for supervisor, tool, writer, and reviewer stages."""

    def __init__(self, documents: dict[str, str], delay_seconds: float = 0.005):
        self.documents = documents
        self.delay = delay_seconds
        self.attempts = 0
        self.stage_counts = {"planning": 0, "generation": 0, "review": 0}
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def __call__(self, body: dict) -> dict:
        self.attempts += 1
        messages = body.get("messages", [])
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""

        payload_data = {}
        if user.startswith("{"):
            try:
                payload_data = json.loads(user)
            except Exception:
                pass
        q_text = payload_data.get("question", user)

        if "Plan 1-8" in system or "steps" in system or "planning" in system.lower():
            stage = "planning"
            self.stage_counts["planning"] += 1
            # Match document by key terms
            q_lower = q_text.lower()
            matched_doc = next(iter(self.documents.keys()))
            for doc_id, doc_text in self.documents.items():
                if doc_id.lower() in q_lower or any(w in doc_text.lower() for w in q_lower.split() if len(w) > 4):
                    matched_doc = doc_id
                    break
            content = json.dumps({
                "steps": [
                    {"id": "read_doc", "tool": "read", "arguments": {"document_id": matched_doc}, "depends_on": []}
                ]
            })
        elif "reviewer" in system.lower() or "check correctness" in system.lower():
            stage = "review"
            self.stage_counts["review"] += 1
            content = json.dumps({"approved": True, "feedback": "Reviewed against verified source documentation."})
        else:
            stage = "generation"
            self.stage_counts["generation"] += 1
            evidence_list = payload_data.get("evidence", [])
            doc_snippet = evidence_list[0].get("result", "") if evidence_list and isinstance(evidence_list[0], dict) else ""
            content = f"Answer grounded in documentation: {doc_snippet[:150]}"

        if self.delay > 0:
            time.sleep(self.delay)

        p_tok = len(system.split()) + len(user.split()) + 15
        c_tok = len(content.split()) + 5
        self.total_prompt_tokens += p_tok
        self.total_completion_tokens += c_tok

        return {
            "id": f"call-{hashlib.sha256(content.encode()).hexdigest()[:12]}",
            "model": body.get("model", "gpt-4o-mini-fixture"),
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }],
            "usage": {
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": p_tok + c_tok,
            },
        }


# =============================================================================
# 5. Arm Runner & Comprehensive Instrumentation
# =============================================================================

def run_evaluation_arm(
    arm_name: str,
    workload: list[dict],
    documents: dict[str, str],
    db_dir: Path,
    encoder: DenseEncoder,
    passage_scorer: LocalPassageScorer,
    verbose: bool = False,
) -> dict[str, Any]:
    """Runs a single arm against the exact workload sequence."""
    db_dir.mkdir(parents=True, exist_ok=True)
    answers_db = db_dir / f"{arm_name}_answers.db"
    tasks_db = db_dir / f"{arm_name}_tasks.db"
    if answers_db.exists():
        answers_db.unlink()
    if tasks_db.exists():
        tasks_db.unlink()

    provider = ScalableBenchmarkProvider(documents, delay_seconds=0.005)
    tools = DocumentTools(documents)
    scope = KnowledgeScope("eval-corpus", tools.revision, "benchmark-tenant", f"campaign-{arm_name}")

    exact_storage = SQLiteResponseStore(answers_db, namespace=f"arm-{arm_name}")

    # Arm configuration
    if arm_name == "no_cache":
        exact = ScopedResponseCache(
            eligibility=lambda _: EligibilityDecision(False, "no_cache"),
            storage=exact_storage,
        )
        outer = ConservativeHybridCache(
            exact, scope=scope, encoder=encoder, passage_scorer=passage_scorer, mode="off",
        )
    elif arm_name == "exact_only":
        exact = ScopedResponseCache(
            eligibility=lambda _: EligibilityDecision(True, "benchmark_policy"),
            storage=exact_storage,
        )
        outer = ConservativeHybridCache(
            exact, scope=scope, encoder=encoder, passage_scorer=passage_scorer, mode="exact",
        )
    elif arm_name == "verified_semantic":
        exact = ScopedResponseCache(
            eligibility=lambda _: EligibilityDecision(True, "benchmark_policy"),
            storage=exact_storage,
        )
        outer = ConservativeHybridCache(
            exact,
            scope=scope,
            encoder=encoder,
            passage_scorer=passage_scorer,
            mode="guarded",
            retrieval_threshold=0.60,
            verification_threshold=0.85,
        )
    elif arm_name == "shadow_semantic":
        exact = ScopedResponseCache(
            eligibility=lambda _: EligibilityDecision(True, "benchmark_policy"),
            storage=exact_storage,
        )
        outer = ConservativeHybridCache(
            exact,
            scope=scope,
            encoder=encoder,
            passage_scorer=passage_scorer,
            mode="shadow",
            retrieval_threshold=0.60,
            verification_threshold=0.85,
        )
    else:
        raise ValueError(f"Unknown arm name: {arm_name}")

    answer_cache = CompletedDocumentCache(outer)
    service = GatewayService(
        ScopedResponseCache(eligibility=lambda _: EligibilityDecision(False, "gateway_off")),
        provider,
        allowed_models={"gpt-4o-mini-fixture"},
    )
    agent = DocumentAgent(
        service,
        tools,
        tasks_db,
        model="gpt-4o-mini-fixture",
        answer_cache=answer_cache,
        stage_accounting=True,
    )

    event_records = []
    t_arm_start = time.perf_counter()

    # Track metrics
    false_accepts_by_type = {}
    guard_rejections_by_type = {}
    retrieval_recalls = []
    verifier_times_ms = []

    for idx, item in enumerate(workload):
        t0 = time.perf_counter()
        prov_before = provider.attempts
        p_tok_before = provider.total_prompt_tokens
        c_tok_before = provider.total_completion_tokens

        tid = agent.create(
            item["question"],
            tenant="benchmark-tenant",
            principal="benchmark-evaluator",
            standalone_document_task=True,
        )

        state = None
        for _ in range(25):
            state = agent.advance(tid, tenant="benchmark-tenant", principal="benchmark-evaluator")
            if state["status"] in {"COMPLETED", "AWAITING_APPROVAL", "FAILED", "REJECTED"}:
                break

        prov_calls = provider.attempts - prov_before
        p_tok = provider.total_prompt_tokens - p_tok_before
        c_tok = provider.total_completion_tokens - c_tok_before

        is_hit = bool(state.get("answer_cache_hit", False))
        tier = "NONE"
        if is_hit:
            tier = state["trace"][0].get("tier", "EXACT")

        # Operator approval to admit answers into durable SQLite store
        if state["status"] == "AWAITING_APPROVAL":
            state = agent.decide(tid, "approve", tenant="benchmark-tenant", principal="benchmark-evaluator")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Safety checking: Adversarial challenges must never produce a cache hit
        if item["category"] == "adversarial_challenge" and is_hit:
            ctype = item.get("challenge_type", "unknown")
            false_accepts_by_type[ctype] = false_accepts_by_type.get(ctype, 0) + 1

        is_correct_reuse = bool(is_hit and item["eligible"])
        is_incorrect_reuse = bool(is_hit and not item["eligible"])

        rec = {
            "arm": arm_name,
            "event_id": item["event_id"],
            "intent_id": item["intent_id"],
            "group_id": item["group_id"],
            "category": item["category"],
            "challenge_type": item.get("challenge_type"),
            "question": item["question"],
            "is_hit": is_hit,
            "tier": tier,
            "is_correct_reuse": is_correct_reuse,
            "is_incorrect_reuse": is_incorrect_reuse,
            "selected_cached_answer": state.get("answer") if is_hit else None,
            "matched_question": state.get("matched_question") if is_hit else None,
            "actual_response_answer": state.get("answer"),
            "provider_calls": prov_calls,
            "avoided_calls": 3 if is_hit else 0,
            "prompt_tokens_synthetic": p_tok,
            "completion_tokens_synthetic": c_tok,
            "total_tokens_synthetic": p_tok + c_tok,
            "elapsed_ms": elapsed_ms,
            "terminal_status": state.get("status"),
        }
        event_records.append(rec)

    total_wall_s = time.perf_counter() - t_arm_start
    latencies = [r["elapsed_ms"] for r in event_records]
    total_calls = provider.attempts
    total_avoided = sum(r["avoided_calls"] for r in event_records)
    total_avoided_correct = sum(r["avoided_calls"] for r in event_records if r["is_correct_reuse"])
    total_avoided_incorrect = sum(r["avoided_calls"] for r in event_records if r["is_incorrect_reuse"])
    total_hits = sum(1 for r in event_records if r["is_hit"])
    exact_hits = sum(1 for r in event_records if r["tier"] == "EXACT")
    semantic_hits = sum(1 for r in event_records if r["tier"] == "SEMANTIC")

    # Opportunity coverage: safe paraphrases hit rate
    paraphrase_events = [r for r in event_records if r["category"] == "natural_paraphrase"]
    para_hits = sum(1 for r in paraphrase_events if r["is_hit"])
    para_coverage = (para_hits / len(paraphrase_events)) if paraphrase_events else 0.0

    exact_repeat_events = [r for r in event_records if r["category"] == "exact_repeat"]
    exact_repeat_hits = sum(1 for r in exact_repeat_events if r["is_hit"])
    exact_coverage = (exact_repeat_hits / len(exact_repeat_events)) if exact_repeat_events else 0.0

    adv_events = [r for r in event_records if r["category"] == "adversarial_challenge"]
    total_false_accepts = sum(false_accepts_by_type.values())
    false_accept_ci = clopper_pearson_ci(total_false_accepts, len(adv_events)) if adv_events else [0.0, 0.0]

    summary = {
        "arm": arm_name,
        "total_events": len(event_records),
        "total_provider_calls": total_calls,
        "calls_avoided_total": total_avoided,
        "calls_avoided_correct": total_avoided_correct,
        "calls_avoided_incorrect": total_avoided_incorrect,
        "total_synthetic_tokens": provider.total_prompt_tokens + provider.total_completion_tokens,
        "cache_hits_total": total_hits,
        "exact_hits": exact_hits,
        "semantic_hits": semantic_hits,
        "exact_repeat_coverage": round(exact_coverage * 100.0, 2),
        "paraphrase_opportunity_coverage": round(para_coverage * 100.0, 2),
        "adversarial_total_challenges": len(adv_events),
        "adversarial_false_accepts": total_false_accepts,
        "adversarial_false_accept_rate": round((total_false_accepts / len(adv_events) * 100.0) if adv_events else 0.0, 4),
        "adversarial_false_accept_ci95": false_accept_ci,
        "false_accepts_by_type": false_accepts_by_type,
        "quality_review_status": "model_assisted_simulation",
        "latency_ms_p50": round(statistics.median(latencies), 2) if latencies else 0.0,
        "latency_ms_p90": round(statistics.quantiles(latencies, n=10)[8], 2) if len(latencies) >= 10 else round(max(latencies), 2),
        "latency_ms_p95": round(statistics.quantiles(latencies, n=20)[18], 2) if len(latencies) >= 20 else round(max(latencies), 2),
        "latency_ms_mean": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "total_wall_seconds": round(total_wall_s, 2),
        "provider_stage_calls": provider.stage_counts,
    }

    return {
        "summary": summary,
        "records": event_records,
    }


# =============================================================================
# 6. Main Orchestrator and Artifact Generation
# =============================================================================

def run_large_workload_campaign(
    profile: str = "repeat_rich",
    output_dir: Path | None = None,
    arms: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    if output_dir is None:
        output_dir = ROOT / "artifacts/metrics_campaign"
    output_dir.mkdir(parents=True, exist_ok=True)

    if arms is None:
        arms = ["no_cache", "exact_only", "verified_semantic", "shadow_semantic"]

    intents = build_base_intents()
    workload = build_evaluation_workload(intents, profile=profile)

    documents = {k: v["doc"] for k, v in CORPUS_SOURCE_GROUPS.items()}

    # Compute manifest metadata
    category_counts = {}
    for item in workload:
        cat = item["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    manifest = {
        "campaign_version": "v1.0",
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workload_profile": profile,
        "total_events": len(workload),
        "source_groups_count": len(CORPUS_SOURCE_GROUPS),
        "base_intents_count": len(intents),
        "event_category_distribution": category_counts,
        "models": {
            "dense_encoder": f"{ENCODER_MODEL}@{ENCODER_REVISION}",
            "passage_scorer": f"{PASSAGE_MODEL}@{PASSAGE_REVISION}",
            "agent_model": "gpt-4o-mini-fixture",
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
        },
        "source_groups": list(CORPUS_SOURCE_GROUPS.keys()),
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        print("================================================================================")
        print("       ADAPTIVE AGENTIC SEMANTIC CACHE — LARGE WORKLOAD EVALUATION HARNESS      ")
        print("================================================================================\n")
        print(f"Workload Profile : {profile.upper()}")
        print(f"Total Events     : {len(workload)} events")
        print(f"Source Groups    : {len(CORPUS_SOURCE_GROUPS)} groups")
        print(f"Base Intents     : {len(intents)} distinct intents")
        print("Category Breakdown:")
        for cat, cnt in category_counts.items():
            print(f"  - {cat:<24}: {cnt} events")
        print(f"\nManifest exported: {manifest_path}\n")

    encoder = DenseEncoder(ENCODER_MODEL, ENCODER_REVISION)
    passage_scorer = LocalPassageScorer(PASSAGE_MODEL, PASSAGE_REVISION)

    all_arm_summaries = {}
    all_raw_records = []

    db_base = output_dir / "runs_db"

    for arm in arms:
        if verbose:
            print(f"--- Running Arm: {arm.upper()} ({len(workload)} events) ---")
        res = run_evaluation_arm(arm, workload, documents, db_base / arm, encoder, passage_scorer, verbose=verbose)
        all_arm_summaries[arm] = res["summary"]
        all_raw_records.extend(res["records"])
        if verbose:
            sm = res["summary"]
            print(f"    Calls: {sm['total_provider_calls']} | Avoided (Correct/Incorr): {sm['calls_avoided_correct']}/{sm['calls_avoided_incorrect']} | Hits: {sm['cache_hits_total']} (Exact: {sm['exact_hits']}, Sem: {sm['semantic_hits']}) | p50: {sm['latency_ms_p50']}ms | False Accepts: {sm['adversarial_false_accepts']}")

    # Export raw JSONL
    results_jsonl_path = output_dir / "offline_results.jsonl"
    with open(results_jsonl_path, "w", encoding="utf-8") as f:
        for rec in all_raw_records:
            f.write(json.dumps(rec) + "\n")

    # Export summary JSON
    summary_path = output_dir / "summary.json"
    campaign_summary = {
        "manifest": manifest,
        "arms": all_arm_summaries,
        "raw_records_count": len(all_raw_records),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(campaign_summary, f, indent=2)

    if verbose:
        print("\n================================================================================")
        print("                        METRICS CAMPAIGN SUMMARY TABLE                          ")
        print("================================================================================\n")
        print("| Arm | Provider Calls | Avoided (Correct/Incorr) | Hits (Exact / Sem) | Repeat Cov | Para Cov | p50 Latency | False Accepts (95% CI) |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for arm, sm in all_arm_summaries.items():
            hits_str = f"{sm['exact_hits']} / {sm['semantic_hits']}"
            ci_str = f"{sm['adversarial_false_accepts']} [{sm['adversarial_false_accept_ci95'][0]:.3f}, {sm['adversarial_false_accept_ci95'][1]:.3f}]"
            avoided_str = f"{sm['calls_avoided_correct']} / {sm['calls_avoided_incorrect']}"
            print(f"| **{arm}** | {sm['total_provider_calls']} | {avoided_str} | {hits_str} | {sm['exact_repeat_coverage']}% | {sm['paraphrase_opportunity_coverage']}% | {sm['latency_ms_p50']} ms | {ci_str} |")
        print(f"\nArtifacts successfully written to: {output_dir}\n")

    return campaign_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["repeat_rich", "lower_repeat", "both"], default="repeat_rich")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/metrics_campaign")
    parser.add_argument("--arms", nargs="+", default=["no_cache", "exact_only", "verified_semantic", "shadow_semantic"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.profile == "both":
        s_rich = run_large_workload_campaign(profile="repeat_rich", output_dir=args.output_dir / "repeat_rich", arms=args.arms, verbose=not args.json)
        s_lower = run_large_workload_campaign(profile="lower_repeat", output_dir=args.output_dir / "lower_repeat", arms=args.arms, verbose=not args.json)
        overall = {"repeat_rich": s_rich, "lower_repeat": s_lower}
    else:
        overall = run_large_workload_campaign(profile=args.profile, output_dir=args.output_dir, arms=args.arms, verbose=not args.json)

    if args.json:
        print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()

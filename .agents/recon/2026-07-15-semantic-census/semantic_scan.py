#!/usr/bin/env python3
"""Generate a repository-wide, evidence-oriented semantic census.

The scanner never imports project modules. Python is parsed with the stdlib AST;
other text languages use conservative declaration patterns. Every tracked file
is represented in file-inventory.csv, including non-source artifacts.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


LANGUAGES = {
    ".py": "Python",
    ".c": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".cxx": "C++",
    ".h": "C/C++ Header",
    ".hh": "C++ Header",
    ".hpp": "C++ Header",
    ".cu": "CUDA",
    ".cuh": "CUDA Header",
    ".v": "Verilog",
    ".sv": "SystemVerilog",
    ".vhd": "VHDL",
    ".vhdl": "VHDL",
    ".tcl": "Tcl",
    ".sh": "Shell",
    ".bash": "Shell",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".scala": "Scala",
    ".m": "MATLAB/Objective-C",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".json": "JSON",
    ".xml": "XML",
    ".md": "Markdown",
    ".rst": "reStructuredText",
    ".tex": "TeX",
}

TEXT_EXTENSIONS = set(LANGUAGES) | {
    ".cfg", ".conf", ".ini", ".mk", ".cmake", ".gradle", ".txt",
}

NONPY_PATTERNS = {
    "Verilog": [
        ("module", re.compile(r"^\s*module\s+([A-Za-z_]\w*)")),
        ("function", re.compile(r"^\s*function(?:\s+automatic)?(?:\s+\[[^]]+\])?\s+([A-Za-z_]\w*)")),
        ("task", re.compile(r"^\s*task(?:\s+automatic)?\s+([A-Za-z_]\w*)")),
    ],
    "SystemVerilog": [
        ("module", re.compile(r"^\s*module\s+([A-Za-z_]\w*)")),
        ("interface", re.compile(r"^\s*interface\s+([A-Za-z_]\w*)")),
        ("class", re.compile(r"^\s*class\s+([A-Za-z_]\w*)")),
        ("function", re.compile(r"^\s*function\b.*?([A-Za-z_]\w*)\s*\(")),
        ("task", re.compile(r"^\s*task\b.*?([A-Za-z_]\w*)\s*\(")),
    ],
    "Tcl": [("proc", re.compile(r"^\s*proc\s+([^\s{]+)"))],
    "Shell": [
        ("function", re.compile(r"^\s*(?:function\s+)?([A-Za-z_]\w*)\s*\(\s*\)\s*\{")),
    ],
    "JavaScript": [
        ("class", re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)")),
        ("function", re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)")),
    ],
    "TypeScript": [
        ("class", re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)")),
        ("interface", re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)")),
        ("function", re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)")),
    ],
    "C": [
        ("function", re.compile(r"^\s*(?!if\b|for\b|while\b|switch\b)(?:[\w:*&<>]+\s+)+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{")),
    ],
    "C++": [
        ("class", re.compile(r"^\s*(?:class|struct)\s+([A-Za-z_]\w*)")),
        ("function", re.compile(r"^\s*(?!if\b|for\b|while\b|switch\b)(?:[\w:*&<>~]+\s+)+([A-Za-z_]\w*(?:::\w+)*)\s*\([^;]*\)\s*(?:const\s*)?\{")),
    ],
    "C/C++ Header": [
        ("class", re.compile(r"^\s*(?:class|struct)\s+([A-Za-z_]\w*)")),
        ("function", re.compile(r"^\s*(?!if\b|for\b|while\b|switch\b)(?:[\w:*&<>~]+\s+)+([A-Za-z_]\w*(?:::\w+)*)\s*\([^;]*\)\s*(?:const\s*)?[{;]")),
    ],
    "C++ Header": [
        ("class", re.compile(r"^\s*(?:class|struct)\s+([A-Za-z_]\w*)")),
        ("function", re.compile(r"^\s*(?!if\b|for\b|while\b|switch\b)(?:[\w:*&<>~]+\s+)+([A-Za-z_]\w*(?:::\w+)*)\s*\([^;]*\)\s*(?:const\s*)?[{;]")),
    ],
    "CUDA": [
        ("kernel_or_function", re.compile(r"^\s*(?:(?:__global__|__device__|__host__)\s+)*(?:[\w:*&<>]+\s+)+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{")),
    ],
    "CUDA Header": [
        ("kernel_or_function", re.compile(r"^\s*(?:(?:__global__|__device__|__host__)\s+)*(?:[\w:*&<>]+\s+)+([A-Za-z_]\w*)\s*\([^;]*\)\s*[;{]")),
    ],
}


@dataclass
class FileRecord:
    path: str
    zone: str
    language: str
    extension: str
    bytes: int
    git_blob: str
    sha256: str
    lines: int
    code_lines: int
    blank_lines: int
    parse_status: str


@dataclass
class SymbolRecord:
    path: str
    zone: str
    kind: str
    qualname: str
    name: str
    start_line: int
    end_line: int
    loc: int
    statement_count: int
    branch_count: int
    parameter_count: int
    decorator_count: int
    has_docstring: bool
    calls: str
    semantic_hash: str


@dataclass
class Finding:
    path: str
    line: int
    end_line: int
    zone: str
    symbol: str
    rule: str
    severity: str
    claim_type: str
    confidence: str
    evidence: str


@dataclass(frozen=True)
class GitEntry:
    path: str
    oid: str
    size: int


def git_entries(root: Path, commit: str) -> list[GitEntry]:
    raw = subprocess.check_output(["git", "ls-tree", "-r", "-l", "-z", commit], cwd=root)
    entries: list[GitEntry] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_type, oid, raw_size = metadata.decode("ascii").split()
        if object_type != "blob":
            continue
        path = raw_path.decode("utf-8", "surrogateescape")
        entries.append(GitEntry(path=path, oid=oid, size=int(raw_size)))
    return entries


def committed_blobs(root: Path, entries: list[GitEntry]) -> Iterator[tuple[GitEntry, bytes]]:
    process = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        for entry in entries:
            process.stdin.write(f"{entry.oid}\n".encode("ascii"))
            process.stdin.flush()
            header = process.stdout.readline().decode("ascii").strip().split()
            assert len(header) == 3 and header[0] == entry.oid and header[1] == "blob"
            size = int(header[2])
            data = process.stdout.read(size)
            assert len(data) == size
            assert process.stdout.read(1) == b"\n"
            yield entry, data
    finally:
        process.stdin.close()
        return_code = process.wait()
        assert return_code == 0


def classify(path: str) -> str:
    p = path.replace("\\", "/")
    parts = p.split("/")
    lower_parts = [part.lower() for part in parts]
    name = parts[-1].lower()
    if p.startswith("third/"):
        return "vendor"
    if "/archive/" in f"/{p}" or p.startswith("algorithm/archive/") or p.startswith("hardware/archive/"):
        return "archive"
    if "references" in lower_parts[:-1] or "hardware_reference" in lower_parts[:-1] or p.startswith("references/"):
        return "reference"
    if any(part in {"tests", "test", "tb"} for part in parts[:-1]) or name.startswith("test_") or name.endswith("_test.py"):
        return "test"
    if any(part.lower() in {"artifacts", "reports", "results", "outputs"} for part in parts[:-1]):
        return "artifact"
    if p.startswith("algorithm/analysis/"):
        return "analysis"
    if p.startswith("docs/") or "/docs/" in f"/{p}" or Path(p).suffix.lower() in {".md", ".rst", ".pdf", ".tex"}:
        return "docs"
    return "maintained"


def decode_text(data: bytes) -> tuple[str | None, str]:
    if b"\x00" in data[:8192]:
        return None, "binary"
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding), "text"
        except UnicodeDecodeError:
            continue
    return None, "decode-error"


def line_counts(text: str) -> tuple[int, int, int]:
    lines = text.splitlines()
    blank = sum(not line.strip() for line in lines)
    comments = sum(line.lstrip().startswith(("#", "//", "/*", "*", "--")) for line in lines if line.strip())
    return len(lines), max(0, len(lines) - blank - comments), blank


def node_end(node: ast.AST) -> int:
    return int(getattr(node, "end_lineno", getattr(node, "lineno", 0)))


def branch_count(node: ast.AST) -> int:
    branch_nodes = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.IfExp, ast.comprehension, ast.BoolOp)
    if hasattr(ast, "Match"):
        branch_nodes = branch_nodes + (ast.Match, ast.match_case)  # type: ignore[attr-defined]
    return sum(isinstance(child, branch_nodes) for child in ast.walk(node))


def statement_count(node: ast.AST) -> int:
    return sum(isinstance(child, ast.stmt) for child in ast.walk(node))


def parameter_count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    args = node.args
    return len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs) + (1 if args.vararg else 0) + (1 if args.kwarg else 0)


def call_names(node: ast.AST) -> list[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            chain: list[str] = [func.attr]
            value = func.value
            while isinstance(value, ast.Attribute):
                chain.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                chain.append(value.id)
            names.add(".".join(reversed(chain)))
    return sorted(names)


class SemanticNormalizer(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = self.generic_visit(node)
        node.name = "__symbol__"
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node = self.generic_visit(node)
        node.name = "__symbol__"
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node


def semantic_hash(node: ast.AST) -> str:
    import copy

    normalized = SemanticNormalizer().visit(copy.deepcopy(node))
    dump = ast.dump(normalized, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(dump.encode()).hexdigest()[:20]


def nearest_symbol(line: int, spans: list[tuple[int, int, str]]) -> str:
    containing = [(end - start, name) for start, end, name in spans if start <= line <= end]
    return min(containing)[1] if containing else "<module>"


def python_scan(path: str, zone: str, text: str) -> tuple[list[SymbolRecord], list[Finding], list[tuple[str, str, int]], Counter[str], str]:
    symbols: list[SymbolRecord] = []
    findings: list[Finding] = []
    imports: list[tuple[str, str, int]] = []
    usages: Counter[str] = Counter()
    parse_status = "parsed"
    try:
        tree = ast.parse(text, filename=path, type_comments=True)
    except SyntaxError as exc:
        try:
            tree = ast.parse(text, filename=path, type_comments=False)
        except SyntaxError:
            findings.append(Finding(path, exc.lineno or 1, exc.lineno or 1, zone, "<module>", "python-syntax-error", "high" if zone in {"maintained", "test"} else "medium", "fact", "high", exc.msg))
            return symbols, findings, imports, usages, f"syntax-error:{exc.lineno or 0}"
        parse_status = f"parsed-invalid-type-comment:{exc.lineno or 0}"
        findings.append(Finding(path, exc.lineno or 1, exc.lineno or 1, zone, "<module>", "invalid-type-comment", "medium" if zone in {"maintained", "test"} else "low", "fact", "high", exc.text.strip() if exc.text else exc.msg))

    deep = zone in {"maintained", "analysis", "test"}
    if deep:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                usages[node.id] += 1
            elif isinstance(node, ast.Attribute):
                usages[node.attr] += 1
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((path, alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                imports.append((path, ("." * node.level) + (node.module or ""), node.lineno))

    qualnames: dict[ast.AST, str] = {}
    function_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    class_nodes: list[ast.ClassDef] = []

    def walk_body(nodes: Iterable[ast.stmt], prefix: str = "") -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = f"{prefix}.{node.name}" if prefix else node.name
                qualnames[node] = qual
                function_nodes.append(node)
                walk_body(node.body, qual)
            elif isinstance(node, ast.ClassDef):
                qual = f"{prefix}.{node.name}" if prefix else node.name
                qualnames[node] = qual
                class_nodes.append(node)
                walk_body(node.body, qual)

    walk_body(tree.body)

    for node in [*class_nodes, *function_nodes]:
        start = node.lineno
        end = node_end(node)
        is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        params = parameter_count(node) if is_function else 0
        calls = call_names(node) if deep and is_function else []
        symbols.append(SymbolRecord(
            path=path,
            zone=zone,
            kind=("async-function" if isinstance(node, ast.AsyncFunctionDef) else "function") if is_function else "class",
            qualname=qualnames[node],
            name=node.name,
            start_line=start,
            end_line=end,
            loc=end - start + 1,
            statement_count=statement_count(node) if deep else 0,
            branch_count=branch_count(node) if deep else 0,
            parameter_count=params,
            decorator_count=len(node.decorator_list),
            has_docstring=bool(ast.get_docstring(node, clean=False)),
            calls=";".join(calls),
            semantic_hash=semantic_hash(node) if is_function and deep and (end - start + 1) <= 500 else "",
        ))

    if not deep:
        return symbols, findings, imports, usages, parse_status

    spans = [(item.start_line, item.end_line, item.qualname) for item in symbols]
    active = deep
    for item in symbols:
        if item.kind.endswith("function") and item.loc > 120:
            findings.append(Finding(path, item.start_line, item.end_line, zone, item.qualname, "large-function", "high" if active else "low", "fact", "high", f"function spans {item.loc} lines"))
        elif item.kind.endswith("function") and item.loc > 60:
            findings.append(Finding(path, item.start_line, item.end_line, zone, item.qualname, "large-function", "medium" if active else "low", "fact", "high", f"function spans {item.loc} lines"))
        if item.kind == "class" and item.loc > 300:
            findings.append(Finding(path, item.start_line, item.end_line, zone, item.qualname, "large-class", "high" if active else "low", "fact", "high", f"class spans {item.loc} lines"))
        elif item.kind == "class" and item.loc > 150:
            findings.append(Finding(path, item.start_line, item.end_line, zone, item.qualname, "large-class", "medium" if active else "low", "fact", "high", f"class spans {item.loc} lines"))
        if item.branch_count > 25:
            findings.append(Finding(path, item.start_line, item.end_line, zone, item.qualname, "high-branch-density", "high" if active else "low", "inference", "medium", f"AST branch proxy={item.branch_count}"))
        elif item.branch_count > 12:
            findings.append(Finding(path, item.start_line, item.end_line, zone, item.qualname, "high-branch-density", "medium" if active else "low", "inference", "medium", f"AST branch proxy={item.branch_count}"))
        if item.parameter_count > 10:
            findings.append(Finding(path, item.start_line, item.end_line, zone, item.qualname, "too-many-parameters", "medium" if active else "low", "fact", "high", f"parameter count={item.parameter_count}"))

    for node in ast.walk(tree):
        line = int(getattr(node, "lineno", 1))
        end = node_end(node) or line
        symbol = nearest_symbol(line, spans)
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                findings.append(Finding(path, line, end, zone, symbol, "bare-except", "high" if active else "low", "fact", "high", "bare except catches BaseException"))
            elif isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}:
                findings.append(Finding(path, line, end, zone, symbol, "broad-except", "medium" if active else "low", "fact", "high", f"catches {node.type.id}"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = [*node.args.defaults, *[value for value in node.args.kw_defaults if value is not None]]
            if any(isinstance(value, (ast.List, ast.Dict, ast.Set)) for value in defaults):
                findings.append(Finding(path, line, end, zone, qualnames.get(node, node.name), "mutable-default", "high" if active else "low", "fact", "high", "mutable literal used as default argument"))
        elif isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            findings.append(Finding(path, line, end, zone, symbol, "wildcard-import", "medium" if active else "low", "fact", "high", f"from {node.module or ''} import *"))
        elif isinstance(node, ast.Global):
            findings.append(Finding(path, line, end, zone, symbol, "global-mutation-surface", "medium" if active else "low", "fact", "high", ",".join(node.names)))
        elif isinstance(node, ast.Call):
            target = ""
            if isinstance(node.func, ast.Name):
                target = node.func.id
            elif isinstance(node.func, ast.Attribute):
                target = node.func.attr
            if target in {"eval", "exec"}:
                findings.append(Finding(path, line, end, zone, symbol, "dynamic-execution", "high" if active else "low", "fact", "high", target))
            if target == "print" and zone == "maintained" and "/scripts/" not in f"/{path}":
                findings.append(Finding(path, line, end, zone, symbol, "library-print", "low", "inference", "medium", "print call in maintained non-script module"))
            if any(keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True for keyword in node.keywords):
                findings.append(Finding(path, line, end, zone, symbol, "subprocess-shell-true", "high" if active else "low", "fact", "high", "call uses shell=True"))

    absolute_path = re.compile(r"(?:[A-Za-z]:[\\/]|/(?:home|mnt|Users|tmp)/)")
    for lineno, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        symbol = nearest_symbol(lineno, spans)
        if re.search(r"\b(?:TODO|FIXME|HACK|XXX)\b", raw, re.IGNORECASE):
            findings.append(Finding(path, lineno, lineno, zone, symbol, "work-marker", "low", "fact", "high", stripped[:240]))
        if absolute_path.search(raw):
            findings.append(Finding(path, lineno, lineno, zone, symbol, "hard-coded-absolute-path", "high" if active else "low", "fact", "high", stripped[:240]))
        if len(raw) > 160 and active:
            findings.append(Finding(path, lineno, lineno, zone, symbol, "very-long-line", "low", "fact", "high", f"line length={len(raw)}"))

    return symbols, findings, imports, usages, parse_status


def nonpython_scan(path: str, zone: str, language: str, text: str) -> list[dict[str, object]]:
    patterns = NONPY_PATTERNS.get(language, [])
    records: list[dict[str, object]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for kind, pattern in patterns:
            match = pattern.search(line)
            if match:
                records.append({"path": path, "zone": zone, "language": language, "kind": kind, "name": match.group(1), "line": lineno})
                break
    return records


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", default="HEAD")
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", args.commit], cwd=root, text=True).strip()
    entries = git_entries(root, commit)

    files: list[FileRecord] = []
    symbols: list[SymbolRecord] = []
    findings: list[Finding] = []
    nonpython: list[dict[str, object]] = []
    imports: list[tuple[str, str, int]] = []
    name_usages: Counter[str] = Counter()

    for entry, data in committed_blobs(root, entries):
        rel = entry.path
        suffix = Path(rel).suffix.lower()
        language = LANGUAGES.get(suffix, "Other")
        zone = classify(rel)
        text: str | None = None
        status = "not-text"
        lines = code = blank = 0
        if suffix in TEXT_EXTENSIONS or suffix == "" or language != "Other":
            text, status = decode_text(data)
        if text is not None:
            lines, code, blank = line_counts(text)
        if suffix == ".py" and text is not None:
            py_symbols, py_findings, py_imports, usages, status = python_scan(rel, zone, text)
            symbols.extend(py_symbols)
            findings.extend(py_findings)
            imports.extend(py_imports)
            name_usages.update(usages)
        elif text is not None and language in NONPY_PATTERNS:
            nonpython.extend(nonpython_scan(rel, zone, language, text))
        files.append(FileRecord(
            rel,
            zone,
            language,
            suffix or "<none>",
            entry.size,
            entry.oid,
            hashlib.sha256(data).hexdigest(),
            lines,
            code,
            blank,
            status,
        ))

    duplicate_groups: list[dict[str, object]] = []
    by_hash: defaultdict[str, list[SymbolRecord]] = defaultdict(list)
    for symbol in symbols:
        if symbol.semantic_hash and symbol.statement_count >= 4:
            by_hash[symbol.semantic_hash].append(symbol)
    group_id = 0
    for digest, group in sorted(by_hash.items()):
        unique_locations = {(item.path, item.start_line) for item in group}
        if len(unique_locations) < 2:
            continue
        group_id += 1
        for item in group:
            duplicate_groups.append({
                "group_id": group_id,
                "semantic_hash": digest,
                "path": item.path,
                "zone": item.zone,
                "qualname": item.qualname,
                "line": item.start_line,
                "loc": item.loc,
            })

    dead_candidates: list[dict[str, object]] = []
    for symbol in symbols:
        if symbol.zone != "maintained" or symbol.kind == "class" or symbol.name.startswith("_"):
            continue
        references = name_usages[symbol.name]
        if references <= 1:
            dead_candidates.append({
                "path": symbol.path,
                "line": symbol.start_line,
                "qualname": symbol.qualname,
                "name": symbol.name,
                "name_reference_count": references,
                "claim_type": "inference",
                "confidence": "low",
                "caveat": "dynamic imports, external callers, and attribute aliases are not resolved",
            })

    duplicate_blobs: list[dict[str, object]] = []
    blobs_by_sha: defaultdict[str, list[FileRecord]] = defaultdict(list)
    for item in files:
        if item.bytes > 0:
            blobs_by_sha[item.sha256].append(item)
    blob_group_id = 0
    for digest, group in sorted(blobs_by_sha.items()):
        zones = sorted({item.zone for item in group})
        if len(zones) < 2:
            continue
        blob_group_id += 1
        for item in group:
            duplicate_blobs.append({
                "group_id": blob_group_id,
                "sha256": digest,
                "git_blob": item.git_blob,
                "bytes": item.bytes,
                "zones": ";".join(zones),
                "path": item.path,
                "zone": item.zone,
            })

    write_csv(out / "file-inventory.csv", (asdict(row) for row in files), list(FileRecord.__annotations__))
    write_csv(out / "python-symbols.csv", (asdict(row) for row in symbols), list(SymbolRecord.__annotations__))
    write_csv(out / "line-findings.csv", (asdict(row) for row in findings), list(Finding.__annotations__))
    write_csv(out / "nonpython-symbols.csv", nonpython, ["path", "zone", "language", "kind", "name", "line"])
    write_csv(out / "import-edges.csv", ({"path": p, "module": m, "line": line} for p, m, line in imports), ["path", "module", "line"])
    write_csv(out / "duplicate-functions.csv", duplicate_groups, ["group_id", "semantic_hash", "path", "zone", "qualname", "line", "loc"])
    write_csv(out / "duplicate-blobs.csv", duplicate_blobs, ["group_id", "sha256", "git_blob", "bytes", "zones", "path", "zone"])
    write_csv(out / "dead-code-candidates.csv", dead_candidates, ["path", "line", "qualname", "name", "name_reference_count", "claim_type", "confidence", "caveat"])

    summary = {
        "schema": "hbtxr-semantic-census.v1",
        "commit": commit,
        "source_view": "committed-blobs",
        "tracked_files": len(files),
        "total_bytes": sum(item.bytes for item in files),
        "text_lines": sum(item.lines for item in files),
        "code_lines_proxy": sum(item.code_lines for item in files),
        "files_by_zone": dict(Counter(item.zone for item in files)),
        "files_by_language": dict(Counter(item.language for item in files)),
        "python_parse_status": dict(Counter(item.parse_status for item in files if item.language == "Python")),
        "python_symbols": len(symbols),
        "python_symbol_kinds": dict(Counter(item.kind for item in symbols)),
        "nonpython_symbols": len(nonpython),
        "findings": len(findings),
        "findings_by_rule": dict(Counter(item.rule for item in findings)),
        "findings_by_zone": dict(Counter(item.zone for item in findings)),
        "findings_by_severity": dict(Counter(item.severity for item in findings)),
        "duplicate_groups": group_id,
        "duplicate_symbol_rows": len(duplicate_groups),
        "nonempty_cross_authority_duplicate_blob_groups": blob_group_id,
        "nonempty_cross_authority_duplicate_blob_rows": len(duplicate_blobs),
        "dead_code_candidates": len(dead_candidates),
        "method": {
            "source": "git ls-tree and git cat-file --batch committed blobs; working-tree content is never read",
            "python": "stdlib AST parse attempt for every committed .py blob; project modules are never imported",
            "non_python": "conservative declaration regexes for supported source languages",
            "line_counts": "decoded tracked text files; comment count is a language-agnostic proxy",
            "limitations": [
                "dynamic dispatch and runtime plugin registration are not resolved",
                "non-Python regex symbols are structural candidates, not compiler-verified declarations",
                "dead-code candidates are low-confidence name-reference heuristics",
                "duplicate-blobs.csv excludes zero-byte blobs because empty placeholders are not cleanup-equivalent content",
                "binary, dataset, image, notebook cell, and paper semantics are outside this static source census",
            ],
        },
    }
    (out / "semantic-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

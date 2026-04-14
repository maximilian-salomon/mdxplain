"""Lint docstrings in mdxplain/ for ReadTheDocs / Sphinx build errors.

Checks
------
- **E-PARSE**      — Python file cannot be parsed (SyntaxError).
- **E-SECTION**    — NumPy-style section underline length mismatch.
- **E-INDENT**     — List after paragraph without blank line.
- **E-STRONG**     — Unescaped ``**`` or ``*`` treated as RST markup.
- **E-REF**        — ``word_`` interpreted as RST hyperlink reference.
- **E-ROLE**       — Malformed Sphinx cross-reference role.
- **E-EMPTY-DOC**  — Public object has empty docstring.
- **W-NO-DOC**     — Public function / class / method has no docstring.

Usage::

    python check_docstring_errors.py                   # Full report (public only)
    python check_docstring_errors.py --summary         # Counts only
    python check_docstring_errors.py --errors-only     # Errors only
    python check_docstring_errors.py --include-private # Include private (_name) objects
    python check_docstring_errors.py --fail-on-error   # Exit 1 on errors
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUMPY_SECTIONS = {
    "attributes", "deprecated", "examples", "fields", "methods",
    "note", "notes", "other parameters", "parameters", "raises",
    "references", "returned metadata", "returns", "see also",
    "warning", "warnings", "yields",
}

_BASE_ROLES = {
    "attr", "class", "const", "data", "exc", "func", "meth", "mod",
    "obj", "ref", "term", "type", "doc", "download", "envvar",
    "guilabel", "kbd", "mailheader", "makevar", "manpage",
    "menuselection", "mimetype", "newsgroup", "option", "program",
    "regexp", "samp", "pep", "rfc",
}
SPHINX_ROLES = _BASE_ROLES | {f"py:{r}" for r in _BASE_ROLES}

RE_ROLE = re.compile(r":(\w[\w.]*):(`[^`]*`?)")
RE_UNDERLINE = re.compile(r"^(\s*)([-=~]{3,})\s*$")
RE_RST_REF = re.compile(r"(\w+)_(?=[^\w_*{<]|$)")

# Inline emphasis / strong: (finder, closer, marker, label)
_MARKUP_PATTERNS = [
    (re.compile(r"(?<!\\)\*\*(\w+)"),
     re.compile(r"\*\*"),
     "**", "strong"),
    (re.compile(r"(?<!\\)(?<!\*)\*(?!\*)(\w+)"),
     re.compile(r"(?<!\\)(?<!\*)\*(?!\*)"),
     "*", "emphasis"),
]

# Trailing emphasis / strong: word**, word*  (finder, opener, marker, label)
_TRAILING_MARKUP_PATTERNS = [
    (re.compile(r"(\w+)\*\*(?!\*)"),
     re.compile(r"(?<!\\)\*\*"),
     "**", "strong"),
    (re.compile(r"(\w+)\*(?!\*)"),
     re.compile(r"(?<!\\)(?<!\*)\*(?!\*)"),
     "*", "emphasis"),
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """A single linting finding."""

    file: str
    line: int
    code: str
    message: str
    object_name: str = ""
    is_public: bool = True

    @property
    def is_error(self) -> bool:
        return self.code.startswith("E-")

    def __str__(self) -> str:
        obj = f"  ({self.object_name})" if self.object_name else ""
        return f"{self.file}:{self.line}: [{self.code}]{obj} {self.message}"


@dataclass
class DocstringInfo:
    """Metadata about a single docstring extracted from the AST."""

    object_name: str
    docstring: str
    lineno: int          # definition line (def / class keyword)
    doc_lineno: int      # first line of docstring *content*
    is_public: bool
    _lines: list[str] | None = field(default=None, repr=False)

    @property
    def lines(self) -> list[str]:
        """Cached ``splitlines()`` of the docstring."""
        if self._lines is None:
            self._lines = self.docstring.splitlines()
        return self._lines

    def finding(self, line_offset: int, code: str, message: str) -> Finding:
        """Create a Finding anchored at ``doc_lineno + line_offset``."""
        return Finding(
            file="", line=self.doc_lineno + line_offset,
            code=code, message=message, object_name=self.object_name,
            is_public=self.is_public,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_project_root() -> Path:
    """Return the project root (directory containing ``pyproject.toml``)."""
    here = Path(__file__).resolve().parent
    root = here.parent.parent
    if not (root / "pyproject.toml").exists():
        sys.exit(f"Cannot find project root from {here}")
    return root


def _iter_py_files(pkg_dir: Path) -> Iterator[Path]:
    """Yield all ``.py`` files under *pkg_dir*, skipping hidden dirs."""
    for dirpath, dirnames, filenames in os.walk(pkg_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "__"))]
        for fn in sorted(filenames):
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# AST docstring extraction
# ---------------------------------------------------------------------------


def _content_start_line(source_lines: list[str], node_lineno: int) -> int:
    """1-based line where docstring text begins (after opening quotes)."""
    after = source_lines[node_lineno - 1].lstrip()
    for q in ('"""', "'''"):
        if after.startswith(q):
            after = after[len(q):]
            break
    return node_lineno if after.strip() else node_lineno + 1


def _extract_docstrings(
    tree: ast.Module, source_lines: list[str],
) -> list[DocstringInfo]:
    """Walk the AST and collect docstrings with metadata."""
    results: list[DocstringInfo] = []

    def _visit(node: ast.AST, stack: list[str], in_function: bool = False) -> None:
        is_module = isinstance(node, ast.Module)
        is_func = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        is_def = is_func or isinstance(node, ast.ClassDef)
        if not (is_module or is_def):
            return

        name = node.name if is_def else "<module>"  # type: ignore[union-attr]
        qname = ".".join(stack + [name]) if is_def else "<module>"
        public = not name.startswith("_") if is_def else True
        ds = ast.get_docstring(node)

        if ds is not None:
            doc_node = node.body[0].value  # type: ignore[union-attr]
            results.append(DocstringInfo(
                object_name=qname, docstring=ds,
                lineno=getattr(node, "lineno", 1),
                doc_lineno=_content_start_line(source_lines, doc_node.lineno),
                is_public=public,
            ))
        elif is_def and public and not name.startswith("__") and not in_function:
            results.append(DocstringInfo(
                object_name=qname, docstring="",
                lineno=node.lineno,  # type: ignore[union-attr]
                doc_lineno=node.lineno,  # type: ignore[union-attr]
                is_public=True,
            ))

        child_stack = stack + [name] if isinstance(node, ast.ClassDef) else stack
        child_in_function = in_function or is_func
        for child in ast.iter_child_nodes(node):
            _visit(child, child_stack, child_in_function)

    _visit(tree, [])
    return results


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_empty(info: DocstringInfo) -> list[Finding]:
    """E-EMPTY-DOC / W-NO-DOC."""
    if not info.docstring:
        if info.is_public and info.object_name != "<module>":
            return [Finding(
                file="", line=info.lineno, code="W-NO-DOC",
                message="Public object has no docstring.",
                object_name=info.object_name,
            )]
        return []
    if not info.docstring.strip():
        return [info.finding(0, "E-EMPTY-DOC",
                             "Docstring is empty / whitespace-only.")]
    return []


def _check_sections(info: DocstringInfo) -> list[Finding]:
    """E-SECTION: underline length != header length."""
    findings: list[Finding] = []
    for i, line in enumerate(info.lines):
        m = RE_UNDERLINE.match(line)
        if not m or i == 0:
            continue
        header = info.lines[i - 1].strip()
        if not header:
            continue
        header_lower = header.rstrip(":.").lower()
        if header_lower not in NUMPY_SECTIONS:
            continue
        ulen = len(m.group(2))
        if ulen != len(header) and ulen != len(header_lower):
            findings.append(info.finding(i, "E-SECTION",
                f"Section underline length ({ulen}) does not match "
                f"header '{header}' (length {len(header)})."))
    return findings


def _check_roles(info: DocstringInfo) -> list[Finding]:
    """E-ROLE: malformed Sphinx roles."""
    findings: list[Finding] = []
    for i, line in enumerate(info.lines):
        for m in RE_ROLE.finditer(line):
            if not m.group(2).endswith("`"):
                findings.append(info.finding(i, "E-ROLE",
                    f"Unclosed backtick in Sphinx role "
                    f":{m.group(1)}:{m.group(2)}..."))
        for m in re.finditer(r":(\w[\w.]*):(\w+)", line):
            if m.group(1).lower() in SPHINX_ROLES:
                findings.append(info.finding(i, "E-ROLE",
                    f"Sphinx role :{m.group(1)}: is not followed by "
                    f"a backtick-quoted argument."))
    return findings


def _check_inline_markup(info: DocstringInfo) -> list[Finding]:
    """E-STRONG: unescaped ``**`` or ``*`` that RST treats as markup."""
    findings: list[Finding] = []
    for i, line in enumerate(info.lines):
        stripped = line.strip()
        if stripped.startswith((">>>", "...")):
            continue
        # Remove backtick-quoted spans to avoid false positives
        clean = re.sub(r"``[^`]*``", " ", line)
        clean = re.sub(r"`[^`]*`", " ", clean)
        # Remove leading RST list markers (* , - , + ) so they don't
        # interfere with emphasis / strong detection
        clean = re.sub(r"^(\s*)[-*+]\s+", lambda m: " " * len(m.group()), clean)

        for finder, closer, marker, label in _MARKUP_PATTERNS:
            for m in finder.finditer(clean):
                if closer.search(clean[m.end():]):
                    continue  # properly closed on same line
                word = m.group(1)
                esc = marker.replace("*", "\\*") + word
                findings.append(info.finding(i, "E-STRONG",
                    f"Unescaped inline {label} markup '{marker}{word}' "
                    f"— escape as '{esc}' or wrap in backticks."))

        for finder, opener, marker, label in _TRAILING_MARKUP_PATTERNS:
            for m in finder.finditer(clean):
                if opener.search(clean[:m.start()]):
                    continue  # properly opened on same line
                word = m.group(1)
                # RST does not start/end emphasis after _ (word character)
                if word.endswith("_"):
                    continue
                esc = word + marker.replace("*", "\\*")
                findings.append(info.finding(i, "E-STRONG",
                    f"Unescaped inline {label} markup '{word}{marker}' "
                    f"— escape as '{esc}' or wrap in backticks."))
    return findings


def _check_rst_reference(info: DocstringInfo) -> list[Finding]:
    """E-REF: word_ patterns that RST interprets as hyperlink references."""
    findings: list[Finding] = []
    for i, line in enumerate(info.lines):
        stripped = line.strip()
        if stripped.startswith((">>>", "...")):
            continue
        # Remove backtick-quoted spans to avoid false positives
        clean = re.sub(r"``[^`]*``", " ", line)
        clean = re.sub(r"`[^`]*`", " ", clean)
        # Remove Sphinx roles :role:`target` (already consumed above)
        clean = re.sub(r":\w[\w.]*:`[^`]*`", " ", clean)

        for m in RE_RST_REF.finditer(clean):
            word = m.group(1)
            # Skip dunder names (__word__) — RST won't treat these as refs
            if word.startswith("__") or word.startswith("_"):
                continue
            findings.append(info.finding(i, "E-REF",
                f"'{word}_' is interpreted as RST hyperlink reference "
                f"(Sphinx error: 'Unknown target name') "
                f"— wrap in backticks or escape as '{word}\\_'."))
    return findings


def _check_unexpected_indent(info: DocstringInfo) -> list[Finding]:
    """E-INDENT: list item after paragraph without blank line."""
    findings: list[Finding] = []
    prev_blank = True

    for i, curr in enumerate(info.lines):
        if not curr.strip():
            prev_blank = True
            continue
        if (not prev_blank and i > 0
                and re.match(r"\s*[-*+]\s+\S", curr)):
            prev = info.lines[i - 1]
            if (prev.strip()
                    and not re.match(r"\s*[-*+]\s+\S", prev)
                    and not RE_UNDERLINE.match(prev)
                    and len(curr) - len(curr.lstrip())
                    >= len(prev) - len(prev.lstrip())):
                findings.append(info.finding(i, "E-INDENT",
                    "List item directly follows non-list text without a "
                    "separating blank line — causes 'Unexpected indentation' "
                    "in RST/Sphinx."))
        prev_blank = False

    return findings


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    _check_empty,
    _check_sections,
    _check_roles,
    _check_inline_markup,
    _check_rst_reference,
    _check_unexpected_indent,
]


def lint_file(py_path: Path, root: Path) -> list[Finding]:
    """Lint all docstrings in a single Python file."""
    rel = _relative(py_path, root)
    try:
        source = py_path.read_text(encoding="utf-8")
    except Exception as exc:
        return [Finding(file=rel, line=1, code="E-PARSE",
                        message=f"Cannot read file: {exc}")]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source, filename=str(py_path))
    except SyntaxError as exc:
        return [Finding(file=rel, line=exc.lineno or 1, code="E-PARSE",
                        message=f"SyntaxError: {exc.msg}")]

    findings: list[Finding] = []
    for info in _extract_docstrings(tree, source.splitlines()):
        for check in ALL_CHECKS:
            for f in check(info):
                f.file = rel
                findings.append(f)
    return findings


def lint_package(root: Path) -> list[Finding]:
    """Lint the entire ``mdxplain/`` package."""
    pkg_dir = root / "mdxplain"
    if not pkg_dir.is_dir():
        sys.exit(f"Package directory not found: {pkg_dir}")
    findings: list[Finding] = []
    for py_path in _iter_py_files(pkg_dir):
        findings.extend(lint_file(py_path, root))
    findings.sort(key=lambda f: (f.file, f.line))
    return findings


# ---------------------------------------------------------------------------
# Reporting & CLI
# ---------------------------------------------------------------------------


def _short_message(f: Finding) -> str:
    """One-line compact description for a finding."""
    obj = f.object_name
    if f.code == "E-INDENT":
        return f"List after text without blank line  ({obj})"
    if f.code == "E-SECTION":
        return f"{f.message}  ({obj})"
    if f.code == "E-STRONG":
        # Extract the quoted markup from the full message
        m = re.search(r"markup '([^']+)'", f.message)
        token = m.group(1) if m else "?"
        return f"Unescaped {token}  ({obj})"
    if f.code == "E-REF":
        m = re.search(r"'([^']+_)'", f.message)
        token = m.group(1) if m else "?"
        return f"RST reference '{token}'  ({obj})"
    if f.code == "E-ROLE":
        return f"{f.message}  ({obj})"
    if f.code == "W-NO-DOC":
        return f"Missing docstring  ({obj})"
    if f.code == "E-EMPTY-DOC":
        return f"Empty docstring  ({obj})"
    return f"{f.message}  ({obj})"


def _print_report(findings: list[Finding], *, summary_only: bool = False) -> None:
    errors = [f for f in findings if f.is_error]
    warnings = [f for f in findings if not f.is_error]

    counts: dict[str, int] = {}
    for f in findings:
        counts[f.code] = counts.get(f.code, 0) + 1

    if not summary_only:
        # Group by file, then print per-file blocks
        from itertools import groupby

        for section_label, section_items in [
            ("ERRORS (will likely break Sphinx / ReadTheDocs build)", errors),
            ("WARNINGS (may cause incomplete documentation)", warnings),
        ]:
            if not section_items:
                continue
            sep = "=" if section_items is errors else "-"
            print(f"\n{sep * 72}")
            print(section_label)
            print(sep * 72)

            for file_path, group in groupby(section_items, key=lambda f: f.file):
                print(f"\n  {file_path}")
                for f in group:
                    print(f"    :{f.line:<5d} [{f.code}]  {_short_message(f)}")

    # Summary
    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    for code in sorted(counts):
        kind = "ERROR  " if code.startswith("E-") else "WARNING"
        print(f"  {kind}  {code:<20s} {counts[code]:>5d}")
    print(f"\n  Total errors:   {len(errors)}")
    print(f"  Total warnings: {len(warnings)}")
    print(f"  Total findings: {len(findings)}")
    if not findings:
        print("\n  No issues found — docstrings look clean!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lint mdxplain/ docstrings for ReadTheDocs build issues.",
    )
    parser.add_argument("--summary", action="store_true",
                        help="Print only the summary.")
    parser.add_argument("--fail-on-error", action="store_true",
                        help="Exit with code 1 if any errors are found.")
    parser.add_argument("--errors-only", action="store_true",
                        help="Show only errors (E-*), suppress warnings.")
    parser.add_argument("--include-private", action="store_true",
                        help="Include findings in private (_name) objects "
                             "(hidden by default).")
    args = parser.parse_args()

    root = find_project_root()
    findings = lint_package(root)
    if not args.include_private:
        findings = [f for f in findings if f.is_public]
    if args.errors_only:
        findings = [f for f in findings if f.is_error]
    _print_report(findings, summary_only=args.summary)
    if args.fail_on_error and any(f.is_error for f in findings):
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Comparison script: mdxplain/ modules vs. docs/api/ RST documentation.

Compares the Python modules under mdxplain/ with the ``.. automodule::``
references in the RST files under docs/api/.

Conventions:
  - __init__.py files are referenced as packages (folder path),
    e.g. mdxplain/analysis/__init__.py -> mdxplain.analysis
  - If a folder contains only one script + __init__.py, the __init__.py
    (i.e. the package) is NOT referenced separately — UNLESS the __init__.py
    has a detailed (multi-line) docstring.

Output:
  1. Modules in mdxplain/ that are missing from docs/api/
  2. References in docs/api/ that no longer exist in mdxplain/
  3. (optional, --show-inits) Detailed overview of all __init__.py packages:
     which have a detailed docstring and which are ignored.

Usage:
  python check_api_coverage.py                # Standard output
  python check_api_coverage.py --show-inits   # + init detail overview
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

SEPARATOR = "=" * 70


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class InitDetail:
    """Metadata for a __init__.py file."""

    package: str
    category: str  # "multi" | "single_detailed" | "single_trivial"
    included: bool
    detailed_docstring: bool
    num_scripts: int
    num_subpackages: int
    name_mismatch: bool = False


@dataclass
class ComparisonResult:
    """Result of the module/API comparison."""

    mdxplain_modules: set[str]
    api_modules: set[str]
    init_details: list[InitDetail]

    missing_in_api: list[str] = field(default_factory=list)
    truly_missing: list[str] = field(default_factory=list)
    api_refs_ignored_inits: list[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(self.missing_in_api or self.truly_missing)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def find_project_root() -> Path:
    """Find the project root (directory containing pyproject.toml)."""
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent.parent  # docs/dev -> root
    if (root / "pyproject.toml").exists():
        return root
    # Fallback: search from cwd
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    print("ERROR: Could not find project root (with pyproject.toml).")
    sys.exit(1)


def _has_detailed_docstring(init_path: Path) -> bool:
    """Check whether a __init__.py has a detailed (multi-line) docstring."""
    try:
        tree = ast.parse(init_path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return False
    docstring = ast.get_docstring(tree)
    if not docstring:
        return False
    content_lines = [ln for ln in docstring.strip().splitlines() if ln.strip()]
    return len(content_lines) > 1


# Folder names that are pure data-container conventions where the script
# name is always domain-specific (e.g. entities/feature_data.py).
_DATA_CONTAINER_FOLDERS: set[str] = {"entities"}

# Generic folder names where the script suffix alone does not guarantee
# a semantic relationship to the parent package.  For these folders an
# additional *parent-context* check is applied.
_PARENT_CHECK_FOLDERS: set[str] = {"helper"}


def _parent_context_in_script(parent_name: str, script_name: str) -> bool:
    """Return True if *script_name* contains recognisable context from *parent_name*.

    Handles common variations:
      - Exact substring (``membership`` in ``membership_helper``).
      - Singular form (``plots`` -> ``plot``).
      - Cross-word matching: any significant word (>=5 chars) from one
        name appears in the other (``cluster`` <-> ``clustering``).
    """
    if parent_name in script_name:
        return True
    # Singular form of parent
    if parent_name.endswith("s"):
        singular = parent_name[:-1]
        if singular and singular in script_name:
            return True
    # Cross-word matching between parent parts and script parts
    _MIN_WORD = 5
    script_parts = [p for p in script_name.split("_") if len(p) >= _MIN_WORD]
    parent_parts = [p for p in parent_name.split("_") if len(p) >= _MIN_WORD]
    for sp in script_parts:
        if sp in parent_name:
            return True
    for pp in parent_parts:
        if pp in script_name:
            return True
    return False


def _is_name_mismatch(
    script_name: str,
    folder_name: str,
    parent_name: str = "",
) -> bool:
    """Decide whether *script_name* truly mismatches *folder_name*.

    A mismatch means the script cannot be found via the folder name alone,
    so the package-level RST entry is needed for navigation.

    Standard patterns that are NOT considered mismatches:
      - Script name contains the folder name
        (e.g. ``analysis_manager`` contains ``manager``).
      - Script name contains the singular form (trailing ``s`` removed)
        (e.g. ``feature_importance_add_service`` contains ``service``).
      - Folder is a known data-container name (``entities``) where
        scripts always have domain-specific names by convention.

    For folders in ``_PARENT_CHECK_FOLDERS`` (e.g. ``helper``) an extra
    check verifies that the script name also relates to the *parent*
    package context (e.g. ``block_optimizer_helper`` under
    ``membership/helper`` is a mismatch because ``membership`` does not
    appear in the script name).
    """
    if folder_name in _DATA_CONTAINER_FOLDERS:
        return False

    folder_found = folder_name in script_name
    if not folder_found and folder_name.endswith("s"):
        singular = folder_name[:-1]
        folder_found = bool(singular and singular in script_name)

    if not folder_found:
        return True  # script has no trace of the folder name at all

    # For generic folders, additionally verify parent context
    if folder_name in _PARENT_CHECK_FOLDERS and parent_name:
        return not _parent_context_in_script(parent_name, script_name)

    return False


def _classify_init(
    init_path: Path,
    package_dotted: str,
    num_scripts: int,
    num_subpackages: int,
    script_names: list[str] | None = None,
) -> InitDetail:
    """Classify a __init__.py and decide whether it should be included.

    A single-script package whose only script has a genuinely different
    name than the folder is promoted to ``single_detailed`` even without
    a detailed docstring, because the package-level RST entry is needed
    for navigation.
    """
    detailed = _has_detailed_docstring(init_path)
    is_single = num_scripts == 1 and num_subpackages == 0

    folder_name = init_path.parent.name
    parts = package_dotted.split(".")
    parent_name = parts[-2] if len(parts) >= 2 else ""
    name_mismatch = (
        is_single
        and script_names is not None
        and _is_name_mismatch(script_names[0], folder_name, parent_name)
    )

    if not is_single:
        category = "multi"
    elif detailed or name_mismatch:
        category = "single_detailed"
    else:
        category = "single_trivial"

    included = not is_single or detailed or name_mismatch

    return InitDetail(
        package=package_dotted,
        category=category,
        included=included,
        detailed_docstring=detailed,
        num_scripts=num_scripts,
        num_subpackages=num_subpackages,
        name_mismatch=name_mismatch,
    )


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def collect_mdxplain_modules(mdxplain_dir: Path) -> tuple[set[str], list[InitDetail]]:
    """Collect all Python modules under mdxplain/ as dotted-path strings.

    Returns:
        modules: Set of all module paths.
        init_details: List with info for each __init__.py.
    """
    modules: set[str] = set()
    init_details: list[InitDetail] = []
    package_root = mdxplain_dir.parent

    for dirpath, dirnames, filenames in os.walk(mdxplain_dir):
        dirnames[:] = [
            d for d in dirnames if d != "__pycache__" and not d.startswith(".")
        ]
        current = Path(dirpath)
        py_files = [f for f in filenames if f.endswith(".py")]
        if not py_files:
            continue

        scripts = [f for f in py_files if f != "__init__.py"]
        package_dotted = ".".join(current.relative_to(package_root).parts)
        subpackages = [d for d in dirnames if (current / d / "__init__.py").exists()]

        if "__init__.py" in py_files:
            script_stems = [s.removesuffix(".py") for s in scripts]
            detail = _classify_init(
                current / "__init__.py", package_dotted, len(scripts), len(subpackages),
                script_names=script_stems,
            )
            init_details.append(detail)
            if detail.included:
                modules.add(package_dotted)

        for script in scripts:
            modules.add(f"{package_dotted}.{script.removesuffix('.py')}")

    return modules, init_details


def collect_api_automodules(api_dir: Path) -> set[str]:
    """Collect all ``.. automodule::`` references from RST files under docs/api/."""
    pattern = re.compile(r"\.\.\s+automodule::\s+(.+)")
    modules: set[str] = set()
    for rst_file in api_dir.rglob("*.rst"):
        for line in rst_file.read_text(encoding="utf-8").splitlines():
            if match := pattern.match(line.strip()):
                modules.add(match.group(1).strip())
    return modules


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_modules(
    mdxplain_modules: set[str],
    api_modules: set[str],
    init_details: list[InitDetail],
) -> ComparisonResult:
    """Compare collected modules with API references.

    Splits the differences into:
      - missing_in_api: Present in mdxplain/ but not documented.
      - truly_missing: Referenced in docs/api/ but no module exists.
      - api_refs_ignored_inits: Referenced in docs/api/ but intentionally
        ignored init package (single-script, trivial docstring).
    """
    ignored = {d.package for d in init_details if not d.included}
    raw_missing_in_mdxplain = api_modules - mdxplain_modules

    return ComparisonResult(
        mdxplain_modules=mdxplain_modules,
        api_modules=api_modules,
        init_details=init_details,
        missing_in_api=sorted(mdxplain_modules - api_modules),
        truly_missing=sorted(raw_missing_in_mdxplain - ignored),
        api_refs_ignored_inits=sorted(raw_missing_in_mdxplain & ignored),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_section(title: str, items: list[str]) -> None:
    """Print a section with title and item listing."""
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    for item in items or ["(None)"]:
        print(f"  - {item}" if items else f"  {item}")
    print()


def _inits_by_category(
    details: list[InitDetail], category: str,
) -> list[InitDetail]:
    """Filter and sort init details by category."""
    return sorted(
        (d for d in details if d.category == category),
        key=lambda d: d.package,
    )


def _format_multi(d: InitDetail, name_width: int = 0) -> str:
    ds = "detailed" if d.detailed_docstring else "trivial"
    padded = d.package.ljust(name_width)
    return (
        f"{padded}  ({d.num_scripts} script(s), "
        f"{d.num_subpackages} subpkg(s), docstring: {ds})"
    )


def _format_trivial(d: InitDetail, api_refs: set[str]) -> str:
    suffix = "  WARNING: referenced in API" if d.package in api_refs else ""
    return f"{d.package}{suffix}"


def print_report(result: ComparisonResult, *, show_inits: bool = False) -> None:
    """Print the full comparison report."""
    # Header
    print(SEPARATOR)
    print("COMPARISON: mdxplain/ modules  <->  docs/api/ documentation")
    print(SEPARATOR)
    print()
    print(f"  Module in mdxplain/:     {len(result.mdxplain_modules)}")
    print(f"  References in docs/api/: {len(result.api_modules)}")
    print()

    # Core sections
    _print_section(
        f"PRESENT IN mdxplain/ BUT MISSING FROM docs/api/ "
        f"({len(result.missing_in_api)})",
        result.missing_in_api,
    )
    _print_section(
        f"REFERENCED IN docs/api/ BUT NOT PRESENT IN mdxplain/ "
        f"({len(result.truly_missing)})",
        result.truly_missing,
    )

    if result.api_refs_ignored_inits:
        _print_section(
            f"REFERENCED IN docs/api/ BUT IGNORED INIT PACKAGE "
            f"| single-script, trivial docstring "
            f"({len(result.api_refs_ignored_inits)})",
            [
                f"{pkg}  (package exists, __init__.py has only a trivial docstring)"
                for pkg in result.api_refs_ignored_inits
            ],
        )

    # Optional init detail overview
    if show_inits:
        _print_init_details(result)

    # Summary
    _print_summary(result)


def _print_init_details(result: ComparisonResult) -> None:
    """Print the three init category sections."""
    details = result.init_details
    api_refs = set(result.api_refs_ignored_inits)

    multi_details = _inits_by_category(details, "multi")
    multi_width = max((len(d.package) for d in multi_details), default=0)

    sections: list[tuple[str, str, list[str]]] = [
        (
            "multi",
            "__init__.py MULTI-SCRIPT PACKAGES | multiple scripts/subpackages",
            [_format_multi(d, multi_width) for d in multi_details],
        ),
        (
            "single_detailed",
            "__init__.py SINGLE-SCRIPT PACKAGES WITH DETAILED DOCSTRING "
            "OR NAME MISMATCH | included in API",
            [
                f"{d.package}  (name mismatch)" if d.name_mismatch
                else d.package
                for d in _inits_by_category(details, "single_detailed")
            ],
        ),
        (
            "single_trivial",
            "__init__.py SINGLE-SCRIPT PACKAGES WITH TRIVIAL DOCSTRING | ignored",
            [
                _format_trivial(d, api_refs)
                for d in _inits_by_category(details, "single_trivial")
            ],
        ),
    ]

    for _cat, title, items in sections:
        _print_section(f"{title} ({len(items)})", items)


def _print_summary(result: ComparisonResult) -> None:
    """Print the comparison summary."""
    print(SEPARATOR)
    total = len(result.missing_in_api) + len(result.truly_missing)
    if total == 0 and not result.api_refs_ignored_inits:
        print("RESULT: Full match!")
    else:
        parts = []
        if total:
            parts.append(f"{total} discrepancy/ies")
        if result.api_refs_ignored_inits:
            parts.append(
                f"{len(result.api_refs_ignored_inits)} API ref(s) to "
                f"ignored init packages"
            )
        print(f"RESULT: {', '.join(parts)}.")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Comparison: mdxplain/ modules vs. docs/api/ documentation",
    )
    parser.add_argument(
        "--show-inits",
        action="store_true",
        help="Show a detailed overview of all __init__.py packages "
             "(detailed docstring vs. ignored).",
    )
    args = parser.parse_args()

    root = find_project_root()
    for name, subdir in [("mdxplain", "mdxplain"), ("docs/api", "docs/api")]:
        if not (root / subdir).is_dir():
            print(f"ERROR: {name} directory not found: {root / subdir}")
            sys.exit(1)

    modules, init_details = collect_mdxplain_modules(root / "mdxplain")
    api_modules = collect_api_automodules(root / "docs" / "api")

    result = compare_modules(modules, api_modules, init_details)
    print_report(result, show_inits=args.show_inits)

    return 1 if result.has_issues else 0


if __name__ == "__main__":
    sys.exit(main())

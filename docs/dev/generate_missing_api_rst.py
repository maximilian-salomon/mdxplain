"""Generate missing API RST files based on check_api_coverage.py results.

Detects modules in mdxplain/ that have no ``.. automodule::`` reference in
docs/api/ and generates the appropriate RST stubs.  Also identifies parent
toctrees that need to be updated to include the new entries.

Three RST patterns are recognised:

1. **Multi** — package with multiple scripts / sub-packages:
   ``automodule`` on the package (no ``:members:``) + ``toctree`` listing
   all children.

2. **Single-trivial** — package with exactly one script and a trivial
   (one-line or empty) ``__init__.py`` docstring:
   GitHub link + ``automodule`` on the *script* (with ``:members:``).
   The ``__init__.py`` is **not** referenced.

3. **Single-detailed** — package with exactly one script and a detailed
   (multi-line) ``__init__.py`` docstring:
   ``automodule`` on the *package* (without ``:members:``) **+** a
   sub-section with ``automodule`` on the script (with ``:members:``).

Usage::

    python generate_missing_api_rst.py                 # Dry-run
    python generate_missing_api_rst.py --write         # Write files + show toctree patches
    python generate_missing_api_rst.py --write --patch # Write files + auto-patch toctrees
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Import helpers from the companion script
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_api_coverage import (  # noqa: E402
    ComparisonResult,
    InitDetail,
    collect_api_automodules,
    collect_mdxplain_modules,
    compare_modules,
    find_project_root,
)

GITHUB_BASE = "https://github.com/maximilian-salomon/mdxplain/blob/main"
SEPARATOR = "=" * 70


# ---------------------------------------------------------------------------
# Title helpers
# ---------------------------------------------------------------------------


def _humanise(name: str) -> str:
    """``some_thing`` → ``Some Thing``."""
    return name.replace("_", " ").title()


def _module_to_title(module_path: str) -> str:
    """Derive a readable RST title from a dotted module path.

    Uses the *last two* segments when the last segment is a very generic
    word (helper, manager, entities, services …) so the title stays
    informative.
    """
    parts = module_path.split(".")
    generic = {"helper", "manager", "entities", "services", "interfaces", "utils"}
    if len(parts) >= 2 and parts[-1] in generic:
        return _humanise(f"{parts[-2]}_{parts[-1]}")
    return _humanise(parts[-1])


def _rst_heading(title: str, char: str = "=") -> str:
    """Return an RST section heading (title + underline)."""
    return f"{title}\n{char * len(title)}"


# ---------------------------------------------------------------------------
# RST content generators
# ---------------------------------------------------------------------------


def generate_leaf_rst(module_path: str) -> str:
    """**Type 2 leaf**: GitHub link + ``automodule`` with ``:members:``."""
    title = _module_to_title(module_path)
    file_path = module_path.replace(".", "/") + ".py"
    return (
        f"{_rst_heading(title)}\n"
        f"\n"
        f"GitHub Link to `Code <{GITHUB_BASE}/{file_path}>`_.\n"
        f"\n"
        f".. automodule:: {module_path}\n"
        f"   :members:\n"
        f"   :special-members: __init__\n"
        f"   :undoc-members:\n"
    )


def generate_multi_rst(
    module_path: str,
    children: list[str],
    *,
    maxdepth: int = 1,
    titlesonly: bool = False,
) -> str:
    """**Type 1**: ``automodule`` on package + ``toctree`` of children."""
    title = _module_to_title(module_path)
    toctree_opts = f"   :maxdepth: {maxdepth}\n"
    if titlesonly:
        toctree_opts += "   :titlesonly:\n"
    children_str = "\n".join(f"   {c}" for c in sorted(children))
    return (
        f"{_rst_heading(title)}\n"
        f"\n"
        f".. automodule:: {module_path}\n"
        f"\n"
        f".. toctree::\n"
        f"{toctree_opts}\n"
        f"{children_str}\n"
    )


def generate_single_detailed_rst(
    package_path: str,
    script_module: str,
) -> str:
    """**Type 3**: ``automodule`` on package + sub-section for the script."""
    title = _module_to_title(package_path)
    script_title = _module_to_title(script_module)
    file_path = script_module.replace(".", "/") + ".py"
    return (
        f"{_rst_heading(title)}\n"
        f"\n"
        f"GitHub Link to `Code <{GITHUB_BASE}/{file_path}>`_.\n"
        f"\n"
        f".. automodule:: {package_path}\n"
        f"\n"
        f"{_rst_heading(script_title, '-')}\n"
        f"\n"
        f".. automodule:: {script_module}\n"
        f"   :members:\n"
        f"   :special-members: __init__\n"
        f"   :undoc-members:\n"
    )


# ---------------------------------------------------------------------------
# Locate the correct docs/api/ sub-directory for a module
# ---------------------------------------------------------------------------


def _find_rst_dir(module_path: str, api_dir: Path) -> Path:
    """Walk the docs/api/ tree to find the deepest matching sub-folder.

    The first segment after ``mdxplain`` maps to a top-level folder.
    Deeper segments are matched against existing sub-folders (including a
    ``+s`` plural variant such as ``plot_type`` → ``plot_types``).
    """
    parts = module_path.split(".")
    if len(parts) < 2:
        return api_dir

    current = api_dir / parts[1]  # e.g. docs/api/clustering
    for part in parts[2:]:
        # Check exact match or plural
        candidate = current / part
        candidate_plural = current / (part + "s")
        if candidate.is_dir():
            current = candidate
        elif candidate_plural.is_dir():
            current = candidate_plural
        else:
            break
    return current


# ---------------------------------------------------------------------------
# Determine what needs to happen for every missing module
# ---------------------------------------------------------------------------


def _discover_children(pkg_dir: Path, module_path: str) -> list[str]:
    """List direct children (sub-packages + scripts) of a package dir."""
    children: list[str] = []
    for item in sorted(pkg_dir.iterdir()):
        if item.name.startswith(".") or item.name == "__pycache__":
            continue
        if item.is_dir() and (item / "__init__.py").exists():
            children.append(f"{module_path}.{item.name}")
        elif item.is_file() and item.suffix == ".py" and item.name != "__init__.py":
            children.append(f"{module_path}.{item.stem}")
    return children


def _find_single_script(pkg_dir: Path) -> str | None:
    """Return the stem of the single .py script in *pkg_dir* (excl. __init__)."""
    scripts = [
        f.stem
        for f in pkg_dir.iterdir()
        if f.is_file() and f.suffix == ".py" and f.name != "__init__.py"
    ]
    return scripts[0] if len(scripts) == 1 else None


# ---------------------------------------------------------------------------
# Action dataclass
# ---------------------------------------------------------------------------


@dataclass
class Action:
    """A single RST generation / update action."""

    type: str  # "leaf" | "multi" | "single_detailed"
    module: str
    rst_dir: Path
    rst_file: str
    content: str
    parent_module: str | None = None
    existing_rst: Path | None = None

    @property
    def is_replacement(self) -> bool:
        return self.existing_rst is not None


# ---------------------------------------------------------------------------
# Action builders (one per type)
# ---------------------------------------------------------------------------


def _action_single_detailed(
    module_path: str, pkg_dir: Path, rst_dir: Path, rst_path: Path,
) -> Action | None:
    """Build action for a single-script package with detailed docstring."""
    script_stem = _find_single_script(pkg_dir)
    if script_stem is None:
        return None
    script_module = f"{module_path}.{script_stem}"
    is_replacement = rst_path.exists()
    return Action(
        type="single_detailed",
        module=module_path,
        rst_dir=rst_dir,
        rst_file=f"{module_path}.rst",
        content=generate_single_detailed_rst(module_path, script_module),
        existing_rst=rst_path if is_replacement else None,
        # New files need a toctree entry in the parent RST.
        parent_module=None if is_replacement else module_path.rsplit(".", 1)[0],
    )


def _action_multi(
    module_path: str, pkg_dir: Path, rst_dir: Path, rst_path: Path,
) -> Action:
    """Build action for a multi-script / multi-subpackage package."""
    children = _discover_children(pkg_dir, module_path)
    return Action(
        type="multi",
        module=module_path,
        rst_dir=rst_dir,
        rst_file=f"{module_path}.rst",
        content=generate_multi_rst(module_path, children),
        existing_rst=rst_path if rst_path.exists() else None,
    )


def _action_leaf(module_path: str, rst_dir: Path) -> Action:
    """Build action for a plain script module."""
    return Action(
        type="leaf",
        module=module_path,
        rst_dir=rst_dir,
        rst_file=f"{module_path}.rst",
        content=generate_leaf_rst(module_path),
        parent_module=module_path.rsplit(".", 1)[0],
    )


# ---------------------------------------------------------------------------
# Collect all actions
# ---------------------------------------------------------------------------


def build_actions(
    result: ComparisonResult,
    mdxplain_dir: Path,
    api_dir: Path,
) -> list[Action]:
    """Return a list of generation / update actions for every missing module."""
    init_map: dict[str, InitDetail] = {d.package: d for d in result.init_details}
    package_root = mdxplain_dir.parent
    actions: list[Action] = []

    for module_path in sorted(result.missing_in_api):
        rst_dir = _find_rst_dir(module_path, api_dir)
        rst_path = rst_dir / f"{module_path}.rst"

        if module_path in init_map:
            detail = init_map[module_path]
            pkg_dir = package_root / Path(*module_path.split("."))
            if detail.category == "single_detailed":
                action = _action_single_detailed(module_path, pkg_dir, rst_dir, rst_path)
                if action:
                    actions.append(action)
            elif detail.category == "multi":
                actions.append(_action_multi(module_path, pkg_dir, rst_dir, rst_path))
            # single_trivial → ignored by design
        else:
            actions.append(_action_leaf(module_path, rst_dir))

    return actions


# ---------------------------------------------------------------------------
# Toctree patching
# ---------------------------------------------------------------------------

_TOCTREE_RE = re.compile(
    r"(?P<before>.*\.\.\s+toctree::.*?\n(?:\s+:.*\n)*\n)"
    r"(?P<entries>(?:\s+\S+\n)*)",
    re.DOTALL,
)


def _patch_toctree(rst_text: str, new_entry: str) -> str | None:
    """Insert *new_entry* into an existing toctree block (sorted).

    Returns the patched text, or ``None`` if no toctree was found.
    """
    m = _TOCTREE_RE.search(rst_text)
    if not m:
        return None
    before = m.group("before")
    raw_entries = m.group("entries")
    entries = [ln.strip() for ln in raw_entries.splitlines() if ln.strip()]
    if new_entry in entries:
        return None  # already present
    entries.append(new_entry)
    entries.sort()
    indent = "   "
    new_block = "\n".join(f"{indent}{e}" for e in entries) + "\n"
    return rst_text[: m.start()] + before + new_block + rst_text[m.end() :]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_action(action: Action, project_root: Path) -> None:
    """Pretty-print a single action."""
    rel = action.rst_dir.relative_to(project_root) / action.rst_file
    tag = action.type.upper()
    verb = "REPLACE" if action.is_replacement else "CREATE"
    print(f"  [{tag:>17s}]  {verb:7s}  {rel}")


def _collect_toctree_updates(
    actions: list[Action], api_dir: Path,
) -> dict[str, list[str]]:
    """Map parent RST paths to the child entries that need inserting."""
    updates: dict[str, list[str]] = {}
    for a in actions:
        if a.parent_module:
            parent_rst = _find_rst_dir(a.parent_module, api_dir) / f"{a.parent_module}.rst"
            updates.setdefault(str(parent_rst), []).append(a.module)
    return updates


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


def _print_overview(
    actions: list[Action],
    toctree_updates: dict[str, list[str]],
    project_root: Path,
) -> None:
    """Print the summary tables (files to create/replace, toctrees)."""
    new_files = [a for a in actions if not a.is_replacement]
    replacements = [a for a in actions if a.is_replacement]

    print(f"\n{SEPARATOR}")
    print(f"RST FILES TO CREATE ({len(new_files)})")
    print(SEPARATOR)
    for a in new_files:
        _print_action(a, project_root)

    if replacements:
        print(f"\n{SEPARATOR}")
        print(f"EXISTING RST FILES TO REPLACE ({len(replacements)})")
        print(f"  (single-trivial -> single-detailed upgrade)")
        print(SEPARATOR)
        for a in replacements:
            _print_action(a, project_root)

    if toctree_updates:
        print(f"\n{SEPARATOR}")
        print(f"PARENT TOCTREES TO UPDATE ({len(toctree_updates)})")
        print(SEPARATOR)
        for rst_path_str, entries in sorted(toctree_updates.items()):
            rst_path = Path(rst_path_str)
            rel = rst_path.relative_to(project_root)
            exists = "OK" if rst_path.exists() else "NOT FOUND"
            print(f"  {rel}  [{exists}]")
            for e in sorted(entries):
                print(f"      + {e}")


def _print_rst_content_preview(
    actions: list[Action], project_root: Path,
) -> None:
    """Print the full generated RST content for every action."""
    print(f"\n{SEPARATOR}")
    print("GENERATED RST CONTENT PREVIEW")
    print(SEPARATOR)
    for a in actions:
        rel = a.rst_dir.relative_to(project_root) / a.rst_file
        print(f"\n>>> {rel} [{a.type.upper()}]")
        print(a.content)


# ---------------------------------------------------------------------------
# Write & patch
# ---------------------------------------------------------------------------


def _write_files(actions: list[Action]) -> tuple[int, int]:
    """Write all RST files to disk. Returns (created, replaced) counts."""
    created = replaced = 0
    for a in actions:
        a.rst_dir.mkdir(parents=True, exist_ok=True)
        (a.rst_dir / a.rst_file).write_text(a.content, encoding="utf-8")
        if a.is_replacement:
            replaced += 1
        else:
            created += 1
    return created, replaced


def _patch_toctrees(
    toctree_updates: dict[str, list[str]], project_root: Path,
) -> int:
    """Patch parent toctrees on disk. Returns number of patched files."""
    patched = 0
    for rst_path_str, entries in sorted(toctree_updates.items()):
        rst_path = Path(rst_path_str)
        if not rst_path.exists():
            print(f"  WARNING: Cannot patch (file missing): {rst_path.relative_to(project_root)}")
            continue
        text = rst_path.read_text(encoding="utf-8")
        changed = False
        for entry in sorted(entries):
            result_text = _patch_toctree(text, entry)
            if result_text is not None:
                text = result_text
                changed = True
        if changed:
            rst_path.write_text(text, encoding="utf-8")
            patched += 1
            print(f"  OK Patched toctree: {rst_path.relative_to(project_root)}")
    return patched


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def execute_actions(
    actions: list[Action],
    project_root: Path,
    api_dir: Path,
    *,
    write: bool = False,
    patch: bool = False,
) -> None:
    """Execute or preview all generation actions."""
    if not actions:
        print("\nNo missing RST files detected. Nothing to do.")
        return

    toctree_updates = _collect_toctree_updates(actions, api_dir)

    _print_overview(actions, toctree_updates, project_root)
    _print_rst_content_preview(actions, project_root)

    if not write:
        print(f"\n  Dry-run. Use --write to create/replace files.")
        return

    created, replaced = _write_files(actions)
    print(f"\n  OK Created {created} file(s), replaced {replaced} file(s).")

    if not patch:
        if toctree_updates:
            print("  Pass --patch to auto-insert entries into parent toctrees.")
        return

    n = _patch_toctrees(toctree_updates, project_root)
    print(f"  OK Patched {n} toctree(s).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate missing API RST files for mdxplain.",
    )
    parser.add_argument(
        "--write", action="store_true",
        help="Actually write the RST files (default: dry-run).",
    )
    parser.add_argument(
        "--patch", action="store_true",
        help="Auto-insert new entries into parent toctrees (implies --write).",
    )
    parser.add_argument(
        "--preview", metavar="MODULE",
        help="Print the generated RST for a single module and exit.",
    )
    args = parser.parse_args()
    if args.patch:
        args.write = True
    return args


def _validate_dirs(root: Path) -> tuple[Path, Path]:
    """Return (mdxplain_dir, api_dir) or exit on error."""
    mdxplain_dir = root / "mdxplain"
    api_dir = root / "docs" / "api"
    for name, path in [("mdxplain", mdxplain_dir), ("docs/api", api_dir)]:
        if not path.is_dir():
            print(f"ERROR: {name} directory not found: {path}")
            sys.exit(1)
    return mdxplain_dir, api_dir


def main() -> int:
    args = _parse_args()
    root = find_project_root()
    mdxplain_dir, api_dir = _validate_dirs(root)

    modules, init_details = collect_mdxplain_modules(mdxplain_dir)
    api_modules = collect_api_automodules(api_dir)
    result = compare_modules(modules, api_modules, init_details)

    if not result.missing_in_api:
        print("All modules are documented. Nothing to generate.")
        return 0

    actions = build_actions(result, mdxplain_dir, api_dir)

    if args.preview:
        match = next((a for a in actions if a.module == args.preview), None)
        if match:
            print(match.content)
            return 0
        print(f"Module '{args.preview}' not found in missing list.")
        return 1

    print(SEPARATOR)
    print("GENERATE MISSING API RST FILES")
    print(SEPARATOR)
    print(f"  Missing modules: {len(result.missing_in_api)}")
    print(f"  Actions:         {len(actions)}")

    execute_actions(actions, root, api_dir, write=args.write, patch=args.patch)
    return 0


if __name__ == "__main__":
    sys.exit(main())

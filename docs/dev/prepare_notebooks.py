#!/usr/bin/env python3
"""
Script to copy notebooks from root/tutorials to docs/tutorials/notebooks
and generate toctree entries for them.
"""
import re
import shutil
from pathlib import Path


def prepare_notebooks():
    """Copy notebooks and generate toctree entries."""
    # Define paths
    docs_dir = Path(__file__).parent.parent
    root_dir = docs_dir.parent  # Go up to repository root
    source_dir = root_dir / "tutorials"
    target_dir = docs_dir / "tutorials" / "notebooks"  # docs/tutorials/notebooks
    learning_rst = docs_dir / "tutorials" / "learning.rst"  # docs/tutorials/learning.rst

    # Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all notebooks in source directory
    notebooks = sorted(source_dir.glob("*.ipynb"))
    
    if not notebooks:
        print("No notebooks found to copy.")
        return
    
    # Cleanup: Remove notebooks from target that no longer exist in source
    source_names = {nb.name for nb in notebooks}
    existing_targets = list(target_dir.glob("*.ipynb"))
    
    for target_file in existing_targets:
        if target_file.name not in source_names:
            print(f"Removing deleted notebook: {target_file.name}")
            target_file.unlink()
    
    # Copy notebooks (replace if already exists)
    print(f"Copying {len(notebooks)} notebook(s)...")
    for notebook in notebooks:
        target_file = target_dir / notebook.name
        
        # Check if file exists and remove it first (failsafe)
        if target_file.exists():
            print(f"  Replacing existing: {notebook.name}")
            target_file.unlink()
        else:
            print(f"  Copying: {notebook.name}")
        
        # Copy the notebook
        shutil.copy2(notebook, target_file)
    
    # Generate toctree entries (without .ipynb extension)
    toctree_entries = "\n".join(f"   notebooks/{nb.stem}" for nb in notebooks)
    
    # Read the learning.rst file
    if not learning_rst.exists():
        print(f"Warning: {learning_rst} not found.")
        return
    
    content = learning_rst.read_text()
    
    # Failsafe: Replace existing toctree content between markers
    # Pattern matches placeholder, toctree directive, any content (including whitespace), and end marker
    # \s* allows any amount of whitespace (including multiple newlines)
    pattern = r'(.. placeholder for notebooks\s*\n\s*\n.. toctree::\s*\n\s*:maxdepth: 1)\s*(.*?)\s*(.. toctree notebooks end)'
    
    replacement = f'\\1\n\n{toctree_entries}\n\n\\3'
    
    # Check if pattern exists and replace
    if re.search(pattern, content, re.DOTALL):
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        learning_rst.write_text(new_content)
        print(f"\nUpdated {learning_rst.name} with {len(notebooks)} notebook(s).")
    else:
        print("\nWarning: Placeholder pattern not found in learning.rst")
        print("Expected pattern:")
        print(".. placeholder for notebooks")
        print("")
        print(".. toctree::")
        print("   :maxdepth: 1")
        print("")
        print(".. toctree notebooks end")
        print("\nGenerated toctree entries:")
        print(toctree_entries)


if __name__ == "__main__":
    prepare_notebooks()

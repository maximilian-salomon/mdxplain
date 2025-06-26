#!/usr/bin/env python3
"""
Manual code quality check script.
Run this when you want to check code quality manually.
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and print results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"Exit code: {result.returncode}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def main():
    """Run comprehensive code quality checks."""
    # Check if we're in the right directory
    if not os.path.exists('mdxplain'):
        print("Error: Run this script from the project root directory")
        sys.exit(1)
    
    print("Starting comprehensive code quality checks...")
    
    # 1. Black formatting check
    print("\n" + "="*60)
    print("1. FORMATTING CHECK WITH BLACK")
    run_command(['black', '--check', '--diff', 'mdxplain/'], 
                "Check formatting with black")
    
    # 2. Import sorting check
    print("\n" + "="*60)
    print("2. IMPORT SORTING CHECK WITH ISORT") 
    run_command(['isort', '--check-only', '--diff', 'mdxplain/'], 
                "Check import sorting with isort")
    
    # 3. Style checks
    print("\n" + "="*60) 
    print("3. STYLE CHECK WITH FLAKE8")
    run_command(['flake8', 'mdxplain/', '--max-line-length=100'], 
                "Style check with flake8")
    
    print("\n" + "="*60)
    print("4. PEP8 STYLE CHECK WITH PYCODESTYLE")
    run_command(['pycodestyle', 'mdxplain/', '--max-line-length=100'], 
                "PEP8 style check with pycodestyle")
    
    print("\n" + "="*60)
    print("5. ERROR CHECK WITH PYFLAKES")
    run_command(['pyflakes', 'mdxplain/'], 
                "Error check with pyflakes")
    
    # 4. Docstring checks
    print("\n" + "="*60)
    print("6. DOCSTRING CHECK WITH PYDOCSTYLE")
    run_command(['pydocstyle', 'mdxplain/', '--count'], 
                "Docstring check with pydocstyle")
    
    # 5. Type checking
    print("\n" + "="*60)
    print("7. TYPE CHECK WITH MYPY") 
    run_command(['mypy', 'mdxplain/', '--ignore-missing-imports'], 
                "Type check with mypy")
    
    # 6. Security checks
    print("\n" + "="*60)
    print("8. SECURITY CHECK WITH BANDIT")
    run_command(['bandit', '-r', 'mdxplain/', '-f', 'txt'], 
                "Security check with bandit")
    
    # 7. Dead code detection
    print("\n" + "="*60)
    print("9. DEAD CODE CHECK WITH VULTURE")
    run_command(['vulture', 'mdxplain/', '--min-confidence', '80'], 
                "Dead code check with vulture")
    
    # 8. Code complexity analysis
    print("\n" + "="*60)
    print("10. CODE COMPLEXITY CHECK WITH RADON")
    run_command(['radon', 'cc', 'mdxplain/', '--total-average'], 
                "Code complexity check with radon")
    
    # 9. Documentation coverage
    print("\n" + "="*60)
    print("11. DOCSTRING COVERAGE WITH INTERROGATE")
    run_command(['interrogate', 'mdxplain/', '--verbose', '--fail-under=100'], 
                "Documentation coverage check with interrogate (requires 100%)")
    
    # 10. Dependency security check
    print("\n" + "="*60)
    print("12. DEPENDENCY SECURITY CHECK WITH SAFETY")
    run_command(['safety', 'check'], 
                "Dependency security check with safety")
    
    # 11. Comprehensive linting
    print("\n" + "="*60)
    print("13. COMPREHENSIVE LINT WITH PYLINT")
    run_command(['pylint', 'mdxplain/', '--jobs=1'], 
                "Comprehensive linting with pylint")
    
    # 12. Auto-formatting check (what autopep8 would change)
    print("\n" + "="*60)
    print("14. AUTO-FORMATTING CHECK WITH AUTOPEP8")
    run_command(['autopep8', '--diff', '--recursive', 'mdxplain/'], 
                "Auto-formatting check with autopep8")
    
    print(f"\n{'='*60}")
    print("Comprehensive code quality checks completed!")
    print("Total checks: 14 different tools")
    print("="*60)


if __name__ == "__main__":
    main() 
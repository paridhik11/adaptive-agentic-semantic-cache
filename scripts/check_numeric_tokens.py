"""Strict token-level numeric token check.

Computes the strict set difference of extracted numeric tokens between
target documents (README.md, DESIGN.md, demo/SCRIPT.md) and docs/FACTS.md.
Zero substring matching. Reports every missing token and its source line numbers.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Set


def extract_numeric_tokens(filepath: Path) -> Dict[str, List[int]]:
    """Extract numeric tokens and their 1-indexed line numbers.
    
    A numeric token is an integer or decimal number bounded by non-word characters.
    Zero substring matching.
    """
    text = filepath.read_text(encoding="utf-8")
    tokens: Dict[str, List[int]] = {}
    
    # Matches integers and decimals not bounded by letters/digits/underscores
    pattern = re.compile(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)*(?![A-Za-z0-9_])")
    
    for line_idx, line in enumerate(text.splitlines(), start=1):
        for match in pattern.finditer(line):
            tok = match.group(0)
            if tok not in tokens:
                tokens[tok] = []
            tokens[tok].append(line_idx)
            
    return tokens


def main() -> int:
    workspace_root = Path(__file__).resolve().parent.parent
    facts_file = workspace_root / "docs" / "FACTS.md"
    
    if not facts_file.exists():
        print(f"Error: {facts_file} not found.", file=sys.stderr)
        return 1
        
    facts_tokens = extract_numeric_tokens(facts_file)
    facts_set: Set[str] = set(facts_tokens.keys())
    
    print(f"Loaded {len(facts_set)} unique numeric tokens from {facts_file.name}.\n")
    
    target_rel_paths = [
        Path("README.md"),
        Path("DESIGN.md"),
        Path("demo") / "SCRIPT.md",
    ]
    
    total_violations = 0
    
    for rel_path in target_rel_paths:
        target_path = workspace_root / rel_path
        if not target_path.exists():
            print(f"Error: {target_path} not found.", file=sys.stderr)
            total_violations += 1
            continue
            
        doc_tokens = extract_numeric_tokens(target_path)
        doc_set: Set[str] = set(doc_tokens.keys())
        
        diff = doc_set - facts_set
        
        print(f"=== Checking {rel_path} ===")
        print(f"Total extracted numeric tokens: {len(doc_set)}")
        print(f"Missing tokens count: {len(diff)}")
        
        if diff:
            total_violations += len(diff)
            print("Missing numeric tokens (not present in docs/FACTS.md):")
            def sort_key(tok: str):
                try:
                    return (0, float(tok))
                except ValueError:
                    return (1, tok)
                    
            for tok in sorted(diff, key=sort_key):
                lines = doc_tokens[tok]
                print(f"  - Token: '{tok}' | Line(s): {lines}")
        else:
            print("  Status: All numeric tokens strictly verified against docs/FACTS.md.")
        print()
        
    if total_violations == 0:
        print("OVERALL RESULT: PASS (0 unledgered numeric tokens across all audited documents)")
        return 0
    else:
        print(f"OVERALL RESULT: FAIL ({total_violations} unledgered numeric token instances found)")
        return 1


if __name__ == "__main__":
    sys.exit(main())

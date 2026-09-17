#!/usr/bin/env python3
r"""
OpenTheory Blueprint Exporter.
Extracts mathematical declarations, \uses{}, \lean{}, and \leanok tags
from blueprint/src/content.tex and builds blueprint.json for the web visualizer.
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parent.parent
CONTENT_TEX = ROOT_DIR / "blueprint" / "src" / "content.tex"
DEFAULT_OUTPUT = ROOT_DIR / "blueprint" / "blueprint.json"

def parse_blueprint(output_path=None):
    if not CONTENT_TEX.exists():
        print(f"Error: {CONTENT_TEX} does not exist", file=sys.stderr)
        sys.exit(1)

    out_file = Path(output_path) if output_path else DEFAULT_OUTPUT
    text = CONTENT_TEX.read_text(encoding="utf-8")

    # Match blocks like \begin{theorem}[Title]\label{code} ... \end{theorem}
    env_regex = re.compile(
        r"\\begin\{(theorem|lemma|definition|conjecture)\}(?:\[(.*?)\])?\\label\{(.*?)\}(.*?)\\end\{\1\}",
        re.DOTALL
    )

    items = []
    for match in env_regex.finditer(text):
        kind, title, label, body = match.groups()
        
        # Check for \lean{...}
        lean_match = re.search(r"\\lean\{(.*?)\}", body)
        lean_name = lean_match.group(1).strip() if lean_match else None

        # Check for \leanok
        is_leanok = "\\leanok" in body

        # Check for \uses{...}
        uses_match = re.search(r"\\uses\{(.*?)\}", body)
        dependencies = [u.strip() for u in uses_match.group(1).split(",")] if uses_match else []

        items.append({
            "id": label,
            "kind": kind,
            "title": title or label,
            "leanName": lean_name,
            "isVerified": is_leanok,
            "dependencies": dependencies
        })

    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "toolchain": "leanprover/lean4:v4.11.0",
        "nodesCount": len(items),
        "nodes": items
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[Blueprint Export] Successfully wrote {len(items)} nodes to {out_file}")

if __name__ == "__main__":
    out_arg = sys.argv[1] if len(sys.argv) > 1 else None
    parse_blueprint(out_arg)

#!/usr/bin/env python3
"""Export blueprint/src/content.tex to blueprint.json for the web platform.

Massot convention:
  - \\leanok on a definition/statement  → statement is in Lean
  - \\leanok inside the following proof → proof is kernel-checked
  - isVerified is true only when the proof is tagged (or the item is a
    definition with \\leanok and no proof environment)
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CONTENT_TEX = ROOT_DIR / "blueprint" / "src" / "content.tex"
DEFAULT_OUTPUT = ROOT_DIR / "blueprint" / "blueprint.json"

ENV_RE = re.compile(
    r"\\begin\{(theorem|lemma|definition|conjecture|proposition|corollary)\}"
    r"(?:\[(.*?)\])?\s*\\label\{(.*?)\}(.*?)\\end\{\1\}",
    re.DOTALL,
)

PROOF_RE = re.compile(r"^\s*\\begin\{proof\}(.*?)\\end\{proof\}", re.DOTALL)


def split_lean_names(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def strip_macros(body: str) -> str:
    cleaned = re.sub(r"\\lean\{.*?\}", "", body)
    cleaned = re.sub(r"\\leanok", "", cleaned)
    cleaned = re.sub(r"\\uses\{.*?\}", "", cleaned)
    cleaned = re.sub(r"\\notready", "", cleaned)
    cleaned = re.sub(r"\\mathlibok", "", cleaned)
    return cleaned.strip()


def parse_blueprint(output_path: Path | None = None) -> dict:
    if not CONTENT_TEX.exists():
        print(f"Error: {CONTENT_TEX} does not exist", file=sys.stderr)
        sys.exit(1)

    text = CONTENT_TEX.read_text(encoding="utf-8")
    items = []

    for match in ENV_RE.finditer(text):
        kind, title, label, body = match.groups()
        lean_names = []
        for lean_match in re.finditer(r"\\lean\{(.*?)\}", body):
            lean_names.extend(split_lean_names(lean_match.group(1)))

        statement_ok = r"\leanok" in body
        uses_match = re.search(r"\\uses\{(.*?)\}", body)
        dependencies = (
            [u.strip() for u in uses_match.group(1).split(",") if u.strip()]
            if uses_match
            else []
        )

        after = text[match.end() :]
        proof_match = PROOF_RE.match(after)
        proof_ok = False
        proof_sketch = None
        bounty_usd = None
        if proof_match:
            proof_body = proof_match.group(1)
            proof_ok = r"\leanok" in proof_body
            proof_sketch = re.sub(r"\\leanok", "", proof_body).strip()
            bounty_match = re.search(
                r"Bounty\s+\\?\$?([0-9]+(?:\{,\}[0-9]+)?)",
                proof_sketch,
            )
            if bounty_match:
                bounty_usd = int(re.sub(r"[^0-9]", "", bounty_match.group(1)))

        if kind == "definition":
            is_verified = statement_ok
        else:
            is_verified = proof_ok

        items.append(
            {
                "id": label,
                "kind": kind,
                "title": title or label,
                "leanName": lean_names[0] if lean_names else None,
                "leanNames": lean_names,
                "statementFormalized": statement_ok,
                "proofFormalized": proof_ok,
                "isVerified": is_verified,
                "dependencies": dependencies,
                "bountyUsd": bounty_usd,
                "informalSketch": proof_sketch,
            }
        )

    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "toolchain": "leanprover/lean4:v4.11.0",
        "convention": "massot-leanblueprint",
        "nodesCount": len(items),
        "nodes": items,
    }

    out_file = Path(output_path) if output_path else DEFAULT_OUTPUT
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[Blueprint Export] Wrote {len(items)} nodes to {out_file}")
    return payload


if __name__ == "__main__":
    out_arg = sys.argv[1] if len(sys.argv) > 1 else None
    parse_blueprint(out_arg)

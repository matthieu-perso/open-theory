# OpenTheory (Lean core)

This is a **Lean 4 + Mathlib** repository with a Patrick Massot `leanblueprint`. It is not a Next.js app.

## Source of truth

- Mathematics: `OpenTheory/**/*.lean` and `blueprint/src/content.tex`
- Platform feed: `blueprint/blueprint.json` (regenerate with `python3 scripts/export_blueprint.py`)
- Local agents: `mcp/` (stdio). Do not add an HTTP API on this repo.

## Honesty rules

- Every `\lean{Name}` in `content.tex` must be a real declaration (`python3 scripts/check_lean_decls.py`).
- `\leanok` on a lemma/conjecture **proof** means no `sorry`. Statement-only `\leanok` means the claim is in Lean and may still be `sorry`.
- Do not mark stubs, `True` fillers, or trivial `use` proofs as verified.
- Allowed axioms: `propext`, `Classical.choice`, `Quot.sound`, and `sorryAx` on open targets.

## Commands

```bash
lake build
lake env lean --run scripts/audit_axioms.lean
python3 scripts/check_lean_decls.py
python3 scripts/export_blueprint.py blueprint/blueprint.json
```

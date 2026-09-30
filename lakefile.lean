import Lake
open Lake DSL

package «open_theory» where
  -- Lean 4.11 + Mathlib v4.11.0. Blueprint lives in blueprint/ (leanblueprint).

@[default_target]
lean_lib «OpenTheory» where
  -- Library configuration options

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.11.0"

lean_exe audit_axioms where
  root := `scripts.audit_axioms

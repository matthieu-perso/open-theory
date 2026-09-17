import Lake
open Lake DSL

package «open_theory» where
  -- Package configuration options

@[default_target]
lean_lib «OpenTheory» where
  -- Library configuration options

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.11.0"

lean_exe audit_axioms where
  root := `scripts.audit_axioms

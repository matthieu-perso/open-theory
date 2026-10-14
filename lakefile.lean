import Lake
open Lake DSL

package «open_theory» where
  -- Lean / Mathlib 4.34.0. Blueprint lives in blueprint/ (leanblueprint).

@[default_target]
lean_lib «OpenTheory» where
  -- Library configuration options

require TorchLean from git
  "https://github.com/lean-dojo/TorchLean.git" @ "2d148d85839d97b02b06e8e02be61e3d57be3ad7"

-- Keep Mathlib last: its support-package revisions take precedence.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "5ed2965256430c3649e86755f9576b54eca72435"

lean_exe audit_axioms where
  root := `scripts.audit_axioms

-- Narrow, provenance-recorded FoML port for the pinned Mathlib version.
lean_lib FoML where
  srcDir := "vendor/FoML"

-- Narrow, provenance-recorded StatLean port for finite-class PAC bounds.
lean_lib StatLean where
  srcDir := "vendor/StatLean"

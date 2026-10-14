import Lean

open Lean

def isAllowedAxiom (n : Name) : Bool :=
  n == ``Classical.choice || n == ``Quot.sound || n == ``propext || n == ``sorryAx

def isAllowedForCompleted (n : Name) : Bool :=
  isAllowedAxiom n && n != ``sorryAx

/-- Existing research targets only. New declarations are strict by default. -/
def openTargets : List Name := []

def axiomsOf (env : Environment) (declName : Name) : IO (Array Name) :=
  (Lean.collectAxioms declName : CoreM (Array Name)).toIO'
    { fileName := "audit", fileMap := default } { env := env }

/-- Resolve blueprint references against the actual imported Lean environment. -/
def blueprintNames (source : String) : List Name :=
  ((source.splitOn "\\lean{").drop 1).flatMap fun block =>
    ((block.splitOn "}").head!).splitOn "," |>.map fun name => name.trimAscii.toString.toName

def main : IO UInt32 := do
  let env ← importModules #[{ module := `OpenTheory }] {}
  let names := blueprintNames (← IO.FS.readFile "blueprint/src/content.tex")
  for name in names do
    if (env.find? name).isNone then
      IO.println s!"[FATAL] Blueprint declaration not found in Lean: {name}"
      return 1
  IO.println s!"[VERIFIED] {names.length} blueprint references resolve in Lean (including imports)."
  let mut violations : List (Name × Name) := []
  let mut sorryDecls : List Name := []

  for (declName, info) in env.constants.toList do
    unless declName.toString.startsWith "OpenTheory" do
      continue
    -- Unsafe compiler intermediates may contain erased proofs such as lcProof.
    -- They cannot be used in kernel proofs; audit the safe declarations instead.
    if info.isUnsafe then
      continue
    for ax in ← axiomsOf env declName do
      if ax == ``sorryAx then
        sorryDecls := declName :: sorryDecls
      if !isAllowedForCompleted ax && !(ax == ``sorryAx && openTargets.contains declName) then
        violations := (declName, ax) :: violations

  let sorryUnique := sorryDecls.eraseDups
  IO.println s!"[INFO] OpenTheory declarations using sorryAx: {sorryUnique.length}"
  for decl in sorryUnique do
    IO.println s!"  sorry: {decl}"

  if !violations.isEmpty then
    IO.println "[FATAL] Unauthorized axioms detected:"
    for (decl, ax) in violations do
      IO.println s!"  Declaration: {decl} -> Illegal Axiom: {ax}"
    return 1

  IO.println "[VERIFIED] Safe OpenTheory declarations use only standard axioms; sorryAx is confined to named open targets."
  return 0

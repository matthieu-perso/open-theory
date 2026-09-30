import Lean

open Lean

def isAllowedAxiom (n : Name) : Bool :=
  n == ``Classical.choice || n == ``Quot.sound || n == ``propext || n == ``sorryAx

def axiomsOf (env : Environment) (declName : Name) : Array Name :=
  let (_, s) := ((CollectAxioms.collect declName).run env).run {}
  s.axioms

def main : IO UInt32 := do
  let env ← importModules [{ module := `OpenTheory }] {}
  let mut violations : List (Name × Name) := []
  let mut sorryDecls : List Name := []

  for (declName, _) in env.constants.toList do
    unless declName.toString.startsWith "OpenTheory" do
      continue
    for ax in axiomsOf env declName do
      if ax == ``sorryAx then
        sorryDecls := declName :: sorryDecls
      else if !isAllowedAxiom ax then
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

  IO.println "[VERIFIED] No axioms outside {propext, Classical.choice, Quot.sound, sorryAx}."
  return 0

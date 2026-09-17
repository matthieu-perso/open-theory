import Lean

open Lean

def standardAxioms : List Name := [
  `Classical.choice,
  `Quot.sound,
  `propext
]

def main : IO UInt32 := do
  let env ← importModules [{ module := `OpenTheory }] {}
  let mut violations : List (Name × Name) := []

  for (declName, _) in env.constants.toList do
    if declName.toString.startsWith "OpenTheory" then
      let (_, axioms) := (env.find? declName).get!.value.collectAxioms env
      for ax in axioms do
        if ¬ standardAxioms.contains ax then
          violations := (declName, ax) :: violations

  if ¬ violations.isEmpty then
    IO.println "[FATAL] Unauthorized axioms detected in proof submission:"
    for (decl, ax) in violations do
      IO.println s!"  Declaration: {decl} -> Illegal Axiom: {ax}"
    return 1

  IO.println "[VERIFIED] All declarations are sound against standard Mathlib axioms."
  return 0

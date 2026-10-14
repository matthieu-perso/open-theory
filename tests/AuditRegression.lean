import scripts.audit_axioms

-- A completed claim must reject sorryAx even when inherited from another declaration.
example : isAllowedForCompleted ``sorryAx = false := by decide
example : isAllowedForCompleted ``Classical.choice = true := by decide
example : isAllowedForCompleted `arbitraryExtraAxiom = false := by decide
#eval show IO Unit from do
  unless blueprintNames "text \\lean{Nat.add, Nat.mul} end" == [`Nat.add, `Nat.mul] do
    throw (IO.userError "Blueprint reference parsing failed")

namespace AuditFixture
axiom untrusted : False
theorem inherited : False := untrusted
theorem unfinished : True := by sorry
theorem inheritedSorry : True := unfinished
end AuditFixture

run_cmd do
  let axioms ← Lean.collectAxioms `AuditFixture.inherited
  unless axioms.contains `AuditFixture.untrusted do
    throwError "Audit failed to find a transitive custom axiom"
  let axioms ← Lean.collectAxioms `AuditFixture.inheritedSorry
  unless axioms.contains ``sorryAx do
    throwError "Audit failed to find transitive sorryAx"

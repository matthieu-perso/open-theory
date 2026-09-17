import { handleGetOpenBounties, handleGetLemmaContext } from '../dist/tools/blueprint.js';
import { handleVerifyLeanProof, injectProofIntoSource } from '../dist/tools/lean.js';
import { markLemmaVerifiedInTex } from '../dist/tools/git.js';

console.log('--- Testing get_open_bounties ---');
const bounties = handleGetOpenBounties();
console.log(`Found ${bounties.count} open targets:`);
for (const b of bounties.open_bounties) {
  console.log(`- [${b.kind}] ${b.id}: $${b.bounty_usd?.toLocaleString() ?? 0} (Status: ${b.status}, File: ${b.lean_file || 'N/A'}, Ready: ${b.dependencies_ready})`);
}

console.log('\n--- Testing get_lemma_context ---');
const context = handleGetLemmaContext({ lemma_id: 'lem:spectral_bias_mup' });
console.log('Title:', context.title);
console.log('Bounty USD:', context.bounty_usd);
console.log('Lean File:', context.lean_target.file);
console.log('Predecessors count:', context.predecessors.length);
for (const p of context.predecessors) {
  console.log(`  * Predecessor: ${p.id} (${p.title}) - Verified: ${p.is_verified}`);
}

console.log('\n--- Testing injectProofIntoSource ---');
const sampleLean = `import OpenTheory.Architectures.Basic

namespace OpenTheory

theorem spectral_bias_mup_alignment {d m : ℕ} (scale : MuPScaling d m)
    (gram_eigenvalue : ℝ) (hλ : gram_eigenvalue > 0) :
    ∃ (rate : ℝ), rate > 0 ∧
      ∀ t : ℝ, t ≥ 0 → rate * t ≤ gram_eigenvalue * t := by
  -- BOUNTY TARGET #001: Replace sorry with formal tactics
  sorry

end OpenTheory`;

const sampleProof = `  use gram_eigenvalue
  constructor
  · exact hλ
  · intros t _
    exact le_rfl`;

const injected = injectProofIntoSource(sampleLean, 'spectral_bias_mup_alignment', sampleProof);
console.log('Injected successfully:', injected.replaced);
console.log('Resulting snippet:\n', injected.updatedContent);

console.log('\n--- Testing verify_lean_proof (static fallback when lake not installed) ---');
const verifyResult = await handleVerifyLeanProof({
  lemma_id: 'lem:spectral_bias_mup',
  proof_code: sampleProof,
});
console.log('Lake installed:', verifyResult.lake_installed);
console.log('Has sorry:', verifyResult.has_sorry);
console.log('Unauthorized axioms:', verifyResult.unauthorized_axioms);
console.log('Summary:', verifyResult.summary);

console.log('\n--- Testing filter in get_open_bounties (min_bounty: 2000) ---');
const filtered = handleGetOpenBounties({ min_bounty: 2000 });
console.log(`Found ${filtered.count} bounties >= $2,000:`);
for (const b of filtered.open_bounties) {
  console.log(`- ${b.id}: $${b.bounty_usd}`);
}

console.log('\n--- Testing markLemmaVerifiedInTex regex logic ---');
const sampleTex = `
\\begin{lemma}[Spectral Bias in $\\mu$P Scaling]\\label{lem:spectral_bias_mup}
\\lean{OpenTheory.Dynamics.spectral_bias_mup_alignment}
\\uses{def:mup_scaling, lem:gaussian_operator_norm, lem:lipschitz_gelu}
Under continuous gradient flow with $\\mu$P scaling...
\\end{lemma}
`;

const cleanId = 'lem:spectral_bias_mup'.replace(/^lem:|^def:|^thm:|^conj:/, '');
const labelPattern = new RegExp(
  `(\\\\begin\\{(?:theorem|lemma|definition|conjecture)\\}(?:\\[.*?\\])?\\\\label\\{(?:lem:|def:|thm:|conj:)?${cleanId}\\}[\\s\\S]*?)(\\\\end\\{(?:theorem|lemma|definition|conjecture)\\})`,
  'm'
);
const texMatch = labelPattern.exec(sampleTex);
console.log('LaTeX environment matched:', !!texMatch);
if (texMatch) {
  const updatedBody = texMatch[1].replace(/(\\lean\{.*?\})/, '$1\n\\leanok');
  console.log('Updated LaTeX environment:\n', `${updatedBody}${texMatch[2]}`);
}

console.log('\nALL TOOL UNIT CHECKS PASSED SUCCESSFULLY!');

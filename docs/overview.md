# OpenTheory: Machine-Checked Foundations of Deep Learning

OpenTheory is an open mathematical platform for formalizing deep learning theory in **Lean 4**. 
By replacing 80-page non-reproducible PDF preprints with a machine-checked dependency DAG and milestone-funded proof bounties, OpenTheory accelerates verifiable AI theory.

---

## 1. Project Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        OPENTHEORY PLATFORM                             │
│                                                                        │
│   [ Interactive Blueprint DAG ] ── (@xyflow/react + ELK layering)     │
│                 │                                                      │
│                 ▼                                                      │
│   [ Theorem Inspector Modal ]   ── (LaTeX Informal + Lean 4 + Axioms)  │
│                 │                                                      │
│                 ▼                                                      │
│   [ Live Verification Engine ] ── (In-browser Lean 4 simulation & API) │
│                 │                                                      │
│                 ▼                                                      │
│   [ CI Kernel & Axiom Audit ]   ── (Lean 4.11 + Mathlib4 + Escrow)     │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Frontend**: Next.js 16 with React 19, Tailwind CSS v4, and `@xyflow/react`.
2. **Mathematical Rendering**: KaTeX math typesetter with intuitive proof sketches.
3. **Formal Verification Kernel**: Lean 4.11.0 with Mathlib4.
4. **Security Filter**: `lean/scripts/audit_axioms.lean` checks `#print axioms` to reject unauthorized `axiom cheat : False` declarations.
5. **Blueprint Specification**: Modeled on Patrick Massot's `leanblueprint` standard (`blueprint/src/content.tex`, `references.bib`).
6. **Zero-Setup Researcher Environment**: 1-click cloud VS Code via `.devcontainer/devcontainer.json` (GitHub Codespaces).
7. **Incentive Layer**: Smart-contract / multi-sig backed bounties automatically disbursed upon CI passing.

---

## 2. Research Standard: The Dual-Track Pipeline

OpenTheory follows the standard established by high-profile formal math initiatives (e.g., Terence Tao's PFR project, Patrick Massot's Sphere Eversion):

1. **Informal Human Manuscript (`blueprint/src/content.tex`):**
   - LaTeX-formatted statement with mathematical intuition and human proof sketches.
   - Formal literature citations with BibTeX (`blueprint/src/references.bib`) and arXiv permalinks.
   - Declarations use `\lean{...}` identifiers and `\uses{...}` dependency links.
2. **Formal Machine Library (`lean/OpenTheory/`):**
   - Pinned toolchain in `lean/lean-toolchain` (`v4.11.0`).
   - Standardized modules: `Architectures/`, `Dynamics/`, `Generalization/`.
   - Continuous verification via `lake build` and Mathlib cache fetching.
3. **Axiom Audit in CI (`.github/workflows/verify.yml`):**
   - Every pull request executes `lake env lean --run scripts/audit_axioms.lean`.
   - Only standard foundational Mathlib axioms (`Classical.choice`, `Quot.sound`, `propext`) are accepted.
   - Custom axiomatic assertions cause immediate CI failure.

---

## 3. Genesis Formalization Tree

### Grand Conjecture (Apex Target 🔴)
- `CONJ-01`: **Global Feature Learning Alignment in Deep ResNets under µP**
  - **Statement**: Shows that deep residual networks trained with gradient flow under Maximal Update Parametrization align representations with the principal data manifold eigenspaces, strictly surpassing the static Neural Tangent Kernel (NTK) regime.
  - **Bounty**: $15,000 USD

### Unresolved Intermediate Lemmas (Bounty Targets 🔵)
- `LEM-2.1`: **Spectral Bias of Continuous Gradient Flow in µP Scaling** (Bounty: $2,500)
- `LEM-2.2`: **Representation Contraction in Deep Residual Blocks** (Bounty: $1,500)
- `LEM-2.3`: **Stationarity of Neural Tangent Kernel in Standard Scaling** (Bounty: $1,000)
- `LEM-2.4`: **Hessian Spectral Concentration in Overparameterized Limits** (Bounty: $2,000)

### Informal Literature Proofs (Yellow 🟡)
- `THM-3.1`: **Asymptotic Equivalence to Kernel Ridge Regression in NTK Limit** (Bounty: $3,000)
- `THM-3.2`: **Non-Trivial Feature Learning Condition under µP Parameterization** (Bounty: $4,500)

### Verified Foundation Leaves (Green 🟢)
- `DEF-01`: TwoLayerNet Architecture & Forward Map
- `DEF-02`: Maximal Update Parametrization (`µP`) Normalization
- `LEM-1.1`: Lipschitz Continuity of Smooth Activations (GeLU / ReLU)
- `LEM-1.2`: Rademacher Complexity Symmetrization Bound
- `LEM-1.3`: Spectral Norm Bound for Random Gaussian Matrices
- `THM-1.4`: Linear Mode Connectivity along Low-Loss Manifolds

---

## 4. Contributor Workflows

### 1-Click Browser Development (Zero Local Setup)
Mathematicians can launch a complete Lean 4 cloud IDE directly from any theorem in the web UI or via GitHub Codespaces:
- Click **"Codespace"** in the Node Inspector or Bounty Board.
- The environment builds using `.devcontainer/devcontainer.json`, downloads Mathlib `.olean` cache files, and starts the Lean 4 Language Server Protocol (LSP).

### Local Development:
```bash
# 1. Run the Next.js visualizer:
npm run dev

# 2. Run the Lean 4 kernel & axiom audit locally:
cd lean
lake exe cache get
lake build
lake env lean --run scripts/audit_axioms.lean

# 3. Export blueprint graph for the visualizer:
python3 scripts/export_blueprint.py
```

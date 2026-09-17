# OpenTheory Core: Machine-Checked Foundations of Deep Learning

[![Lean 4](https://img.shields.io/badge/Lean_4-v4.11.0-blue.svg)](https://leanprover.github.io/)
[![Mathlib 4](https://img.shields.io/badge/Mathlib_4-pinned-brightgreen.svg)](https://github.com/leanprover-community/mathlib4)
[![Axiom Audit](https://img.shields.io/badge/Axiom_Audit-Strict_Mathlib_Only-emerald.svg)](scripts/audit_axioms.lean)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](LICENSE)

OpenTheory is an open mathematical research initiative formalizing the mathematical foundations of deep learning in **Lean 4** with **Mathlib**.

By replacing 80-page non-reproducible PDF preprints with a machine-checked dependency DAG and milestone-funded proof bounties, OpenTheory accelerates verifiable AI theory.

---

## 1. Mathematical Scope

The library formalizes non-asymptotic and asymptotic representation learning in deep neural networks:

- **Architectures & Encodings (`OpenTheory/Architectures/`)**: Feedforward networks over real Hilbert spaces, Gaussian initialization scaling, and Maximal Update Parametrization ($\mu\text{P}$).
- **Dynamics & Alignment (`OpenTheory/Dynamics/`)**: Continuous gradient flow ODEs, spectral bias dynamics, representation contraction across depth, and the central **Global Feature Learning Alignment** conjecture.
- **Generalization & Kernels (`OpenTheory/Generalization/`)**: Rademacher symmetrization bounds, Neural Tangent Kernel (NTK) stationarity, and mode connectivity.

---

## 2. Quick Start for Mathematicians

### Option A: 1-Click Cloud IDE (No Local Setup)
Click below to launch an interactive browser VS Code instance with Lean 4.11, Mathlib pre-compiled cache, and the language server pre-loaded:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=main)

### Option B: Local Installation

1. **Install `elan`** (the Lean toolchain manager):
   ```bash
   curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
   source ~/.bashrc  # or ~/.zshrc
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/open-theory/open-theory-core.git
   cd open-theory-core
   ```

3. **Fetch pre-compiled Mathlib binaries** (avoids hours of compilation):
   ```bash
   lake exe cache get
   ```

4. **Build and verify proofs**:
   ```bash
   lake build
   ```

5. **Run the Axiom Security Auditor**:
   ```bash
   lake env lean --run scripts/audit_axioms.lean
   ```

---

## 3. Strict Axiom Policy & Security

To prevent "axiom cheating" (such as asserting `axiom cheat : False` to bypass hard proof obligations), OpenTheory enforces strict CI kernel audits via `scripts/audit_axioms.lean`.

Every Pull Request must strictly rely **only** on the three standard foundational Mathlib axioms:
1. `propext` (Propositional extensionality)
2. `Classical.choice` (Axiom of choice)
3. `Quot.sound` (Quotient soundness)

Any declaration using custom axioms, unchecked `trustMe`, or unverified constants fails CI immediately.

---

## 4. The Dual-Track Blueprint (`blueprint/`)

Following Patrick Massot's `leanblueprint` standard, the mathematical manuscript lives in `blueprint/src/content.tex`.

- **Informal Paper**: Every lemma has an informal LaTeX statement, intuition sketch, and BibTeX reference (`blueprint/src/references.bib`).
- **Lean Mapping**: Statements are annotated with `\lean{OpenTheory.declaration_name}`.
- **Verification Status**: Once a proof has `0 sorry` and passes the axiom audit, it receives the `\leanok` tag.

To export the machine-readable graph for the web visualizer:
```bash
python3 scripts/export_blueprint.py blueprint/blueprint.json
```

---

## 5. How to Contribute a Proof

1. Check open lemma targets and bounties on the [OpenTheory Web Platform](https://open-theory.org).
2. Look for declarations with `-- sorry` in `OpenTheory/`.
3. Replace `sorry` with valid Lean 4 tactics.
4. Verify locally using `lake build && lake env lean --run scripts/audit_axioms.lean`.
5. Submit a Pull Request. Bounties disburse automatically upon passing CI.

---

## 6. Project Structure

```
.
├── OpenTheory.lean          # Root library import
├── OpenTheory/
│   ├── Architectures/       # Network models, activations, µP scaling
│   ├── Dynamics/            # Gradient flow, spectral bias, alignment
│   └── Generalization/      # Rademacher bounds, NTK regimes
├── blueprint/
│   ├── leanblueprint.ini    # Blueprint tool configuration
│   └── src/
│       ├── content.tex      # LaTeX manuscript with \lean{} tags
│       └── references.bib   # Real academic BibTeX bibliography
├── scripts/
│   ├── audit_axioms.lean    # CI security kernel audit
│   └── export_blueprint.py  # Blueprint JSON graph generator
├── .devcontainer/           # 1-click Codespaces configuration
├── .github/workflows/       # CI verification actions
├── lakefile.lean            # Lake build configuration
└── lean-toolchain           # Pinned toolchain (v4.11.0)
```

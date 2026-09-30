---
usemathjax: true
---

# OpenTheory

Machine-checked foundations of **deep learning** in Lean&nbsp;4.

Informal papers are not the source of truth. The source of truth is a Lean library with a Massot blueprint: every definition and lemma is written in \(\mathrm{\LaTeX}\), mapped to a declaration with `\lean{...}`, and coloured on a dependency graph according to whether the *statement* and the *proof* are kernel-checked.

## Read the mathematics

- [Blueprint (HTML)]({{ site.url }}{{ site.baseurl }}/blueprint/)
- [Dependency graph]({{ site.url }}{{ site.baseurl }}/blueprint/dep_graph_document.html)
- [Blueprint (PDF)]({{ site.url }}{{ site.baseurl }}/blueprint.pdf)
- [Lean API docs]({{ site.url }}{{ site.baseurl }}/docs/)

## Work on a proof

1. Open a node that is **ready to prove** (blue fill: dependencies complete, proof still `sorry`).
2. Replace `sorry` in the corresponding `OpenTheory/` file.
3. Run `lake build` and `lake env lean --run scripts/audit_axioms.lean`.
4. Tag `\leanok` **inside the proof** in `blueprint/src/content.tex`, not on the statement alone.
5. Open a pull request. CI will refuse custom axioms and unknown `\lean` names.

[GitHub repository](https://github.com/matthieu-perso/open-theory) · [Zulip](https://leanprover.zulipchat.com/)

The interactive dashboard that consumes `blueprint/blueprint.json` lives in a separate web repo. This site is the mathematical artifact.

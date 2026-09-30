# Blueprint (Massot / leanblueprint)

This is a **Patrick Massot Lean blueprint**, not a custom JSON graph with stub macros.

## Layout

```text
blueprint/src/
  web.tex          plasTeX / HTML entry
  print.tex        XeLaTeX / PDF entry
  content.tex      the mathematics (no \documentclass)
  macros/          common + web + print
  plastex.cfg
  latexmkrc
  extra_styles.css
home_page/         Jekyll landing page deployed to GitHub Pages
```

## Commands

```bash
# After `lake build`
pip install leanblueprint
sudo apt install graphviz libgraphviz-dev   # Linux; see pygraphviz docs on macOS

leanblueprint checkdecls   # every \lean{Name} exists
leanblueprint pdf          # blueprint/print.pdf via latexmk
leanblueprint web          # blueprint/web/ HTML + dep graph
leanblueprint serve        # http://0.0.0.0:8000/
```

Platform JSON (committed, consumed by open-theory-platform):

```bash
python3 scripts/export_blueprint.py blueprint/blueprint.json
```

`isVerified` is true only when the **proof** has `\leanok` (or a definition is `\leanok` with no proof). Statement-only `\leanok` means the claim is in Lean, possibly still `sorry`.

## GitHub Pages

After the `Compile blueprint` workflow succeeds, enable **Settings → Pages → Source: GitHub Actions**.

Expected URLs:

- `https://matthieu-perso.github.io/open-theory/`
- `https://matthieu-perso.github.io/open-theory/blueprint/`
- `https://matthieu-perso.github.io/open-theory/docs/`

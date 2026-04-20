# VLA Foundry Homepage + Documentation

This directory holds the site-build tooling: the custom homepage
(`../index.html`), the MkDocs configuration (`mkdocs.yml`), and the
local-development script (`serve_local.sh`). The published site lives at
<https://tri-ml.github.io/vla_foundry>.

## Local Development

### Quick Start

From the repo root, run:

```bash
./website/serve_local.sh
```

The script `cd`s into the repo root internally, so it works no matter where
you invoke it from. It builds the documentation, stages the homepage and
static assets into `_site/`, and serves the result on port `4001`.

Then open your browser to:
- **Homepage**: http://127.0.0.1:4001
- **Documentation**: http://127.0.0.1:4001/docs/

### Manual Steps

If you prefer to run things manually (from the repo root):

1. **Build the documentation:**
   ```bash
   mkdocs build -f website/mkdocs.yml --site-dir _site/docs
   ```

2. **Copy the homepage and paper:**
   ```bash
   cp index.html _site/
   mkdir -p _site/assets
   cp assets/vla_foundry_technical_report.pdf _site/assets/
   ```

3. **Start the server:**
   ```bash
   cd _site && python3 -m http.server 4001
   ```

## File Structure

```
/
├── index.html              # Homepage (root landing page)
├── docs/                   # MkDocs documentation source
├── assets/                 # Top-level assets served at /assets/
│   ├── logo.svg            # ...copied into _site/assets/ by the build
│   └── vla_foundry_technical_report.pdf
├── website/
│   ├── mkdocs.yml          # MkDocs configuration (docs_dir: ../docs)
│   ├── _config.yml         # Jekyll configuration (legacy)
│   ├── serve_local.sh      # Local development server script
│   └── README.md           # (this file)
└── _site/                  # Built site (auto-generated, not committed)
    ├── index.html          # Homepage (copied from root)
    ├── assets/             # Homepage assets
    └── docs/               # Built MkDocs documentation
```

## Editing the Homepage

Edit `index.html` directly. It's a self-contained HTML file with inline CSS.

### Updating the Paper PDF

The paper lives at `assets/vla_foundry_technical_report.pdf` and is
referenced from `index.html` via `href="assets/vla_foundry_technical_report.pdf"`.
To swap it out, drop a new PDF at that path (or update the `href` to point
elsewhere).

### Adding Results

Replace the placeholder in `index.html` at the "Key Results" section with
your content:

```html
<div class="content-section">
    <h2>Key Results</h2>
    <!-- Add your results here: images, charts, tables, etc. -->
</div>
```

## Editing Documentation

1. Edit markdown files in `docs/`.
2. Update navigation in `website/mkdocs.yml`.
3. Rebuild with `mkdocs build -f website/mkdocs.yml --site-dir _site/docs`.

Or use `mkdocs serve -f website/mkdocs.yml` for live reload during
documentation development:

```bash
mkdocs serve -f website/mkdocs.yml --dev-addr=127.0.0.1:8000
```

## Deploying to GitHub Pages

Deploys run automatically via `.github/workflows/deploy.yml` on every push
to `main`. The workflow builds the docs with `mkdocs build -f website/mkdocs.yml`,
stages the homepage and paper into `_site/`, and pushes the result to the
`gh-pages` branch using [`peaceiris/actions-gh-pages`](https://github.com/peaceiris/actions-gh-pages).

To point the deploy at a different repo, update `repo_url` / `site_url` in
`website/mkdocs.yml` and (if present) `website/_config.yml`.

## Notes

- The `_site/` directory is auto-generated and should not be committed to git
- Add `_site/` to your `.gitignore`
- MkDocs Material theme is already configured

# VLA Foundry Homepage + Documentation

This repository uses a combination of a custom homepage (`index.html`) and MkDocs for documentation.

## Local Development

### Quick Start

Run the local development server:

```bash
./serve_local.sh
```

Then open your browser to:
- **Homepage**: http://127.0.0.1:4001
- **Documentation**: http://127.0.0.1:4001/docs/

### Manual Steps

If you prefer to run things manually:

1. **Build the documentation:**
   ```bash
   mkdocs build --site-dir _site/docs
   ```

2. **Copy the homepage:**
   ```bash
   cp index.html _site/
   ```

3. **Start the server:**
   ```bash
   cd _site && python3 -m http.server 4001
   ```

## File Structure

```
/
├── index.html              # Homepage (root landing page)
├── _config.yml            # Jekyll configuration  
├── docs/                  # MkDocs documentation source
├── mkdocs.yml            # MkDocs configuration
├── _site/                # Built site (auto-generated, not committed)
│   ├── index.html        # Homepage (copied from root)
│   └── docs/             # Built MkDocs documentation
└── serve_local.sh        # Development server script
```

## Editing the Homepage

Edit `index.html` directly. It's a self-contained HTML file with inline CSS.

### Adding Your Paper PDF

Place your paper PDF in the root directory as `paper.pdf`, or update the link in `index.html`:

```html
<a href="paper.pdf" class="btn btn-secondary">
    📄 Read Paper
</a>
```

### Adding Results

Replace the placeholder in `index.html` at the "Key Results" section with your content:

```html
<div class="content-section">
    <h2>Key Results</h2>
    <!-- Add your results here: images, charts, tables, etc. -->
</div>
```

## Editing Documentation

1. Edit markdown files in `docs/`
2. Update navigation in `mkdocs.yml`
3. Rebuild with `mkdocs build --site-dir _site/docs`

Or use `mkdocs serve` for live reload during documentation development:

```bash
mkdocs serve --dev-addr=127.0.0.1:8000
```

## Deploying to GitHub Pages

When ready to deploy:

1. **Update repository URLs** in both `_config.yml` and `mkdocs.yml` if needed

2. **Create a GitHub Actions workflow** (`.github/workflows/deploy.yml`):
   ```yaml
   name: Deploy to GitHub Pages
   
   on:
     push:
       branches: [main]
   
   jobs:
     deploy:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         
         - name: Setup Python
           uses: actions/setup-python@v5
           with:
             python-version: '3.x'
             
         - name: Install dependencies
           run: pip install -r docs/requirements.txt
           
         - name: Build MkDocs
           run: mkdocs build --site-dir _site/docs
           
         - name: Copy homepage
           run: cp index.html _site/
           
         - name: Deploy to GitHub Pages
           uses: peaceiris/actions-gh-pages@v4
           with:
             github_token: ${{ secrets.GITHUB_TOKEN }}
             publish_dir: ./_site
   ```

3. **Enable GitHub Pages** in repository settings:
   - Go to Settings → Pages
   - Source: Deploy from a branch
   - Branch: `gh-pages` / `/ (root)`

## Notes

- The `_site/` directory is auto-generated and should not be committed to git
- Add `_site/` to your `.gitignore`
- Jekyll is configured but we're using plain HTML for simplicity
- MkDocs Material theme is already configured

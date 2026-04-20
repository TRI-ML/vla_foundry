#!/bin/bash
# Local development server for Jekyll homepage + MkDocs documentation.
# Always runs from the repo root, regardless of where it is invoked from.
set -e
cd "$(dirname "$0")/.."

echo "Building documentation..."
mkdocs build -f website/mkdocs.yml --site-dir _site/docs

echo "Copying homepage and assets..."
cp index.html _site/
mkdir -p _site/assets
mkdir -p _site/assets/carousel_failure
mkdir -p _site/assets/carousel_success
cp assets/vla_foundry_technical_report.pdf _site/assets/
cp docs/assets/logo_dark.svg _site/assets/
cp docs/assets/logo.svg _site/assets/
cp docs/assets/tri-logo-dark.png _site/assets/
cp docs/assets/favicon.png _site/assets/
cp docs/assets/fig1a.png _site/assets/
cp docs/assets/fig1b.png _site/assets/
cp docs/assets/B_hop_close_current_floor_fixed.mp4 _site/assets/ 2>/dev/null || true
cp docs/assets/carousel_failure/*.mp4 _site/assets/carousel_failure/ 2>/dev/null || true
cp docs/assets/carousel_success/*.mp4 _site/assets/carousel_success/ 2>/dev/null || true

HOSTNAME=$(hostname -I | awk '{print $1}')

echo "Starting local server..."
echo ""
echo "  Local access:"
echo "    Homepage:       http://127.0.0.1:4001"
echo "    Documentation:  http://127.0.0.1:4001/docs/"
echo ""
echo "  Network access (from other devices):"
echo "    Homepage:       http://${HOSTNAME}:4001"
echo "    Documentation:  http://${HOSTNAME}:4001/docs/"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd _site && python3 -m http.server 4001 --bind 0.0.0.0

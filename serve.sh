#!/usr/bin/env bash
# Local dev server for the Ocean Currents viewer.
# Syncs latest data from S3 (if newer) before starting.
cd "$(dirname "$0")"

echo "Checking for new data..."
Python_SSCOFS/sync_from_s3.sh
echo ""

echo "Serving at http://localhost:8000/map-viewer-mobile.html"
python -m http.server 8000

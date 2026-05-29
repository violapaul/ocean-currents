#!/usr/bin/env bash
# Local dev server for the Ocean Currents viewer.
# Syncs latest data from S3 (if newer) before starting.
cd "$(dirname "$0")"

echo "Checking for new data..."
./sync_from_s3.sh
echo ""

echo "Serving at http://localhost:8080/map-viewer-mobile.html"
python3 -m http.server 8080

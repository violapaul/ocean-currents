#!/usr/bin/env bash
# Download the latest processed current data from S3 for local development.
# No AWS credentials needed — the bucket is public-read.
#
# Usage:
#   ./sync_from_s3.sh          # download if newer data exists
#   ./sync_from_s3.sh --force  # always re-download

set -euo pipefail
cd "$(dirname "$0")"

S3_BASE="https://viola-ocean-currents.s3.us-west-2.amazonaws.com/ocean-currents"
OUT_DIR="current_data"
FORCE=false
[[ "${1:-}" == "--force" ]] && FORCE=true

# Check what we have locally
LOCAL_RUN=""
if [[ -f "$OUT_DIR/latest.json" ]]; then
  LOCAL_RUN=$(python3 -c "import json; print(json.load(open('$OUT_DIR/latest.json'))['run'])")
fi

# Check what's on S3
LATEST=$(curl -sf "$S3_BASE/latest.json")
REMOTE_RUN=$(echo "$LATEST" | python3 -c "import sys,json; print(json.load(sys.stdin)['run'])")

if [[ "$LOCAL_RUN" == "$REMOTE_RUN" && "$FORCE" == false ]]; then
  echo "Local data is already up to date ($LOCAL_RUN)."
  exit 0
fi

echo "Syncing: ${LOCAL_RUN:-none} → $REMOTE_RUN"

RUN_DIR="$OUT_DIR/$REMOTE_RUN"
mkdir -p "$RUN_DIR"

curl -sf "$S3_BASE/$REMOTE_RUN/manifest.json" -o "$RUN_DIR/manifest.json"
curl -sf "$S3_BASE/$REMOTE_RUN/geometry.bin"   -o "$RUN_DIR/geometry.bin"

HOURS=$(python3 -c "
import json
m = json.load(open('$RUN_DIR/manifest.json'))
print(' '.join(f'f{h:03d}' for h in m['forecast_hours']))
")

echo "Downloading $(echo $HOURS | wc -w | tr -d ' ') forecast files..."
for f in $HOURS; do
  curl -sf "$S3_BASE/$REMOTE_RUN/$f.bin" -o "$RUN_DIR/$f.bin" &
  while [ "$(jobs -rp | wc -l)" -ge 10 ]; do sleep 0.05; done
done
wait

echo "$LATEST" > "$OUT_DIR/latest.json"
echo "Done. Local data is now $REMOTE_RUN."

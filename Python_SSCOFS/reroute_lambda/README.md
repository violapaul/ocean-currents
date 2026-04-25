# Race Reroute Lambda

On-demand race route recompute. The PWA's GO button POSTs the boat's
current GPS to a Function URL; this Lambda runs the same `SectorRouter`
the precompute workflow uses, but with `start = current_position`, and
returns the new GeoJSON inline. Cold start ≈ 30 s; warm solve ≈ 25-45 s
for typical mid-race distances.

Lives at `OceanCurrents/Python_SSCOFS/reroute_lambda/`.

## Files

- `Dockerfile` — container image (`public.ecr.aws/lambda/python:3.11`-based).
- `handler.py` — `def handler(event, context)`. Wired to AWS_PROXY-style Function URL events.
- `deploy.sh` — build + push + Lambda code update.
- `setup.sh` — one-time AWS provisioning (ECR, IAM role, Lambda fn, Function URL).
- `test_event.json` — sample Function URL event for local handler testing.

## First-time deploy

Two paths; pick whichever fits. Both leave you with the same Lambda + Function URL.

### Path A — local Docker (OrbStack / Docker Desktop)

Prereqs: `aws` CLI configured with admin perms, a container runtime (`brew install orbstack` is the lightest).

```bash
cd OceanCurrents/Python_SSCOFS/reroute_lambda

# 1. Create ECR repo + IAM role (no Lambda fn yet — needs an image first).
./setup.sh                # will warn that the image isn't in ECR yet; that's fine

# 2. Build + push image to ECR.
./deploy.sh push

# 3. Re-run setup to create the Lambda fn + Function URL.
./setup.sh
# -> prints the Function URL when done.
```

### Path B — cloud build via GitHub Actions (no local Docker)

```bash
cd OceanCurrents/Python_SSCOFS/reroute_lambda

# 1. Create ECR repo + IAM role (skips Lambda; same as Path A step 1).
./setup.sh

# 2. Grant the existing GHA IAM user perms to push to ECR + update Lambda.
./grant_gha_perms.sh

# 3. Trigger the deploy workflow (builds + pushes from ubuntu-latest).
gh workflow run deploy-reroute-lambda.yml --ref main

# 4. Re-run setup to create the Lambda fn now that the image exists.
./setup.sh
# -> prints the Function URL.
```

### Wiring the URL into the PWA

After either path:

```bash
# Replace LAMBDA_REROUTE_URL in map-viewer-mobile.html with the Function URL
sed -i '' "s|LAMBDA_REROUTE_URL: ''|LAMBDA_REROUTE_URL: '<paste URL here>'|" OceanCurrents/map-viewer-mobile.html

# Commit + redeploy to GitHub Pages
git add OceanCurrents/map-viewer-mobile.html
git commit -m "Wire reroute Lambda URL into PWA"
git push origin main
git push ocean-currents $(git subtree split --prefix=OceanCurrents):main --force
```

## Iterating on handler code

```bash
./deploy.sh deploy   # build + push + update-function-code
aws logs tail /aws/lambda/race-reroute --follow --region us-west-2
```

## Local handler test (no AWS)

The AWS Lambda Runtime Interface Emulator lets us hit the handler locally
through the same Function URL contract. Build and run:

```bash
./deploy.sh build
docker run --rm -p 9000:8080 race-reroute:latest

# In another terminal:
curl -sf -X POST 'http://localhost:9000/2015-03-31/functions/function/invocations' \
  -d @test_event.json | jq -r '.body' | jq '.summary'
```

Expected:
```json
{
  "remaining_distance_nm": 17.x,
  "remaining_time_hr": 2.x,
  "legs": 2,
  "next_waypoint_index": 1
}
```

## Request schema

```jsonc
POST /  (Function URL)
Content-Type: application/json
{
  "race_slug":            "shilshole_double_bluff_return",
  "start":                { "lat": 47.85, "lon": -122.48 },
  "depart_utc":           "2026-04-25T17:30:00Z",  // optional, default = now()
  "next_waypoint_index":  1                          // optional, default = 1
}
```

`next_waypoint_index` semantics: the index of the *next* waypoint the user
intends to round, into the original race YAML's waypoint list. `1` means
"go from current GPS to all marks except start" (the common case). `2`
means "user has already rounded mark 1; resume from mark 2." `0` is
reserved/invalid.

## Response schema

```jsonc
200 OK
{
  "slug":          "shilshole_double_bluff_return",
  "reroute":       true,
  "generated_at":  "2026-04-25T17:30:42Z",
  "summary": {
    "remaining_distance_nm":  17.4,
    "remaining_time_hr":      2.83,
    "legs":                   2,
    "next_waypoint_index":    1
  },
  "geojson":  { /* FeatureCollection, same schema as race_publish.write_geojson */ }
}
```

Errors return `4xx`/`5xx` with `{"error": "..."}` JSON body.

## Operational notes

- IAM: Lambda role gets `s3:GetObject` on the SSCOFS bucket only. **No write perms** — keeps the publish path (precompute workflow) and the read path (this Lambda) cleanly separated.
- Memory: 3008 MB. The router is CPU-bound, not memory-bound, but Lambda
  scales vCPU with memory; bumping memory cuts solve time roughly 1:1.
- Timeout: 90 s. Race solves in 25-45 s typically; cold start adds another
  ~30 s. The 90 s ceiling gives headroom for SSCOFS fetch jitter.
- Architecture: amd64. Most numpy/scipy/pyproj stacks have arm64 wheels
  now too — switch later for ~20% cost savings if all deps cooperate.
- Concurrency: Lambda default (1000). Race-day spike is tiny (one or two
  taps per boat); no need to provision concurrency.

## Cost estimate

- Cold start ≈ 30 s × 3008 MB = ~90 GB-s.
- Warm invocation ≈ 35 s × 3008 MB = ~105 GB-s.
- Lambda price: ~$0.0000167 per GB-s.
- **Per-tap cost: ~$0.0015 cold, ~$0.0018 warm.** A whole race day of
  testing fits in pennies. ECR storage: ~$0.01/GB-month for the image.

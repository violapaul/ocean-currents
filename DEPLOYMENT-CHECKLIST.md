# Deployment Checklist

Follow these steps in order to deploy the Ocean Currents viewer.

## Pre-Deployment

- [ ] Verify all files are present in this directory:
  - [ ] index.html
  - [ ] map-viewer-mobile.html
  - [ ] manifest.json
  - [ ] app-icon.svg
  - [ ] app-icon-192.png
  - [ ] app-icon-512.png
  - [ ] apple-touch-icon.png
  - [ ] .nojekyll
  - [ ] proxy-worker.js
  - [ ] wrangler.toml

## Cloudflare Worker Deployment

- [ ] Create Cloudflare account at https://dash.cloudflare.com/sign-up
- [ ] Install Wrangler: `npm install -g wrangler`
- [ ] Login to Cloudflare: `wrangler login`
- [ ] Deploy Worker: `wrangler deploy`
- [ ] Note your Worker URL: ________________________.workers.dev
- [ ] Test Worker health: https://YOUR-WORKER.workers.dev/healthz

## Update Configuration

- [ ] Open `map-viewer-mobile.html`
- [ ] Find line ~220 with `YOUR-CLOUDFLARE-SUBDOMAIN`
- [ ] Replace with your actual Cloudflare subdomain
- [ ] Save the file

## GitHub Pages Deployment

- [ ] Create new GitHub repository
- [ ] Initialize git: `git init`
- [ ] Add files: `git add .`
- [ ] Commit: `git commit -m "Initial deployment"`
- [ ] Add remote: `git remote add origin https://github.com/USERNAME/REPO.git`
- [ ] Push: `git push -u origin main`
- [ ] Enable GitHub Pages in repository Settings
- [ ] Select main branch and root folder
- [ ] Wait for deployment (check Actions tab)
- [ ] Note your GitHub Pages URL: https://________________________.github.io/________________________/

## Testing

### Desktop Testing
- [ ] Open GitHub Pages URL in Chrome/Firefox
- [ ] Open browser console (F12)
- [ ] Verify no CORS errors
- [ ] Check that map loads
- [ ] Verify current overlay tiles appear
- [ ] Test time slider functionality

### Mobile Testing
- [ ] Open URL on iPhone/iPad
- [ ] Open URL on Android device
- [ ] Test touch gestures (pan, zoom)
- [ ] Test time slider
- [ ] Try "Add to Home Screen"
- [ ] Launch from home screen
- [ ] Verify full-screen mode works

## Final Verification

- [ ] Cloudflare Worker responding to requests
- [ ] GitHub Pages site accessible
- [ ] Tiles loading without CORS errors
- [ ] PWA installable on mobile devices
- [ ] Time slider updates currents display
- [ ] Model information shows in header

## Post-Deployment

- [ ] Check Cloudflare analytics after 24 hours
- [ ] Monitor for any error reports
- [ ] Document the final URLs:

### Your Deployed URLs

**GitHub Pages (App):**
```
https://________________________.github.io/________________________/
```

**Cloudflare Worker (Proxy):**
```
https://________________________.workers.dev
```

**Date Deployed:** ________________________

## Notes

Add any deployment-specific notes here:

---

Deployment completed successfully! 🎉

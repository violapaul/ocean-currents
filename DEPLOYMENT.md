# Deployment Guide

This guide covers deploying the Ocean Currents viewer using GitHub Pages and Cloudflare Worker.

## Prerequisites

- GitHub account
- Cloudflare account (free)
- Node.js installed (for Wrangler CLI)

## Step 1: Deploy Cloudflare Worker Proxy

### 1.1 Install Wrangler CLI
```bash
npm install -g wrangler
```

### 1.2 Login to Cloudflare
```bash
wrangler login
```

### 1.3 Set up workers.dev subdomain
- Go to https://dash.cloudflare.com
- Navigate to "Workers & Pages"
- Choose your subdomain (e.g., "yourname" → `*.yourname.workers.dev`)
- This is permanent for your account

### 1.4 Deploy the Worker
```bash
cd ocean-currents-deploy/
wrangler deploy
```

You'll get a URL like: `https://ocean-currents-proxy.yourname.workers.dev`

### 1.5 Update the Proxy URL
Edit `map-viewer-mobile.html` line ~228:
```javascript
// Change from:
return 'https://ocean-currents-proxy.YOUR-CLOUDFLARE-SUBDOMAIN.workers.dev';

// To your actual subdomain:
return 'https://ocean-currents-proxy.yourname.workers.dev';
```

## Step 2: Deploy to GitHub Pages

### 2.1 Create GitHub Repository
- Go to https://github.com/new
- Name: `ocean-currents` (or any name you prefer)
- Keep it Public
- DON'T initialize with README

### 2.2 Push Code to GitHub
```bash
cd ocean-currents-deploy/
git init
git add .
git commit -m "Deploy Ocean Currents viewer"
git remote add origin https://github.com/YOUR-USERNAME/ocean-currents.git
git branch -M main
git push -u origin main
```

### 2.3 Enable GitHub Pages
1. Go to your repository on GitHub
2. Click Settings → Pages (left sidebar)
3. Source: Deploy from a branch
4. Branch: main, Folder: / (root)
5. Click Save

Wait 1-2 minutes for deployment.

## Step 3: Test Your Deployment

### Desktop Browser
- Visit: `https://YOUR-USERNAME.github.io/ocean-currents/`
- Open browser console (F12) to check for errors
- Verify map loads and tiles appear

### Mobile Device
- Open the URL on iOS/Android
- Test touch gestures and time slider
- Install as PWA:
  - iOS: Share → Add to Home Screen
  - Android: Menu → Install App

## Deployed URLs

After successful deployment:
- **App**: `https://YOUR-USERNAME.github.io/ocean-currents/`
- **Proxy**: `https://ocean-currents-proxy.YOUR-SUBDOMAIN.workers.dev/`

## Updating the App

To update after making changes:

### Update Worker
```bash
wrangler deploy
```

### Update GitHub Pages
```bash
git add .
git commit -m "Update description"
git push
```

GitHub Pages auto-deploys on push (1-2 minutes).

## Troubleshooting

### Tiles Not Loading
- Check Cloudflare Worker is deployed: `wrangler tail`
- Verify proxy URL is correct in HTML file
- Test proxy directly: `https://your-proxy.workers.dev/healthz`
- Check browser console for CORS errors

### GitHub Pages Not Working
- Ensure repository is Public
- Verify Pages is enabled in Settings
- Wait a few minutes for initial deployment
- Check Actions tab for deployment status

### PWA Not Installing
- Must be served over HTTPS (GitHub Pages handles this)
- Check manifest.json is valid
- Verify all icon files are present

## Monitoring

### Cloudflare Analytics
- Dashboard → Workers & Pages → Your Worker → Analytics
- Shows requests, errors, response times

### GitHub Pages
- No built-in analytics
- Repository → Insights → Traffic shows visitor counts

## Configuration Files

### wrangler.toml
```toml
name = "ocean-currents-proxy"
main = "proxy-worker.js"
compatibility_date = "2024-01-01"
workers_dev = true
account_id = "your-account-id"
```

### manifest.json
```json
{
  "name": "Ocean Currents Viewer",
  "short_name": "Currents",
  "start_url": "./map-viewer-mobile.html",
  "display": "standalone",
  "orientation": "portrait",
  "theme_color": "#0077be"
}
```

## Free Tier Limits

- **Cloudflare Worker**: 100,000 requests/day
- **GitHub Pages**: 100GB bandwidth/month

Both are very generous for this application.

## Support

If you encounter issues:
1. Check browser console for JavaScript errors
2. Verify all URLs are correctly configured
3. Test proxy endpoint separately
4. Review Cloudflare Worker logs: `wrangler tail`

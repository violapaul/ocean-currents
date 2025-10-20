# 🚀 Cloudflare Worker Setup - Quick Fix

## The Issue
You need to register a workers.dev subdomain before you can deploy. This is a one-time setup for your Cloudflare account.

## ✅ Quick Solution (2 minutes)

### Step 1: Register Your Workers Subdomain

1. **Open this link in your browser:**
   ```
   https://dash.cloudflare.com/6f08cd01caaee709588f3568a5c8cc6f/workers/onboarding
   ```

2. **Choose a subdomain** (this will be permanent for your account):
   - Example: `viola` → gives you `*.viola.workers.dev`
   - Pick something short and memorable
   - This subdomain is yours forever on Cloudflare

3. **Click "Continue" to confirm**

### Step 2: Deploy Your Worker

Once you have your subdomain, run the deploy command again:

```bash
cd "/Users/viola/Resilio Sync/Dropbox/Python/WaysWaterMoves/Web/deployment/ocean-currents-deploy"
wrangler deploy
```

You should see output like:
```
Uploaded ocean-currents-proxy
Published ocean-currents-proxy
https://ocean-currents-proxy.YOUR-SUBDOMAIN.workers.dev
```

### Step 3: Update Your HTML File

1. Open `map-viewer-mobile.html`
2. Find line ~220 with `YOUR-CLOUDFLARE-SUBDOMAIN`
3. Replace with your actual subdomain
4. Save the file

Example:
```javascript
// Change from:
return 'https://ocean-currents-proxy.YOUR-CLOUDFLARE-SUBDOMAIN.workers.dev';

// To (if your subdomain is "viola"):
return 'https://ocean-currents-proxy.viola.workers.dev';
```

## 📝 Important Notes

- **The subdomain is account-wide**: Once you pick it, all your workers will use `*.your-subdomain.workers.dev`
- **It's permanent**: Choose carefully, you can't change it later
- **Free tier is generous**: 100,000 requests per day

## 🎯 Next Steps

After successful deployment:
1. Test the proxy: `https://ocean-currents-proxy.YOUR-SUBDOMAIN.workers.dev/healthz`
2. Push to GitHub Pages
3. Test the complete app

## 🆘 Troubleshooting

If you still get errors:
- Make sure you're logged in: `wrangler login`
- Check your account ID is correct
- Try `wrangler whoami` to verify your account

Your deployment is almost ready - just need this one-time subdomain setup! 🌊

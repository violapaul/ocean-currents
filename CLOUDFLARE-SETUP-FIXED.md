# 🚀 Cloudflare Worker Setup - Updated Instructions

## The Issue
You need to register a workers.dev subdomain before deploying. The onboarding URL has changed.

## ✅ Solution: Set Up Your Subdomain

### Option 1: Via Cloudflare Dashboard (Recommended)

1. **Go to Cloudflare Dashboard:**
   ```
   https://dash.cloudflare.com
   ```

2. **Navigate to Workers & Pages:**
   - Look for "Workers & Pages" in the left sidebar
   - Or try this direct link: https://dash.cloudflare.com/?to=/:account/workers

3. **Create your subdomain:**
   - If you haven't set up a subdomain yet, you'll see a prompt
   - Choose your subdomain (e.g., "viola" gives you `*.viola.workers.dev`)
   - This is permanent for your account

### Option 2: Via Wrangler CLI (Interactive)

Try running the deploy command with interactive mode:

```bash
cd "/Users/viola/Resilio Sync/Dropbox/Python/WaysWaterMoves/Web/deployment/ocean-currents-deploy"

# This should prompt you to set up subdomain if needed
wrangler deploy --yes
```

When prompted "Would you like to register a workers.dev subdomain now?", type `yes` and follow the prompts.

### Option 3: Check Your Account Settings

Your subdomain might already be set up. Check with:

```bash
wrangler whoami
```

This will show your account details and subdomain if it exists.

## 🔍 Finding Your Subdomain

If you already have a subdomain but don't remember it:

1. **Via Dashboard:**
   - Go to https://dash.cloudflare.com
   - Click on "Workers & Pages"
   - Your subdomain will be shown in the overview

2. **Via CLI:**
   ```bash
   wrangler whoami
   ```

## 📝 After Setting Up Your Subdomain

Once you have your subdomain (let's say it's "myname"):

1. **Deploy the worker:**
   ```bash
   wrangler deploy
   ```
   
   You should see:
   ```
   Published ocean-currents-proxy
   https://ocean-currents-proxy.myname.workers.dev
   ```

2. **Update map-viewer-mobile.html:**
   
   Find line ~220:
   ```javascript
   // Change from:
   return 'https://ocean-currents-proxy.YOUR-CLOUDFLARE-SUBDOMAIN.workers.dev';
   
   // To (using your actual subdomain):
   return 'https://ocean-currents-proxy.myname.workers.dev';
   ```

3. **Test the proxy:**
   ```
   https://ocean-currents-proxy.myname.workers.dev/healthz
   ```
   
   Should return: `ok`

## 🆘 Alternative: Force Subdomain Setup

If the dashboard isn't working, try this direct approach:

```bash
# Login first
wrangler login

# Try to list your workers (this might trigger subdomain setup)
wrangler list

# Or try publishing with a specific route
wrangler deploy --compatibility-date 2024-01-01
```

## 📞 If Still Having Issues

1. **Clear Wrangler cache and re-login:**
   ```bash
   wrangler logout
   wrangler login
   ```

2. **Check your account type:**
   - Make sure you have a Cloudflare account (not just a domain account)
   - Free accounts work fine for Workers

3. **Try the direct Workers dashboard:**
   ```
   https://workers.cloudflare.com
   ```

## ✅ Success Indicators

You'll know it worked when:
- `wrangler deploy` completes without subdomain errors
- You get a URL like: `https://ocean-currents-proxy.YOUR-SUBDOMAIN.workers.dev`
- The healthz endpoint responds with "ok"

Your worker is ready to deploy - we just need to get past this subdomain setup! 🌊

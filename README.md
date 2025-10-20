# Ocean Currents Mobile Viewer Deployment

This directory contains all the files needed to deploy the Ocean Currents mobile viewer using GitHub Pages for hosting and Cloudflare Worker for the proxy.

## 📁 Files Included

- `index.html` - Landing page that redirects to mobile viewer
- `map-viewer-mobile.html` - Main mobile PWA application
- `manifest.json` - PWA manifest for installability
- `app-icon.svg` - Vector app icon
- `app-icon-192.png` - PWA icon (192x192)
- `app-icon-512.png` - PWA icon (512x512)
- `apple-touch-icon.png` - iOS home screen icon
- `.nojekyll` - Prevents GitHub Pages from processing files
- `proxy-worker.js` - Cloudflare Worker proxy script
- `wrangler.toml` - Cloudflare Worker configuration

## 🚀 Deployment Steps

### Step 1: Deploy Cloudflare Worker (Proxy)

1. **Create a Cloudflare account** (free):
   - Go to https://dash.cloudflare.com/sign-up
   - Verify your email

2. **Install Wrangler CLI**:
   ```bash
   npm install -g wrangler
   ```

3. **Login to Cloudflare**:
   ```bash
   wrangler login
   ```
   This will open your browser to authenticate.

4. **Deploy the Worker**:
   ```bash
   # From this directory
   wrangler deploy
   ```
   
   You'll see output like:
   ```
   Uploaded ocean-currents-proxy (1.23 sec)
   Published ocean-currents-proxy (0.45 sec)
   https://ocean-currents-proxy.YOUR-SUBDOMAIN.workers.dev
   ```

5. **Note your Worker URL** - you'll need this for the next step!

### Step 2: Update the Proxy URL

1. **Edit `map-viewer-mobile.html`**:
   - Find line ~220 where it says:
     ```javascript
     return 'https://ocean-currents-proxy.YOUR-CLOUDFLARE-SUBDOMAIN.workers.dev';
     ```
   - Replace `YOUR-CLOUDFLARE-SUBDOMAIN` with your actual subdomain from Step 1

2. **Save the file**

### Step 3: Deploy to GitHub Pages

1. **Create a new GitHub repository**:
   - Go to https://github.com/new
   - Name it something like `ocean-currents`
   - Make it public
   - Don't initialize with README (we have our own)

2. **Push the files**:
   ```bash
   # Initialize git in this directory
   git init
   
   # Add all files
   git add .
   
   # Commit
   git commit -m "Initial deployment of Ocean Currents viewer"
   
   # Add your GitHub repo as origin
   git remote add origin https://github.com/YOUR-USERNAME/ocean-currents.git
   
   # Push to main branch
   git push -u origin main
   ```

3. **Enable GitHub Pages**:
   - Go to your repository on GitHub
   - Click **Settings** tab
   - Scroll down to **Pages** section (left sidebar)
   - Under **Source**, select:
     - Deploy from a branch: `main`
     - Folder: `/ (root)`
   - Click **Save**

4. **Wait for deployment** (1-2 minutes):
   - GitHub will show a green checkmark when ready
   - Your site will be available at:
     ```
     https://YOUR-USERNAME.github.io/ocean-currents/
     ```

## ✅ Testing Your Deployment

### Desktop Browser
1. Open `https://YOUR-USERNAME.github.io/ocean-currents/`
2. Check browser console (F12) for any errors
3. Verify tiles are loading (you should see ocean current overlays)

### Mobile Device (iOS/Android)
1. Open the URL on your mobile device
2. The interface should be optimized for mobile
3. Try the time slider to change forecast hours
4. **Install as PWA**:
   - **iOS**: Tap Share → Add to Home Screen
   - **Android**: Menu → Install App or Add to Home Screen

## 🔍 Troubleshooting

### Tiles Not Loading
- Check browser console for CORS errors
- Verify your Cloudflare Worker URL is correct in `map-viewer-mobile.html`
- Test the proxy directly: `https://YOUR-WORKER.workers.dev/healthz`

### GitHub Pages Not Working
- Make sure repository is public
- Check that GitHub Pages is enabled in Settings
- Wait a few minutes for initial deployment
- Try hard refresh (Ctrl+Shift+R or Cmd+Shift+R)

### PWA Not Installing
- Must be served over HTTPS (GitHub Pages does this)
- Check that all icon files are present
- Verify manifest.json is valid JSON

## 📊 Monitoring

### Cloudflare Analytics
1. Log into Cloudflare dashboard
2. Go to Workers & Pages
3. Click on your worker
4. View Analytics tab for:
   - Request count
   - Error rate
   - Response times

### GitHub Pages
- No built-in analytics
- Consider adding Google Analytics if needed

## 🔧 Making Updates

To update the app after initial deployment:

1. Make changes to files locally
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push
   ```
3. GitHub Pages will automatically redeploy (1-2 minutes)

To update the Cloudflare Worker:
```bash
wrangler deploy
```

## 📝 Notes

- **Free Tier Limits**:
  - Cloudflare Worker: 100,000 requests/day
  - GitHub Pages: 100GB bandwidth/month
  
- **No Custom Domain Required**: Both services provide their own URLs

- **Data Source**: Tiles are proxied from coral.apl.uw.edu (NOAA/UW data)

## 🆘 Support

If you encounter issues:
1. Check the browser console for errors
2. Verify all URLs are correct
3. Ensure Cloudflare Worker is deployed and accessible
4. Check that GitHub Pages is enabled

## 📜 License

This viewer interfaces with NOAA public data services. Refer to NOAA's data usage policies for commercial applications.

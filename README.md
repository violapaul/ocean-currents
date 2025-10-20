# Ocean Currents Mobile Viewer

A mobile-optimized Progressive Web App (PWA) for viewing real-time ocean current forecasts from NOAA's Salish Sea Coastal Ocean Forecast System (SSCOFS).

## 🌊 Live Demo

**App**: https://violapaul.github.io/ocean-currents/

## Features

- **Real-time ocean current visualization** for Puget Sound
- **Mobile-optimized interface** with touch-friendly controls
- **Time slider** to view forecasts up to 72 hours ahead
- **Progressive Web App** - installable on iOS and Android
- **Automatic model detection** - always shows the latest available SSCOFS data

## Architecture

- **Static Hosting**: GitHub Pages (free, HTTPS, CDN)
- **Proxy Server**: Cloudflare Worker (serverless, handles CORS)
- **Data Source**: NOAA/UW coral.apl.uw.edu tile server
- **Map Library**: MapLibre GL JS

## Files

- `index.html` - Landing page with redirect
- `map-viewer-mobile.html` - Main PWA application
- `manifest.json` - PWA configuration
- `app-icon-*.png/svg` - App icons for various platforms
- `proxy-worker.js` - Cloudflare Worker proxy script
- `wrangler.toml` - Cloudflare Worker configuration

## How It Works

1. User visits the GitHub Pages site
2. Mobile-optimized interface loads with MapLibre GL
3. JavaScript requests ocean current tiles through Cloudflare Worker proxy
4. Proxy bypasses CORS restrictions and fetches tiles from NOAA server
5. Tiles are displayed as overlays on the map
6. Time slider allows viewing different forecast hours

## SSCOFS Model Information

The app automatically detects and uses the latest available SSCOFS model run:
- Model runs: 00z, 03z, 09z, 15z, 21z UTC daily
- Forecast range: 0-72 hours
- Data typically available 3-5 hours after model run time

## Development

To run locally:

1. Start a local proxy server (if testing tile loading):
```bash
cd Web/wwm-proxy
npm install
node server.js
```

2. Open `map-viewer-mobile.html` in a browser

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## URLs

- **Production App**: https://violapaul.github.io/ocean-currents/
- **Cloudflare Proxy**: https://ocean-currents-proxy.violapaul.workers.dev
- **GitHub Repository**: https://github.com/violapaul/ocean-currents

## License

This viewer interfaces with NOAA public data services. Refer to NOAA's data usage policies for commercial applications.

## Credits

- Ocean current data: NOAA/NOS/CO-OPS Salish Sea Coastal Ocean Forecast System
- Tile service: University of Washington Applied Physics Lab
- Map library: MapLibre GL JS
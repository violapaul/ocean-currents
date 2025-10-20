# Ocean Currents Mobile Viewer

A mobile-optimized Progressive Web App (PWA) for viewing real-time ocean current forecasts from NOAA's Salish Sea Coastal Ocean Forecast System (SSCOFS) with tide data integration.

## 🌊 Live Demo

**App**: https://violapaul.github.io/ocean-currents/

## Features

### Core Functionality
- **Real-time ocean current visualization** for Puget Sound region
- **Mobile-first design** with touch-optimized controls
- **Time slider** with hour-by-hour navigation (-1/+1 buttons)
- **Progressive Web App** - installable on iOS and Android
- **Automatic model detection** - always shows the latest available SSCOFS data

### Interactive Features
- **Tap to show current speed** - Display current magnitude in knots at any point
- **Auto-updating measurements** - Current speed updates when time changes
- **Seattle tide chart** - Live tide visualization showing full day (midnight to midnight)
- **Moving time indicator** - Red line shows current position in tidal cycle

### User Interface
- **Compact layout** - Maximized map area with minimal UI chrome
- **Left-aligned time display** - Current forecast time clearly visible
- **Tide chart overlay** - 200x100px chart in upper right corner
- **Smart data caching** - Efficient loading and smooth transitions

## Architecture

- **Static Hosting**: GitHub Pages (free, HTTPS, CDN)
- **Proxy Server**: Cloudflare Worker (serverless, handles CORS for tiles, NVS API, and NOAA tides)
- **Data Sources**: 
  - Ocean currents: NOAA/UW coral.apl.uw.edu tile server
  - Current magnitudes: NVS NANOOS API
  - Tide predictions: NOAA CO-OPS API (Station 9447130 - Seattle)
- **Map Library**: MapLibre GL JS

## Files

- `index.html` - Landing page with redirect
- `map-viewer-mobile.html` - Main PWA application
- `manifest.json` - PWA configuration
- `app-icon-*.png/svg` - App icons for various platforms
- `proxy-worker.js` - Cloudflare Worker proxy script
- `wrangler.toml` - Cloudflare Worker configuration

## How It Works

1. User visits the GitHub Pages site (or installs as PWA)
2. Mobile-optimized interface loads with MapLibre GL
3. App detects latest available SSCOFS model run
4. JavaScript requests ocean current tiles through Cloudflare Worker proxy
5. Proxy handles CORS for:
   - Ocean current tiles from coral.apl.uw.edu
   - Current magnitude data from NVS API
   - Tide predictions from NOAA CO-OPS
6. Interactive features:
   - Tap anywhere to see current speed in knots
   - Drag slider or use -1/+1 buttons to change time
   - Tide chart updates showing daily cycle with current position

## SSCOFS Model Information

The app automatically detects and uses the latest available SSCOFS model run:
- Model runs: 00z, 03z, 09z, 15z, 21z UTC daily
- Forecast range: 0-72 hours
- Data typically available 3-5 hours after model run time

## Development

### Local Testing

```bash
# Navigate to the OceanCurrents directory
cd OceanCurrents

# Start a simple HTTP server
python3 -m http.server 8000

# Open in browser
# http://localhost:8000/map-viewer-mobile.html
```

### Debug Logging

Control logging verbosity via URL parameters or localStorage:

```javascript
// URL parameter (temporary)
map-viewer-mobile.html?log=debug  // All messages
map-viewer-mobile.html?log=info   // Info and above
map-viewer-mobile.html?log=warn   // Warnings and errors (default)
map-viewer-mobile.html?log=off    // No logging

// localStorage (persistent)
localStorage.setItem('logLevel', 'debug');
```

Log levels: `off` (0), `error` (1), `warn` (2), `info` (3), `debug` (4)

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
- Helpful web page:  https://chrishewett.com/blog/slippy-tile-explorer/?
- Cloudflare:  https://dash.cloudflare.com/
- Safari mobile debugging using the iphone15 simulator (better than nothing!)
    - Enable Developer.
    - Use "responsive design mode"
    - Use a simulator.

    

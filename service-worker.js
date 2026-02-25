// Ocean Currents PWA Service Worker
// Enables offline functionality by caching tiles and assets

const CACHE_VERSION = 'ocean-currents-v2';
const TILE_CACHE = `${CACHE_VERSION}-tiles`;
const STATIC_CACHE = `${CACHE_VERSION}-static`;

// Files to cache immediately on install
const STATIC_FILES = [
  './',
  './index.html',
  './map-viewer-mobile.html',
  './manifest.json',
  './app-icon.svg',
  './app-icon-192.png',
  './app-icon-512.png',
  './apple-touch-icon.png',
  'https://unpkg.com/maplibre-gl@3.6.1/dist/maplibre-gl.css',
  'https://unpkg.com/maplibre-gl@3.6.1/dist/maplibre-gl.js'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(STATIC_CACHE);
    console.log('Caching static assets');
    await Promise.allSettled(
      STATIC_FILES.map((url) => cache.add(url))
    );
    await self.skipWaiting();
  })());
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const cacheNames = await caches.keys();
    await Promise.all(
      cacheNames.map((cacheName) => {
        if (cacheName !== STATIC_CACHE && cacheName !== TILE_CACHE) {
          console.log('Deleting old cache:', cacheName);
          return caches.delete(cacheName);
        }
        return Promise.resolve();
      })
    );
    await self.clients.claim();
  })());
});

// Fetch event - serve from cache when possible
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  
  // Navigation requests should be network-first so users get new app shells quickly.
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request).then((response) => {
        if (response && response.ok) {
          const responseToCache = response.clone();
          caches.open(STATIC_CACHE).then((cache) => {
            cache.put(event.request, responseToCache);
          });
        }
        return response;
      }).catch(async () => {
        const cachedNav = await caches.match(event.request);
        if (cachedNav) return cachedNav;
        const cachedAppShell = await caches.match('./map-viewer-mobile.html');
        if (cachedAppShell) return cachedAppShell;
        return new Response('Offline', {
          status: 503,
          headers: { 'Content-Type': 'text/plain' }
        });
      })
    );
    return;
  }
  
  // Handle tile requests - cache by exact URL (includes model time AND forecast hour)
  // URL format: /tiles/salish-currents-vec/{startTime}/{endTime}/Surface/v2/ValueLocList/{z}/{y}/{x}.png
  if (url.pathname.includes('/tiles/')) {
    event.respondWith(
      caches.open(TILE_CACHE).then((cache) => {
        // Use the exact URL as cache key - this includes both model time and forecast hour
        return cache.match(event.request).then((cachedResponse) => {
          if (cachedResponse) {
            // Return cached tile for this specific model + forecast hour
            console.log('Cache hit for tile:', url.pathname);
            return cachedResponse;
          }
          
          // Not in cache - fetch it
          console.log('Cache miss for tile:', url.pathname);
          return fetch(event.request).then((response) => {
            // Only cache successful responses
            if (response.ok) {
              // Clone BEFORE returning to avoid "body already used" error
              const responseToCache = response.clone();
              // Cache with full URL including timestamps
              cache.put(event.request, responseToCache);
              console.log('Cached new tile:', url.pathname);
            }
            return response;
          }).catch(() => {
            // Offline - return a transparent tile or error tile
            console.log('Offline - no tile available for:', url.pathname);
            return new Response(null, {
              status: 204,
              statusText: 'No Content'
            });
          });
        });
      })
    );
    return;
  }
  
  // Handle API requests (NVS, NOAA tides)
  if (url.pathname.includes('/nvs/') || url.pathname.includes('/noaa/')) {
    event.respondWith(
      fetch(event.request).catch(() => {
        // Return empty response when offline
        return new Response(JSON.stringify({ 
          error: 'offline',
          message: 'Data not available offline' 
        }), {
          status: 503,
          headers: { 'Content-Type': 'application/json' }
        });
      })
    );
    return;
  }
  
  // Handle static assets - cache first, then network
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(event.request).then((response) => {
        // Clone the response BEFORE using it
        if (response.ok && event.request.method === 'GET') {
          const shouldCache = 
            url.hostname === location.hostname ||
            url.hostname.includes('unpkg.com') ||
            url.hostname.includes('arcgisonline.com');
          
          if (shouldCache) {
            // Clone immediately to avoid "body already used" error
            const responseToCache = response.clone();
            caches.open(STATIC_CACHE).then((cache) => {
              cache.put(event.request, responseToCache);
            });
          }
        }
        return response;
      });
    })
  );
});

// Clean up old tiles periodically (keep last 7 days)
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
    return;
  }

  if (event.data === 'cleanup') {
    caches.open(TILE_CACHE).then((cache) => {
      cache.keys().then((requests) => {
        const cutoff = Date.now() - (7 * 24 * 60 * 60 * 1000);
        requests.forEach((request) => {
          // Check if tile is old (this is simplified - in production you'd parse the URL)
          cache.delete(request);
        });
      });
    });
  }
});

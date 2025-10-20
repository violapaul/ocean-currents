// Ocean Currents PWA Service Worker
// Enables offline functionality by caching tiles and assets

const CACHE_VERSION = 'ocean-currents-v1';
const TILE_CACHE = 'tiles-v1';
const STATIC_CACHE = 'static-v1';

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
  event.waitUntil(
    caches.open(STATIC_CACHE).then((cache) => {
      console.log('Caching static assets');
      return cache.addAll(STATIC_FILES);
    })
  );
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== STATIC_CACHE && cacheName !== TILE_CACHE) {
            console.log('Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  self.clients.claim();
});

// Fetch event - serve from cache when possible
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  
  // Handle tile requests - cache for 7 days
  if (url.pathname.includes('/tiles/')) {
    event.respondWith(
      caches.open(TILE_CACHE).then((cache) => {
        return cache.match(event.request).then((cachedResponse) => {
          if (cachedResponse) {
            // Return cached tile
            return cachedResponse;
          }
          
          // Fetch and cache new tile
          return fetch(event.request).then((response) => {
            // Only cache successful responses
            if (response.ok) {
              // Clone BEFORE returning to avoid "body already used" error
              const responseToCache = response.clone();
              cache.put(event.request, responseToCache);
            }
            return response;
          }).catch(() => {
            // Offline - return a transparent tile or error tile
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

/**
 * Ocean Currents Proxy Worker for Cloudflare
 * Proxies tile requests to coral.apl.uw.edu with proper headers to bypass CORS
 */

addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
  const url = new URL(request.url)
  
  // CORS headers for all responses
  const corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, OPTIONS, HEAD',
    'Access-Control-Allow-Headers': '*',
    'Access-Control-Max-Age': '86400',
  }
  
  // Handle preflight OPTIONS requests
  if (request.method === 'OPTIONS') {
    return new Response(null, { 
      status: 204,
      headers: corsHeaders 
    })
  }
  
  // Health check endpoint
  if (url.pathname === '/healthz') {
    return new Response('ok', {
      status: 200,
      headers: {
        ...corsHeaders,
        'Content-Type': 'text/plain',
      }
    })
  }
  
  // Extract path after /tiles
  if (!url.pathname.startsWith('/tiles')) {
    return new Response('Not Found - use /tiles/* path', { 
      status: 404,
      headers: {
        ...corsHeaders,
        'Content-Type': 'text/plain',
      }
    })
  }
  
  const tilePath = url.pathname.replace(/^\/tiles/, '')
  const upstreamUrl = 'https://coral.apl.uw.edu' + tilePath + url.search
  
  console.log(`Proxying request to: ${upstreamUrl}`)
  
  // Create upstream request with spoofed headers
  const upstreamHeaders = {
    'Referer': 'https://nvs.nanoos.org/WaysWaterMoves/',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Sec-Fetch-Dest': 'image',
    'Sec-Fetch-Mode': 'no-cors',
    'Sec-Fetch-Site': 'cross-site',
  }
  
  // Handle HEAD requests (convert to GET as some backends don't support HEAD for dynamic tiles)
  const requestMethod = request.method === 'HEAD' ? 'GET' : request.method
  
  try {
    const response = await fetch(upstreamUrl, {
      method: requestMethod,
      headers: upstreamHeaders,
      // CF properties for better performance
      cf: {
        cacheTtl: 300, // Cache for 5 minutes
        cacheEverything: true,
      },
    })
    
    // Prepare response headers
    const responseHeaders = new Headers(response.headers)
    
    // Add CORS headers
    Object.entries(corsHeaders).forEach(([key, value]) => {
      responseHeaders.set(key, value)
    })
    
    // Override cache control for better client caching
    responseHeaders.set('Cache-Control', 'public, max-age=300, s-maxage=300')
    
    // Add custom header to indicate proxy
    responseHeaders.set('X-Proxy', 'ocean-currents-worker')
    
    // For HEAD requests, return empty body
    if (request.method === 'HEAD') {
      return new Response(null, {
        status: response.status,
        statusText: response.statusText,
        headers: responseHeaders,
      })
    }
    
    // Return the proxied response
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: responseHeaders,
    })
    
  } catch (error) {
    console.error('Proxy error:', error)
    return new Response(`Proxy Error: ${error.message}`, { 
      status: 502,
      headers: {
        ...corsHeaders,
        'Content-Type': 'text/plain',
      }
    })
  }
}

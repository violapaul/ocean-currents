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
  
  // Handle NVS API proxy for magnitude values
  if (url.pathname === '/nvs/get_values') {
    const params = url.searchParams
    
    // Build the NVS API URL with all parameters
    const nvsUrl = new URL('https://nvs.nanoos.org/ssa/get_imov_values.php')
    
    // Copy all query parameters to the NVS URL
    for (const [key, value] of params) {
      nvsUrl.searchParams.append(key, value)
    }
    
    console.log(`Proxying NVS API request to: ${nvsUrl.toString()}`)
    
    try {
      const response = await fetch(nvsUrl.toString(), {
        headers: {
          'Referer': 'https://nvs.nanoos.org/',
          'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        }
      })
      
      const data = await response.text()
      
      return new Response(data, {
        status: response.status,
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json',
          'Cache-Control': 'public, max-age=60',
        }
      })
    } catch (error) {
      console.error('NVS proxy error:', error)
      return new Response(JSON.stringify({
        success: false,
        error: error.message
      }), {
        status: 500,
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json',
        }
      })
    }
  }
  
  // Handle NOAA tide API proxy
  if (url.pathname === '/noaa/tides') {
    const params = url.searchParams
    
    // Build the NOAA API URL with all parameters
    const noaaUrl = new URL('https://api.tidesandcurrents.noaa.gov/api/prod/datagetter')
    
    // Copy all query parameters to the NOAA URL
    for (const [key, value] of params) {
      noaaUrl.searchParams.append(key, value)
    }
    
    console.log(`Proxying NOAA API request to: ${noaaUrl.toString()}`)
    
    try {
      const response = await fetch(noaaUrl.toString(), {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
          'Accept': 'application/json',
        }
      })
      
      const data = await response.text()
      
      return new Response(data, {
        status: response.status,
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json',
          'Cache-Control': 'public, max-age=3600', // Cache tide data for 1 hour
        }
      })
    } catch (error) {
      console.error('NOAA proxy error:', error)
      return new Response(JSON.stringify({
        error: error.message
      }), {
        status: 500,
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json',
        }
      })
    }
  }
  
  // Extract path after /tiles
  if (!url.pathname.startsWith('/tiles')) {
    return new Response('Not Found - use /tiles/*, /nvs/*, or /noaa/* path', { 
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

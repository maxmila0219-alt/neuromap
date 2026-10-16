/* NeuroMap service worker: always fresh when online, works offline */
const CACHE = 'neuromap-v6';
const CORE = ['./', 'index.html', 'journey.html', 'neurogl.js', 'manifest.webmanifest', 'icons/icon-192.png', 'icons/icon-512.png'];
self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(CORE)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
/* network-first for our own files (updates show up immediately), cache as offline fallback */
self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  const own = url.origin === location.origin, fonts = /fonts\.(googleapis|gstatic)\.com$/.test(url.hostname);
  if (!own && !fonts) return;
  e.respondWith(caches.open(CACHE).then(async c => {
    if (fonts) { const hit = await c.match(req); if (hit) return hit; const r = await fetch(req); if (r && (r.ok || r.type === 'opaque')) c.put(req, r.clone()); return r }
    try { const r = await fetch(req, { cache: 'no-cache' }); if (r && r.ok) c.put(req, r.clone()); return r }
    catch (err) { return (await c.match(req, { ignoreSearch: true })) || Response.error() }
  }));
});

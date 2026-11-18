import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './style.css';

document.querySelector('#app').innerHTML = `
<header><a class="brand" href="./"><span class="logo">⌁</span>PathSnap<span class="tag">ML HACKATHON</span></a><span class="session">Evaluator workspace</span></header>
<main><div class="intro"><div><p class="eyebrow">GPS → ROAD NETWORK</p><h1>Every trip, a clearer path.</h1><p class="muted">Select a trip and explore its predicted route.</p></div><span id="health" class="badge" role="status">Checking API…</span></div>
<section class="connection"><form id="connection"><label for="base">Backend URL</label><input id="base" type="url" required placeholder="http://192.168.1.20:8000"><button id="connect" class="secondary">Connect / refresh</button></form><p id="connection-note" class="muted" role="status"></p></section>
<div class="workspace"><aside><section class="card"><p class="eyebrow">01 / SELECT & PREDICT</p><h2>Your next trip</h2><form id="prediction"><label for="trip">Trip ID</label><select id="trip" disabled><option>Connect to load trips</option></select><button id="run" disabled>Run prediction <span>↗</span></button></form><p id="status" role="status" aria-live="polite">Connect to the API to get started.</p></section>
<section class="card results"><div class="result-heading"><h2>Prediction metrics</h2><span id="result-trip">—</span></div><div class="metric"><span>Pointwise accuracy</span><strong id="accuracy">—</strong></div><div class="metric"><span>Processing time</span><strong id="seconds">—</strong></div><div class="metric"><span>GPS observations</span><strong id="count">—</strong></div></section><p class="footnote">Metrics are scored by the API.<br>Reference labels stay on the server.</p></aside>
<section class="map-card"><div class="map-heading"><div><p class="eyebrow">02 / EXPLORE</p><h2 id="map-title">Trip overview</h2></div><button id="fit" class="secondary" disabled>Fit trip ↗</button></div><div class="map-wrap"><div id="map" role="region" aria-label="Interactive trip map"></div><p id="map-warning" role="status" hidden></p><div id="empty">Your predicted route will appear here</div></div><div class="legend"><label><input id="route-toggle" type="checkbox" checked><i class="route-key"></i>Predicted route</label><label><input id="gps-toggle" type="checkbox" checked><i class="gps-key"></i>Raw GPS points</label><span id="geometry-note">Waiting for a prediction</span></div></section></div>
<footer>PathSnap <span>Live trip prediction · Evaluator dashboard</span></footer></main>`;

const $ = id => document.getElementById(id);
let map, routes, gps, bounds, busy = false, connecting = false, apiBase = '', tripIds = [];
try {

  map = L.map('map', { preferCanvas: true }).setView([20.59, 78.96], 4);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19,
  }).addTo(map).on('tileerror', () => {
    $('map-warning').hidden = false;
    $('map-warning').textContent = 'Basemap tiles unavailable. Route and GPS layers still work.';
  });
  routes = L.featureGroup().addTo(map); gps = L.featureGroup().addTo(map);
  new ResizeObserver(() => map.invalidateSize()).observe($('map'));
} catch {
  map = null;
  $('empty').textContent = 'Map library unavailable. Check internet access and reload.';
}
try { $('base').value = localStorage.getItem('pathsnap-api') || import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'; }
catch { $('base').value = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'; }

function setStatus(message, state = 'idle') {
  $('status').textContent = message;
  $('status').dataset.state = state;
}
function connectionChanged() {
  try { return normalizeBase($('base').value) !== apiBase; } catch { return true; }
}
function controls() {
  $('prediction').setAttribute('aria-busy', String(busy));
  $('connection').setAttribute('aria-busy', String(connecting));
  $('run').disabled = busy || connecting || !tripIds.length || connectionChanged();
  $('trip').disabled = busy || connecting || !tripIds.length;
  $('base').disabled = busy || connecting;
  $('connect').disabled = busy || connecting;
  $('run').innerHTML = busy ? '<span class="spinner"></span> Running prediction…' : 'Run prediction <span>↗</span>';
}
function clear() {
  routes?.clearLayers(); gps?.clearLayers(); bounds = null;
  for (const key of ['accuracy', 'seconds', 'count', 'result-trip']) $(key).textContent = '—';
  $('map-title').textContent = 'Trip overview'; $('fit').disabled = true;
  if (map) { $('empty').hidden = false; $('empty').textContent = 'Your predicted route will appear here'; map.setView([20.59,78.96],4); }
  $('geometry-note').textContent = 'Waiting for a prediction';
}
function normalizeBase(value) {
  let u;
  try { u = new URL(value.trim()); }
  catch { throw new Error('Enter a complete backend URL, for example http://192.168.1.20:8000.'); }
  if (location.protocol === 'https:' && u.protocol === 'http:') throw new Error('Open the UI over HTTP for an HTTP LAN backend; browsers block mixed content.');
  if (!['http:', 'https:'].includes(u.protocol) || u.username || u.password || u.search || u.hash) throw new Error('Enter an HTTP or HTTPS backend URL without credentials, query, or fragment.');
  return u.href.replace(/\/+$/, '');
}
async function request(path, method = 'GET') {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), method === 'POST' ? 300000 : 10000);
  try {
    const response = await fetch(apiBase + path, { method, signal: controller.signal, cache: 'no-store' });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(typeof data?.detail === 'string' ? data.detail : 'API returned HTTP ' + response.status);
    }
    try { return await response.json(); }
    catch (e) { if (e.name === 'AbortError') throw e; throw new Error('API returned invalid JSON. Check the backend URL and response.'); }
  } catch (e) {
    if (e.name === 'AbortError') throw new Error('Request timed out. Check the backend, then retry.');
    if (e instanceof TypeError) throw new Error('Cannot reach API. Check the LAN address, backend, firewall, and CORS settings.');
    throw e;
  } finally { clearTimeout(timer); }
}
async function connect(event) {
  event?.preventDefault(); if (busy || connecting) return;
  connecting = true; tripIds = []; clear(); controls();
  $('health').textContent = 'Checking API…'; $('health').className = 'badge';
  $('trip').replaceChildren(new Option('Loading trips…', ''));
  try {
    apiBase = normalizeBase($('base').value);
    const health = await request('/health');
    if (health?.status !== 'ready') throw new Error('API is reachable but is not ready.');
    $('health').textContent = '● API reachable'; $('health').className = 'badge online';
    const data = await request('/trips');
    if (!Array.isArray(data?.trip_ids) || !data.trip_ids.every(id => typeof id === 'string')) throw new Error('Invalid trip list: API must return trip_ids as strings.');
    tripIds = [...new Set(data.trip_ids)];
    $('trip').replaceChildren(...tripIds.map(id => new Option(id, id)));
    try { localStorage.setItem('pathsnap-api', apiBase); } catch {}
    $('connection-note').textContent = tripIds.length + ' trips available · ' + apiBase;
    setStatus(tripIds.length ? 'Choose any trip to run a prediction.' : 'The API returned no trips. Refresh when trips are available.');
  } catch (e) {
    $('health').textContent = 'Connection needs attention'; $('health').className = 'badge error';
    $('connection-note').textContent = e.message;
    $('trip').replaceChildren(new Option('No trips loaded', ''));
    $('status').textContent = 'Check the connection and select Connect / refresh to retry.';
  } finally { connecting = false; controls(); }
}
function point(coordinate) {
  if (!Array.isArray(coordinate) || coordinate.length < 2 || !Number.isFinite(coordinate[0]) || !Number.isFinite(coordinate[1]) || Math.abs(coordinate[0]) > 180 || Math.abs(coordinate[1]) > 90) throw new Error('Invalid coordinates in API response.');
  return [coordinate[1], coordinate[0]];
}
$('prediction').addEventListener('submit', async event => {
  event.preventDefault(); if (busy || connecting || !tripIds.length || connectionChanged()) return;
  const id = $('trip').value;
  busy = true; clear(); controls(); setStatus('Predicting trip ' + id + '…', 'loading');
  try {
    const data = await request('/trips/' + encodeURIComponent(id) + '/predict', 'POST');
    if (data?.trip_id !== id || !Array.isArray(data.observations) || !Array.isArray(data.predicted_route) ||
        !Number.isInteger(data.observation_count) || data.observation_count < 0 ||
        data.observation_count !== data.observations.length ||
        !Number.isFinite(data.metrics?.pointwise_accuracy) || data.metrics.pointwise_accuracy < 0 || data.metrics.pointwise_accuracy > 1 ||
        !Number.isFinite(data.metrics?.processing_seconds) || data.metrics.processing_seconds < 0) throw new Error('Prediction response does not match the API contract.');
    const segments = data.predicted_route.map(segment => {
      if (!Array.isArray(segment?.coordinates)) throw new Error('Missing route segment coordinates.');
      return segment.coordinates.map(point);
    });
    const observations = data.observations.map(point);
    if (map) {
      segments.forEach(segment => { if (segment.length) L.polyline(segment, { color: '#6cbbff', weight: 5, opacity: .85 }).addTo(routes); });
      observations.forEach(p => L.circleMarker(p, { radius: 4, color: '#402b20', weight: 1, fillColor: '#ffaf70', fillOpacity: .95 }).addTo(gps));
      bounds = L.latLngBounds([]);
      segments.forEach(segment => segment.forEach(p => bounds.extend(p)));
      observations.forEach(p => bounds.extend(p));
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [36,36], maxZoom: 17 });
        $('fit').disabled = false; $('empty').hidden = true;
      } else { $('empty').textContent = 'This trip has no geometry to display.'; }
      $('geometry-note').textContent = segments.length + ' route segments';
    }
    $('accuracy').textContent = (data.metrics.pointwise_accuracy * 100).toFixed(1) + '%';
    $('seconds').textContent = data.metrics.processing_seconds.toFixed(3) + ' s';
    $('count').textContent = String(data.observation_count);
    $('result-trip').textContent = id; $('map-title').textContent = 'Trip ' + id;
    $('health').textContent = '● API reachable'; $('health').className = 'badge online';
    setStatus('Prediction complete for trip ' + id + (map ? '.' : '. Map unavailable; reload the page.'), 'success');
  } catch (e) {
    clear();
    if (/Cannot reach API|timed out/.test(e.message)) {
      $('health').textContent = 'Connection needs attention'; $('health').className = 'badge error';
    }
    setStatus(e.message + ' Select Run prediction to retry.', 'error');
  }
  finally { busy = false; controls(); }
});
$('trip').addEventListener('change', () => { clear(); setStatus('Ready to predict trip ' + $('trip').value + '.'); });
$('connection').addEventListener('submit', connect);
$('base').addEventListener('input', () => {
  controls();
  if (connectionChanged()) {
    clear();
    $('health').textContent = 'URL changed · reconnect'; $('health').className = 'badge';
    setStatus('Select Connect / refresh to use this backend address.');
  }
});
$('fit').addEventListener('click', () => bounds && map.fitBounds(bounds, {padding:[36,36],maxZoom:17}));
for (const [id, group] of [['route-toggle', () => routes], ['gps-toggle', () => gps]]) {
  $(id).addEventListener('change', e => { if (map) e.target.checked ? group().addTo(map) : map.removeLayer(group()); });
}
connect();

// Exports also let browser tests inspect the real Leaflet layers without globals.
export { map, routes, gps, bounds };

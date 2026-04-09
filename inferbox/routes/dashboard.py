"""Web dashboard at /ui — single-page status view."""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["dashboard"])


_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>InferBox Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {
  --bg: #0f1115;
  --fg: #e6e8eb;
  --muted: #8b8f96;
  --accent: #5eead4;
  --warn: #fbbf24;
  --err: #f87171;
  --card: #161a21;
  --border: #262b34;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "SF Mono", Menlo, monospace;
  background: var(--bg);
  color: var(--fg);
  margin: 0;
  padding: 24px;
  font-size: 13px;
  line-height: 1.5;
}
h1 { font-size: 18px; margin: 0 0 4px 0; font-weight: 600; }
h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin: 24px 0 8px; font-weight: 600; }
.subtitle { color: var(--muted); margin-bottom: 24px; }
.grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 16px;
}
.card .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }
.card .value { font-size: 22px; margin-top: 4px; font-weight: 500; }
.card .sub { color: var(--muted); font-size: 11px; margin-top: 2px; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 500; text-transform: uppercase; font-size: 10px; letter-spacing: 0.08em; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 500; }
.pill.loaded { background: rgba(94, 234, 212, 0.15); color: var(--accent); }
.pill.unloaded { background: rgba(139, 143, 150, 0.15); color: var(--muted); }
.pill.draining { background: rgba(251, 191, 36, 0.15); color: var(--warn); }
.pill.ok { background: rgba(94, 234, 212, 0.15); color: var(--accent); }
.bar {
  height: 6px; background: var(--border); border-radius: 3px; overflow: hidden;
  margin-top: 6px;
}
.bar > span { display: block; height: 100%; background: var(--accent); transition: width 0.4s; }
.warn { color: var(--warn); }
.err { color: var(--err); }
.refresh { color: var(--muted); font-size: 11px; }
.dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--accent); margin-right: 4px; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
</style>
</head>
<body>
<h1>InferBox <span style="color: var(--muted); font-weight: 400;">/ dashboard</span></h1>
<div class="subtitle">Live status · <span id="refresh" class="refresh"><span class="dot"></span>refreshing every 2s</span></div>

<div class="grid" id="overview"></div>

<h2>Models</h2>
<table id="models-table">
  <thead><tr><th>Model</th><th>Type</th><th>Status</th><th>VRAM</th><th>Idle</th></tr></thead>
  <tbody></tbody>
</table>

<h2>Endpoints</h2>
<table id="stats-table">
  <thead><tr><th>Model</th><th>Endpoint</th><th>Requests</th><th>Errors</th><th>Items</th><th>p50</th><th>p95</th></tr></thead>
  <tbody></tbody>
</table>

<script>
const fmt = (n) => n.toLocaleString();
const fmtMB = (mb) => mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : mb + ' MB';
const fmtSec = (s) => {
  if (s < 60) return s + 's';
  const m = Math.floor(s / 60);
  if (m < 60) return m + 'm';
  const h = Math.floor(m / 60);
  return h + 'h ' + (m % 60) + 'm';
};

async function refresh() {
  try {
    const [healthR, modelsR, statsR] = await Promise.all([
      fetch('/v1/health'),
      fetch('/v1/models', { headers: { 'X-API-Key': window.localStorage.getItem('inferbox_key') || '' } }),
      fetch('/v1/stats', { headers: { 'X-API-Key': window.localStorage.getItem('inferbox_key') || '' } }),
    ]);
    const health = await healthR.json();
    const models = modelsR.ok ? await modelsR.json() : { models: [] };
    const stats = statsR.ok ? await statsR.json() : { endpoints: [], uptime_seconds: 0 };

    // Overview cards
    const totalVram = health.gpu?.vram_total_mb || 0;
    const usedVram = health.vram_used_mb || 0;
    const pct = totalVram > 0 ? (usedVram / totalVram) * 100 : 0;
    const draining = health.status === 'draining';
    document.getElementById('overview').innerHTML = `
      <div class="card">
        <div class="label">Status</div>
        <div class="value">
          <span class="pill ${draining ? 'draining' : 'ok'}">${health.status}</span>
        </div>
        <div class="sub">${health.in_flight} in flight</div>
      </div>
      <div class="card">
        <div class="label">GPU</div>
        <div class="value" style="font-size: 14px;">${health.gpu?.gpu || 'CPU only'}</div>
        <div class="sub">${fmtMB(usedVram)} / ${fmtMB(totalVram)}</div>
        <div class="bar"><span style="width: ${pct}%"></span></div>
      </div>
      <div class="card">
        <div class="label">Models loaded</div>
        <div class="value">${health.models_loaded}</div>
        <div class="sub">${models.models?.length || 0} registered</div>
      </div>
      <div class="card">
        <div class="label">Uptime</div>
        <div class="value">${fmtSec(stats.uptime_seconds)}</div>
        <div class="sub">${stats.endpoints?.reduce((a, e) => a + e.requests, 0) || 0} total requests</div>
      </div>
    `;

    // Models table
    const mtbody = document.querySelector('#models-table tbody');
    mtbody.innerHTML = (models.models || []).map(m => `
      <tr>
        <td><strong>${m.id}</strong><br><span style="color: var(--muted); font-size: 11px;">${m.model_id}</span></td>
        <td>${m.type}</td>
        <td><span class="pill ${m.status}">${m.status}</span></td>
        <td>${fmtMB(m.vram_mb)}</td>
        <td>${m.idle_seconds !== undefined ? fmtSec(m.idle_seconds) : '—'}</td>
      </tr>
    `).join('') || '<tr><td colspan="5" style="color: var(--muted); text-align: center;">authenticate to see models (set localStorage.inferbox_key)</td></tr>';

    // Stats table
    const stbody = document.querySelector('#stats-table tbody');
    stbody.innerHTML = (stats.endpoints || []).map(e => `
      <tr>
        <td>${e.model}</td>
        <td>${e.endpoint}</td>
        <td>${fmt(e.requests)}</td>
        <td class="${e.errors > 0 ? 'err' : ''}">${fmt(e.errors)}</td>
        <td>${fmt(e.items)}</td>
        <td>${e.p50_ms} ms</td>
        <td>${e.p95_ms} ms</td>
      </tr>
    `).join('') || '<tr><td colspan="7" style="color: var(--muted); text-align: center;">no requests yet</td></tr>';

  } catch (err) {
    document.getElementById('refresh').innerHTML = '<span class="err">connection error</span>';
  }
}

// Allow user to set API key via prompt or URL param
const urlKey = new URLSearchParams(location.search).get('key');
if (urlKey) {
  localStorage.setItem('inferbox_key', urlKey);
  history.replaceState(null, '', location.pathname);
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


@router.get("/ui", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(_HTML)

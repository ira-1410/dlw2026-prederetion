// ─── Map setup ───────────────────────────────────────────────────────────────

const map = L.map("map").setView([1.3521, 103.8198], 12);

L.tileLayer(
  "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
  { attribution: "© OpenStreetMap © CARTO" }
).addTo(map);

const API_BASE = "https://dlw2026-prederetion-node-server-production.up.railway.app";

// ─── State ────────────────────────────────────────────────────────────────────

let startMarker   = null;
let endMarker     = null;
let routeSegments = [];    // Leaflet polylines for the drawn route
let ltaSegments   = [];    // LTA road segment geometry { LinkID, StartLat, ... }
let spatialGrid   = {};    // spatial index over ltaSegments
let riskByLinkId  = {};    // { [LinkID]: risk_index }
let cameraMarkers = [];   // Leaflet layers for camera X markers
let geometryReady = false;

let startLocation = null;
let endLocation   = null;
let typingTimer   = null;


// ─── Autocomplete ─────────────────────────────────────────────────────────────

async function autocomplete(query, containerId, isStart) {
  if (query.length < 3) return;

  const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(
    query + ", Singapore"
  )}&addressdetails=1&limit=5`;

  const res  = await fetch(url);
  const data = await res.json();

  const container = document.getElementById(containerId);
  container.innerHTML = "";

  data.forEach(place => {
    const div       = document.createElement("div");
    div.className   = "result-item";
    div.textContent = place.display_name;
    div.onclick     = () => {
      if (isStart) startLocation = place;
      else         endLocation   = place;
      container.innerHTML = "";
      document.getElementById(isStart ? "start" : "end").value = place.display_name;
    };
    container.appendChild(div);
  });
}

document.getElementById("start").addEventListener("input", e => {
  clearTimeout(typingTimer);
  typingTimer = setTimeout(() => autocomplete(e.target.value, "start-results", true), 200);
});

document.getElementById("end").addEventListener("input", e => {
  clearTimeout(typingTimer);
  typingTimer = setTimeout(() => autocomplete(e.target.value, "end-results", false), 200);
});


// ─── Color ────────────────────────────────────────────────────────────────────

function getColor(value) {
  if (value == null) return "#aaaaaa";   // grey  = no data
  if (value >= 75)   return "#e53935";   // red   = high risk
  if (value >= 50)   return "#fb8c00";   // orange = medium risk
  return "#43a047";                      // green = low risk
}


// ─── Geometry loading + spatial index ────────────────────────────────────────

async function loadGeometry() {
  document.getElementById("route-button").disabled    = true;
  document.getElementById("route-button").textContent = "Loading map data...";

  const res = await fetch(`${API_BASE}/segments/geometry`);
  const data = await res.json();
  ltaSegments = data.geometry;

  buildSpatialIndex();

  geometryReady = true;
  document.getElementById("route-button").disabled    = false;
  document.getElementById("route-button").textContent = "Get Route";
  console.log(`Geometry loaded: ${ltaSegments.length} segments`);
}

const GRID_SIZE = 0.01;   // ~1km cells

function buildSpatialIndex() {
  spatialGrid = {};
  for (const seg of ltaSegments) {
    const midLat  = (seg.StartLat + seg.EndLat) / 2;
    const midLon  = (seg.StartLon + seg.EndLon) / 2;
    const key     = `${Math.floor(midLat / GRID_SIZE)},${Math.floor(midLon / GRID_SIZE)}`;
    if (!spatialGrid[key]) spatialGrid[key] = [];
    spatialGrid[key].push(seg);
  }
  console.log(`Spatial index built: ${Object.keys(spatialGrid).length} cells`);
}


// ─── Geometry helpers ─────────────────────────────────────────────────────────

function distancePointToSegment(px, py, x1, y1, x2, y2) {
  const A = px - x1, B = py - y1;
  const C = x2 - x1, D = y2 - y1;
  const dot   = A * C + B * D;
  const lenSq = C * C + D * D;
  let param   = lenSq !== 0 ? dot / lenSq : -1;

  const xx = param < 0 ? x1 : param > 1 ? x2 : x1 + param * C;
  const yy = param < 0 ? y1 : param > 1 ? y2 : y1 + param * D;

  return Math.sqrt((px - xx) ** 2 + (py - yy) ** 2);
}

function findMatchingRoadSegment(lat, lon) {
  const cx = Math.floor(lat / GRID_SIZE);
  const cy = Math.floor(lon / GRID_SIZE);

  let best = null, minDist = Infinity;

  for (let dx = -1; dx <= 1; dx++) {
    for (let dy = -1; dy <= 1; dy++) {
      const candidates = spatialGrid[`${cx + dx},${cy + dy}`] || [];
      for (const seg of candidates) {
        const d = distancePointToSegment(
          lon, lat,
          seg.StartLon, seg.StartLat,
          seg.EndLon,   seg.EndLat
        );
        if (d < minDist) { minDist = d; best = seg; }
      }
    }
  }

  return minDist < 0.0003 ? best : null;   // ~30 metre threshold
}


// ─── Routing ──────────────────────────────────────────────────────────────────

document.getElementById("route-button").addEventListener("click", async () => {
  if (!startLocation || !endLocation) {
    alert("Please select both addresses from the dropdown.");
    return;
  }
  if (!geometryReady) {
    alert("Map data still loading, please wait a moment.");
    return;
  }

  startMarker?.remove();
  endMarker?.remove();

  startMarker = L.marker([startLocation.lat, startLocation.lon])
    .addTo(map).bindPopup("Start").openPopup();
  endMarker = L.marker([endLocation.lat, endLocation.lon])
    .addTo(map).bindPopup("End").openPopup();

  map.fitBounds([
    [startLocation.lat, startLocation.lon],
    [endLocation.lat,   endLocation.lon]
  ]);

  await drawRoute();
});




async function drawRoute() {
  const start = startMarker.getLatLng();
  const end   = endMarker.getLatLng();

  const url  = `https://router.project-osrm.org/route/v1/driving/${start.lng},${start.lat};${end.lng},${end.lat}?overview=full&geometries=geojson`;
  const res  = await fetch(url);
  const data = await res.json();
  if (!data.routes?.length) return;

  // Clear previous route
  routeSegments.forEach(s => map.removeLayer(s));
  routeSegments = [];

  const coords = data.routes[0].geometry.coordinates;

  // 1. Draw grey immediately so the user sees the route while scores load
  for (let i = 0; i < coords.length - 1; i++) {
    const line = L.polyline(
      [[coords[i][1], coords[i][0]], [coords[i+1][1], coords[i+1][0]]],
      { color: "#aaaaaa", weight: 6, opacity: 0.9 }
    ).addTo(map);
    line._linkId = null;
    routeSegments.push(line);
  }

  // 2. Match each polyline segment to an LTA road segment
  for (let i = 0; i < coords.length - 1; i++) {
    const midLat = (coords[i][1]   + coords[i+1][1]) / 2;
    const midLon = (coords[i][0]   + coords[i+1][0]) / 2;
    const road   = findMatchingRoadSegment(midLat, midLon);
    routeSegments[i]._linkId = road?.LinkID ?? null;
  }

  const uniqueLinkIds = [...new Set(routeSegments.map(s => s._linkId).filter(Boolean))];
  if (!uniqueLinkIds.length) {
    console.warn("No LTA segments matched — route will stay grey");
    return;
  }
  console.log(`Matched ${uniqueLinkIds.length} unique LTA segments`);

  // 3. Score matched segments AND fetch cameras in parallel
  try {
    const [scoreRes, cameras] = await Promise.all([
      fetch(`${API_BASE}/predict/route`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ linkIds: uniqueLinkIds })
      }),
      fetchRouteCameras(coords).catch(err => {
        console.warn("Camera fetch failed:", err);
        return [];
      })
    ]);

    const scoreData = await scoreRes.json();

    let reasons = [];

    scoreData.results.forEach(r => {
    if (r.risk_index > 60 && r.reasons) {
        Object.entries(r.reasons).forEach(([reason, isActive]) => {
        if (isActive && !reasons.includes(reason)) {
            reasons.push(reason);
        }
        });
    }
    });
    if (reasons.length>0) { alert(reasons); }

    // Update risk lookup
    riskByLinkId = {};
    for (const row of scoreData.results) {
      riskByLinkId[row.link_id] = row.risk_index;
    }

    // 4. Recolour each polyline segment
    routeSegments.forEach(line => {
      const risk = riskByLinkId[line._linkId] ?? null;
      line._risk = risk;
      line.setStyle({ color: getColor(risk) });
    });

    // 5. Render camera strip
    renderCameraStrip(cameras);
    //renderCameraMarkers(cameras);

    console.log(`Route coloured — risks: ${JSON.stringify(
      scoreData.results.slice(0, 3).map(r => ({ id: r.link_id, risk: r.risk_index }))
    )}`);
    console.log(`Camera strip: ${cameras.length} cameras shown`);

  } catch (err) {
    console.error("Route scoring failed:", err);
  }
}

function alert(reasons) {
  if (!reasons || reasons.length === 0) return;
  
  const existing = document.getElementById("risk-alert");
  if (existing) existing.remove();

  const alertBox = document.createElement("div");
  alertBox.id = "risk-alert";
  alertBox.innerHTML = `
    <strong>⚠️ Be cautious of </strong>
    <div class="risk-reasons">
      ${reasons.map(r => `<div>• ${formatReason(r)}</div>`).join("")}
    </div>
  `;

  document.body.appendChild(alertBox);  // ← append to body, not #controls

  setTimeout(() => {
    alertBox.classList.add("fade-out");
    setTimeout(() => alertBox.remove(), 500);
  }, 6000);
}

function formatReason(reason) {
  const labels = {
    flooding: "Flooding detected",
    incidents: "Traffic incidents",
    faultyLights: "Faulty traffic lights",
    vms: "VMS issues",
    highSpeed: "High speed traffic"
  };

  return labels[reason] || reason;
}

// ─── Traffic Camera Images ────────────────────────────────────────────────────

/**
 * Fetch cameras from the server, filtered to the route's bounding box.
 * Returns the N cameras whose positions are closest to evenly-spaced
 * waypoints along the route, so we get good coverage from start to end.
 */
async function fetchRouteCameras(coords, maxCameras = 6) {
  // Build bounding box with a small padding around the route
  const lats = coords.map(c => c[1]);
  const lons = coords.map(c => c[0]);
  const pad = 0.01;  // ~1 km padding
  const minLat = Math.min(...lats) - pad;
  const maxLat = Math.max(...lats) + pad;
  const minLon = Math.min(...lons) - pad;
  const maxLon = Math.max(...lons) + pad;

  const res  = await fetch(
    `${API_BASE}/lta/traffic-images?minLat=${minLat}&maxLat=${maxLat}&minLon=${minLon}&maxLon=${maxLon}`
  );
  const data = await res.json();
  if (!data.cameras?.length) return [];

  // Pick evenly-spaced sample points along the route
  const sampleCount = maxCameras;
  const samplePoints = [];
  for (let i = 0; i < sampleCount; i++) {
    const idx = Math.round((i / (sampleCount - 1)) * (coords.length - 1));
    samplePoints.push(coords[Math.min(idx, coords.length - 1)]);
  }

  // For each sample point, find the nearest camera (haversine-lite with squared distance)
  const used = new Set();
  const selected = [];

  for (const [sLon, sLat] of samplePoints) {
    let bestCam  = null;
    let bestDist = Infinity;

    for (const cam of data.cameras) {
      if (used.has(cam.CameraID)) continue;
      const dLat = cam.Latitude  - sLat;
      const dLon = cam.Longitude - sLon;
      const dist = dLat * dLat + dLon * dLon;   // squared — fine for relative comparison
      if (dist < bestDist) { bestDist = dist; bestCam = cam; }
    }

    if (bestCam) {
      used.add(bestCam.CameraID);
      selected.push(bestCam);
    }
  }

  return selected;
}

async function scoreImage(imageUrl) {
  try {
    const res = await fetch(`${API_BASE}/predict/image`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ imageUrl }),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

function renderCameraStrip(cameras) {
  let strip = document.getElementById("camera-strip");

  // Create strip container if it doesn't exist yet
  if (!strip) {
    strip = document.createElement("div");
    strip.id = "camera-strip";
    strip.style.cssText = `
      position: fixed; bottom: 0; left: 0; right: 0;
      display: flex; gap: 8px; padding: 10px 14px;
      background: rgba(20,20,20,0.88); backdrop-filter: blur(6px);
      overflow-x: auto; z-index: 1000;
      scrollbar-width: thin; scrollbar-color: #555 transparent;
    `;
    document.body.appendChild(strip);
  }

  strip.innerHTML = "";

  if (!cameras.length) {
    strip.innerHTML = `<span style="color:#aaa;font-size:13px;align-self:center;padding:4px 8px">
      No traffic cameras found along this route.</span>`;
    return;
  }

  for (const cam of cameras) {
    const wrapper = document.createElement("div");
    wrapper.style.cssText = `
      flex: 0 0 auto; position: relative; cursor: pointer;
      border-radius: 6px; overflow: hidden;
      border: 2px solid #444; transition: border-color 0.2s;
    `;
    wrapper.onmouseenter = () => wrapper.style.borderColor = "#90caf9";
    wrapper.onmouseleave = () => wrapper.style.borderColor = "#444";

    // Clicking expands the image in a simple overlay
    wrapper.onclick = () => showCameraOverlay(cam);

    const img = document.createElement("img");
    img.src = cam.ImageLink;
    img.alt = `Camera ${cam.CameraID}`;
    img.style.cssText = "display:block; width:280px; height:158px; object-fit:cover;";
    img.onerror = () => { wrapper.style.display = "none"; };   // hide broken images

    const label = document.createElement("div");
    label.textContent = `Cam ${cam.CameraID}`;
    label.style.cssText = `
      position:absolute; bottom:0; left:0; right:0;
      background:rgba(0,0,0,0.55); color:#fff;
      font-size:14px; padding:4px 8px; text-align:center;
    `;

    // Collision badge — updated once async inference resolves
    const badge = document.createElement("div");
    badge.textContent = "Analysing…";
    badge.style.cssText = `
      position:absolute; top:5px; right:5px;
      background:rgba(0,0,0,0.65); color:#ccc;
      font-size:12px; font-weight:600; padding:4px 8px;
      border-radius:4px; letter-spacing:0.3px;
    `;

    wrapper.appendChild(img);
    wrapper.appendChild(label);
    wrapper.appendChild(badge);
    strip.appendChild(wrapper);

    // Fire-and-forget: score the image, then update the badge
    scoreImage(cam.ImageLink).then(result => {
      if (!result) { badge.style.display = "none"; return; }
      const isCollision = result.collision === true || result.label === "collision";
      badge.textContent = isCollision ? "⚠ Collision Detected" : "✓ Normal";
      badge.style.background = isCollision ? "rgba(198,40,40,0.85)" : "rgba(27,94,32,0.85)";
      badge.style.color = "#fff";
      if (isCollision) {
        wrapper.style.borderColor = "#ef5350";
        wrapper.onmouseleave = () => wrapper.style.borderColor = "#ef5350";
      }
    });
  }
}

function showCameraOverlay(cam) {
  let overlay = document.getElementById("camera-overlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.id = "camera-overlay";
    overlay.style.cssText = `
      position:fixed; inset:0; background:rgba(0,0,0,0.82);
      display:flex; flex-direction:column; align-items:center; justify-content:center;
      z-index:2000; cursor:pointer;
    `;
    overlay.onclick = () => overlay.remove();
    document.body.appendChild(overlay);
  } else {
    overlay.innerHTML = "";
  }

  const img = document.createElement("img");
  img.src = cam.ImageLink;
  img.style.cssText = "max-width:90vw; max-height:80vh; border-radius:8px; border:2px solid #90caf9;";

  const caption = document.createElement("div");
  caption.textContent = `Camera ${cam.CameraID} · Click anywhere to close`;
  caption.style.cssText = "color:#ccc; margin-top:10px; font-size:13px;";

  overlay.appendChild(img);
  overlay.appendChild(caption);
}

loadGeometry();

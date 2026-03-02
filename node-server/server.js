import { buildFeatureRow, toFeatureVector } from "./feature.js";

import dotenv from "dotenv";
dotenv.config();

import express from "express";
import fetch from "node-fetch";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json());

const PORT          = 3000;
const BASE_URL      = "https://datamall2.mytransport.sg/ltaodataservice";
const INFERENCE_URL = "http://localhost:5001";

const prevSpeedBandByLink = new Map();

// ─── API Key Rotation ─────────────────────────────────────────────────────────

const API_KEYS = [
  process.env.LTA_API_KEY_1,
  process.env.LTA_API_KEY_2,
].filter(Boolean);

if (API_KEYS.length === 0) console.warn("⚠️  WARNING: No LTA API keys set in .env");

let activeKeyIndex = 0;

function getActiveKey() {
  return API_KEYS[activeKeyIndex];
}

setInterval(() => {
  activeKeyIndex = (activeKeyIndex + 1) % API_KEYS.length;
  console.log(`[key rotation] Switched to key index ${activeKeyIndex}`);
}, 10 * 60 * 1000);


// ─── Snapshot cache ───────────────────────────────────────────────────────────

const CACHE_TTL_MS = 10 * 60 * 1000;   // 10 minutes

const cache = {
  snapshot:  null,
  fetchedAt: null,
  pending:   null,
};

let warmPromise = null;

function cacheValid() {
  return cache.snapshot && cache.fetchedAt && Date.now() - cache.fetchedAt < CACHE_TTL_MS;
}

async function getCachedSnapshot() {
  // Always return what we have immediately if anything exists
  const hasData = !!cache.snapshot;
  const expired = !cacheValid();

  // Trigger a background refresh if stale/missing and not already fetching
  if ((expired || !hasData) && !cache.pending && !warmPromise) {
    cache.pending = fetchSnapshot()
      .then(snapshot => {
        cache.snapshot  = snapshot;
        cache.fetchedAt = Date.now();
        cache.pending   = null;
        console.log(`[cache] Refreshed — ${snapshot.speedBands.length} segments`);
      })
      .catch(err => {
        cache.pending = null;
        console.warn(`[cache] Refresh failed: ${err.message}`);
      });
  }

  // Return stale data immediately if available — never await
  if (hasData) return cache.snapshot;

  // Only block if we have absolutely nothing (cold start with no data at all)
  if (cache.pending) {
    await cache.pending;
    return cache.snapshot;
  }

  throw new Error("No data available and no fetch in progress");
}


// ─── Geometry cache ───────────────────────────────────────────────────────────

let geometryCache = null;
let geometryReady = false;

async function warmGeometry() {
  try {
    const speedBands = await getSpeedBands();
    geometryCache = speedBands.map(s => ({
      LinkID:   s.LinkID,
      StartLat: s.StartLat, StartLon: s.StartLon,
      EndLat:   s.EndLat,   EndLon:   s.EndLon,
    }));
    geometryReady = true;
    console.log(`[geometry] Loaded ${geometryCache.length} segments`);
  } catch (err) {
    console.warn(`[geometry] Warm failed: ${err.message}`);
  }
}

app.get("/segments/geometry", (req, res) => {
  if (!geometryReady) return res.status(503).json({ error: "Geometry not ready yet" });
  res.json({ geometry: geometryCache });
});

/*
app.get("/segments/geometry", async (req, res) => {
  try {
    const now = Date.now();

    if (geometryCache && now - geometryCacheAge <= GEOMETRY_TTL) {
      return res.json({ geometry: geometryCache });
    }

    if (!geometryPending) {
      geometryPending = getSpeedBands()
        .then(speedBands => {
          geometryCache    = speedBands.map(s => ({
            LinkID:   s.LinkID,
            StartLat: s.StartLat, StartLon: s.StartLon,
            EndLat:   s.EndLat,   EndLon:   s.EndLon,
          }));
          geometryCacheAge = Date.now();
          geometryPending  = null;
          console.log(`[geometry cache] refreshed — ${geometryCache.length} segments`);
          return geometryCache;
        })
        .catch(err => {
          geometryPending = null;
          throw err;
        });
    } else {
      console.log("[geometry cache] WAIT (fetch already in flight)");
    }

    await geometryPending;
    res.json({ geometry: geometryCache });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});
*/


// ─── Inference client ─────────────────────────────────────────────────────────

async function scoreBatch(featDicts) {
  const res = await fetch(`${INFERENCE_URL}/score/batch`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(featDicts)
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "(unreadable)");
    throw new Error(`Inference server error ${res.status}: ${body}`);
  }
  return res.json();
}

async function checkInferenceHealth() {
  try {
    const res = await fetch(`${INFERENCE_URL}/health`, { timeout: 2000 });
    return res.ok;
  } catch {
    return false;
  }
}


// ─── Core fetcher ─────────────────────────────────────────────────────────────

async function fetchPaginated(endpoint) {
  if (API_KEYS.length === 0) {
    throw new Error(`No LTA API keys set in .env — cannot fetch [${endpoint}]`);
  }

  let results = [];
  let skip    = 0;
  const limit = 500;

  while (true) {
    const url      = `${BASE_URL}/${endpoint}?$skip=${skip}`;
    const response = await fetch(url, {
      headers: { AccountKey: getActiveKey(), Accept: "application/json" }
    });

    if (!response.ok) {
      const body = await response.text().catch(() => "(unreadable)");
      throw new Error(`LTA API [${endpoint}] returned HTTP ${response.status}: ${body}`);
    }

    const json  = await response.json();
    const value = json.value || [];
    results.push(...value);

    if (value.length < limit) break;
    skip += limit;
  }

  return results;
}


// ─── Per-source fetchers ──────────────────────────────────────────────────────

async function getSpeedBands() {
  const records = await fetchPaginated("v4/TrafficSpeedBands");
  return records.map(r => ({
    ...r,
    SpeedBand:    Number(r.SpeedBand),
    MinimumSpeed: Number(r.MinimumSpeed),
    MaximumSpeed: Number(r.MaximumSpeed),
    StartLon:     Number(r.StartLon),
    StartLat:     Number(r.StartLat),
    EndLon:       Number(r.EndLon),
    EndLat:       Number(r.EndLat),
    RoadCategory: Number(r.RoadCategory),
    timestamp:    new Date().toISOString()
  }));
}

async function getIncidents() {
  const records = await fetchPaginated("TrafficIncidents");
  return records.map(r => ({
    ...r,
    Latitude:  Number(r.Latitude),
    Longitude: Number(r.Longitude),
    timestamp: new Date().toISOString()
  }));
}

async function getFaultyLights() {
  const records = await fetchPaginated("FaultyTrafficLights");
  return records.map(r => ({
    ...r,
    Type:      Number(r.Type),
    timestamp: new Date().toISOString()
  }));
}

async function getVms() {
  const records = await fetchPaginated("VMS");
  return records.map(r => ({
    ...r,
    Latitude:  Number(r.Latitude),
    Longitude: Number(r.Longitude),
    timestamp: new Date().toISOString()
  }));
}

async function getFloods() {
  const records = await fetchPaginated("PubFloodAlerts");
  return records.map(r => {
    let flood_lat = null, flood_lon = null, flood_radius = null;
    if (r.circle) {
      try {
        const [latlon, radius] = r.circle.split(" ");
        const [lat, lon]       = latlon.split(",");
        flood_lat    = Number(lat);
        flood_lon    = Number(lon);
        flood_radius = Number(radius);
      } catch {}
    }
    return { ...r, flood_lat, flood_lon, flood_radius, timestamp: new Date().toISOString() };
  });
}


// ─── Snapshot ─────────────────────────────────────────────────────────────────

async function fetchSnapshot() {
  const [speedBands, incidents, faultyLights, vms, floods] = await Promise.all([
    getSpeedBands(),
    getIncidents().catch(err    => { console.warn("incidents skipped:",    err.message); return []; }),
    getFaultyLights().catch(err => { console.warn("faultyLights skipped:", err.message); return []; }),
    getVms().catch(err          => { console.warn("vms skipped:",          err.message); return []; }),
    getFloods().catch(err       => { console.warn("floods skipped:",       err.message); return []; })
  ]);
  return { speedBands, incidents, faultyLights, vms, floods };
}


// ─── Individual proxy routes ──────────────────────────────────────────────────

app.get("/lta/traffic-speed-bands",   async (req, res) => { try { res.json(await getSpeedBands());    } catch (e) { res.status(500).json({ error: e.message }); } });
app.get("/lta/traffic-incidents",     async (req, res) => { try { res.json(await getIncidents());     } catch (e) { res.status(500).json({ error: e.message }); } });
app.get("/lta/faulty-traffic-lights", async (req, res) => { try { res.json(await getFaultyLights());  } catch (e) { res.status(500).json({ error: e.message }); } });
app.get("/lta/vms",                   async (req, res) => { try { res.json(await getVms());           } catch (e) { res.status(500).json({ error: e.message }); } });
app.get("/lta/flood-alerts",          async (req, res) => { try { res.json(await getFloods());        } catch (e) { res.status(500).json({ error: e.message }); } });


// ─── Traffic images ───────────────────────────────────────────────────────────

const IMAGE_CACHE_TTL = 60_000;
let imageCacheData    = null;
let imageCacheAge     = 0;
let imagePending      = null;

async function getTrafficImages() {
  const records = await fetchPaginated("Traffic-Imagesv2");
  return records.map(r => ({
    CameraID:  r.CameraID,
    Latitude:  Number(r.Latitude),
    Longitude: Number(r.Longitude),
    ImageLink: r.ImageLink,
    timestamp: new Date().toISOString()
  }));
}

app.get("/lta/traffic-images", async (req, res) => {
  try {
    const now = Date.now();

    if (imageCacheData && now - imageCacheAge <= IMAGE_CACHE_TTL) {
      // cache hit — fall through to filter + respond
    } else {
      if (!imagePending) {
        imagePending = getTrafficImages()
          .then(cameras => {
            imageCacheData = cameras;
            imageCacheAge  = Date.now();
            imagePending   = null;
            console.log(`[image cache] refreshed — ${cameras.length} cameras`);
            return cameras;
          })
          .catch(err => {
            imagePending = null;
            throw err;
          });
      } else {
        console.log("[image cache] WAIT (fetch already in flight)");
      }
      await imagePending;
    }

    const { minLat, maxLat, minLon, maxLon } = req.query;
    let cameras = imageCacheData;
    if (minLat && maxLat && minLon && maxLon) {
      cameras = cameras.filter(c =>
        c.Latitude  >= Number(minLat) && c.Latitude  <= Number(maxLat) &&
        c.Longitude >= Number(minLon) && c.Longitude <= Number(maxLon)
      );
    }

    res.json({ cameras });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


// ─── Route scoring ────────────────────────────────────────────────────────────
function extractReasons(seg, snapshot) {
  const linkId = seg.LinkID;

  const hasFlooding = snapshot.floods?.some(f =>
    f.LinkID === linkId || f.linkIds?.includes(linkId)
  );

  const hasIncident = snapshot.incidents?.some(i =>
    i.LinkID === linkId || i.linkIds?.includes(linkId)
  );

  const hasFaultyLight = snapshot.faultyLights?.some(l =>
    l.LinkID === linkId || l.linkIds?.includes(linkId)
  );

  const hasVms = snapshot.vms?.some(v =>
    v.LinkID === linkId || v.linkIds?.includes(linkId)
  );

  const hasHighSpeed = seg.SpeedBand ===7 || seg.SpeedBand ===8;

  return {
    highSpeed: hasHighSpeed,
    flooding: hasFlooding,
    incidents: hasIncident,
    faultyLights: hasFaultyLight,
    vms: hasVms
  };
}


app.post("/predict/route", async (req, res) => {
  try {
    const { linkIds } = req.body;
    if (!linkIds?.length) return res.status(400).json({ error: "linkIds required" });

    const now      = new Date();
    const snapshot = await getCachedSnapshot();

    const relevant = snapshot.speedBands.filter(s => linkIds.includes(s.LinkID));
    if (!relevant.length) return res.json({ results: [] });

    const featDicts   = [];
    const segByLinkId = {};

    for (const seg of relevant) {
      const linkId    = seg.LinkID;
      const prevSpeed = prevSpeedBandByLink.get(linkId);

      const features = buildFeatureRow(
        seg, prevSpeed,
        snapshot.incidents, snapshot.faultyLights,
        snapshot.vms, snapshot.floods, now
      );

      prevSpeedBandByLink.set(linkId, Number(seg.SpeedBand));
      featDicts.push({ link_id: linkId, timestamp: now.toISOString(), ...toFeatureVector(features) });
      segByLinkId[linkId] = seg;
    }

    const scores      = await scoreBatch(featDicts);
    const scoreByLink = Object.fromEntries(scores.map(s => [s.link_id, s]));

    /*const results = featDicts.map(f => {
      const seg = segByLinkId[f.link_id];
      return {
        link_id:    f.link_id,
        risk_index: scoreByLink[f.link_id]?.risk_index ?? null,
        StartLat:   seg.StartLat, StartLon: seg.StartLon,
        EndLat:     seg.EndLat,   EndLon:   seg.EndLon,
      };
    });*/

    const results = featDicts.map(f => {
    const seg     = segByLinkId[f.link_id];
    const reasons = extractReasons(seg, snapshot);

    return {
        link_id:     f.link_id,
        risk_index:  scoreByLink[f.link_id]?.risk_index ?? null,

        reasons, // 👈 NEW

        StartLat: seg.StartLat,
        StartLon: seg.StartLon,
        EndLat:   seg.EndLat,
        EndLon:   seg.EndLon
    };
    });

    res.json({ timestamp: now.toISOString(), results });

  } catch (err) {
    console.error("[/predict/route]", err.message);
    res.status(500).json({ error: err.message });
  }
});


// ─── Image collision detection ────────────────────────────────────────────────

app.post("/predict/image", async (req, res) => {
  try {
    const { imageUrl } = req.body;
    if (!imageUrl) return res.status(400).json({ error: "imageUrl required" });

    // Fetch the traffic camera image as a raw buffer
    const imgRes = await fetch(imageUrl);
    if (!imgRes.ok) return res.status(502).json({ error: "Failed to fetch image from URL" });

    const arrayBuffer = await imgRes.arrayBuffer();
    const imgBuffer   = Buffer.from(arrayBuffer);
    const contentType = imgRes.headers.get("content-type") || "image/jpeg";

    // Build multipart/form-data manually — avoids form-data vs node-fetch compatibility issues
    const boundary = `----FormBoundary${Date.now().toString(16)}`;
    const header   = Buffer.from(
      `--${boundary}\r\nContent-Disposition: form-data; name="image"; filename="camera.jpg"\r\nContent-Type: ${contentType}\r\n\r\n`
    );
    const footer   = Buffer.from(`\r\n--${boundary}--\r\n`);
    const body     = Buffer.concat([header, imgBuffer, footer]);

    const pyRes = await fetch(`${INFERENCE_URL}/score/image`, {
      method:  "POST",
      headers: {
        "Content-Type":   `multipart/form-data; boundary=${boundary}`,
        "Content-Length": body.length,
      },
      body,
    });

    if (!pyRes.ok) {
      const errBody = await pyRes.text().catch(() => "(unreadable)");
      return res.status(502).json({ error: `Inference error ${pyRes.status}: ${errBody}` });
    }

    const result = await pyRes.json();
    res.json(result);
  } catch (err) {
    console.error("[/predict/image]", err.message);
    res.status(500).json({ error: err.message });
  }
});


// ─── Health ───────────────────────────────────────────────────────────────────

app.get("/health", async (req, res) => {
  const inferenceOk = await checkInferenceHealth();
  res.json({
    status:    "ok",
    inference: inferenceOk ? "ok" : "unreachable",
    lta_keys:  API_KEYS.length,
    active_key_index: activeKeyIndex,
    cache: {
      valid:    cacheValid(),
      age_sec:  cache.fetchedAt ? Math.round((Date.now() - cache.fetchedAt) / 1000) : null,
      segments: cache.snapshot?.speedBands?.length ?? 0,
    }
  });
});


// ─── Background polling ───────────────────────────────────────────────────────

// setTimeout chain — next poll starts 5 min after the previous fetch *completes*,
// preventing drift past the 7-min TTL regardless of how long LTA takes to respond
async function backgroundPoll() {
  try {
    await getCachedSnapshot();
  } catch (err) {
    console.warn(`[poller] Refresh failed: ${err.message}`);
  } finally {
    const age     = cache.fetchedAt ? Date.now() - cache.fetchedAt : 0;
    const waitMs  = Math.max(0, CACHE_TTL_MS - age - 60_000); // refresh 1min before expiry
    setTimeout(backgroundPoll, waitMs);
  }
}


// ─── Start ────────────────────────────────────────────────────────────────────

app.listen(PORT, async () => {
  console.log(`LTA proxy running on http://localhost:${PORT}`);
  if (API_KEYS.length === 0) console.warn("⚠️  WARNING: No LTA API keys set in .env");

  const inferenceOk = await checkInferenceHealth();
  if (!inferenceOk) {
    console.warn("⚠️  WARNING: Inference server unreachable. Start with: python inference_server.py");
  } else {
    console.log("✓  Inference server connected");
  }

  if (API_KEYS.length > 0) {
    warmPromise = Promise.all([
      getCachedSnapshot(),
      warmGeometry()  
    ]).finally(() => {
      warmPromise = null;
      setTimeout(backgroundPoll, 300_000);
    });
  }
});
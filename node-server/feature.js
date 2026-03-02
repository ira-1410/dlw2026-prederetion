
/***********************************************************
 * Feature Builder – JS Port of Python Training Pipeline
 * Parity target: EXACT
 ***********************************************************/


const FEATURE_COLS = [
  'speed_band', 'min_speed', 'max_speed',
  'speed_deviation', 'speed_drop_rate',
  'incident_score', 'incident_count', 'nearest_incident_dist',
  'flag_accident', 'flag_breakdown', 'flag_weather', 'flag_roadwork',
  'faulty_light_score', 'blackout_count', 'flashing_count',
  'emas_breakdown_score', 'emas_rain_score',
  'flood_severity_score', 'flood_nearby',
  'road_cat_1', 'road_cat_2', 'road_cat_3',
  'road_cat_4', 'road_cat_5', 'road_cat_6', 'road_cat_8',
  'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
  'is_peak_hour', 'is_late_night'
];


/* ------------------ Constants ------------------ */

const EARTH_RADIUS_KM = 6371.0;
const PROXIMITY_THRESHOLD_KM = 1.0;
const EMAS_RADIUS_KM = 2.0;
const INCIDENT_RISK_WEIGHTS = {
    'Accident':           1.0,
    'Vehicle breakdown':  0.6,
    'Roadwork':           0.4,
    'Weather':            0.7,
    'Obstacle':           0.5,
    'Road Block':         0.5,
    'Heavy Traffic':      0.3,
    'Miscellaneous':      0.2,
    'Diversion':          0.3,
    'Unattended Vehicle': 0.5,
    'Fire':               0.9,
    'Plant Failure':      0.4,
    'Reverse Flow':       0.8,
}
const DOMAIN_MULTIPLIERS = {
    'active_accident':       2.0,
    'flood_extreme':         1.8,
    'flood_severe':          1.5,
    'flood_moderate':        1.3,
    'flood_minor':           1.1,
    'faulty_light_blackout': 1.6,
    'faulty_light_flashing': 1.3,
    'late_night':            1.3,  // 0200–0500 hrs
    'peak_hour':             1.1,   //0730–0930 or 1730–1930
    'active_roadworks':      1.2,
    'emas_breakdown':        1.4,
    'emas_rain':             1.2,
}
const EMAS_BREAKDOWN_KEYWORDS = ["breakdown", "brkdown", "veh break", "stalled"];
const EMAS_RAIN_KEYWORDS = ["rain", "wet", "flood", "slippery"];
const SEVERITY_SCORE = {
    Extreme: 4,
    Severe: 3,
    Moderate: 2,
    Minor: 1
};
const ROAD_CAT_EXPECTED_BAND = {
    1: 7.0,   //Expressways: expect SpeedBand ~7 (60-69 km/h) normally
    2: 5.5,   // Major Arterial Roads
    3: 4.5,   // Arterial Roads
    4: 3.5,   // Minor Arterial Roads
    5: 3.0,   // Small Roads
    6: 3.0,   // Slip Roads
    8: 5.0,   // Short Tunnels
}
const ROAD_CAT_ONEHOT = [1, 2, 3, 4, 5, 6, 8];

/* ------------------ Geometry ------------------ */
function haversine(lat1, lon1, lat2, lon2) {
    const toRad = v => (v * Math.PI) / 180;
    const φ1 = toRad(lat1);
    const φ2 = toRad(lat2);
    const Δφ = toRad(lat2 - lat1);
    const Δλ = toRad(lon2 - lon1);

    const a =
        Math.sin(Δφ / 2) ** 2 +
        Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) ** 2;

    return EARTH_RADIUS_KM * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}
function segmentMidpoint(seg) {
    const lat =
        (Number(seg.StartLat || 0) + Number(seg.EndLat || 0)) / 2;
    const lon =
        (Number(seg.StartLon || 0) + Number(seg.EndLon || 0)) / 2;
    return [lat, lon];
}

/* ------------------ Incident Features ------------------ */
function computeIncidentFeatures(segLat, segLon, incidents) {
    const feats = {
        incident_score: 0.0,
        incident_count: 0,
        nearest_incident_dist: PROXIMITY_THRESHOLD_KM,
        flag_accident: 0,
        flag_breakdown: 0,
        flag_weather: 0,
        flag_roadwork: 0
    };

    if (!incidents || incidents.length === 0) return feats;

    const nearby = [];

    for (const inc of incidents) {
        if (inc.Latitude == null || inc.Longitude == null) continue;

        const dist = haversine(
        segLat,
        segLon,
        Number(inc.Latitude),
        Number(inc.Longitude)
        );

        if (dist <= PROXIMITY_THRESHOLD_KM) {
        nearby.push({ dist, type: inc.Type || "Miscellaneous" });
        }
    }

    if (nearby.length === 0) return feats;

    feats.incident_count = nearby.length;
    feats.nearest_incident_dist = Math.min(
        ...nearby.map(n => n.dist)
    );

    for (const n of nearby) {
        const w = INCIDENT_RISK_WEIGHTS[n.type] ?? 0.2;
        const invDist = 1.0 / Math.max(n.dist, 0.05);
        feats.incident_score += w * invDist;

        const t = n.type.toLowerCase();
        if (t.includes("accident")) feats.flag_accident = 1;
        if (t.includes("breakdown")) feats.flag_breakdown = 1;
        if (t.includes("weather")) feats.flag_weather = 1;
        if (t.includes("roadwork") || t.includes("road work"))
        feats.flag_roadwork = 1;
    }

    feats.incident_score = Math.min(feats.incident_score, 20.0);
    return feats;
}

/* ------------------ Faulty Traffic Lights ------------------ */
function computeLightFeatures(lights) {
    const feats = {
        faulty_light_score: 0.0,
        blackout_count: 0,
        flashing_count: 0
    };

    if (!lights || lights.length === 0) return feats;

    for (const l of lights) {
        if (Number(l.Type) === 4) feats.blackout_count += 1;
        if (Number(l.Type) === 13) feats.flashing_count += 1;
    }

    feats.faulty_light_score =
        feats.blackout_count * 1.5 +
        feats.flashing_count * 0.8;

    return feats;
}

/* ------------------ VMS / EMAS ------------------ */
function computeVmsFeatures(segLat, segLon, vms) {
    const feats = {
        emas_breakdown_score: 0.0,
        emas_rain_score: 0.0
    };

    if (!vms || vms.length === 0) return feats;

    for (const row of vms) {
        if (row.Latitude == null || row.Longitude == null) continue;

        const dist = haversine(
        segLat,
        segLon,
        Number(row.Latitude),
        Number(row.Longitude)
        );

        if (dist > EMAS_RADIUS_KM) continue;

        const msg = String(row.Message || "").toLowerCase();
        const weight = 1.0 / Math.max(dist, 0.1);

        if (EMAS_BREAKDOWN_KEYWORDS.some(k => msg.includes(k))) {
        feats.emas_breakdown_score += weight;
        }
        if (EMAS_RAIN_KEYWORDS.some(k => msg.includes(k))) {
        feats.emas_rain_score += weight;
        }
    }

    feats.emas_breakdown_score = Math.min(
        feats.emas_breakdown_score,
        10.0
    );
    feats.emas_rain_score = Math.min(
        feats.emas_rain_score,
        10.0
    );

    return feats;
}

/* ------------------ Flood ------------------ */
function computeFloodFeatures(segLat, segLon, floods) {
  const feats = {
    flood_severity_score: 0.0,
    flood_nearby: 0
  };

  if (!floods || floods.length === 0) return feats;

  for (const row of floods) {
    if (row.flood_lat == null || row.flood_lon == null) continue;

    const dist = haversine(
      segLat,
      segLon,
      Number(row.flood_lat),
      Number(row.flood_lon)
    );

    const radius = Math.max(Number(row.flood_radius || 0.5), 0.5);

    if (dist <= radius) {
      const sev =
        SEVERITY_SCORE[row.severity] ?? 1;
      feats.flood_severity_score = Math.max(
        feats.flood_severity_score,
        sev
      );
      feats.flood_nearby = 1;
    }
  }

  return feats;
}

/* ------------------ Time Features ------------------ */
function cyclicalEncode(value, maxValue) {
  const angle = (2 * Math.PI * value) / maxValue;
  return [Math.sin(angle), Math.cos(angle)];
}
function timeFeatures(dt) {
  const hour = dt.getHours();
  const minute = dt.getMinutes();
  const dow = dt.getDay(); // Sunday=0 (matches Python weekday shift risk acceptable)

  const [hour_sin, hour_cos] = cyclicalEncode(hour, 24);
  const [dow_sin, dow_cos] = cyclicalEncode(dow, 7);

  const hFloat = hour + minute / 60;

  const is_peak_hour =
    (hFloat >= 7.5 && hFloat <= 9.5) ||
    (hFloat >= 17.5 && hFloat <= 19.5)
      ? 1
      : 0;

  const is_late_night =
    hour >= 2 && hour <= 5 ? 1 : 0;

  return {
    hour_sin,
    hour_cos,
    dow_sin,
    dow_cos,
    is_peak_hour,
    is_late_night
  };
}

/* ------------------ MASTER FEATURE BUILDER ------------------ */
function buildFeatureRow(
  seg,
  prevSpeedBand,
  incidents,
  lights,
  vms,
  floods,
  dt
) {
  const [segLat, segLon] = segmentMidpoint(seg);

  const roadCat = Number(seg.RoadCategory || 5);
  const speedBand = Number(seg.SpeedBand || 4);
  const minSpeed = Number(seg.MinimumSpeed || 0);
  const maxSpeed = Number(seg.MaximumSpeed || 0);

  const expectedBand =
    ROAD_CAT_EXPECTED_BAND[roadCat] ?? 4.0;

  const speed_deviation = expectedBand - speedBand;
  const speed_drop_rate =
    prevSpeedBand != null
      ? prevSpeedBand - speedBand
      : 0.0;

  const roadCatOH = {};
  for (const c of ROAD_CAT_ONEHOT) {
    roadCatOH[`road_cat_${c}`] = roadCat === c ? 1 : 0;
  }

  const row = {
    link_id: seg.LinkID || "UNKNOWN",
    timestamp: dt.toISOString(),
    road_name: seg.RoadName || "",
    seg_lat: segLat,
    seg_lon: segLon,

    speed_band: speedBand,
    min_speed: minSpeed,
    max_speed: maxSpeed,
    speed_deviation,
    speed_drop_rate,

    ...roadCatOH,
    ...timeFeatures(dt),
    ...computeIncidentFeatures(segLat, segLon, incidents),
    ...computeLightFeatures(lights),
    ...computeVmsFeatures(segLat, segLon, vms),
    ...computeFloodFeatures(segLat, segLon, floods)
  };

  return row;
}

function toFeatureVector(featureRow) {
    const missing = FEATURE_COLS.filter(col => featureRow[col] === undefined);
    if (missing.length > 0) {
        console.warn("toFeatureVector: missing keys defaulting to 0:", missing);
    }
    return Object.fromEntries(
        FEATURE_COLS.map(col => [col, Number(featureRow[col] ?? 0)])
    );
}

export {
    buildFeatureRow,
    toFeatureVector
};
# AetherSound AI

**Physics-informed underwater marine-debris & anomaly detection from side-scan sonar imagery.**
Smart India Hackathon 2026 · Problem Statement **[Your PS Number]** · Theme: Disaster Management · Category: Software
**Team Virasat** (ID **181076**).

AetherSound AI ingests a raw side-scan sonar (SSS) transect, detects man-made seabed hazards
(shipwrecks, pipelines/cables, mine-like cylinders, entangled "ghost" nets) with **oriented
bounding boxes** that follow true debris orientation, runs every candidate through a
**6-Pillar physics decision engine** before it ever reaches an operator, geotags confirmed
detections to latitude/longitude with an honest search radius, and streams the full telemetry
breakdown to a live tactical dashboard — the inference path is **torch-free, ONNX-only, ~50 MB**,
and edge-deployable on an AUV with no cloud dependency.

> Why not just fine-tune a stock YOLO model and call it done? Standard deep-learning pipelines
> treat sonar as if it were a photograph. They hallucinate on rock ripples and reverberation,
> draw axis-aligned boxes that don't match the strike angle of real debris, and destroy the
> radiometric data needed to measure shadows and elevation the moment they apply naive CLAHE
> across the whole image. AetherSound AI splits visual detection and physical verification into
> two separate streams so neither compromises the other — see §4 below.

---

## 1. Repository layout

```
apps/
  api/              [Your backend framework]  →  job orchestration, detection storage, review API
    detections/     models · views/routes · serializers · the ML job runner
  dashboard/        Next.js tactical interface  →  the operator console
    src/components/ CanvasViewport (rotated OBB rendering) · TelemetryConsole · ReviewQueue
    src/pages/       UploadResultsPage · LiveFeed · GlobalMap · Export
ml/
  scripts/          dataset assembly · preprocess (dual-stream) · train · calibrate · evaluate · export_onnx
  inference/
    stream_a/       MMSE Lee + local CLAHE  →  feeds the visual model only
    stream_b/       float32 MMSE Lee + global min-max  →  feeds the physics engine only
    detector/       ONNX YOLOv11-OBB wrapper (5-param oriented boxes: x_c, y_c, w, h, θ)
    physics_engine/ the 6-Pillar decision engine (calibration, shadow, SNR, texture, geometry, projection)
  geotagging/       XTF header parser · nav CSV · slant-to-ground projection · run_geotag
  reporting/        schema · json_export · csv_export
  models/
    checkpoints/    best_detector.pt   (training-time only, needs torch)
    exported/       best_detector.onnx + calibrator.pkl   (what actually ships)
edge/               torch-free inference: edge_infer.py · onnx_runtime_server.py · benchmark.py
demo/               curated upload bundle · dual-stream before/after visualizer · sample tiles
docs/               problem & physics · data & preprocessing · the model · the physics engine · the system
```

*(This layout mirrors how a project like this is typically organized — adjust paths to match
however your team has actually structured the repo.)*

Training data and weight backups are **not in git** — see §6.

---

## 2. Run it locally

*(Fill in with your actual stack once decided — this is a placeholder based on a typical
Django/FastAPI + Next.js split. Swap in whatever you're really running.)*

```bash
cp .env.example .env
docker compose up -d --build
docker compose exec api python manage.py migrate      # or your framework's equivalent

# frontend
cd apps/dashboard
npm install
npm run dev                                            # → http://localhost:3000
```

### Smoke test

```bash
curl -F "file=@demo/sample_tiles/ghost_net.png" \
     -F "nav=@demo/sample_tiles/navigation.csv" \
     http://localhost:8000/api/upload/                 # → {"job_id": "..."}
```

---

## 3. The dashboard

The Next.js tactical interface renders every field of the backend telemetry payload:

- **Interactive HTML5 Canvas viewport** — draws precise oriented target contours rather than
  static boxes: the canvas translates to each target's centre `(x_c, y_c)` and rotates via
  `ctx.rotate(θ)` to match the exact strike angle on the seafloor.
- **Colour-coded confidence tiers** — Emerald for verified `CONFIRMED_ANOMALY` contacts, Amber
  for secondary `PROBABLE_TARGET` scatters.
- **Operator telemetry console** — for every contact, shows:
  - **Metric dimensions (L × W)** derived from sensor resolution
  - **3D relief / protrusion height (H)**, from acoustic shadow trigonometry:
    `H = (L_s × A) / (L_s + R_s)`
  - **Acoustic channel** — `PORT` vs `STARBOARD` relative to the nadir track, plus strike heading
  - **Geospatial cross-track position** — lat/lon with a circular error probable (CEP) radius,
    never a bare pin
  - **Multi-evidence breakdown** — the individual score from all 6 physics pillars, with zero
    hidden parameters
- **Review queue** — analyst Confirm/Reject on any `PROBABLE_TARGET`, logged for fine-tuning.
- **Export** — JSON / CSV / GeoJSON per survey job.

---

## 4. The model & pipeline

### 4.1 Decoupled dual-stream architecture

Visual feature extraction and physical measurement are deliberately kept on separate paths so
that enhancing an image for a neural network never corrupts the radiometric data needed to
measure it physically:

```
                       ┌──► Stream A (MMSE Lee + local CLAHE) ────► ONNX YOLOv11-OBB (visual AI)
Raw sonar + telemetry ─┤
                       └──► Stream B (float32 MMSE Lee + global min-max) ─► 6-Pillar Physics Engine
```

- **Stream A — visual AI stream:** adaptive MMSE Lee filter + local CLAHE (clip limit 3.0, 8×8
  grid) to boost edge gradients for convolutional feature extraction. This stream feeds the
  detector only — its non-linear contrast stretch is never used for measurement.
- **Stream B — radiometric physics stream:** MMSE Lee despeckling in float32 precision, then a
  *global linear* min-max stretch (0–255) instead of CLAHE, preserving the true radiometric
  ratios between highlight, background, and shadow needed for every physics pillar below.

### 4.2 Detector

- **ONNX YOLOv11-OBB**, PyTorch stripped at inference time, ~50 MB memory footprint, runs on
  `onnxruntime` alone.
- **Oriented bounding boxes** (`x_c, y_c, w, h, θ`) instead of axis-aligned boxes, so detections
  align with the true physical orientation of elongated debris (pipes, nets, wreckage) instead
  of padding out to an axis-aligned box full of empty seabed.

### 4.3 The 6-Pillar physics decision engine

Every raw detection from the visual stream must clear all six checks (computed against
Stream B's untouched radiometric data) before it can reach the operator:

| # | Pillar | What it checks |
|---|--------|-----------------|
| 1 | Platt-scaled AI calibration | Converts raw overconfident network scores into calibrated probabilities |
| 2 | Acoustic shadow contrast | Confirms a genuine shadow void exists behind the target |
| 3 | Local signal-to-noise ratio | Ensures the highlight stands clear of background reverberation |
| 4 | Natural-feature exclusion | Sobel gradient variance rejects repetitive sand ripples / geology |
| 5 | Metric geometry & aspect ratio | Real-world dimensions must respect physical bounds (e.g. mine-like cylinders ≤ 3.5 m) |
| 6 | Slant-to-ground projection | `R_g = √(R_s² − A²)` — converts diagonal slant range to true ground distance using towfish altitude `A` |

- **Hard-veto gates:** a candidate with no acoustic shadow void, or that violates physical
  aspect-ratio constraints, is rejected before it ever reaches the operator payload
  (`REJECTED_NO_PHYSICAL_EVIDENCE`, `REJECTED_GEOMETRIC_VIOLATION`) — this is what suppresses
  false alarms on natural seabed texture that a vision-only model would hallucinate on.

### 4.4 Pipeline (per job)

```
raw SSS transect + nav log
        │
        ▼
  tile 640×640, overlap                                 ── for every tile ──┐
        │                                                                   │
        ▼                                                                   │
  dual-stream preprocess   Stream A (visual) / Stream B (physics)           │
        ▼                                                                   │
  detect     ONNX YOLOv11-OBB → oriented box + class + raw score            │
        ▼                                                                   │
  6-pillar physics engine   calibration → shadow → SNR → texture →          │
                            geometry → slant-to-ground projection           │
        ▼                                                                   │
  geotag     ground range + towfish nav → (lat, lon) + CEP radius           │
        └───────────────────────────────────────────────────────────────────┘
        ▼
  merge duplicates across overlapping tiles
        ▼
  route by calibrated score   confirmed / probable-review / vetoed
        ▼
  JSON / CSV / GeoJSON  +  live tactical dashboard
```

### 4.5 Edge / ONNX

| Model | Footprint | Runtime | Role |
|---|---|---|---|
| ONNX YOLOv11-OBB | ~50 MB | `onnxruntime` + `numpy` + `opencv`, no PyTorch | shipped — CPU-deployable on AUV hardware |

```bash
python ml/scripts/export_onnx.py --model ml/models/checkpoints/best_detector.pt --benchmark
python edge/edge_infer.py --image tile.png
uvicorn edge.onnx_runtime_server:app --port 8100
```

---

## 5. Results

*(Fill in your own held-out test numbers once you've evaluated — placeholder table below.)*

| Metric | Value |
|---|---|
| mAP@50 | *[fill in]* |
| mAP@50–95 | *[fill in]* |
| Precision / Recall | *[fill in]* |
| Calibration ECE (post Platt-scaling) | *[fill in]* |
| Ghost-net AP50 | *[fill in — flag clearly if evaluated on synthetic vs. real data]* |

> Be explicit in your docs about which numbers come from synthetic vs. real sonar data — this
> is a real credibility point judges will probe, not a formality.

---

## 6. Data & weights

*(Placeholder — document your actual sources, licences, and what's committed vs. fetched.)*

| Asset | Notes |
|---|---|
| `best_detector.onnx` + `calibrator.pkl` | committed to git — clone-and-run |
| `.pt` checkpoints, per-epoch backups | *[link to your model hosting, if any]* |
| Training splits | *[your dataset source + licence]* |

---

## 7. API

*(Placeholder contract — adjust to your actual routes.)*

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/upload/` | multipart `file` (+ optional `xtf`, `nav`) → `{job_id}` |
| `GET` | `/api/jobs/<id>/` | job status + progress |
| `GET` | `/api/detections/<job_id>/` | detection records incl. all 6-pillar scores |
| `PATCH` | `/api/detections/<id>/review/` | analyst confirm/reject → audit entry |
| `GET` | `/api/export/<job_id>/?format=json\|csv\|geojson` | report download |
| `WS` | `/ws/jobs/<job_id>/` | live per-tile detection stream |

---

## 8. Status

| Part | State |
|---|---|
| Dual-stream preprocessing | *[fill in]* |
| ONNX YOLOv11-OBB detector | *[fill in]* |
| 6-Pillar physics engine | *[fill in]* |
| Geotagging (slant-to-ground) | *[fill in]* |
| Next.js tactical dashboard | *[fill in]* |
| Edge / ONNX runtime path | *[fill in]* |
| Real-sonar validation for ghost-nets | *[fill in — flag honestly if still synthetic-only]* |
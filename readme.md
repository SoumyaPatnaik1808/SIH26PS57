# AetherSound AI | Tactical Sonar Debris & Anomaly Detection

**AI-Powered Automated Underwater Marine Debris Detection using Side-Scan Sonar**  
*Smart India Hackathon (SIH 2026) · Problem Statement: 26057 · Theme: Disaster Management / Ocean Tech*

---

## 📌 Executive Summary

AetherSound AI is an end-to-end autonomous tactical system engineered to process raw side-scan sonar (SSS) transects, detect benthic marine hazards with oriented bounding boxes (YOLO-OBB), and validate every candidate through a rigorous **6-Pillar Acoustic Physics Decision Engine** before alerting human operators.

Unlike generic computer vision approaches that hallucinate over natural seabed ripples or seabed backscatter noise, AetherSound AI enforces acoustic physics principles (acoustic shadow verification, radiometric dynamic range preservation, and towfish trigonometry) to guarantee that non-protruding artifacts are filtered out.

---

## ⚡ 1-Command Submission Run (Docker)

To run the entire integrated software stack (FastAPI Backend + Next.js Tactical Dashboard + ONNX YOLO-OBB + 6-Pillar Physics Decision Engine) in a single command:

```bash
docker compose up --build
```

Once running:
* **Tactical Operator Console**: [http://localhost:3000](http://localhost:3000)
* **Backend API & Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check Endpoint**: [http://localhost:8000/](http://localhost:8000/)

To stop the containers:
```bash
docker compose down
```

---

## ☁️ Cloud Deployment Architecture

This project is configured for cloud deployment:
1. **Frontend (Vercel)**: Deployed directly from the `frontend/` folder. Next.js handles server-side rendering, CDN edge caching, and interactive canvas overlays.
2. **Backend (Render)**: Deployed from `backend/Dockerfile`. Containerized Python environment with OpenCV, SciPy, and ONNX Runtime.

### Configuration for Cloud Deployment

* **Render (Backend)**:
  * **Root Directory**: `backend`
  * **Build Command**: Automatically detected from `backend/Dockerfile`
  * **Environment Variable**: `ALLOWED_ORIGINS` = `https://your-frontend.vercel.app,http://localhost:3000`
* **Vercel (Frontend)**:
  * **Root Directory**: `frontend`
  * **Framework Preset**: Next.js
  * **Environment Variable**: `NEXT_PUBLIC_API_URL` = `https://your-backend.onrender.com`

---

## 📂 Repository Structure & Standards

```
SONAR_SIH26_57/
├── docker-compose.yml          # Top-level multi-container orchestrator
├── .gitignore                  # Production ignore file for Python, Node, and IDEs
├── readme.md                   # Complete system and deployment documentation
│
├── backend/                    # FastAPI Inference & Physics Engine
│   ├── Dockerfile              # Python 3.11-slim with OpenCV & health checks
│   ├── .dockerignore           # Prevents cache & virtualenv container leakage
│   ├── requirements.txt        # Pinned dependencies (FastAPI, ONNX, OpenCV, SciPy)
│   ├── main.py                 # API router, CORS handling, and validation pipeline
│   ├── models/
│   │   └── sonar_model.onnx    # High-speed ONNX YOLOv11-OBB model weights (~10.3 MB)
│   └── utils/
│       ├── preprocessing.py    # Decoupled Dual-Stream Acoustic Physics Filter
│       ├── decision_engine.py  # 6-Pillar Multi-Evidence Fusion Engine
│       └── sonar_validator.py  # Input shield protecting against non-sonar imagery
│
└── frontend/                   # Next.js 16 + React 19 Tactical Dashboard
    ├── Dockerfile              # Multi-stage production container (Node 20-alpine)
    ├── .dockerignore           # Excludes node_modules & development build cache
    ├── next.config.ts          # Configured with standalone output for lean containers
    ├── package.json            # Scripts & frontend dependencies (lucide-react, tailwindcss)
    ├── app/
    │   ├── page.tsx            # Tactical operator dashboard with metric tooltips & reports
    │   ├── layout.tsx          # Root layout and theme providers
    │   └── globals.css         # Lightweight GPU-accelerated CSS animations and reticles
    └── public/                 # Static vector assets and icons
```

---

## 🔬 Core Innovations & Scientific Pipeline

### 1. Decoupled Dual-Stream Preprocessing Architecture
Conventional deep learning applies standard histogram equalization across the whole image, which destroys true physical acoustic shadow intensity ratios. AetherSound AI decouples preprocessing into two distinct streams:
* **Stream A (Visual AI Stream)**: Adaptive MMSE Lee filter + localized CLAHE (clip limit 3.0, 8×8 grid) to boost spatial edge gradients for oriented bounding box neural detection.
* **Stream B (Radiometric Physics Stream)**: Unwarped float32 linear dynamic range normalization ($0-255$), preserving exact backscatter-to-shadow ratios required for physical shadow depth calculations.

### 2. The 6-Pillar Multi-Evidence Decision Engine
Every raw candidate detected by the neural network must pass through six independent physics verification checks before an alert is raised:
1. **Calibrated AI Confidence**: Platt logistic regression calibration mapping raw model activations to genuine statistical probability.
2. **Acoustic Shadow Contrast**: Verifies that a physical acoustic shadow void exists behind the object. A real 3D protrusion on the seabed *must* obstruct sound waves.
3. **Seabed Signal-to-Noise Ratio (SNR)**: Ensures the highlight stands distinctly clear of ambient seafloor reverberation.
4. **Natural Feature Exclusion**: Sobel directional gradient analysis that penalizes natural geological ridges, rock outcrops, and sand ripples.
5. **Metric Geometry Plausibility**: Compares measured pixel dimensions against real-world metric constraints (e.g. mine cylinder diameters between $0.3\text{m}$ and $3.5\text{m}$).
6. **Towfish Slant-to-Ground Projection**: Converts diagonal acoustic slant range ($R_s$) to true ground distance ($R_g$) using towfish altitude ($A$):
   $$R_g = \sqrt{R_s^2 - A^2}$$
   Calculates 3D protrusion relief height ($H$) from acoustic shadow length ($L_s$):
   $$H = \frac{L_s \cdot A}{L_s + R_s}$$

### 3. Non-Sonar Input Validation Shield
An 8-heuristic statistical test suite (aspect ratio, edge density, histogram distribution, and texture entropy) intercepts non-sonar uploads (such as document photographs, camera snapshots, or blank files) and rejects them before they reach the neural network, preventing false alarms.

### 4. Tactical Operator Console
* **Rotated Canvas OBB Overlays**: Renders true oriented bounding contours via `ctx.rotate(\theta)` matching seafloor strike headings.
* **Metric Decoders & Tooltips**: Plain-English explanations for every technical metric to ensure operational clarity for non-specialist naval and survey operators.
* **Filter Toggle**: One-click toggle between "Verified Targets Only" and "All Candidates (including suppressed false alarms)".
* **1-Click Mission Report Export**: Exports geodetic GPS coordinates, metric dimensions, and telemetry breakdown to a standardized `.json` format for dive or ROV recovery teams.

---

## 💻 Local Development Setup (Without Docker)

### Backend
```bash
cd backend
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.
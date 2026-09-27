# AetherSound AI | Tactical Sonar Debris & Anomaly Detection

**AI-Powered Automated Underwater Marine Debris Detection using Side-Scan Sonar**  
*SIH 2026 · Problem Statement: 26057 · Theme: Disaster Management / Ocean Tech*

---

## ⚡ 1-Command Submission Run (Docker)

To run the entire integrated software stack (FastAPI Backend + Next.js Tactical Dashboard + ONNX YOLO-OBB + 6-Pillar Physics Decision Engine):

```bash
docker compose up --build
```

Once built and running:
* **Tactical Frontend**: [http://localhost:3000](http://localhost:3000)
* **Inference API & Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Architecture & Containerization Details

The project is containerized with isolated, production-grade Docker containers orchestrated via `docker-compose.yml`:

```
SONAR_SIH26_57/
├── docker-compose.yml          # Top-level multi-container orchestrator
├── backend/
│   ├── Dockerfile              # Python 3.11-slim with OpenCV, ONNX runtime & health checks
│   ├── .dockerignore           # Prevents local venv & cache pollution
│   ├── main.py                 # FastAPI edge endpoint + validation gate
│   ├── models/
│   │   └── sonar_model.onnx    # High-speed ONNX YOLO-OBB weights
│   └── utils/
│       ├── preprocessing.py    # Dual-stream acoustic physics filter (Lee filter + CLAHE)
│       ├── decision_engine.py  # 6-Pillar acoustic physics validation engine
│       └── sonar_validator.py  # Input validation gate against non-sonar imagery
└── frontend/
    ├── Dockerfile              # Multi-stage production build (Node 20-alpine + standalone runner)
    ├── .dockerignore           # Prevents node_modules & build artifacts leakage
    ├── next.config.ts          # Standalone optimized Next.js server configuration
    ├── app/
    │   ├── page.tsx            # Tactical operator dashboard with metric tooltips & reports
    │   └── globals.css         # Hardware-accelerated CSS animations & reticles
    └── package.json
```

---

## 🛠️ Individual Container Build & Run

If you wish to test or run containers individually:

### Backend (Port 8000)
```bash
cd backend
docker build -t aethersound-backend .
docker run -p 8000:8000 aethersound-backend
```

### Frontend (Port 3000)
```bash
cd frontend
docker build --build-arg NEXT_PUBLIC_API_URL=http://localhost:8000 -t aethersound-frontend .
docker run -p 3000:3000 aethersound-frontend
```

---

## 🔬 Core Innovation Highlights

1. **Dual-Stream Preprocessing Architecture**:
   - **Stream A (Visual AI)**: MMSE Lee despeckling + local CLAHE for oriented bounding box neural detection.
   - **Stream B (Radiometric Physics)**: Unwarped float32 linear dynamic range for acoustic shadow trigonometry ($H = \frac{L_s \cdot A}{L_s + R}$).
2. **6-Pillar Multi-Evidence Decision Engine**:
   - Converts raw bounding boxes into verified tactical intelligence with Platt calibration, shadow contrast checks, seabed SNR, natural geological exclusion, and geometric bounding checks.
3. **Input Validation Shield**:
   - Heuristic guard rejecting non-sonar images (documents, regular photographs) before inference.
4. **Interactive Tactical Dashboard**:
   - Canvas-based oriented bounding boxes ($x, y, w, h, \theta$), hoverable metric decoders, filtered views for verified vs false alarms, and 1-click JSON mission report export.
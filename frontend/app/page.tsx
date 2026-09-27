"use client";

import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import {
  ArrowDown, ArrowUpRight, Check, ChevronRight, Compass,
  Crosshair, FileImage, Focus, Layers3, MapPin, Radar, Radio, ScanLine,
  ShieldCheck, Upload, Waves, X,
} from "lucide-react";

type Target = {
  class_name: string;
  class_id: number;
  yolo_confidence: number;
  verdict: string;
  fused_score: number;
  bbox_obb: [number, number, number, number, number];
  evidence_breakdown: Record<string, number>;
  physical_dimensions: Record<string, number>;
  coordinates: { latitude?: number; longitude?: number };
  tactical_telemetry: {
    threat_classification?: string;
    acoustic_channel?: string;
    target_strike_heading_deg?: number;
    estimated_3d_relief_height_m?: number;
  };
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const evidenceLabels: [string, string][] = [
  ["Calibrated AI probability", "calibrated_ai_probability"],
  ["Acoustic shadow strength", "acoustic_shadow_strength"],
  ["Local SNR / image quality", "image_quality_index"],
  ["Natural feature exclusion", "natural_feature_exclusion"],
  ["Geometric plausibility", "geometric_plausibility"],
];

function percent(value: number) {
  return `${Math.max(0, Math.min(100, value * 100))}%`;
}

function verdictTone(verdict: string) {
  if (verdict === "CONFIRMED_ANOMALY") return "confirmed";
  if (verdict.startsWith("REJECTED_")) return "rejected";
  return "probable";
}

function displayNumber(value: number | undefined, digits = 2) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "N/A";
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState("");
  const [targets, setTargets] = useState<Target[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [altitude, setAltitude] = useState("5.0");
  const [latitude, setLatitude] = useState("12.9716");
  const [longitude, setLongitude] = useState("77.5946");
  const [busy, setBusy] = useState(false);
  const [hasRun, setHasRun] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [viewportVersion, setViewportVersion] = useState(0);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const consoleRef = useRef<HTMLElement>(null);
  const imageUrlRef = useRef("");

  const selected = targets[selectedIndex];

  useEffect(() => () => {
    if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image || !image.complete || !image.naturalWidth) return;
    const bounds = image.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(bounds.width * dpr);
    canvas.height = Math.round(bounds.height * dpr);
    canvas.style.width = `${bounds.width}px`;
    canvas.style.height = `${bounds.height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, bounds.width, bounds.height);
    const scaleX = bounds.width / image.naturalWidth;
    const scaleY = bounds.height / image.naturalHeight;

    targets.forEach((target, index) => {
      const [cx, cy, width, height, angle] = target.bbox_obb;
      const active = index === selectedIndex;
      const color = target.verdict === "CONFIRMED_ANOMALY" ? "#49e6b3" : target.verdict.startsWith("REJECTED_") ? "#dc7468" : "#f2b65e";
      ctx.save();
      ctx.translate(cx * scaleX, cy * scaleY);
      ctx.rotate(angle);
      ctx.strokeStyle = color;
      ctx.lineWidth = active ? 2.5 : 1.5;
      ctx.shadowColor = color;
      ctx.shadowBlur = active ? 12 : 5;
      ctx.strokeRect(-(width * scaleX) / 2, -(height * scaleY) / 2, width * scaleX, height * scaleY);
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(0, -(height * scaleY) / 2 - 13);
      ctx.strokeStyle = "#38d6e8";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.restore();

      ctx.save();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(cx * scaleX, cy * scaleY, active ? 3.5 : 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = "600 10px ui-monospace, monospace";
      const label = `T-${String(index + 1).padStart(2, "0")}`;
      ctx.fillStyle = "#081110";
      ctx.fillRect(cx * scaleX - 18, cy * scaleY + (height * scaleY) / 2 + 5, 36, 16);
      ctx.fillStyle = color;
      ctx.fillText(label, cx * scaleX - 13, cy * scaleY + (height * scaleY) / 2 + 16);
      ctx.restore();
    });
  }, [targets, selectedIndex, imageUrl, viewportVersion]);

  useEffect(() => {
    const stage = canvasRef.current?.parentElement;
    if (!stage) return;
    const observer = new ResizeObserver(() => setViewportVersion((version) => version + 1));
    observer.observe(stage);
    return () => observer.disconnect();
  }, [imageUrl]);

  function acceptFile(nextFile?: File) {
    if (!nextFile) return;
    if (!nextFile.type.startsWith("image/")) {
      setError("Select a sonar image in JPG, PNG, or another supported image format.");
      return;
    }
    if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current);
    imageUrlRef.current = URL.createObjectURL(nextFile);
    setImageUrl(imageUrlRef.current);
    setFile(nextFile);
    setTargets([]);
    setHasRun(false);
    setError("");
    setSelectedIndex(0);
    window.setTimeout(() => consoleRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    acceptFile(event.target.files?.[0]);
    event.target.value = "";
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files[0]);
  }

  async function executeAnalysis() {
    if (!file || busy) return;
    setBusy(true);
    setError("");
    setHasRun(false);
    const body = new FormData();
    body.append("file", file);
    const query = new URLSearchParams({ altitude, towfish_lat: latitude, towfish_lon: longitude });
    try {
      const response = await fetch(`${API_URL}/api/v1/detect?${query}`, { method: "POST", body });
      if (!response.ok) throw new Error(`Analysis service returned ${response.status}. Check that the backend is running.`);
      const result: Target[] = await response.json();
      setTargets(result);
      setSelectedIndex(0);
      setHasRun(true);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to reach the analysis service.");
    } finally {
      setBusy(false);
    }
  }

  function selectOnCanvas(event: React.MouseEvent<HTMLCanvasElement>) {
    const image = imageRef.current;
    if (!image) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (image.naturalWidth / rect.width);
    const y = (event.clientY - rect.top) * (image.naturalHeight / rect.height);
    const found = targets.findIndex((target) => {
      const [cx, cy, width, height, angle] = target.bbox_obb;
      const dx = x - cx;
      const dy = y - cy;
      const localX = dx * Math.cos(angle) + dy * Math.sin(angle);
      const localY = -dx * Math.sin(angle) + dy * Math.cos(angle);
      return Math.abs(localX) <= width / 2 + 8 && Math.abs(localY) <= height / 2 + 8;
    });
    if (found >= 0) setSelectedIndex(found);
  }

  function jumpToConsole() {
    consoleRef.current?.scrollIntoView({ behavior: "smooth" });
    window.setTimeout(() => fileRef.current?.click(), 250);
  }

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Aether Sound AI home">
          <span className="brand-mark"><Waves size={18} strokeWidth={1.8} /></span>
          <span>AETHER<span className="brand-light">SOUND</span> AI</span>
        </a>
        <div className="topbar-right">
          
          <button className="nav-upload" onClick={jumpToConsole}>INGEST SONAR LOG <ArrowUpRight size={14} /></button>
        </div>
      </header>

      <section className="hero" id="top">
        <div className="hero-grid" />
        
        <div className="hero-content">
         
          <h1>ILLUMINATING THE<br /><span>ABYSSAL SILENCE.</span></h1>
          <p className="hero-copy">Autonomous sonar intelligence detecting ghost nets, benthic debris, and submerged hazards across Indian EEZ waters with millimetric precision.</p>
          <div className="hero-actions">
            <button className="primary-button" onClick={jumpToConsole}>UPLOAD YOUR SONAR LOG <ArrowUpRight size={16} /></button>
            <a className="secondary-button" href="#mission"><span className="play-icon"><ArrowDown size={13} /></span> EXPLORE THE SYSTEM</a>
          </div>
          
        </div>
        <div className="hero-index"><span>01</span><i /> <span>02</span></div>
        <a href="#mission" className="scroll-cue"><span>SCROLL TO EXPLORE</span><ArrowDown size={13} /></a>
      </section>

      

      <section className="console-section" ref={consoleRef} id="console">
        <div className="console-heading">
          <div><span className="section-kicker">FIELD CONSOLE / 01</span><h2>Acoustic analysis</h2></div>
         
        </div>
        <div className="flight-bar">
          <div className="flight-title"><Compass size={16} /><span>SURVEY PARAMETERS</span></div>
          <label className="field-group"><span>ALTITUDE <small>M</small></span><input aria-label="Altitude in meters" type="number" step="0.1" min="0" value={altitude} onChange={(event) => setAltitude(event.target.value)} /></label>
          <label className="field-group"><span>TOWFISH LAT</span><input aria-label="Towfish latitude" type="number" step="0.0001" value={latitude} onChange={(event) => setLatitude(event.target.value)} /></label>
          <label className="field-group"><span>TOWFISH LON</span><input aria-label="Towfish longitude" type="number" step="0.0001" value={longitude} onChange={(event) => setLongitude(event.target.value)} /></label>
          <button className="execute-button" disabled={!file || busy} onClick={executeAnalysis}>{busy ? <><span className="button-spinner" /> ANALYZING</> : <><ScanLine size={15} /> EXECUTE ANALYSIS</>}</button>
        </div>

        <input ref={fileRef} className="visually-hidden" type="file" accept="image/*" onChange={onFileChange} />
        {error && <div className="error-banner" role="alert"><X size={15} /> {error}</div>}

        {!file ? (
          <div className={`ingest-zone ${dragging ? "is-dragging" : ""}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
            <div className="ingest-icon"><FileImage size={22} /></div>
            <div><h3>Awaiting sonar imagery</h3><p>Drop a side-scan image here, or browse your files</p></div>
            <button className="browse-button" onClick={() => fileRef.current?.click()}><Upload size={15} /> CHOOSE IMAGE</button>
            <span className="file-hint">JPG · PNG · TIFF</span>
          </div>
        ) : (
          <div className="workspace-grid">
            <div className="viewport-column">
              <div className="viewport-toolbar"><div><span className="live-dot" /> SONAR VIEWPORT <span className="toolbar-divider">/</span> {file.name}</div><button className="icon-button" aria-label="Remove image" title="Remove image" onClick={() => { if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current); imageUrlRef.current = ""; setImageUrl(""); setFile(null); setTargets([]); setHasRun(false); }}><X size={15} /></button></div>
              <div className="image-stage" onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
                <img ref={imageRef} src={imageUrl} alt="Uploaded sonar scan" onLoad={() => setViewportVersion((version) => version + 1)} />
                <canvas ref={canvasRef} className="obb-canvas" aria-label="Detection overlay. Click a target box to inspect." onClick={selectOnCanvas} />
                {!hasRun && <div className="awaiting-overlay"><Focus size={19} /><span>IMAGE LOADED · READY FOR ANALYSIS</span></div>}
                {busy && <div className="scan-overlay"><div className="scan-line" /><span><Radar size={16} /> RUNNING 6-PILLAR FUSION</span></div>}
                <div className="viewport-corner corner-tl" /><div className="viewport-corner corner-tr" /><div className="viewport-corner corner-bl" /><div className="viewport-corner corner-br" />
              </div>
              <div className="viewport-footer"><span><Crosshair size={13} /> {targets.length ? `${targets.length} MODEL CANDIDATE${targets.length === 1 ? "" : "S"}` : hasRun ? "NO MODEL DETECTIONS" : "NO ACTIVE OVERLAYS"}</span><span>OBB / RADIAN ROTATION</span></div>
              {hasRun && targets.length > 1 && <div className="target-list"><span className="target-list-label">MODEL CANDIDATES</span>{targets.map((target, index) => <button className={`target-row ${selectedIndex === index ? "active" : ""}`} key={`${target.class_id}-${index}`} onClick={() => setSelectedIndex(index)}><span className={`target-dot ${verdictTone(target.verdict)}`} /><span>T-{String(index + 1).padStart(2, "0")} / {target.class_name.replaceAll("_", " ").toUpperCase()}</span><span className="target-row-score">{Math.round(target.fused_score * 100)}%</span><ChevronRight size={14} /></button>)}</div>}
            </div>

            <aside className="telemetry-panel">
              <div className="panel-heading"><div><span className="section-kicker">INTELLIGENCE REPORT</span><h3>Target telemetry</h3></div><Layers3 size={17} /></div>
              {!hasRun ? <div className="panel-placeholder"><Radar size={25} /><p>Analysis telemetry will populate here after the acoustic sweep.</p></div> : targets.length === 0 ? (
                <div className="empty-state"><div className="empty-state-icon"><ShieldCheck size={24} /></div><span className="empty-state-tag"><Check size={12} /> SWEEP COMPLETE</span><h3>No model detections found.</h3><p>The model returned no oriented-box candidates for this image.</p><div className="empty-rule"><span /> TRY ANOTHER SONAR IMAGE</div></div>
              ) : selected ? (
                <div className="telemetry-content">
                  <article className="telemetry-card verification-card">
                    <div className="card-title"><span>01</span> VERIFICATION &amp; TACTICAL VERDICT</div>
                    <div className="target-name-row"><div><h4>{selected.class_name.replaceAll("_", " ")}</h4><span className="class-id">MODEL CLASS {selected.class_id}</span></div><span className={`verdict-badge ${verdictTone(selected.verdict)}`}>{selected.verdict.replaceAll("_", " ")}</span></div>
                    <div className="tag-row"><span className="tactical-tag"><ShieldCheck size={12} /> {selected.tactical_telemetry.threat_classification ?? "CLASSIFICATION UNAVAILABLE"}</span><span className="channel-tag"><Radio size={12} /> {selected.tactical_telemetry.acoustic_channel ?? "CHANNEL N/A"}</span></div>
                    <div className="score-block"><div className="score-head"><span>FUSED ACOUSTIC CONFIDENCE</span><strong>{Math.round(selected.fused_score * 100)}<small>%</small></strong></div><div className="progress-track score-track"><span style={{ width: percent(selected.fused_score) }} /></div></div>
                    <div className="baseline-row"><span>RAW YOLO CONFIDENCE</span><strong>{(selected.yolo_confidence * 100).toFixed(1)}%</strong></div>
                  </article>

                  <article className="telemetry-card">
                    <div className="card-title"><span>02</span> PHYSICS EVIDENCE / 6 PILLARS</div>
                    <div className="evidence-list">{evidenceLabels.map(([label, key]) => { const score = selected.evidence_breakdown[key]; return <div className="evidence-item" key={key}><div className="evidence-label"><span>{label}</span><strong>{score == null ? "N/A" : <>{(score * 100).toFixed(0)}<small>%</small></>}</strong></div><div className="progress-track"><span style={{ width: score == null ? "0%" : percent(score) }} /></div></div>; })}</div>
                    <div className="uncertainty-row"><span>GEOSPATIAL UNCERTAINTY</span><strong>{displayNumber(selected.evidence_breakdown.geospatial_accuracy_radius_m)} <small>m</small></strong></div>
                  </article>

                  <article className="telemetry-card">
                    <div className="card-title"><span>03</span> METRIC DIMENSIONS &amp; RELIEF</div>
                    <div className="metric-grid"><div className="metric-cell"><span>REAL LENGTH</span><strong>{displayNumber(selected.physical_dimensions.length_meters)}<small> m</small></strong></div><div className="metric-cell"><span>REAL WIDTH</span><strong>{displayNumber(selected.physical_dimensions.width_meters)}<small> m</small></strong></div><div className="metric-cell"><span>GROUND RANGE <i>R<sub>g</sub></i></span><strong>{displayNumber(selected.physical_dimensions.ground_range_meters)}<small> m</small></strong></div><div className="metric-cell"><span>ACOUSTIC 3D RELIEF <i>H</i></span><strong>{displayNumber(selected.tactical_telemetry.estimated_3d_relief_height_m)}<small> m</small></strong></div></div>
                    <p className="relief-note">Protrusion height calculated via acoustic shadow trigonometry</p>
                    <div className="heading-row"><span><Compass size={13} /> STRIKE ORIENTATION</span><strong>{displayNumber(selected.tactical_telemetry.target_strike_heading_deg, 1)}°</strong></div>
                  </article>

                  <article className="telemetry-card">
                    <div className="card-title"><span>04</span> MODEL OUTPUT / ORIENTED BOX</div>
                    <div className="metric-grid geometry-grid"><div className="metric-cell"><span>MODEL CLASS ID</span><strong>{selected.class_id}</strong></div><div className="metric-cell"><span>CENTER X</span><strong>{displayNumber(selected.bbox_obb[0], 1)}<small> px</small></strong></div><div className="metric-cell"><span>CENTER Y</span><strong>{displayNumber(selected.bbox_obb[1], 1)}<small> px</small></strong></div><div className="metric-cell"><span>BOX WIDTH</span><strong>{displayNumber(selected.bbox_obb[2], 1)}<small> px</small></strong></div><div className="metric-cell"><span>BOX HEIGHT</span><strong>{displayNumber(selected.bbox_obb[3], 1)}<small> px</small></strong></div><div className="metric-cell"><span>ROTATION</span><strong>{displayNumber(selected.bbox_obb[4] * 180 / Math.PI, 1)}<small>°</small></strong></div></div>
                  </article>

                  <article className="telemetry-card geo-card">
                    <div className="card-title"><span>05</span> GEODETIC TELEMETRY</div>
                    <div className="coordinate-row"><MapPin size={14} /><span>LATITUDE</span><strong>{displayNumber(selected.coordinates.latitude, 6)}{selected.coordinates.latitude == null ? "" : "°"}</strong></div>
                    <div className="coordinate-row"><MapPin size={14} /><span>LONGITUDE</span><strong>{displayNumber(selected.coordinates.longitude, 6)}{selected.coordinates.longitude == null ? "" : "°"}</strong></div>
                    {typeof selected.coordinates.latitude === "number" && typeof selected.coordinates.longitude === "number" && <a className="map-link" href={`https://www.openstreetmap.org/?mlat=${selected.coordinates.latitude}&mlon=${selected.coordinates.longitude}#map=18/${selected.coordinates.latitude}/${selected.coordinates.longitude}`} target="_blank" rel="noreferrer">VIEW ON OPENSTREETMAP <ArrowUpRight size={14} /></a>}
                  </article>
                </div>
              ) : null}
            </aside>
          </div>
        )}
      </section>

      <footer className="footer"><a className="brand footer-brand" href="#top"><span className="brand-mark"><Waves size={16} /></span><span>AETHER<span className="brand-light">SOUND</span> AI</span></a><span>ACOUSTIC INTELLIGENCE FOR A CLEARER OCEAN</span><span>SIH 2026 <i>/</i> PS 26057</span></footer>
    </main>
  );
}

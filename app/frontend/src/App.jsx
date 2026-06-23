import React, { useState, useEffect, useRef } from 'react';

// Formatter Helpers
const formatMB = (bytes) => {
  return (bytes / (1024 * 1024)).toFixed(1);
};

const formatSpeed = (bytesPerSec) => {
  if (bytesPerSec > 1024 * 1024) {
    return `${(bytesPerSec / (1024 * 1024)).toFixed(1)} MB/s`;
  } else if (bytesPerSec > 1024) {
    return `${(bytesPerSec / 1024).toFixed(1)} KB/s`;
  }
  return `${bytesPerSec.toFixed(0)} B/s`;
};

const getStatusMessage = (step) => {
  switch (step) {
    case 0: return 'CARGANDO VOLUMEN Y SERIE DICOM...';
    case 1: return 'EXTRAYENDO FICHA DE METADATOS DICOM...';
    case 2: return 'ANALIZANDO PLANO SAGITAL (SWIN-NET IA)...';
    case 3: return 'ANALIZANDO PLANO CORONAL (SWIN-NET IA)...';
    case 4: return 'ANALIZANDO PLANO AXIAL (SWIN-NET IA)...';
    case 5: return 'CONSOLIDANDO ENSEMBLE Y CALIBRACIÓN...';
    default: return 'PROCESANDO RESONANCIA MAGNÉTICA...';
  }
};

// Interactive PACS-like Volume Scroller Component
function PACSVolumeViewer({ plane, planeData, showHeatmap, heatmapOpacity, threshold, activeIdx, setActiveIdx, selectedModel }) {
  const slices = planeData.slices;
  
  const safeActiveIdx = activeIdx !== undefined && activeIdx !== null ? activeIdx : 0;
  const activeSlice = slices[safeActiveIdx] || slices[0];
  const viewportRef = useRef(null);

  // Mouse wheel scroll navigation bound natively with { passive: false }
  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;

    const handleWheel = (e) => {
      e.preventDefault();
      if (e.deltaY < 0) {
        setActiveIdx(Math.max(0, safeActiveIdx - 1));
      } else if (e.deltaY > 0) {
        setActiveIdx(Math.min(slices.length - 1, safeActiveIdx + 1));
      }
    };

    viewport.addEventListener('wheel', handleWheel, { passive: false });
    return () => {
      viewport.removeEventListener('wheel', handleWheel);
    };
  }, [slices.length, safeActiveIdx, setActiveIdx]);


  if (!activeSlice) return null;

  // Rank active slice attention in that plane among predicted slices
  const rankedIndices = React.useMemo(() => {
    if (!slices || slices.length === 0) return [];
    return [...slices]
      .map((s, idx) => ({ idx, attention: s.has_prediction ? s.swin_attention : -1 }))
      .sort((a, b) => b.attention - a.attention)
      .map(item => item.idx);
  }, [slices]);

  const hasPred = activeSlice.has_prediction;

  let rankClass = 'slice-attention-badge rank-low';
  let rankLabel = 'Atención Normal';
  if (!hasPred) {
    rankClass = 'slice-attention-badge rank-ref';
    rankLabel = 'Corte de Referencia';
  } else {
    const rankPos = rankedIndices.indexOf(safeActiveIdx);
    if (rankPos === 0) {
      rankClass = 'slice-attention-badge rank-1';
      rankLabel = 'Atención Crítica';
    } else if (rankPos === 1) {
      rankClass = 'slice-attention-badge rank-2';
      rankLabel = 'Atención Moderada';
    }
  }

  return (
    <div className="pacs-viewer-container">
      <div className="pacs-main-panel">
        {/* Main Viewport */}
        <div 
          className="pacs-viewport"
          ref={viewportRef}
        >
          <img 
            className="mri-slice-img" 
            src={`data:image/jpeg;base64,${activeSlice.raw_image}`} 
            alt={`Corte ${activeSlice.slice_index}`} 
          />
          {hasPred && activeSlice.heatmap_overlay && (
            <img 
              className="saliency-heatmap-img" 
              src={`data:image/png;base64,${activeSlice.heatmap_overlay}`} 
              alt="Mapa Grad-CAM"
              style={{ opacity: showHeatmap ? heatmapOpacity : 0 }}
            />
          )}

          {/* Hud Overlay */}
          <div className="pacs-hud top-left">
            <span className="pacs-hud-label">CORTE</span>
            <span className="pacs-hud-val">#{activeSlice.slice_index} ({safeActiveIdx + 1}/{slices.length})</span>
          </div>
          <div className="pacs-hud top-right">
            <span className={rankClass}>{rankLabel}</span>
          </div>
          <div className="pacs-hud bottom-left">
            <span className="pacs-hud-label">PROB. CORTE</span>
            <span className="pacs-hud-val" style={{ color: hasPred && activeSlice.slice_probability >= threshold ? 'var(--crimson)' : 'var(--emerald)' }}>
              {hasPred ? `${Math.round(activeSlice.slice_probability * 100)}%` : 'N/A'}
            </span>
          </div>
          <div className="pacs-hud bottom-right">
            <span className="pacs-hud-label">ATENCIÓN</span>
            <span className="pacs-hud-val" style={{ color: hasPred ? 'var(--neon-teal)' : 'var(--color-text-secondary)' }}>
              {hasPred ? `${Math.round(activeSlice.swin_attention * 100)}%` : 'N/A'}
            </span>
          </div>
        </div>

        {/* Scrub Bar */}
        <div className="pacs-scrubber-bar">
          <input 
            type="range" 
            min="0" 
            max={slices.length - 1} 
            value={safeActiveIdx} 
            onChange={(e) => setActiveIdx(parseInt(e.target.value))} 
            className="pacs-slider-scrub"
          />
          <div className="pacs-scrubber-labels">
            <span>Corte {slices[0].slice_index}</span>
            <span style={{ fontSize: '0.7rem', color: 'var(--color-text-secondary)' }}>
              Usa el control deslizante, las miniaturas o la rueda del ratón para navegar entre cortes
            </span>
            <span>Corte {slices[slices.length - 1].slice_index}</span>
          </div>
        </div>
      </div>

      {/* Metric List Side panel */}
      <div className="pacs-metrics-panel">
        <div className="pacs-metrics-header">
          <h4>Métricas del Corte #{activeSlice.slice_index}</h4>
          <span className="pacs-metrics-sub">
            {hasPred ? 'Mapeo del modelo en tiempo real' : 'Corte anatómico de soporte'}
          </span>
        </div>

        <div className="pacs-metrics-list">
          <div className="pacs-metric-row" style={{ opacity: hasPred ? 1 : 0.4 }}>
            <span className="pacs-metric-lbl">Localizador (MobileNetV2):</span>
            <span className="pacs-metric-val">
              {hasPred ? `${Math.round(activeSlice.selector_prob * 100)}%` : 'N/A'}
            </span>
          </div>
          <div className="pacs-metric-row" style={{ opacity: hasPred ? 1 : 0.4 }}>
            <span className="pacs-metric-lbl">
              {selectedModel === 'swin' ? 'Peso de Atención Swin:' : 
               selectedModel === 'vit' ? 'Peso de Atención ViT:' : 
               plane === 'coronal' ? 'Pooling de Corte (Max):' : 'Peso de Atención CNN:'}
            </span>
            <span className="pacs-metric-val">
              {hasPred ? (
                (selectedModel === 'cnn' && plane === 'coronal') ? 'Max' : `${Math.round(activeSlice.swin_attention * 100)}%`
              ) : 'N/A'}
            </span>
          </div>
          <div className="pacs-metric-row" style={{ opacity: hasPred ? 1 : 0.4 }}>
            <span className="pacs-metric-lbl">Prob. de Lesión:</span>
            <span className="pacs-metric-val" style={{ color: hasPred ? (activeSlice.slice_probability >= threshold ? 'var(--neon-pink)' : 'var(--neon-teal)') : 'inherit' }}>
              {hasPred ? `${Math.round(activeSlice.slice_probability * 100)}%` : 'N/A'}
            </span>
          </div>
        </div>

        {!hasPred ? (
          <div className="pacs-clinical-alert alert-reference">
            <div className="alert-icon" style={{ fontSize: '1.2rem' }}>ℹ️</div>
            <div className="alert-body">
              <h5 style={{ color: 'var(--color-text-secondary)' }}>Fuera de Ventana de Interés IA</h5>
              <p>
                Este corte se encuentra fuera de la ventana de inferencia principal de la IA. Utilícelo para contextualizar la anatomía general de la rodilla.
              </p>
            </div>
          </div>
        ) : (
          <div className={`pacs-clinical-alert ${activeSlice.slice_probability >= threshold ? 'alert-tear' : 'alert-healthy'}`}>
            <div className="alert-icon">{activeSlice.slice_probability >= threshold ? '⚠️' : '✓'}</div>
            <div className="alert-body">
              <h5>{activeSlice.slice_probability >= threshold ? 'Hallazgo Positivo' : 'Sin Hallazgos'}</h5>
              <p>
                {activeSlice.slice_probability >= threshold 
                  ? `El modelo indica sospecha clínica de rotura con una probabilidad del ${Math.round(activeSlice.slice_probability * 100)}% en este corte.` 
                  : 'La morfología pormenorizada de este corte sugiere que las fibras del LCA están intactas.'}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Thumbnails strip */}
      <div className="pacs-thumbnails-container">
        <div className="pacs-thumbnails-strip">
          {slices.map((slice, idx) => {
            const isRankedCrit = hasPred && idx === rankedIndices[0];
            const isRef = !slice.has_prediction;
            return (
              <div 
                key={slice.slice_index} 
                className={`pacs-thumb-item ${idx === safeActiveIdx ? 'active' : ''} ${isRankedCrit ? 'crit-attention' : ''} ${isRef ? 'is-reference' : ''}`}
                onClick={() => setActiveIdx(idx)}
              >
                <img src={`data:image/jpeg;base64,${slice.raw_image}`} alt={`Thumbnail ${slice.slice_index}`} />
                <div className="pacs-thumb-badge">#{slice.slice_index}</div>
                {isRef && <div className="pacs-thumb-ref-dot"></div>}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// Live medical logs console
function LiveMedicalConsole({ logs }) {
  const consoleEndRef = useRef(null);

  useEffect(() => {
    if (consoleEndRef.current) {
      consoleEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  return (
    <div className="medical-console-card">
      <div className="console-terminal-header">
        <div className="console-dots">
          <span className="dot-red"></span>
          <span className="dot-yellow"></span>
          <span className="dot-green"></span>
        </div>
        <span className="console-title">CONSOLA CLÍNICA - ANÁLISIS DE RED NEURONAL EN TIEMPO REAL</span>
        <span className="console-badge">GPU ACTIVA</span>
      </div>
      <div className="console-body">
        {logs.length === 0 ? (
          <div className="console-line system-line">
            <span className="console-time">[00:00:00]</span>
            <span className="console-arrow">❯</span>
            <span className="console-msg">Iniciando pipeline de diagnóstico... Esperando datos del servidor...</span>
          </div>
        ) : (
          logs.map((log, idx) => (
            <div key={idx} className={`console-line ${log.message.includes('✓') || log.message.includes('éxito') ? 'success-line' : log.message.startsWith('[INFO]') ? 'info-line' : 'standard-line'}`}>
              <span className="console-time">[{log.time}]</span>
              <span className="console-arrow">❯</span>
              <span className="console-msg">{log.message}</span>
            </div>
          ))
        )}
        <div ref={consoleEndRef} />
      </div>
    </div>
  );
}

export default function App() {
  // Application State
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, uploading, analyzing, completed, error
  const [errorMsg, setErrorMsg] = useState('');
  const [progressStep, setProgressStep] = useState(0);
  const [selectedModel, setSelectedModel] = useState('swin'); // swin, vit or cnn
  
  // Real-time incremental backend data streams
  const [patientInfo, setPatientInfo] = useState(null);
  const [planes, setPlanes] = useState({});
  const [diagnosis, setDiagnosis] = useState(null);

  // PACS navigation/switching state (hoisted)
  const [activePlaneTab, setActivePlaneTab] = useState('sagittal');
  const [activeSliceIdxs, setActiveSliceIdxs] = useState({ sagittal: 0, coronal: 0, axial: 0 });
  const [topViewerMode, setTopViewerMode] = useState('critical'); // Default to critical slices view

  const [prevPlaneTab, setPrevPlaneTab] = useState('sagittal');
  const [slideDirection, setSlideDirection] = useState('next');

  // Track activePlaneTab shifts to calculate sliding animation directions automatically
  useEffect(() => {
    const planesOrder = ['sagittal', 'coronal', 'axial'];
    const prevIdx = planesOrder.indexOf(prevPlaneTab);
    const currIdx = planesOrder.indexOf(activePlaneTab);
    
    if (currIdx !== prevIdx) {
      // Forward/Next direction: sagittal(0) -> coronal(1) -> axial(2) -> sagittal(0)
      const isNext = (currIdx > prevIdx && !(prevIdx === 0 && currIdx === 2)) || (prevIdx === 2 && currIdx === 0);
      setSlideDirection(isNext ? 'next' : 'prev');
      setPrevPlaneTab(activePlaneTab);
    }
  }, [activePlaneTab, prevPlaneTab]);

  // Real-time progress log streaming
  const [consoleLogs, setConsoleLogs] = useState([]);
  const [consoleExpanded, setConsoleExpanded] = useState(true);

  // Auto-collapse console when diagnosis finishes
  useEffect(() => {
    if (diagnosis) {
      setConsoleExpanded(false);
    }
  }, [diagnosis]);

  // Real-time upload progress tracking
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadSpeed, setUploadSpeed] = useState(0);
  const [uploadBytes, setUploadBytes] = useState({ loaded: 0, total: 0 });
  
  // Interactive Dashboard Customizations
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.35);
  const [maskPatientData, setMaskPatientData] = useState(true);

  // Vista comparativa de los 3 modelos + casos de ejemplo
  const [examples, setExamples] = useState([]);
  const [compareOpen, setCompareOpen] = useState(false);
  const [compareStatus, setCompareStatus] = useState('idle'); // idle, running, done, error
  const [compareModels, setCompareModels] = useState({});
  const [compareConsensus, setCompareConsensus] = useState(null);
  const [comparePatient, setComparePatient] = useState(null);
  const [compareLabel, setCompareLabel] = useState('');
  const [compareError, setCompareError] = useState('');
  const [currentStudy, setCurrentStudy] = useState(null); // {safeName, label}
  const [drawerOpen, setDrawerOpen] = useState(true); // cajón de casos abierto/cerrado

  useEffect(() => {
    fetch('/api/examples').then((r) => r.json()).then((d) => setExamples(d.examples || [])).catch(() => {});
  }, []);

  const runComparison = (safeName, label, asModal = true) => {
    if (asModal) setCompareOpen(true);
    setCompareStatus('running');
    setCompareModels({});
    setCompareConsensus(null);
    setComparePatient(null);
    setCompareError('');
    setCompareLabel(label || safeName);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/compare-existing');
    xhr.setRequestHeader('Content-Type', 'application/json');
    let lastLength = 0;
    let buffer = '';
    xhr.onreadystatechange = () => {
      if (xhr.readyState === 3 || xhr.readyState === 4) {
        const txt = xhr.responseText;
        buffer += txt.substring(lastLength);
        lastLength = txt.length;
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.trim()) continue;
          let d;
          try { d = JSON.parse(line); } catch (e) { continue; }
          if (d.error) { setCompareError(d.error); setCompareStatus('error'); xhr.abort(); return; }
          if (d.type === 'patient_info') setComparePatient(d.patient_info);
          else if (d.type === 'model_result') setCompareModels((prev) => ({ ...prev, [d.model]: d.result }));
          else if (d.type === 'comparison') { setCompareConsensus(d.consensus); setComparePatient(d.patient_info); }
        }
      }
      if (xhr.readyState === 4) {
        if (xhr.status >= 200 && xhr.status < 300) setCompareStatus((s) => (s === 'error' ? s : 'done'));
        else if (xhr.status !== 0) { setCompareError('Fallo en la comparativa de modelos.'); setCompareStatus('error'); }
      }
    };
    xhr.onerror = () => { setCompareError('Error de red al conectar con el servidor.'); setCompareStatus('error'); };
    xhr.send(JSON.stringify({ safeName }));
  };

  // En modo "Diagnóstico múltiple": al terminar el análisis detallado (visor), lanza
  // la comparativa de los 3 modelos y la muestra inline (sin modal).
  useEffect(() => {
    if (status === 'completed' && selectedModel === 'multi' && currentStudy?.safeName) {
      runComparison(currentStudy.safeName, currentStudy.label, false);
    }
  }, [status, selectedModel, currentStudy]);

  // Carga un caso del cajón (clic o arrastre al centro) y lo analiza
  const loadExample = (safeName, label) => {
    setCurrentStudy({ safeName, label: label || safeName });
    analyzeExisting(safeName, { name: safeName, size: 0 });
  };

  // Theme state: default to 'dark' (negro por defecto)
  const [theme, setTheme] = useState('dark');

  // Synchronize theme with body element class
  useEffect(() => {
    if (theme === 'light') {
      document.body.classList.add('light-theme');
    } else {
      document.body.classList.remove('light-theme');
    }
  }, [theme]);

  // Smooth, continuous progress animation state for inference
  const [smoothProgress, setSmoothProgress] = useState(0);

  useEffect(() => {
    if (status !== 'analyzing') {
      if (status === 'completed') {
        setSmoothProgress(100);
      } else {
        setSmoothProgress(0);
      }
      return;
    }

    // Define target range mapping for each sequential progress step
    const stepTargets = {
      0: { start: 5, max: 18 },    // Reading & Loading DICOM
      1: { start: 20, max: 35 },   // Extracting patient metadata
      2: { start: 40, max: 55 },   // Sagittal inference
      3: { start: 60, max: 75 },   // Coronal inference
      4: { start: 80, max: 90 },   // Axial inference
      5: { start: 92, max: 98 },   // Consolidation & Calibration
    };

    const targetRange = stepTargets[progressStep] || { start: 5, max: 98 };
    
    // Smoothly kick off the step start if not already there
    setSmoothProgress((prev) => (prev < targetRange.start ? targetRange.start : prev));

    // Crawl forward at a tiny pace to make the loader feel active and continuous
    const interval = setInterval(() => {
      setSmoothProgress((prev) => {
        if (prev < targetRange.max) {
          return Math.min(prev + 0.35, targetRange.max);
        }
        return prev;
      });
    }, 150);

    return () => clearInterval(interval);
  }, [status, progressStep]);

  // Locked Clinical Threshold constants per model
  const THRESHOLDS = {
    swin: 0.3734,
    vit: 0.3804,
    cnn: 0.5455
  };
  const THRESHOLD = THRESHOLDS[selectedModel] || 0.3734;

  const fileInputRef = useRef(null);
  const pacsViewerRef = useRef(null);

  // Clinical steps tracking matching the progressive calculations
  const pipelineSteps = [
    { title: 'Lectura y Carga DICOM', desc: 'Validando archivo comprimido ZIP y comprobando firmas de cabecera.' },
    { title: 'Ficha de Metadatos', desc: 'Mapeando registros del paciente a través de tags de metadatos DICOM.' },
    { title: 'Inferencia de Plano Sagital', desc: `Evaluando cortes sagitales clave usando modelos ${selectedModel === 'swin' ? 'Swin Transformer' : selectedModel === 'vit' ? 'ViT Transformer' : 'ResNet50 Deep Ensemble'} de atención local.` },
    { title: 'Inferencia de Plano Coronal', desc: `Evaluando cortes coronales críticos con ${selectedModel === 'swin' ? 'Swin Transformer' : selectedModel === 'vit' ? 'ViT Transformer' : 'ResNet50 Deep Ensemble'}.` },
    { title: 'Inferencia de Plano Axial', desc: `Evaluando cortes axiales consecutivos enfocados en la inserción del LCA con ${selectedModel === 'swin' ? 'Swin Transformer' : selectedModel === 'vit' ? 'ViT Transformer' : 'ResNet50 Deep Ensemble'}.` },
    { title: 'Consolidación Médica', desc: 'Calculando predicción ensemble consolidada y calibrando confianza diagnóstica.' }
  ];

  // Handle Drag & Drop events
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    // Caso arrastrado desde el cajón de la derecha
    const caseName = e.dataTransfer.getData('application/x-acl-case');
    if (caseName) {
      const label = e.dataTransfer.getData('text/plain') || caseName;
      loadExample(caseName, label);
      return;
    }

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.endsWith('.zip')) {
        setFile(droppedFile);
        checkAndAnalyze(droppedFile);
      } else {
        alert('Formato no soportado. Por favor, sube un archivo ZIP conteniendo series DICOM.');
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.name.endsWith('.zip')) {
        setFile(selectedFile);
        checkAndAnalyze(selectedFile);
      } else {
        alert('Formato no soportado. Por favor, sube un archivo ZIP conteniendo series DICOM.');
      }
    }
  };

  const onButtonClick = () => {
    fileInputRef.current.click();
  };

  // Pre-flight file check to see if we can bypass upload phase
  const checkAndAnalyze = (selectedFile) => {
    setStatus('checking');
    setErrorMsg('');

    fetch(`/api/check-file?filename=${encodeURIComponent(selectedFile.name)}`)
      .then((res) => {
        if (!res.ok) throw new Error('Check request failed');
        return res.json();
      })
      .then((data) => {
        if (data.exists) {
          analyzeExisting(data.safeName, selectedFile);
        } else {
          uploadAndAnalyze(selectedFile);
        }
      })
      .catch((err) => {
        console.warn('[BFF Client] Pre-flight file check failed, falling back to upload:', err);
        uploadAndAnalyze(selectedFile);
      });
  };

  // Run diagnostics on an already uploaded zip file (bypass upload UI completely)
  const analyzeExisting = (safeName, originalFile) => {
    setCurrentStudy({ safeName, label: (originalFile && originalFile.name) || safeName });
    setStatus('analyzing');
    setErrorMsg('');
    setPatientInfo(null);
    setPlanes({});
    setDiagnosis(null);
    setProgressStep(0);
    setUploadProgress(100);
    setUploadSpeed(0);
    setUploadBytes({ loaded: originalFile.size, total: originalFile.size });
    setConsoleLogs([]);
    setConsoleExpanded(true);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/analyze-existing');
    xhr.setRequestHeader('Content-Type', 'application/json');

    let lastLength = 0;
    let buffer = '';

    const handleChunks = () => {
      if (xhr.readyState === 3 || xhr.readyState === 4) {
        const currentText = xhr.responseText;
        const newChunk = currentText.substring(lastLength);
        lastLength = currentText.length;

        buffer += newChunk;
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep partial line in buffer

        for (const line of lines) {
          if (!line.trim()) continue;
          let data;
          try {
            data = JSON.parse(line);
          } catch (e) {
            console.error('Error parsing line:', e, line);
            continue;
          }

          if (data.error || data.success === false) {
            setErrorMsg(data.error || 'Error de procesamiento en el servidor.');
            setStatus('error');
            xhr.abort();
            return;
          }

          if (data.type === 'progress') {
            const timestamp = new Date().toLocaleTimeString();
            setConsoleLogs((prev) => [
              ...prev,
              { time: timestamp, message: data.message, step: data.step_index }
            ]);
            if (data.step_index !== undefined && data.step_index !== null) {
              setProgressStep(data.step_index);
            }
          } else if (data.type === 'patient_info') {
            setPatientInfo(data.patient_info);
            setProgressStep(1);
          } else if (data.type === 'plane_result') {
            setPlanes((prev) => ({
              ...prev,
              [data.plane]: {
                average_probability: data.average_probability,
                slices: data.slices
              }
            }));

            // Pre-calculate active slice for the newly received plane using the highest Swin attention predicted slice
            let maxAtt = -1;
            let maxIdx = 0;
            let foundPredictive = false;
            data.slices.forEach((s, idx) => {
              if (s.has_prediction && s.swin_attention > maxAtt) {
                maxAtt = s.swin_attention;
                maxIdx = idx;
                foundPredictive = true;
              }
            });
            if (!foundPredictive) {
              maxIdx = Math.floor(data.slices.length / 2);
            }
            setActiveSliceIdxs((prev) => ({
              ...prev,
              [data.plane]: maxIdx
            }));

            // Auto-switch to the newly completed plane
            setActivePlaneTab(data.plane);

            if (data.plane === 'sagittal') setProgressStep(2);
            else if (data.plane === 'coronal') setProgressStep(3);
            else if (data.plane === 'axial') setProgressStep(4);
          } else if (data.type === 'diagnosis') {
            setDiagnosis(data.diagnosis);
            setProgressStep(5);
          }
        }
      }

      if (xhr.readyState === 4) {
        if (xhr.status >= 200 && xhr.status < 300) {
          setStatus('completed');
          setProgressStep(6);
        } else {
          if (xhr.status !== 0) {
            setErrorMsg('Fallo en la inferencia del pipeline de diagnóstico.');
            setStatus('error');
          }
        }
      }
    };

    xhr.onreadystatechange = handleChunks;

    xhr.onerror = () => {
      setErrorMsg('Error de red al conectar con el servidor.');
      setStatus('error');
    };

    xhr.send(JSON.stringify({ safeName, model: selectedModel }));
  };

  // Perform upload and parse real-time NDJSON stream chunk-by-chunk using XMLHttpRequest
  const uploadAndAnalyze = (uploadFile) => {
    setStatus('uploading');
    setErrorMsg('');
    setPatientInfo(null);
    setPlanes({});
    setDiagnosis(null);
    setProgressStep(0);
    setUploadProgress(0);
    setUploadSpeed(0);
    setUploadBytes({ loaded: 0, total: uploadFile.size });
    setConsoleLogs([]);
    setConsoleExpanded(true);

    const formData = new FormData();
    formData.append('dicomZip', uploadFile);
    formData.append('model', selectedModel);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/analyze');

    const startTime = Date.now();

    // Track upload progress and speed
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        const percent = Math.round((e.loaded / e.total) * 100);
        const elapsed = (Date.now() - startTime) / 1000; // in seconds
        const speed = elapsed > 0 ? (e.loaded / elapsed) : 0; // bytes per second
        setUploadProgress(percent);
        setUploadSpeed(speed);
        setUploadBytes({ loaded: e.loaded, total: e.total });
      }
    });

    xhr.upload.addEventListener('load', () => {
      setStatus('analyzing');
    });

    let lastLength = 0;
    let buffer = '';

    // Handle stream parsing from the response body in state 3 (LOADING) or 4 (DONE)
    const handleChunks = () => {
      if (xhr.readyState === 3 || xhr.readyState === 4) {
        const currentText = xhr.responseText;
        const newChunk = currentText.substring(lastLength);
        lastLength = currentText.length;

        buffer += newChunk;
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep partial line in buffer

        for (const line of lines) {
          if (!line.trim()) continue;
          let data;
          try {
            data = JSON.parse(line);
          } catch (e) {
            console.error('Error parsing line:', e, line);
            continue;
          }

          if (data.error || data.success === false) {
            setErrorMsg(data.error || 'Error de procesamiento en el servidor.');
            setStatus('error');
            xhr.abort();
            return;
          }

          if (data.type === 'progress') {
            const timestamp = new Date().toLocaleTimeString();
            setConsoleLogs((prev) => [
              ...prev,
              { time: timestamp, message: data.message, step: data.step_index }
            ]);
            if (data.step_index !== undefined && data.step_index !== null) {
              setProgressStep(data.step_index);
            }
          } else if (data.type === 'patient_info') {
            setPatientInfo(data.patient_info);
            setProgressStep(1);
          } else if (data.type === 'plane_result') {
            setPlanes((prev) => ({
              ...prev,
              [data.plane]: {
                average_probability: data.average_probability,
                slices: data.slices
              }
            }));

            // Pre-calculate active slice for the newly received plane using the highest Swin attention predicted slice
            let maxAtt = -1;
            let maxIdx = 0;
            let foundPredictive = false;
            data.slices.forEach((s, idx) => {
              if (s.has_prediction && s.swin_attention > maxAtt) {
                maxAtt = s.swin_attention;
                maxIdx = idx;
                foundPredictive = true;
              }
            });
            if (!foundPredictive) {
              maxIdx = Math.floor(data.slices.length / 2);
            }
            setActiveSliceIdxs((prev) => ({
              ...prev,
              [data.plane]: maxIdx
            }));

            // Auto-switch to the newly completed plane
            setActivePlaneTab(data.plane);

            if (data.plane === 'sagittal') setProgressStep(2);
            else if (data.plane === 'coronal') setProgressStep(3);
            else if (data.plane === 'axial') setProgressStep(4);
          } else if (data.type === 'diagnosis') {
            setDiagnosis(data.diagnosis);
            setProgressStep(5);
          }
        }
      }

      if (xhr.readyState === 4) {
        if (xhr.status >= 200 && xhr.status < 300) {
          setStatus('completed');
          setProgressStep(6);
        } else {
          // If response finished but status is not successful and we didn't already trigger an error
          if (xhr.status !== 0) {
            setErrorMsg('Fallo en la inferencia del pipeline de diagnóstico.');
            setStatus('error');
          }
        }
      }
    };

    xhr.onreadystatechange = handleChunks;
    
    xhr.onerror = () => {
      setErrorMsg('Error de red al conectar con el servidor.');
      setStatus('error');
    };

    xhr.send(formData);
  };

  const resetDiagnostic = () => {
    setFile(null);
    setPatientInfo(null);
    setPlanes({});
    setDiagnosis(null);
    setStatus('idle');
    setErrorMsg('');
    setProgressStep(0);
    setConsoleLogs([]);
    setConsoleExpanded(true);
    setActivePlaneTab('sagittal');
    setActiveSliceIdxs({ sagittal: 0, coronal: 0, axial: 0 });
    setTopViewerMode('critical');
  };

  // Utility to mask patient metadata dynamically (GDPR/HIPAA compliant)
  const maskString = (str, isMasked) => {
    if (!str) return 'N/A';
    if (!isMasked) return str;
    return str.split(' ').map(word => {
      if (word.length <= 1) return word;
      return word[0] + '*'.repeat(Math.max(3, word.length - 1));
    }).join(' ');
  };

  // Calculate diagnostic outcome using locked threshold 0.3734
  const getDynamicDiagnosis = () => {
    if (!diagnosis) return null;
    const ensembleProb = diagnosis.ensemble_probability;
    const isTear = ensembleProb >= THRESHOLD;

    let calibratedConf = 0;
    if (isTear) {
      calibratedConf = 0.5 + 0.5 * (ensembleProb - THRESHOLD) / (1.0 - THRESHOLD + 1e-8);
    } else {
      // Para casos negativos (no está roto), invertir la confianza para ser intuitivo
      calibratedConf = 0.5 * ensembleProb / (THRESHOLD + 1e-8);
      calibratedConf = 1 - calibratedConf; // Invertir: mostrar confianza de que está SANO
    }

    return {
      prediction: isTear ? 'ROTURA DE LCA' : 'RODILLA SANA',
      isTear: isTear,
      ensemble_probability: ensembleProb,
      calibrated_confidence: calibratedConf,
      confidence_level: calibratedConf >= 0.75 ? 'alta' : calibratedConf >= 0.60 ? 'moderada' : 'débil'
    };
  };

  const dynamicDiag = getDynamicDiagnosis();

  const showPacsViewer = () => {
    setTopViewerMode('pacs');
    setTimeout(() => {
      pacsViewerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 60);
  };

  const openSliceInPacs = (plane, sliceIdx) => {
    setActivePlaneTab(plane);
    setActiveSliceIdxs(prev => ({ ...prev, [plane]: sliceIdx }));
    setTopViewerMode('pacs');
    setTimeout(() => {
      pacsViewerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 60);
  };

  return (
    <div className="app-container">
      {/* Cajón deslizante de casos (arrastra un caso al centro para analizarlo) */}
      {examples.length > 0 && (
        <aside className={`cases-drawer ${drawerOpen ? 'open' : 'closed'}`}>
          <button
            className="cases-drawer-handle"
            onClick={() => setDrawerOpen((o) => !o)}
            aria-label={drawerOpen ? 'Cerrar cajón de casos' : 'Abrir cajón de casos'}
          >
            <span className="cases-drawer-handle-text">CASOS</span>
            <span className="cases-drawer-handle-arrow">{drawerOpen ? '›' : '‹'}</span>
          </button>
          <div className="cases-drawer-body">
            <div className="cases-drawer-title">Casos de ejemplo</div>
            <p className="cases-drawer-hint">Arrastra un caso al centro o púlsalo para analizarlo.</p>
            {examples.map((ex) => (
              <div
                key={ex.safeName}
                className="example-card"
                role="button"
                tabIndex={0}
                draggable
                onDragStart={(e) => {
                  e.dataTransfer.setData('application/x-acl-case', ex.safeName);
                  e.dataTransfer.setData('text/plain', ex.label);
                  e.dataTransfer.effectAllowed = 'copy';
                }}
                onClick={() => loadExample(ex.safeName, ex.label)}
                onKeyDown={(e) => { if (e.key === 'Enter') loadExample(ex.safeName, ex.label); }}
              >
                <span className="example-card-mark" aria-hidden="true">RM</span>
                <span className="example-card-text">
                  <span className="example-card-label">{ex.label}</span>
                  <span className="example-card-sub">{selectedModel === 'multi' ? 'Diagnóstico múltiple' : 'Analizar'}</span>
                </span>
              </div>
            ))}
          </div>
        </aside>
      )}

      {/* Modal: comparativa de los 3 modelos */}
      {compareOpen && (
        <div className="compare-overlay" onClick={() => setCompareOpen(false)}>
          <div className="compare-modal" onClick={(e) => e.stopPropagation()}>
            <div className="compare-head">
              <div>
                <h2>Comparativa de arquitecturas</h2>
                <p>{compareLabel}{comparePatient && !maskPatientData ? ` · ${comparePatient.patient_name}` : ''}</p>
              </div>
              <button className="compare-close" onClick={() => setCompareOpen(false)} aria-label="Cerrar">×</button>
            </div>

            {compareError && <div className="compare-error">{compareError}</div>}

            <div className="compare-grid">
              {['cnn', 'vit', 'swin'].map((key) => {
                const r = compareModels[key];
                const names = { cnn: 'ResNet50', vit: 'ViT-Small', swin: 'Swin-Tiny' };
                return (
                  <div key={key} className={`compare-card ${r ? (r.prediction === 'ACL INJURY' ? 'is-injury' : 'is-healthy') : 'is-loading'}`}>
                    <div className="compare-card-name">{names[key]}</div>
                    {!r ? (
                      <div className="compare-card-loading">Evaluando…</div>
                    ) : (
                      <>
                        <div className="compare-verdict">{r.prediction === 'ACL INJURY' ? 'ROTURA LCA' : 'SANO'}</div>
                        <div className="compare-prob">{Math.round((r.prediction === 'ACL INJURY' ? r.calibrated_confidence : 1 - r.calibrated_confidence) * 100)}%<span>confianza {r.prediction === 'ACL INJURY' ? 'de rotura' : 'de sano'}</span></div>
                        <div className="compare-planes">
                          {['sagittal', 'coronal', 'axial'].map((p) => (
                            <div key={p} className="compare-plane-row">
                              <span className="compare-plane-name">{p === 'sagittal' ? 'Sagital' : p === 'coronal' ? 'Coronal' : 'Axial'}</span>
                              <div className="compare-bar"><div className="compare-bar-fill" style={{ width: `${Math.round((r.planes[p] || 0) * 100)}%` }} /></div>
                              <span className="compare-plane-val">{Math.round((r.planes[p] || 0) * 100)}%</span>
                            </div>
                          ))}
                        </div>
                        <div className="compare-thr">umbral clínico {Math.round(r.threshold * 100)}%</div>
                      </>
                    )}
                  </div>
                );
              })}
            </div>

            {compareConsensus && (
              <div className={`compare-consensus ${compareConsensus.prediction === 'ACL INJURY' ? 'is-injury' : 'is-healthy'}`}>
                Consenso: <strong>{compareConsensus.prediction === 'ACL INJURY' ? 'ROTURA DE LCA' : 'RODILLA SANA'}</strong>
                <span> · {compareConsensus.votes_injury}/{compareConsensus.n_models} modelos indican rotura</span>
              </div>
            )}
            {compareStatus === 'running' && <div className="compare-running">Procesando los tres modelos…</div>}
          </div>
        </div>
      )}

      {/* Header section with branding */}
      <header className="app-header">
        <div className="brand-section">
          <span className="brand-mark" aria-hidden="true">{selectedModel === 'swin' ? 'KS' : selectedModel === 'vit' ? 'KV' : 'KC'}</span>
          <div className="brand-title">
            <h1>{selectedModel === 'swin' ? 'KneeSwin-Net' : selectedModel === 'vit' ? 'KneeViT-Net' : 'KneeCNN-Net'}</h1>
            <p>Knee MRI Deep Transformer Diagnostic Suite</p>
          </div>
        </div>
        <div className="system-status" style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          <button 
            className="theme-toggle-btn"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            title="Cambiar Tema de Interfaz"
          >
            {theme === 'dark' ? 'Modo Claro' : 'Modo Oscuro'}
          </button>
          <span style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>
            Estado del Sistema: <strong style={{ color: 'var(--neon-teal)' }}>Activo (GPU CUDA)</strong>
          </span>
        </div>
      </header>

      {/* Landing page state - Upload Area with Selector */}
      {status === 'idle' && (
        <div className="landing-upload-wrapper">
          <div 
            className={`dropzone-container ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            onClick={onButtonClick}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              className="file-input" 
              style={{ display: 'none' }} 
              accept=".zip" 
              onChange={handleFileChange}
            />
            <div className="dropzone-icon">📁</div>
            <div className="dropzone-text">
              <h3>Arrastra y suelta la serie DICOM de rodilla (.zip)</h3>
              <p>Soporta archivos comprimidos exportados directamente de sistemas PACS / Weasis</p>
            </div>
            <button className="upload-button" onClick={(e) => { e.stopPropagation(); onButtonClick(); }}>
              Seleccionar Archivo
            </button>
          </div>

          {/* Model Selector Card Container */}
          <div className="model-selector-container" onClick={(e) => e.stopPropagation()}>
            <div className="model-selector-title">
              <span className="model-icon">⚙️</span>
              <h4>ARQUITECTURA DE IA DIAGNÓSTICA</h4>
            </div>
            <div className="model-options-grid">
              <div 
                className={`model-option-card ${selectedModel === 'swin' ? 'active' : ''}`}
                onClick={() => setSelectedModel('swin')}
              >
                <div className="model-option-header">
                  <span className="model-badge swin-badge">Swin-Net</span>
                  <span className="model-status-indicator">Ensemble</span>
                </div>
                <h5>KneeSwin-Net IA</h5>
                <p>Swin Transformer Tiny (27.8M parámetros) entrenado con ensamble multi-semilla y mapas de saliencia. Mayor sensibilidad del estudio.</p>
                <div className="model-option-stats">
                  <span>AUC test: 0.954</span>
                  <span>Sensibilidad: 90.7%</span>
                </div>
              </div>

              <div 
                className={`model-option-card ${selectedModel === 'vit' ? 'active' : ''}`}
                onClick={() => setSelectedModel('vit')}
              >
                <div className="model-option-header">
                  <span className="model-badge vit-badge">ViT-Net</span>
                  <span className="model-status-indicator">Self-Attention</span>
                </div>
                <h5>KneeViT-Net IA</h5>
                <p>Vision Transformer Small (22M parámetros) con mapas nativos de auto-atención espacial. Supera a ViT-Base (86M) con 4x menos parámetros.</p>
                <div className="model-option-stats">
                  <span>AUC test: 0.943</span>
                  <span>Sensibilidad: 88.4%</span>
                </div>
              </div>

              <div 
                className={`model-option-card ${selectedModel === 'cnn' ? 'active' : ''}`}
                onClick={() => setSelectedModel('cnn')}
              >
                <div className="model-option-header">
                  <span className="model-badge cnn-badge">CNN-Net</span>
                  <span className="model-status-indicator">ResNet50 Ensemble</span>
                </div>
                <h5>KneeCNN-Net IA</h5>
                <p>ResNet50 (23.5M parámetros) con ensamble multi-semilla, dropout por plano y pooling adaptativo. Mayor AUC del estudio.</p>
                <div className="model-option-stats">
                  <span>AUC test: 0.955</span>
                  <span>Sensibilidad: 88.4%</span>
                </div>
              </div>

              <div
                className={`model-option-card multi-option-card ${selectedModel === 'multi' ? 'active' : ''}`}
                onClick={() => setSelectedModel('multi')}
              >
                <div className="model-option-header">
                  <span className="model-badge multi-badge">Multi-Net</span>
                  <span className="model-status-indicator">3 modelos</span>
                </div>
                <h5>Diagnóstico múltiple</h5>
                <p>Ejecuta ResNet50, ViT-Small y Swin-Tiny sobre el mismo estudio y muestra los tres veredictos a la vez, junto al visor. Vista comparativa de arquitecturas.</p>
                <div className="model-option-stats">
                  <span>ResNet · ViT · Swin</span>
                  <span>Consenso</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Uploading & Analyzing Phase Screen (Unified Cinematic Hologram) */}
      {(status === 'uploading' || status === 'analyzing') && (
        <div className="inference-loading-screen-focused">
          <div className="focused-mri-container widescreen-hud">
            
            {/* LEFT HUD PANEL: DICOM and Patient Information */}
            <div className="hud-side-panel hud-left-panel">
              <div className="hud-panel-header">
                <span className="hud-icon">📋</span>
                <h4>INFORMACIÓN DE LA SERIE</h4>
              </div>
              <div className="hud-divider"></div>
              
              <div className="hud-section">
                <div className="hud-metric-row">
                  <span className="hud-label">Paciente:</span>
                  <span className="hud-value font-mono">
                    {patientInfo?.name ? (maskPatientData ? "PACIENTE_RODILLA_#9482" : patientInfo.name) : "ANALIZANDO CABECERA..."}
                  </span>
                </div>
                <div className="hud-metric-row">
                  <span className="hud-label">ID Estudio:</span>
                  <span className="hud-value font-mono text-cyan">
                    {patientInfo?.id ? (maskPatientData ? "STUDY_84920_MR" : patientInfo.id) : "EXTRAYENDO..."}
                  </span>
                </div>
                <div className="hud-metric-row">
                  <span className="hud-label">Equipo MRI:</span>
                  <span className="hud-value">Siemens Magnetom 3.0T</span>
                </div>
                <div className="hud-metric-row">
                  <span className="hud-label">Secuencia:</span>
                  <span className="hud-value font-mono text-cyan">T2-weighted TSE</span>
                </div>
              </div>
              
              <div className="hud-divider"></div>
              
              <div className="hud-section">
                <h5>ESTADÍSTICAS DE CARGA</h5>
                <div className="hud-metric-row">
                  <span className="hud-label">Serie DICOM:</span>
                  <span className="hud-value font-mono">45 Cortes (Slices)</span>
                </div>
                <div className="hud-metric-row">
                  <span className="hud-label">Tamaño:</span>
                  <span className="hud-value font-mono">
                    {uploadBytes.total > 0 ? `${formatMB(uploadBytes.total)} MB` : "Calculando..."}
                  </span>
                </div>
                {status === 'uploading' && (
                  <>
                    <div className="hud-metric-row">
                      <span className="hud-label">Cargado:</span>
                      <span className="hud-value font-mono text-cyan">{uploadProgress}%</span>
                    </div>
                    <div className="hud-metric-row">
                      <span className="hud-label">Velocidad:</span>
                      <span className="hud-value font-mono">{formatSpeed(uploadSpeed)}</span>
                    </div>
                  </>
                )}
                {status === 'analyzing' && (
                  <div className="hud-metric-row">
                    <span className="hud-label">Carga en Red:</span>
                    <span className="hud-value text-emerald font-semibold">Completada (100%)</span>
                  </div>
                )}
              </div>

              <div className="hud-divider"></div>
              
              <div className="hud-section">
                <h5>PARÁMETROS DE ADQUISICIÓN</h5>
                <div className="hud-metric-row">
                  <span className="hud-label">Resolución:</span>
                  <span className="hud-value font-mono">512 × 512 px</span>
                </div>
                <div className="hud-metric-row">
                  <span className="hud-label">Espaciado Slice:</span>
                  <span className="hud-value font-mono">3.0 mm</span>
                </div>
                <div className="hud-metric-row">
                  <span className="hud-label">Campo de Visión:</span>
                  <span className="hud-value font-mono">160 × 160 mm</span>
                </div>
                <div className="hud-metric-row">
                  <span className="hud-label">Eco / Repetición:</span>
                  <span className="hud-value font-mono">TE 30ms / TR 3200ms</span>
                </div>
              </div>
            </div>

            {/* CENTER HUD STAGE: The 3D Bounding Box SVG with Repositioned Non-Overlapping Progress Bar */}
            <div className="hud-center-stage">
              <div className="mri-orthogonal-scanner-large">
                <div className="volume-3d-stage-wrapper">
                  <div className="volume-3d-stage">
                    <svg className="orthogonal-projection-svg" viewBox="0 0 200 160">
                      <defs>
                        <linearGradient id="sagittal-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                          <stop offset="0%" stopColor="var(--neon-purple)" stopOpacity="0.15" />
                          <stop offset="100%" stopColor="var(--neon-purple)" stopOpacity="0.6" />
                        </linearGradient>
                        <linearGradient id="coronal-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                          <stop offset="0%" stopColor="var(--neon-teal)" stopOpacity="0.15" />
                          <stop offset="100%" stopColor="var(--neon-teal)" stopOpacity="0.6" />
                        </linearGradient>
                        <linearGradient id="axial-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                          <stop offset="0%" stopColor="#00b4d8" stopOpacity="0.15" />
                          <stop offset="100%" stopColor="#00b4d8" stopOpacity="0.6" />
                        </linearGradient>
                        <filter id="glow-purple" x="-20%" y="-20%" width="140%" height="140%">
                          <feGaussianBlur stdDeviation="3" result="blur" />
                          <feComposite in="SourceGraphic" in2="blur" operator="over" />
                        </filter>
                        <filter id="glow-teal" x="-20%" y="-20%" width="140%" height="140%">
                          <feGaussianBlur stdDeviation="3" result="blur" />
                          <feComposite in="SourceGraphic" in2="blur" operator="over" />
                        </filter>
                        <filter id="glow-blue" x="-20%" y="-20%" width="140%" height="140%">
                          <feGaussianBlur stdDeviation="3" result="blur" />
                          <feComposite in="SourceGraphic" in2="blur" operator="over" />
                        </filter>
                      </defs>

                      {/* 3D Bounding Box Wireframe */}
                      <line x1="40" y1="120" x2="110" y2="150" className={`vol-wireframe ${status === 'analyzing' && progressStep >= 0 ? 'active' : ''}`} />
                      <line x1="110" y1="150" x2="170" y2="120" className={`vol-wireframe ${status === 'analyzing' && progressStep >= 0 ? 'active' : ''}`} />
                      <line x1="170" y1="120" x2="100" y2="90" className={`vol-wireframe ${status === 'analyzing' && progressStep >= 0 ? 'active' : ''}`} strokeDasharray="3 3" />
                      <line x1="100" y1="90" x2="40" y2="120" className={`vol-wireframe ${status === 'analyzing' && progressStep >= 0 ? 'active' : ''}`} strokeDasharray="3 3" />
                      
                      <line x1="40" y1="40" x2="110" y2="70" className={`vol-wireframe ${status === 'analyzing' && progressStep >= 0 ? 'active' : ''}`} />
                      <line x1="110" y1="70" x2="170" y2="40" className={`vol-wireframe ${status === 'analyzing' && progressStep >= 0 ? 'active' : ''}`} />
                      <line x1="170" y1="40" x2="100" y2="10" className={`vol-wireframe ${status === 'analyzing' && progressStep >= 0 ? 'active' : ''}`} />
                      <line x1="100" y1="10" x2="40" y2="40" className={`vol-wireframe ${status === 'analyzing' && progressStep >= 0 ? 'active' : ''}`} />

                      <line x1="40" y1="40" x2="40" y2="120" className={`vol-wireframe ${status === 'analyzing' && progressStep >= 0 ? 'active' : ''}`} />
                      <line x1="110" y1="70" x2="110" y2="150" className={`vol-wireframe ${status === 'analyzing' && progressStep >= 0 ? 'active' : ''}`} />
                      <line x1="170" y1="40" x2="170" y2="120" className={`vol-wireframe ${status === 'analyzing' && progressStep >= 0 ? 'active' : ''}`} />
                      <line x1="100" y1="10" x2="100" y2="90" className={`vol-wireframe ${status === 'analyzing' && progressStep >= 0 ? 'active' : ''}`} strokeDasharray="3 3" />

                      {/* Stylized Knee Joint Wireframe Silhouette */}
                      <g className="mri-knee-hologram">
                        {/* Femur (Fémur) */}
                        <path 
                          d="M 92,15 L 92,48 C 92,56 82,64 82,76 C 82,86 91,89 103,89 C 115,89 122,86 122,74 C 122,64 114,54 114,48 L 114,15 Z" 
                          className="mri-bone-hologram" 
                        />
                        {/* Tibia */}
                        <path 
                          d="M 80,95 Q 101,93 120,95 C 120,101 114,108 114,142 L 96,142 C 96,108 80,101 80,95 Z" 
                          className="mri-bone-hologram" 
                        />
                        {/* Fibula */}
                        <path 
                          d="M 118,103 C 122,103 123,106 123,109 C 123,116 119,122 119,142 L 114,142 C 114,122 115,116 115,109 C 115,106 116,103 118,103 Z" 
                          className="mri-bone-hologram" 
                        />
                        {/* Patella (Rótula) */}
                        <path 
                          d="M 70,52 C 64,54 62,62 62,69 C 62,75 68,77 71,75 C 74,72 74,56 71,52 Z" 
                          className="mri-bone-hologram" 
                        />
                        
                        {/* Patellar Ligament (Tendón Rotuliano) */}
                        <path 
                          d="M 71,75 L 80,95" 
                          className="mri-tendon-hologram" 
                        />

                        {/* Menisci (Cartílagos de amortiguación) */}
                        <path 
                          d="M 81,95 C 84,92 87,92 90,93 C 86,95 83,96 81,95 Z" 
                          className="mri-meniscus" 
                        />
                        <path 
                          d="M 110,94 C 113,92 116,92 119,95 C 117,96 114,95 110,94 Z" 
                          className="mri-meniscus" 
                        />

                        {/* Glowing Crossing Ligaments (LCA y LCP) */}
                        {/* LCP (Posterior Cruciate Ligament - crossing behind, dim) */}
                        <path 
                          d="M 92,76 L 111,94" 
                          className="mri-pcl-ligament" 
                        />

                        {/* LCA (Glowing ACL Ligament - in front, vibrant neon green) */}
                        <path 
                          d="M 108,74 L 89,94" 
                          className="mri-acl-ligament" 
                          stroke="var(--neon-teal)" 
                          strokeWidth="2.5" 
                          strokeLinecap="round" 
                          filter="url(#glow-teal)" 
                        />
                      </g>

                      {/* PLANES IN THREE DIMENSIONS */}
                      {status === 'analyzing' && progressStep >= 2 ? (
                        <g className={`mri-plane sagittal-plane ${progressStep === 2 ? 'scanning' : 'done'}`}>
                          <polygon 
                            points="75,25 75,105 135,135 135,55" 
                            fill="url(#sagittal-grad)" 
                            stroke="var(--neon-purple)" 
                            strokeWidth={progressStep === 2 ? '2' : '1'}
                            filter={progressStep === 2 ? 'url(#glow-purple)' : ''}
                          />
                        </g>
                      ) : status === 'analyzing' && progressStep === 1 ? (
                        <g className="mri-plane sagittal-plane queued">
                          <polygon points="75,25 75,105 135,135 135,55" fill="rgba(255, 255, 255, 0.01)" stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="2 2" />
                        </g>
                      ) : null}

                      {status === 'analyzing' && progressStep >= 3 ? (
                        <g className={`mri-plane coronal-plane ${progressStep === 3 ? 'scanning' : 'done'}`}>
                          <polygon 
                            points="140,25 140,105 80,135 80,55" 
                            fill="url(#coronal-grad)" 
                            stroke="var(--neon-teal)" 
                            strokeWidth={progressStep === 3 ? '2' : '1'}
                            filter={progressStep === 3 ? 'url(#glow-teal)' : ''}
                          />
                        </g>
                      ) : status === 'analyzing' && progressStep >= 1 ? (
                        <g className="mri-plane coronal-plane queued">
                          <polygon points="140,25 140,105 80,135 80,55" fill="rgba(255, 255, 255, 0.01)" stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="2 2" />
                        </g>
                      ) : null}

                      {status === 'analyzing' && progressStep >= 4 ? (
                        <g className={`mri-plane axial-plane ${progressStep === 4 ? 'scanning' : 'done'}`}>
                          <polygon 
                            points="40,80 110,110 170,80 100,50" 
                            fill="url(#axial-grad)" 
                            stroke="#00b4d8" 
                            strokeWidth={progressStep === 4 ? '2' : '1'}
                            filter={progressStep === 4 ? 'url(#glow-blue)' : ''}
                          />
                        </g>
                      ) : status === 'analyzing' && progressStep >= 1 ? (
                        <g className="mri-plane axial-plane queued">
                          <polygon points="40,80 110,110 170,80 100,50" fill="rgba(255, 255, 255, 0.01)" stroke="rgba(255, 255, 255, 0.08)" strokeDasharray="2 2" />
                        </g>
                      ) : null}
                      
                      {status === 'analyzing' && progressStep >= 5 && (
                        <g className={`mri-ensemble-target ${progressStep === 5 ? 'active' : 'done'}`}>
                          <circle cx="107" cy="80" r="10" className="target-core" />
                          <circle cx="107" cy="80" r="18" className="target-pulse" />
                        </g>
                      )}
                    </svg>
                  </div>

                  {/* Sleek clinical controls grouped outside the volume stage */}
                  <div className="scanner-meta-controls">
                    {/* Sleek clinical progress bar (repositioned out of the stage) */}
                    <div className="scanner-progress-bar-container">
                      <div 
                        className="scanner-progress-bar-fill" 
                        style={{ 
                          width: `${status === 'uploading' ? uploadProgress : Math.round((progressStep / 5) * 100)}%` 
                        }}
                      ></div>
                    </div>

                    {/* Dynamic medical scanning status */}
                    <div className="scanner-status-text">
                      <span className="scanner-status-pulse-dot"></span>
                      <span className="scanner-status-message font-mono">
                        {status === 'uploading' 
                          ? `SUBIENDO VOLUMEN Y SERIE DICOM (${uploadProgress}%)...` 
                          : getStatusMessage(progressStep)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* RIGHT HUD PANEL: Swin-Net IA Neural Layer Metrics */}
            <div className="hud-side-panel hud-right-panel">
              <div className="hud-panel-header">
                <span className="hud-icon">🧠</span>
                <h4>RED NEURONAL {selectedModel === 'swin' ? 'SWIN-NET IA' : selectedModel === 'vit' ? 'VIT-NET IA' : 'CNN-NET IA'}</h4>
              </div>
              <div className="hud-divider"></div>
              
              <div className="hud-section">
                <h5>{selectedModel === 'cnn' ? 'CAPAS DE PROCESAMIENTO' : 'CAPAS DE ATENCIÓN'}</h5>
                
                {/* Feature Extractor Layer */}
                <div className="hud-layer-bar">
                  <div className="hud-metric-row">
                    <span className="hud-label-sm">Backbone {selectedModel === 'swin' ? 'CNN-Swin' : selectedModel === 'vit' ? 'ViT-Base' : 'ResNet50'} (Features)</span>
                    <span className="hud-value font-mono text-xs">{status === 'uploading' ? '0%' : '98.4%'}</span>
                  </div>
                  <div className="hud-bar-visual">
                    <div className={`hud-bar-fill ${status === 'analyzing' ? 'anim-pulse' : ''}`} style={{ width: status === 'uploading' ? '15%' : '98.4%', background: 'var(--neon-teal)' }}></div>
                  </div>
                </div>

                {/* Sagittal Plane Layer */}
                <div className="hud-layer-bar">
                  <div className="hud-metric-row">
                    <span className="hud-label-sm">Filtro Plano Sagital (LCA / LCP)</span>
                    <span className="hud-value font-mono text-xs">
                      {status === 'uploading' ? '0%' : (progressStep > 2 ? '94.1%' : (progressStep === 2 ? 'Procesando...' : 'En Espera'))}
                    </span>
                  </div>
                  <div className="hud-bar-visual">
                    <div className={`hud-bar-fill ${progressStep === 2 ? 'anim-pulse-fast' : ''}`} style={{ 
                      width: status === 'uploading' ? '0%' : (progressStep > 2 ? '94.1%' : (progressStep === 2 ? '65%' : '0%')),
                      background: 'var(--neon-purple)' 
                    }}></div>
                  </div>
                </div>

                {/* Coronal Plane Layer */}
                <div className="hud-layer-bar">
                  <div className="hud-metric-row">
                    <span className="hud-label-sm">Filtro Plano Coronal (Meniscos)</span>
                    <span className="hud-value font-mono text-xs">
                      {status === 'uploading' ? '0%' : (progressStep > 3 ? '89.7%' : (progressStep === 3 ? 'Procesando...' : 'En Espera'))}
                    </span>
                  </div>
                  <div className="hud-bar-visual">
                    <div className={`hud-bar-fill ${progressStep === 3 ? 'anim-pulse-fast' : ''}`} style={{ 
                      width: status === 'uploading' ? '0%' : (progressStep > 3 ? '89.7%' : (progressStep === 3 ? '55%' : '0%')),
                      background: 'var(--neon-teal)' 
                    }}></div>
                  </div>
                </div>

                {/* Axial Plane Layer */}
                <div className="hud-layer-bar">
                  <div className="hud-metric-row">
                    <span className="hud-label-sm">Filtro Plano Axial (Hueso/Cartílago)</span>
                    <span className="hud-value font-mono text-xs">
                      {status === 'uploading' ? '0%' : (progressStep > 4 ? '92.3%' : (progressStep === 4 ? 'Procesando...' : 'En Espera'))}
                    </span>
                  </div>
                  <div className="hud-bar-visual">
                    <div className={`hud-bar-fill ${progressStep === 4 ? 'anim-pulse-fast' : ''}`} style={{ 
                      width: status === 'uploading' ? '0%' : (progressStep > 4 ? '92.3%' : (progressStep === 4 ? '45%' : '0%')),
                      background: '#00b4d8' 
                    }}></div>
                  </div>
                </div>

                {/* Ensemble Integration */}
                <div className="hud-layer-bar">
                  <div className="hud-metric-row">
                    <span className="hud-label-sm">Consolidación de Ensemble IA</span>
                    <span className="hud-value font-mono text-xs">
                      {status === 'uploading' ? '0%' : (progressStep >= 5 ? 'Consolidando...' : 'En Espera')}
                    </span>
                  </div>
                  <div className="hud-bar-visual">
                    <div className={`hud-bar-fill ${progressStep === 5 ? 'anim-pulse-fast' : ''}`} style={{ 
                      width: status === 'uploading' ? '0%' : (progressStep >= 5 ? '90%' : '0%'),
                      background: 'var(--neon-pink)' 
                    }}></div>
                  </div>
                </div>

              </div>

              <div className="hud-divider"></div>
              
              <div className="hud-section">
                <h5>MÉTRICAS DEL MODELO (test MRNet)</h5>
                <div className="hud-metric-row">
                  <span className="hud-label">AUC:</span>
                  <span className="hud-value font-mono text-emerald">{selectedModel === 'swin' ? '0.954' : selectedModel === 'vit' ? '0.943' : '0.955'}</span>
                </div>
                <div className="hud-metric-row">
                  <span className="hud-label">Sensibilidad (LCA):</span>
                  <span className="hud-value font-mono text-cyan">{selectedModel === 'swin' ? '90.7%' : selectedModel === 'vit' ? '88.4%' : '88.4%'}</span>
                </div>
                <div className="hud-metric-row">
                  <span className="hud-label">Especificidad:</span>
                  <span className="hud-value font-mono text-cyan">{selectedModel === 'swin' ? '84.0%' : selectedModel === 'vit' ? '86.1%' : '84.7%'}</span>
                </div>
                <div className="hud-metric-row">
                  <span className="hud-label">Parámetros:</span>
                  <span className="hud-value font-mono">{selectedModel === 'swin' ? '27.8 M' : selectedModel === 'vit' ? '22 M' : '23.5 M'}</span>
                </div>
              </div>
            </div>

          </div>
        </div>
      )}

      {/* Error state display */}
      {status === 'error' && (
        <div className="loading-panel" style={{ borderColor: 'var(--crimson)', boxShadow: 'var(--shadow-crimson)' }}>
          <div className="loading-header">
            <span style={{ fontSize: '3rem', display: 'block', marginBottom: '1rem' }}>⚠️</span>
            <h2 style={{ color: 'var(--crimson)' }}>Error de Diagnóstico</h2>
            <p style={{ marginTop: '0.8rem', color: 'var(--color-text-primary)', fontSize: '1rem' }}>{errorMsg}</p>
          </div>
          <div className="reset-zone" style={{ marginTop: '1rem' }}>
            <button className="secondary-button" onClick={resetDiagnostic}>
              Subir otro Archivo
            </button>
          </div>
        </div>
      )}

      {/* Completed Dashboard Workspace */}
      {status === 'completed' && (
        <div className="workspace-container-fluid fade-in-up">
          {/* Panel comparativo de los 3 modelos (modo Diagnóstico múltiple) */}
          {selectedModel === 'multi' && (
            <div className="multi-panel">
              <div className="multi-panel-head">
                <h3>Diagnóstico múltiple · comparativa de arquitecturas</h3>
                {compareStatus === 'running' && <span className="multi-panel-status">evaluando los 3 modelos…</span>}
              </div>
              <div className="compare-grid">
                {['cnn', 'vit', 'swin'].map((key) => {
                  const r = compareModels[key];
                  const names = { cnn: 'ResNet50', vit: 'ViT-Small', swin: 'Swin-Tiny' };
                  return (
                    <div key={key} className={`compare-card ${r ? (r.prediction === 'ACL INJURY' ? 'is-injury' : 'is-healthy') : 'is-loading'}`}>
                      <div className="compare-card-name">{names[key]}</div>
                      {!r ? (
                        <div className="compare-card-loading">Evaluando…</div>
                      ) : (
                        <>
                          <div className="compare-verdict">{r.prediction === 'ACL INJURY' ? 'ROTURA LCA' : 'SANO'}</div>
                          <div className="compare-prob">{Math.round((r.prediction === 'ACL INJURY' ? r.calibrated_confidence : 1 - r.calibrated_confidence) * 100)}%<span>confianza {r.prediction === 'ACL INJURY' ? 'de rotura' : 'de sano'}</span></div>
                          <div className="compare-planes">
                            {['sagittal', 'coronal', 'axial'].map((p) => (
                              <div key={p} className="compare-plane-row">
                                <span className="compare-plane-name">{p === 'sagittal' ? 'Sagital' : p === 'coronal' ? 'Coronal' : 'Axial'}</span>
                                <div className="compare-bar"><div className="compare-bar-fill" style={{ width: `${Math.round((r.planes[p] || 0) * 100)}%` }} /></div>
                                <span className="compare-plane-val">{Math.round((r.planes[p] || 0) * 100)}%</span>
                              </div>
                            ))}
                          </div>
                          <div className="compare-thr">umbral clínico {Math.round(r.threshold * 100)}%</div>
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
              {compareConsensus && (
                <div className={`compare-consensus ${compareConsensus.prediction === 'ACL INJURY' ? 'is-injury' : 'is-healthy'}`}>
                  Consenso: <strong>{compareConsensus.prediction === 'ACL INJURY' ? 'ROTURA DE LCA' : 'RODILLA SANA'}</strong>
                  <span> · {compareConsensus.votes_injury}/{compareConsensus.n_models} modelos indican rotura</span>
                </div>
              )}
            </div>
          )}
          {/* Top Row: Global Outcome & Patient Ficha Side-by-Side */}
          <div className="top-summary-row">
            {/* Global Outcome Card (oculto en modo Diagnóstico múltiple: ya está el panel de los 3) */}
            {selectedModel !== 'multi' && (!dynamicDiag ? (
              <div className="outcome-card diagnosis-pulsing-card">
                <div className="outcome-left">
                  <div className="outcome-badge" style={{ background: 'rgba(255, 255, 255, 0.05)', color: 'var(--color-text-secondary)', borderColor: 'var(--border-light)' }}>
                    Calculando
                  </div>
                  <div className="outcome-title" style={{ color: 'var(--color-text-secondary)' }}>
                    Consolidando Diagnóstico...
                  </div>
                  <div className="outcome-meta">
                    Analizando interconexiones del ensemble multi-modelo de 15 cortes...
                  </div>
                </div>
                <div className="gauge-wrapper">
                  <div className="shimmer-spinner" />
                </div>
              </div>
            ) : (
              <div className={`outcome-card ${dynamicDiag.isTear ? 'tear-positive' : 'tear-negative'}`}>
                <div className="outcome-left">
                  <div className="outcome-badge">
                    {dynamicDiag.isTear ? 'Rotura Detectada' : 'Estructura Intacta'}
                  </div>
                  <div className="outcome-title">
                    {dynamicDiag.prediction}
                  </div>
                  <div className="outcome-meta">
                    Probabilidad Ensemble: <span>{Math.round(dynamicDiag.ensemble_probability * 100)}%</span> (Umbral: {THRESHOLD}) | Confianza: <span style={{ color: dynamicDiag.isTear ? '#ff6b8b' : '#34d399' }}>{dynamicDiag.confidence_level.toUpperCase()}</span>
                  </div>
                </div>

                <div className="gauge-wrapper">
                  <svg className="circular-gauge" width="120" height="120">
                    <circle className="gauge-bg" cx="60" cy="60" r="50" />
                    <circle 
                      className="gauge-fill" 
                      cx="60" 
                      cy="60" 
                      r="50" 
                      strokeDasharray={2 * Math.PI * 50}
                      strokeDashoffset={2 * Math.PI * 50 * (1 - dynamicDiag.calibrated_confidence)}
                    />
                  </svg>
                  <div className="gauge-text">
                    <span className="gauge-percentage">{Math.round(dynamicDiag.calibrated_confidence * 100)}%</span>
                    <span className="gauge-label">Confianza</span>
                  </div>
                </div>
              </div>
            ))}

            {/* Patient Metadata Details */}
            <div className="side-card">
              <div className="card-title">Ficha del Paciente</div>
              
              {!patientInfo ? (
                <div className="meta-grid">
                  <div className="meta-item">
                    <span className="meta-label">Paciente:</span>
                    <span className="shimmer-text-line" style={{ width: '70%' }}></span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">ID Estudio:</span>
                    <span className="shimmer-text-line" style={{ width: '80%' }}></span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Edad / Sexo:</span>
                    <span className="shimmer-text-line" style={{ width: '40%' }}></span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Fecha RM:</span>
                    <span className="shimmer-text-line" style={{ width: '50%' }}></span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Modalidad:</span>
                    <span className="shimmer-text-line" style={{ width: '60%' }}></span>
                  </div>
                </div>
              ) : (
                <div className="meta-grid">
                  <div className="meta-item">
                    <span className="meta-label">Paciente:</span>
                    <span className={`meta-value ${maskPatientData ? 'masked' : ''}`}>
                      {maskString(patientInfo.patient_name, maskPatientData)}
                    </span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">ID Estudio:</span>
                    <span className={`meta-value ${maskPatientData ? 'masked' : ''}`}>
                      {maskString(patientInfo.patient_id, maskPatientData)}
                    </span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Edad / Sexo:</span>
                    <span className="meta-value">
                      {patientInfo.patient_age || 'N/A'} / {patientInfo.patient_sex || 'N/A'}
                    </span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Fecha RM:</span>
                    <span className="meta-value">
                      {patientInfo.study_date ? 
                        `${patientInfo.study_date.slice(0, 4)}-${patientInfo.study_date.slice(4, 6)}-${patientInfo.study_date.slice(6, 8)}` : 
                        'N/A'
                      }
                    </span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Modalidad:</span>
                    <span className="meta-value">{patientInfo.modality || 'MR'} (Rodilla)</span>
                  </div>
                </div>
              )}

              <button className="mask-toggle" onClick={() => setMaskPatientData(!maskPatientData)} disabled={!patientInfo}>
                {maskPatientData ? '👁 Mostrar Identidad' : '🔒 Ocultar Identidad (GDPR/HIPAA)'}
              </button>
            </div>
          </div>

          {/* Full-width Workspace Panels */}
          <div className="full-width-workspace">

            {/* Global Heatmap Overlay Controllers */}
            <div className="global-heatmap-controller">
              <div className="toggle-switch-wrapper">
                <label className="switch">
                  <input 
                    type="checkbox" 
                    checked={showHeatmap} 
                    onChange={(e) => setShowHeatmap(e.target.checked)} 
                  />
                  <span className="slider-switch"></span>
                </label>
                <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>
                  Superponer Mapas de Atención (Grad-CAM)
                </span>
              </div>

              <div className="opacity-slider-inline">
                <span style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>Opacidad:</span>
                <input 
                  type="range" 
                  min="0.1" 
                  max="1" 
                  step="0.05" 
                  value={heatmapOpacity} 
                  onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))} 
                />
                <span style={{ fontWeight: 700, fontSize: '0.85rem', minWidth: '35px' }}>
                  {Math.round(heatmapOpacity * 100)}%
                </span>
              </div>
            </div>

            {/* AI Top 5 Slices Visual Ribbon Card - Carousel Upgrade */}
            <div ref={pacsViewerRef} className={`top-slices-ribbon-card viewer-mode-${topViewerMode}`}>
              <div className="ribbon-carousel-nav">
                <button 
                  className="ribbon-nav-btn btn-prev"
                  onClick={() => {
                    const prevPlane = activePlaneTab === 'sagittal' ? 'axial' : activePlaneTab === 'coronal' ? 'sagittal' : 'coronal';
                    setSlideDirection('prev');
                    setActivePlaneTab(prevPlane);
                  }}
                >
                  ❮ {activePlaneTab === 'sagittal' ? 'Axial' : activePlaneTab === 'coronal' ? 'Sagital' : 'Coronal'}
                </button>
                
                <div className="ribbon-header" style={{ textAlign: 'center' }}>
                  <h4>{topViewerMode === 'critical' ? 'Cortes Críticos de Mayor Interés IA' : 'Visor PACS'} (Plano {activePlaneTab === 'sagittal' ? 'Sagital' : activePlaneTab === 'coronal' ? 'Coronal' : 'Axial'})</h4>
                  <span className="ribbon-sub">
                    {topViewerMode === 'critical' 
                      ? 'Selecciona un hallazgo para abrirlo aquí mismo en el visor PACS.'
                      : 'Explora el plano activo con rueda, arrastre horizontal, miniaturas y métricas del corte.'}
                  </span>
                  <div className="viewer-mode-switch" role="group" aria-label="Cambiar visor principal">
                    <button
                      type="button"
                      className={`viewer-mode-btn ${topViewerMode === 'critical' ? 'active' : ''}`}
                      onClick={() => setTopViewerMode('critical')}
                    >
                      Cortes críticos
                    </button>
                    <button
                      type="button"
                      className={`viewer-mode-btn ${topViewerMode === 'pacs' ? 'active' : ''}`}
                      onClick={showPacsViewer}
                    >
                      Visor PACS
                    </button>
                  </div>
                </div>

                <button 
                  className="ribbon-nav-btn btn-next"
                  onClick={() => {
                    const nextPlane = activePlaneTab === 'sagittal' ? 'coronal' : activePlaneTab === 'coronal' ? 'axial' : 'sagittal';
                    setSlideDirection('next');
                    setActivePlaneTab(nextPlane);
                  }}
                >
                  {activePlaneTab === 'sagittal' ? 'Coronal' : activePlaneTab === 'coronal' ? 'Axial' : 'Sagital'} ❯
                </button>
              </div>

              <div className="diagnostic-viewer-stage">
                {topViewerMode === 'critical' ? (
                  <div className="ribbon-body-carousel-wrapper viewer-slide-from-left">
                    <div key={`${activePlaneTab}_${slideDirection}`} className={`ribbon-slide-container slide-direction-${slideDirection}`}>
                      {(() => {
                        const plane = activePlaneTab;
                        const planeData = planes[plane];
                        const planeLabel = plane === 'sagittal' ? 'SAGITAL' : plane === 'coronal' ? 'CORONAL' : 'AXIAL';
                        
                        if (!planeData) {
                          return (
                            <div className="ribbon-plane-group-empty" style={{ justifyContent: 'center' }}>
                              <div className="ribbon-shimmer-row" style={{ width: '100%', maxWidth: '600px' }}>
                                <span className="ribbon-shimmer-pulse">Analizando cortes de Plano {planeLabel}...</span>
                              </div>
                            </div>
                          );
                        }

                        const topSlices = [...planeData.slices]
                          .filter(s => s.has_prediction)
                          .sort((a, b) => b.swin_attention - a.swin_attention)
                          .slice(0, 5);

                        return (
                          <div className="ribbon-plane-group" style={{ borderBottom: 'none', paddingBottom: 0 }}>
                            <div className="ribbon-slices-row ribbon-large-slices">
                              {topSlices.map((slice) => {
                                const sliceIdx = planeData.slices.findIndex(s => s.slice_index === slice.slice_index);
                                const isActive = activePlaneTab === plane && activeSliceIdxs[plane] === sliceIdx;
                                return (
                                  <div 
                                    key={slice.slice_index} 
                                    className={`ribbon-slice-thumb ribbon-large ${isActive ? 'active' : ''}`}
                                    onClick={() => openSliceInPacs(plane, sliceIdx)}
                                    title="Abrir este corte en el visor PACS"
                                  >
                                    <div className="ribbon-thumb-wrapper">
                                      <img 
                                        className="ribbon-thumb-img" 
                                        src={`data:image/jpeg;base64,${slice.raw_image}`} 
                                        alt={`Corte ${slice.slice_index}`} 
                                      />
                                      {showHeatmap && slice.heatmap_overlay && (
                                        <img 
                                          className="ribbon-heatmap-overlay" 
                                          src={`data:image/png;base64,${slice.heatmap_overlay}`} 
                                          alt="Grad-CAM"
                                          style={{ opacity: heatmapOpacity }}
                                        />
                                      )}
                                      <div className="ribbon-thumb-hud-top ribbon-large-hud">
                                        <span>#{slice.slice_index}</span>
                                      </div>
                                      <div className="ribbon-thumb-hud-bottom ribbon-large-hud-bottom">
                                        <span className="ribbon-thumb-prob ribbon-large-text" style={{ color: slice.slice_probability >= THRESHOLD ? 'var(--neon-pink)' : 'var(--neon-teal)' }}>
                                          Prob: {Math.round(slice.slice_probability * 100)}%
                                        </span>
                                        <span className="ribbon-thumb-att ribbon-large-text">
                                          Att: {Math.round(slice.swin_attention * 100)}%
                                        </span>
                                      </div>
                                      <button
                                        type="button"
                                        className="ribbon-open-pacs-btn"
                                        onClick={(event) => {
                                          event.stopPropagation();
                                          openSliceInPacs(plane, sliceIdx);
                                        }}
                                      >
                                        Abrir en PACS
                                      </button>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        );
                      })()}
                    </div>
                  </div>
                ) : (
                  <div className="top-pacs-stage viewer-slide-from-right">
                    {(() => {
                      const plane = activePlaneTab;
                      const planeData = planes[plane];
                      const planeLabel = plane === 'sagittal' ? 'SAGITAL' : plane === 'coronal' ? 'CORONAL' : 'AXIAL';

                      if (!planeData) {
                        return (
                          <div className="top-pacs-empty">
                            <span className="ribbon-shimmer-pulse">Analizando cortes de Plano {planeLabel}...</span>
                          </div>
                        );
                      }

                      return (
                        <PACSVolumeViewer 
                          plane={plane} 
                          planeData={planeData} 
                          showHeatmap={showHeatmap} 
                          heatmapOpacity={heatmapOpacity} 
                          threshold={THRESHOLD}
                          activeIdx={activeSliceIdxs[plane]}
                          setActiveIdx={(idx) => setActiveSliceIdxs(prev => ({ ...prev, [plane]: idx }))}
                          selectedModel={selectedModel}
                        />
                      );
                    })()}
                  </div>
                )}
              </div>
            </div>

            {/* Reset / Upload Another Patient Zip Action */}
            {status === 'completed' && (
              <div className="reset-zone">
                <button className="secondary-button" onClick={resetDiagnostic}>
                  Analizar Nuevo Paciente DICOM
                </button>
              </div>
            )}
          </div>

        </div>
      )}
    </div>
  );
}

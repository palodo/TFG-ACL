import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';

// Resolve directory paths in ES module
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 5000;

// Absolute path to virtual environment python interpreter and predict script
const PYTHON_PATH = process.env.PYTHON_PATH || '/home/palodo2/acl_classifier/.venv/bin/python';
const SCRIPT_PATH = path.resolve(__dirname, '../scripts/predict_patient.py');
// Lightweight 3-model comparison script (no per-slice heatmaps)
const COMPARE_SCRIPT = path.resolve(__dirname, '../scripts/compare_models.py');

// Allowed model identifiers (whitelist)
const ALLOWED_MODELS = new Set(['swin', 'vit', 'cnn']);

// Retention of uploaded ZIPs in hours. 0 (default) keeps files indefinitely,
// which is intentional so cached studies can be re-analyzed during presentations.
const UPLOAD_RETENTION_HOURS = parseInt(process.env.UPLOAD_RETENTION_HOURS || '0', 10);

// Security Headers Injection.
// NOTE: no X-Frame-Options and a permissive frame-ancestors so the app can be
// rendered inside VS Code's Simple Browser (a vscode-webview:// iframe) when
// served through an SSH port-forward. This is a localhost-only demo server.
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors *;");
  res.setHeader('X-XSS-Protection', '1; mode=block');
  next();
});

// JSON Parser with strict size limit to prevent DoS
app.use(express.json({ limit: '2mb' }));

// Establish secure upload directory (outside of public web root)
const UPLOAD_DIR = path.resolve(__dirname, 'uploads');
if (!fs.existsSync(UPLOAD_DIR)) {
  fs.mkdirSync(UPLOAD_DIR, { recursive: true });
}

// Optional cleanup of old uploads (disabled when UPLOAD_RETENTION_HOURS=0)
function cleanupOldUploads() {
  if (!UPLOAD_RETENTION_HOURS) return;
  const cutoff = Date.now() - UPLOAD_RETENTION_HOURS * 3600 * 1000;
  fs.readdir(UPLOAD_DIR, (err, files) => {
    if (err) return;
    for (const name of files) {
      if (!name.endsWith('.zip')) continue;
      const filePath = path.join(UPLOAD_DIR, name);
      fs.stat(filePath, (statErr, stats) => {
        if (statErr) return;
        if (stats.mtimeMs < cutoff) {
          fs.unlink(filePath, () => {
            console.log(`[BFF Server] Removed expired upload: ${name}`);
          });
        }
      });
    }
  });
}
cleanupOldUploads();
if (UPLOAD_RETENTION_HOURS) {
  setInterval(cleanupOldUploads, 3600 * 1000).unref();
}

// Sanitize a user-provided filename into a safe basename inside UPLOAD_DIR
function sanitizeZipName(originalName) {
  let safeName = path.basename(String(originalName)).replace(/[^a-zA-Z0-9.-]/g, '_');
  if (!safeName.endsWith('.zip')) {
    safeName += '.zip';
  }
  return safeName;
}

// Multer storage configuration keeping sanitized original filenames
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, UPLOAD_DIR);
  },
  filename: (req, file, cb) => {
    cb(null, sanitizeZipName(file.originalname));
  }
});

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 200 * 1024 * 1024, // 200MB limit to support larger patient Dicom series
    files: 1
  },
  fileFilter: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (ext !== '.zip') {
      return cb(new Error('Only ZIP archive format is permitted for DICOM processing.'), false);
    }
    cb(null, true);
  }
});

// Single-flight guard: the inference engine loads several models on one GPU,
// so concurrent runs would exhaust memory. Extra requests get a clean error chunk.
let inferenceBusy = false;

/**
 * Spawn the Python diagnostic pipeline and stream its NDJSON stdout to the client.
 * Shared by /api/analyze (fresh upload) and /api/analyze-existing (cached zip).
 */
function runInference(zipPath, selectedModel, req, res, label) {
  if (inferenceBusy) {
    console.warn(`[BFF Server] Rejected ${label}: another inference is already running.`);
    res.setHeader('Content-Type', 'application/x-ndjson');
    res.write(JSON.stringify({
      error: 'El servidor está procesando otro estudio en este momento. Inténtelo de nuevo en unos segundos.'
    }) + '\n');
    return res.end();
  }
  inferenceBusy = true;

  console.log(`[BFF Server] ${label}: ${zipPath} (model: ${selectedModel})`);

  // Set response headers for NDJSON real-time streaming
  res.setHeader('Content-Type', 'application/x-ndjson');
  res.setHeader('Transfer-Encoding', 'chunked');

  // Securely spawn the subprocess (shell: false prevents shell command injection)
  const pyProcess = spawn(PYTHON_PATH, [SCRIPT_PATH, '--zip', zipPath, '--model', selectedModel], {
    shell: false,
    env: { ...process.env, PYTHONUNBUFFERED: '1' }
  });

  // Only terminate the inference if the client genuinely aborts BEFORE the
  // response finishes. We listen on `res` (not `req`, whose 'close' can fire
  // early once the request body is consumed) and gate on writableFinished.
  res.on('close', () => {
    if (!res.writableFinished && pyProcess.exitCode === null) {
      console.warn(`[BFF Server] Client aborted; terminating inference (pid ${pyProcess.pid}).`);
      pyProcess.kill('SIGTERM');
    }
  });

  // Directly pipe stdout chunks in real-time to the client
  pyProcess.stdout.on('data', (data) => {
    res.write(data);
  });

  pyProcess.stderr.on('data', (data) => {
    console.error(`[BFF Server] Python process stderr: ${data.toString()}`);
  });

  pyProcess.on('close', (code, signal) => {
    inferenceBusy = false;
    console.log(`[BFF Server] Inference process closed (code ${code}, signal ${signal}). Zip retained: ${zipPath}`);

    if (res.writableEnded) return;

    if (code !== 0 && signal == null) {
      console.error(`[BFF Server] Python process exited with error code ${code}`);
      res.write(JSON.stringify({ error: 'An internal diagnostic failure occurred. Please ensure the uploaded file contains valid knee DICOM slices.' }) + '\n');
    }
    res.end();
  });

  pyProcess.on('error', (err) => {
    inferenceBusy = false;
    console.error(`[BFF Server] Failed to spawn inference process: ${err.message}`);
    if (!res.writableEnded) {
      res.write(JSON.stringify({ error: 'The diagnostic engine could not be started.' }) + '\n');
      res.end();
    }
  });
}

/**
 * Spawn the lightweight 3-model comparison pipeline and stream its NDJSON output.
 */
function runComparison(zipPath, req, res, label) {
  if (inferenceBusy) {
    res.setHeader('Content-Type', 'application/x-ndjson');
    res.write(JSON.stringify({ error: 'El servidor está procesando otro estudio. Inténtelo en unos segundos.' }) + '\n');
    return res.end();
  }
  inferenceBusy = true;
  console.log(`[BFF Server] ${label}: ${zipPath} (comparativa 3 modelos)`);

  res.setHeader('Content-Type', 'application/x-ndjson');
  res.setHeader('Transfer-Encoding', 'chunked');

  const pyProcess = spawn(PYTHON_PATH, [COMPARE_SCRIPT, '--zip', zipPath], {
    shell: false,
    env: { ...process.env, PYTHONUNBUFFERED: '1' }
  });

  res.on('close', () => {
    if (!res.writableFinished && pyProcess.exitCode === null) {
      pyProcess.kill('SIGTERM');
    }
  });

  pyProcess.stdout.on('data', (data) => res.write(data));
  pyProcess.stderr.on('data', (data) => console.error(`[BFF Server] compare stderr: ${data.toString()}`));

  pyProcess.on('close', (code, signal) => {
    inferenceBusy = false;
    if (res.writableEnded) return;
    if (code !== 0 && signal == null) {
      res.write(JSON.stringify({ error: 'Falló la comparativa de modelos.' }) + '\n');
    }
    res.end();
  });

  pyProcess.on('error', (err) => {
    inferenceBusy = false;
    if (!res.writableEnded) {
      res.write(JSON.stringify({ error: 'No se pudo iniciar la comparativa.' }) + '\n');
      res.end();
    }
  });
}

function parseModel(value) {
  return ALLOWED_MODELS.has(value) ? value : 'swin';
}

// Lightweight health/status endpoint for monitoring and the frontend
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    busy: inferenceBusy,
    models: [...ALLOWED_MODELS],
    retention_hours: UPLOAD_RETENTION_HOURS
  });
});

// Diagnostic API Endpoint (fresh upload)
app.post('/api/analyze', upload.single('dicomZip'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ success: false, error: 'No DICOM zip archive uploaded.' });
  }
  const selectedModel = parseModel(req.body.model);
  runInference(req.file.path, selectedModel, req, res, 'Analyze uploaded file');
});

// Endpoint to check if a file already exists securely
app.get('/api/check-file', (req, res) => {
  const { filename } = req.query;
  if (!filename || typeof filename !== 'string') {
    return res.status(400).json({ success: false, error: 'Filename parameter is required.' });
  }

  // Strict normalization & validation to prevent directory traversal
  const safeName = sanitizeZipName(filename);
  const filePath = path.join(UPLOAD_DIR, safeName);

  // Validate the path starts with UPLOAD_DIR and ends with .zip
  if (!filePath.startsWith(UPLOAD_DIR) || !safeName.endsWith('.zip')) {
    return res.status(400).json({ success: false, error: 'Invalid path access or invalid file format.' });
  }

  const exists = fs.existsSync(filePath);
  console.log(`[BFF Server] Pre-flight file check: ${safeName} -> exists: ${exists}`);
  return res.json({ exists, safeName });
});

// Endpoint to run inference on an already uploaded file
app.post('/api/analyze-existing', (req, res) => {
  const { safeName } = req.body;
  if (!safeName || typeof safeName !== 'string') {
    return res.status(400).json({ success: false, error: 'safeName parameter is required.' });
  }

  // Double check directory traversal / boundary injection attempts
  const cleanName = sanitizeZipName(safeName);
  const zipPath = path.join(UPLOAD_DIR, cleanName);

  if (!zipPath.startsWith(UPLOAD_DIR) || !cleanName.endsWith('.zip')) {
    return res.status(400).json({ success: false, error: 'Invalid path access or invalid file format.' });
  }

  if (!fs.existsSync(zipPath)) {
    return res.status(404).json({ success: false, error: 'DICOM zip archive does not exist on server.' });
  }

  const selectedModel = parseModel(req.body.model);
  runInference(zipPath, selectedModel, req, res, 'Re-analyze cached file');
});

// Comparison API (3 models at once) — fresh upload
app.post('/api/compare', upload.single('dicomZip'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ success: false, error: 'No DICOM zip archive uploaded.' });
  }
  runComparison(req.file.path, req, res, 'Compare uploaded file');
});

// Comparison API (3 models at once) — already uploaded file
app.post('/api/compare-existing', (req, res) => {
  const { safeName } = req.body;
  if (!safeName || typeof safeName !== 'string') {
    return res.status(400).json({ success: false, error: 'safeName parameter is required.' });
  }
  const cleanName = sanitizeZipName(safeName);
  const zipPath = path.join(UPLOAD_DIR, cleanName);
  if (!zipPath.startsWith(UPLOAD_DIR) || !cleanName.endsWith('.zip')) {
    return res.status(400).json({ success: false, error: 'Invalid path access or invalid file format.' });
  }
  if (!fs.existsSync(zipPath)) {
    return res.status(404).json({ success: false, error: 'DICOM zip archive does not exist on server.' });
  }
  runComparison(zipPath, req, res, 'Compare cached file');
});

// List the bundled example studies available in the uploads directory
app.get('/api/examples', (req, res) => {
  const examples = [
    { safeName: '2023.zip', label: 'Rodilla 2023 (sana)' },
    { safeName: '2025-10-07-alberola-guillot-marc-rm-rodilla-izquierd.zip', label: 'Rodilla Alberola' }
  ].filter(e => fs.existsSync(path.join(UPLOAD_DIR, e.safeName)));
  res.json({ examples });
});

// Serve compiled frontend files from production build directory
const DIST_DIR = path.resolve(__dirname, 'dist');
if (fs.existsSync(DIST_DIR)) {
  app.use(express.static(DIST_DIR));
  app.get('*', (req, res) => {
    res.sendFile(path.join(DIST_DIR, 'index.html'));
  });
} else {
  app.get('/', (req, res) => {
    res.send('BFF backend is running. Build the frontend to serve the landing page.');
  });
}

// Global error handler
app.use((err, req, res, next) => {
  console.error('[BFF Server] Global Error Handled:', err.message);
  res.status(err.status || 500).json({
    success: false,
    error: err.message || 'An unexpected server error occurred.'
  });
});

// Bind to all interfaces so VS Code / SSH port-forwarding reaches the server
// regardless of whether it connects via IPv4 (127.0.0.1) or IPv6 (::1).
const HOST = process.env.HOST || '0.0.0.0';
app.listen(PORT, HOST, () => {
  console.log(`[BFF Server] Running on http://${HOST}:${PORT} (forward this port)`);
});

#!/usr/bin/env python3
import base64, pathlib
A = pathlib.Path("assets")
def b64(n): return f"data:image/png;base64,{base64.b64encode((A/n).read_bytes()).decode()}"
NAMES = ["pipeline_general.png","app_gradcam_planes.png","mri_slices_sagital.png","roc_test.png",
         "escudo.png","app_dashboard.png","chart_test.png","chart_params.png","chart_croatia.png",
         "chart_multiseed.png","chart_threshold.png","data_planes.png","data_posneg.png"]
IMG = {n: b64(n) for n in NAMES}

CSS = """
*{margin:0;padding:0;box-sizing:border-box;-webkit-font-smoothing:antialiased}
html{scroll-snap-type:y mandatory;scroll-behavior:smooth}
body{font-family:'Inter',system-ui,sans-serif;background:#060E16;color:#E6EEF2}
:root{--teal:#2DD4BF;--sky:#38BDF8;--purp:#A78BFA;--red:#F87171;--mut:#8FA8B8;--faint:#5E7686;
  --card:rgba(255,255,255,.045);--cardb:rgba(255,255,255,.10);--ink:#E6EEF2}
.slide{position:relative;width:100vw;height:56.25vw;max-height:100vh;scroll-snap-align:center;
  overflow:hidden;display:flex;flex-direction:column;padding:4vw 5vw 3.2vw;
  background:radial-gradient(120% 80% at 85% 0%,#0E2433 0%,#081019 55%,#060E16 100%)}
.glow{position:absolute;border-radius:50%;filter:blur(8px);opacity:.5;z-index:0}
.kicker{font-family:'Space Grotesk',sans-serif;font-size:0.92vw;font-weight:600;letter-spacing:0.22em;
  text-transform:uppercase;color:var(--teal);display:flex;align-items:center;gap:0.7vw;margin-bottom:1.3vw;z-index:2}
.kicker .dot{width:0.55vw;height:0.55vw;border-radius:50%;background:var(--teal);box-shadow:0 0 0.8vw var(--teal)}
h1.head{font-family:'Space Grotesk',sans-serif;font-size:2.7vw;font-weight:700;line-height:1.06;
  letter-spacing:-0.02em;color:#fff;max-width:80%;z-index:2}
.sub{font-size:1.18vw;color:var(--mut);line-height:1.55;max-width:46vw;z-index:2}
.foot{position:absolute;left:5vw;bottom:2.2vw;font-size:0.82vw;font-weight:600;letter-spacing:.05em;color:var(--faint);z-index:2}
.pg{position:absolute;right:5vw;bottom:2.2vw;font-size:0.82vw;font-weight:700;color:var(--faint);z-index:2}
.row{display:flex;gap:2.2vw;flex:1;min-height:0;z-index:2}
.col{display:flex;flex-direction:column}
.grow{flex:1}.center{justify-content:center}
.card{background:var(--card);border:1px solid var(--cardb);border-radius:1vw;padding:1.7vw;backdrop-filter:blur(6px)}
.imgcard{background:rgba(255,255,255,.97);border:1px solid var(--cardb);border-radius:1vw;overflow:hidden;
  display:flex;align-items:center;justify-content:center;padding:1vw}
.imgcard.bare{background:transparent;border:none;padding:0}
.imgcard img{width:100%;height:100%;object-fit:contain}
.cap{font-size:0.8vw;color:var(--faint);font-style:italic;margin-top:0.6vw;text-align:center}
.stat{font-family:'Space Grotesk',sans-serif;font-size:3vw;font-weight:700;color:var(--teal);line-height:1;letter-spacing:-.02em}
.stat-l{font-size:0.92vw;color:var(--mut);margin-top:0.5vw;line-height:1.35}
.badge{font-family:'Space Grotesk',sans-serif;display:inline-flex;align-items:center;gap:0.6vw;
  font-size:0.95vw;font-weight:600;color:var(--teal);background:rgba(45,212,191,.12);
  border:1px solid rgba(45,212,191,.35);border-radius:2vw;padding:0.4vw 1.1vw;margin-bottom:1.1vw;width:fit-content}
.why{border-left:0.18vw solid var(--teal);padding-left:1vw;color:var(--ink);font-size:1.12vw;line-height:1.5;max-width:40vw}
.why b{color:var(--teal)}
ul.b{list-style:none;margin-top:0.4vw}
ul.b li{font-size:1.02vw;color:#D4E0E8;line-height:1.45;padding-left:1.4vw;position:relative;margin-bottom:0.8vw}
ul.b li:before{content:"";position:absolute;left:0;top:0.55vw;width:0.5vw;height:0.5vw;border-radius:50%;background:var(--teal)}
ul.b li b{color:#fff}
table.t{width:100%;border-collapse:collapse;font-size:1vw}
table.t th{text-align:left;color:var(--teal);font-weight:600;font-size:0.85vw;letter-spacing:.06em;
  text-transform:uppercase;padding:0.6vw 0.8vw;border-bottom:1px solid var(--cardb)}
table.t td{padding:0.7vw 0.8vw;border-bottom:1px solid rgba(255,255,255,.06);color:#D4E0E8}
table.t td:first-child{color:#fff;font-weight:600}
.chips{display:flex;gap:0.7vw;flex-wrap:wrap}
.chip{font-size:0.92vw;font-weight:600;color:#cfe;background:rgba(255,255,255,.06);
  border:1px solid var(--cardb);border-radius:0.6vw;padding:0.45vw 0.9vw}
.kpi{font-family:'Space Grotesk',sans-serif;font-size:2.2vw;font-weight:700;letter-spacing:-.02em}
.win{display:flex;gap:0.9vw;align-items:flex-start;margin-bottom:1.3vw}
.win .d{width:0.8vw;height:0.8vw;border-radius:50%;margin-top:0.5vw;flex:none}
.win .l{font-size:0.92vw;color:var(--mut)}
.win .v{font-family:'Space Grotesk',sans-serif;font-size:1.3vw;font-weight:700;color:#fff;line-height:1.15}
"""

def slide(inner, page=None, foot="ETSE · Universitat de València", glows=True):
    g = ('<div class="glow" style="top:-10vw;right:-6vw;width:26vw;height:26vw;background:#0E5466"></div>'
         '<div class="glow" style="bottom:-12vw;left:-8vw;width:24vw;height:24vw;background:#103a52"></div>') if glows else ""
    f = f'<div class="foot">{foot}</div>' if foot else ""
    pg = f'<div class="pg">{page:02d}</div>' if page else ""
    return f'<section class="slide">{g}{inner}{f}{pg}</section>'

def kick(t): return f'<div class="kicker"><span class="dot"></span>{t}</div>'
def deci(n): return f'<div class="badge">DECISIÓN {n} · validada en desarrollo</div>'

S=[]

# 1 HERO
S.append(slide(f"""
  {kick('Trabajo Fin de Grado · Ciencia de Datos')}
  <h1 class="head" style="font-size:3.8vw;max-width:76%">Detección automática de rotura del LCA en resonancia de rodilla</h1>
  <p class="sub" style="margin-top:1.6vw;max-width:54vw;font-style:italic">CNN vs. Vision Transformers vs. Swin Transformers — comparados a igualdad de pipeline, con todas las decisiones tomadas en validación.</p>
  <div style="margin-top:3.2vw;font-size:1.08vw;line-height:2;z-index:2">
    <div><span style="color:var(--teal);font-weight:700">Autor&nbsp;&nbsp;</span>Pablo López Domínguez</div>
    <div><span style="color:var(--teal);font-weight:700">Tutores&nbsp;&nbsp;</span>José David Martín Guerrero · Yolanda Vives Gilabert</div>
  </div>
  <img src="{IMG['escudo.png']}" style="position:absolute;right:5vw;bottom:2.8vw;height:10.5vw;z-index:2;opacity:.92">
""", foot=None))

# 2 PROBLEMA
S.append(slide(f"""
  {kick('El problema')}
  <h1 class="head">Leer el LCA en RM es lento y un fallo cuesta caro</h1>
  <div class="row" style="margin-top:2vw">
    <div class="col grow center" style="gap:1.6vw;max-width:40vw">
      <p class="sub" style="max-width:38vw">Una rotura puede verse en muy pocos cortes. Un <b style="color:#fff">falso negativo</b> retrasa el tratamiento: la prioridad clínica es la <b style="color:var(--teal)">sensibilidad</b>.</p>
      <div class="row" style="flex:none;gap:1.4vw">
        <div class="card col" style="flex:1"><div class="stat">~21%</div><div class="stat-l">prevalencia de rotura</div></div>
        <div class="card col" style="flex:1"><div class="stat">3</div><div class="stat-l">planos por estudio</div></div>
        <div class="card col" style="flex:1"><div class="stat">1 corte</div><div class="stat-l">puede bastar para el diagnóstico</div></div>
      </div>
    </div>
    <div class="col" style="width:32vw">
      <div class="imgcard bare grow"><img src="{IMG['data_posneg.png']}"></div>
      <div class="cap">Ejemplos reales del conjunto: rodilla con rotura vs. sana (plano sagital)</div>
    </div>
  </div>
""", page=2))

# 3 DATOS
S.append(slide(f"""
  {kick('Los datos · MRNet')}
  <h1 class="head">1 250 estudios, tres vistas por rodilla</h1>
  <div class="row" style="margin-top:1.8vw">
    <div class="col" style="width:34vw;gap:1.2vw">
      <table class="t">
        <tr><th>Conjunto</th><th>Estudios</th><th>Rotura</th><th>Prevalencia</th></tr>
        <tr><td>Entrenamiento</td><td>875</td><td>183</td><td>20.9%</td></tr>
        <tr><td>Validación</td><td>188</td><td>36</td><td>19.1%</td></tr>
        <tr><td>Test</td><td>187</td><td>43</td><td>23.0%</td></tr>
      </table>
      <p class="sub" style="font-size:1.05vw;max-width:33vw;margin-top:0.6vw">Particiones <b style="color:#fff">a nivel de paciente</b> para evitar fuga de información entre conjuntos.</p>
    </div>
    <div class="col grow">
      <div class="imgcard bare grow"><img src="{IMG['data_planes.png']}"></div>
      <div class="cap">Mismo estudio (rotura de LCA) en los planos sagital, coronal y axial</div>
    </div>
  </div>
""", page=3))

# 4 VALIDACIÓN HONESTA (protocolo)
S.append(slide(f"""
  {kick('Metodología · evaluación honesta')}
  <h1 class="head">El test solo se toca al final</h1>
  <p class="sub" style="margin-top:1vw;max-width:58vw">Toda decisión —arquitectura, cortes, regularización, umbral— se fija mirando <b style="color:var(--teal)">solo validación</b>. El test se mira una vez, al final, sin influir en ninguna elección.</p>
  <div class="row" style="margin-top:2vw;gap:1.6vw;flex:none;height:15vw">
    <div class="card col center" style="flex:1;gap:0.6vw"><div class="kpi" style="color:var(--teal);font-size:1.7vw">Train</div><div class="stat-l" style="font-size:1vw">875 estudios.<br>Aprende los pesos del modelo.</div></div>
    <div class="card col center" style="flex:1;gap:0.6vw"><div class="kpi" style="color:var(--sky);font-size:1.7vw">Validación</div><div class="stat-l" style="font-size:1vw">188 estudios.<br>Aquí se toman <b style="color:#fff">todas</b> las decisiones y se calibra el umbral.</div></div>
    <div class="card col center" style="flex:1;gap:0.6vw"><div class="kpi" style="color:#fff;font-size:1.7vw">Test</div><div class="stat-l" style="font-size:1vw">187 estudios.<br>Se mira una vez. Estimación final, <b style="color:#fff">cero</b> decisiones.</div></div>
    <div class="card col center" style="flex:1;gap:0.6vw"><div class="kpi" style="color:var(--purp);font-size:1.7vw">Croacia</div><div class="stat-l" style="font-size:1vw">917 estudios.<br>Dominio externo: generalización real.</div></div>
  </div>
  <p class="sub" style="margin-top:1.4vw;max-width:56vw;font-size:1vw;color:var(--faint)">El <i>hidden test</i> oficial de MRNet no es accesible: la literatura mezcla protocolos y eso dificulta comparar cifras entre trabajos.</p>
""", page=4))

# 5 PIPELINE
S.append(slide(f"""
  {kick('Visión general del sistema')}
  <h1 class="head">Un pipeline end-to-end multi-vista</h1>
  <div class="imgcard grow" style="margin-top:1.4vw"><img src="{IMG['pipeline_general.png']}"></div>
""", page=5))

# 6 DECISIÓN 1 — selección de cortes
S.append(slide(f"""
  {kick('Selección de cortes')}{deci(1)}
  <h1 class="head">Un selector aprendido, no cortes al azar</h1>
  <div class="row" style="margin-top:1.4vw">
    <div class="col center" style="width:40vw;gap:1.4vw">
      <div class="why">Procesar todo el volumen añade ruido. Un <b>MobileNetV2</b> puntúa cada corte y elige los más informativos.</div>
      <ul class="b">
        <li><b>Sagital K=5</b> · <b>Coronal K=10</b> (selector) · <b>Axial K=10</b> (centrales)</li>
        <li>En validación batió a las estrategias <b>center</b> y <b>random</b>.</li>
        <li>Reduce ruido y coste, y concentra el aprendizaje en la región del LCA.</li>
      </ul>
    </div>
    <div class="col grow">
      <div class="imgcard bare grow"><img src="{IMG['mri_slices_sagital.png']}"></div>
      <div class="cap">Cortes sagitales seleccionados automáticamente, ordenados por relevancia</div>
    </div>
  </div>
""", page=6))

# 7 DECISIÓN 2 — pooling
S.append(slide(f"""
  {kick('Agregación entre cortes')}{deci(2)}
  <h1 class="head">El pooling se elige por plano</h1>
  <div class="row" style="margin-top:1.4vw">
    <div class="col center" style="width:38vw">
      <div class="why">Cada plano tiene una estructura distinta; el mejor modo de combinar los cortes se decidió <b>plano a plano en validación</b>.</div>
    </div>
    <div class="col grow center">
      <table class="t" style="font-size:1.15vw">
        <tr><th>Plano</th><th>Cortes</th><th>Agregación</th></tr>
        <tr><td>Sagital</td><td>5</td><td>Attention Pooling</td></tr>
        <tr><td>Coronal</td><td>10</td><td>Max Pooling</td></tr>
        <tr><td>Axial</td><td>10</td><td>Attention Pooling</td></tr>
      </table>
      <p class="cap" style="text-align:left;margin-top:1vw">Attention Pooling aprende un peso por corte; Max destaca el corte más sospechoso.</p>
    </div>
  </div>
""", page=7))

# 8 DECISIÓN 3 — regularización
S.append(slide(f"""
  {kick('Regularización')}{deci(3)}
  <h1 class="head">Más regularización donde más varía</h1>
  <div class="row" style="margin-top:1.4vw">
    <div class="col center" style="width:36vw">
      <div class="why">El plano <b>coronal</b> es más inestable → más dropout y aumentos. El <b>axial</b>, más limpio → menos. Ajustado en validación (ViT-Small).</div>
    </div>
    <div class="col grow center">
      <table class="t" style="font-size:1.1vw">
        <tr><th>Plano</th><th>Dropout</th><th>Aumentos</th><th>Early stop</th></tr>
        <tr><td>Sagital</td><td>0.20 / 0.15</td><td>conservador</td><td>25</td></tr>
        <tr><td>Coronal</td><td>0.25 / 0.15</td><td>moderado</td><td>15</td></tr>
        <tr><td>Axial</td><td>0.25 / 0.15</td><td>agresivo</td><td>15</td></tr>
      </table>
    </div>
  </div>
""", page=8))

# 9 DECISIÓN 4 — early stopping
S.append(slide(f"""
  {kick('Criterio de parada')}{deci(4)}
  <h1 class="head">Se para por AUC de validación, no por pérdida</h1>
  <div class="row" style="margin-top:1.4vw">
    <div class="col center grow">
      <div class="why" style="max-width:44vw">Se conserva el checkpoint del <b>mejor AUC de validación</b>, no la última época. Que la pérdida de train aún baje es lo esperable: parar antes evita el sobreajuste.</div>
      <ul class="b" style="margin-top:1.6vw">
        <li>Métrica monitorizada: <b>AUC de validación</b> (maximización).</li>
        <li>Paciencia por plano: <b>25 / 15 / 15</b> épocas.</li>
        <li>El ruido de las curvas refleja el tamaño del conjunto (188 estudios).</li>
      </ul>
    </div>
    <div class="col center" style="width:30vw">
      <div class="card"><div class="kpi" style="color:var(--teal)">mejor AUC val</div>
      <div class="stat-l" style="font-size:1.05vw;margin-top:0.6vw">= modelo conservado.<br>La cola de la curva es solo el margen de paciencia.</div></div>
    </div>
  </div>
""", page=9))

# 10 DECISIÓN 5 — multiseed
S.append(slide(f"""
  {kick('Estabilidad')}{deci(5)}
  <h1 class="head">10 semillas: una sola ejecución engaña</h1>
  <div class="row" style="margin-top:1.2vw">
    <div class="imgcard bare grow" style="flex:1.5"><img src="{IMG['chart_multiseed.png']}"></div>
    <div class="col center" style="width:30vw">
      <ul class="b">
        <li>Cada arquitectura/plano se entrena con <b>10 semillas</b> (42–51).</li>
        <li>Se elige el mejor checkpoint <b>por validación</b>.</li>
        <li>El <b>coronal</b> concentra la variabilidad; el <b>axial</b> es el más estable.</li>
      </ul>
    </div>
  </div>
""", page=10))

# 11 DECISIÓN 6 — umbral
S.append(slide(f"""
  {kick('Punto operativo clínico')}{deci(6)}
  <h1 class="head">Umbral calibrado para no perder roturas</h1>
  <div class="row" style="margin-top:1.2vw">
    <div class="imgcard bare grow" style="flex:1.4"><img src="{IMG['chart_threshold.png']}"></div>
    <div class="col center" style="width:32vw">
      <div class="why">τ* = máxima <b>sensibilidad</b> con <b>precisión ≥ 0.75</b>, fijado <b>solo en validación</b>.</div>
      <ul class="b" style="margin-top:1.4vw">
        <li>Un falso negativo es el error más caro → priorizamos recall.</li>
        <li>Cada arquitectura tiene su propio umbral calibrado.</li>
      </ul>
    </div>
  </div>
""", page=11))

# 12 DECISIÓN 7 — ensemble
S.append(slide(f"""
  {kick('Fusión multi-vista')}{deci(7)}
  <h1 class="head">Promedio simple de los tres planos</h1>
  <div class="row" style="margin-top:1.6vw">
    <div class="col center grow">
      <div class="why" style="max-width:46vw">Se promedian las probabilidades de sagital, coronal y axial. <b>Sin parámetros nuevos</b> y sin decisiones dependientes del test.</div>
    </div>
    <div class="col center" style="width:30vw">
      <div class="card" style="text-align:center"><div class="kpi" style="font-size:1.8vw;color:#fff">p = (p_sag + p_cor + p_ax) / 3</div>
      <div class="stat-l" style="margin-top:0.8vw">mejora consistente frente a cualquier plano individual</div></div>
    </div>
  </div>
""", page=12))

# 13 RESULTADOS test
def win(l,v,c): return f'<div class="win"><span class="d" style="background:{c}"></span><div><div class="l">{l}</div><div class="v">{v}</div></div></div>'
S.append(slide(f"""
  {kick('Resultados · test MRNet')}
  <h1 class="head">No hay un ganador absoluto</h1>
  <div class="row" style="margin-top:1.2vw">
    <div class="imgcard bare grow" style="flex:1.5"><img src="{IMG['chart_test.png']}"></div>
    <div class="col center" style="width:28vw">
      {win('Mayor AUC','ResNet50 · 0.954','#94A3B8')}
      {win('Mayor sensibilidad','Swin-Tiny · 0.907 · 4 FN','#38BDF8')}
      {win('Más equilibrado (F1, precisión)','ViT-Small · 0.753','#2DD4BF')}
    </div>
  </div>
""", page=13))

# 14 RESULTADOS ROC
S.append(slide(f"""
  {kick('Resultados · discriminación')}
  <h1 class="head">AUC de test en torno a 0.95</h1>
  <div class="row" style="margin-top:1.2vw">
    <div class="col" style="width:33vw"><div class="imgcard grow"><img src="{IMG['roc_test.png']}"></div></div>
    <div class="col center grow" style="gap:1.4vw">
      <div style="display:flex;align-items:baseline;gap:1vw"><span class="kpi" style="color:#94A3B8">0.954</span><span class="stat-l">ResNet50</span></div>
      <div style="display:flex;align-items:baseline;gap:1vw"><span class="kpi" style="color:var(--teal)">0.943</span><span class="stat-l">ViT-Small</span></div>
      <div style="display:flex;align-items:baseline;gap:1vw"><span class="kpi" style="color:var(--sky)">0.954</span><span class="stat-l">Swin-Tiny</span></div>
      <p class="sub" style="font-size:1.02vw;max-width:30vw;margin-top:0.6vw">Las tres familias discriminan con fiabilidad la presencia de rotura.</p>
    </div>
  </div>
""", page=14))

# 15 EL PORQUÉ — AUC vs params
S.append(slide(f"""
  {kick('El porqué · hallazgo clave')}
  <h1 class="head">Más parámetros no dan más rendimiento</h1>
  <div class="row" style="margin-top:1vw">
    <div class="imgcard bare grow" style="flex:1.6"><img src="{IMG['chart_params.png']}"></div>
    <div class="col center" style="width:28vw;gap:1.3vw">
      <div class="card"><div class="kpi" style="font-size:1.5vw;color:var(--teal)">ViT-Small &gt; ViT-Base</div><div class="stat-l" style="margin-top:0.5vw">22M iguala a 86M (4× menos).</div></div>
      <div class="card"><div class="kpi" style="font-size:1.5vw;color:var(--red)">Swin-Small colapsa</div><div class="stat-l" style="margin-top:0.5vw">49M no converge con datos limitados.</div></div>
    </div>
  </div>
""", page=15))

# 16 EXPLICABILIDAD
S.append(slide(f"""
  {kick('Interpretabilidad')}
  <h1 class="head">El modelo mira la región del LCA</h1>
  <div class="row" style="margin-top:1.2vw">
    <div class="imgcard bare grow" style="flex:1.7"><img src="{IMG['app_gradcam_planes.png']}"></div>
    <div class="col center" style="width:25vw">
      <ul class="b">
        <li>Mapas de saliencia sobre los cortes de los 3 planos.</li>
        <li>Activación en la <b>región intercondílea</b>, donde está el LCA.</li>
        <li>Transparencia clínica en cada predicción.</li>
      </ul>
    </div>
  </div>
""", page=16))

# 17 CROACIA
S.append(slide(f"""
  {kick('Validación externa · Croacia')}
  <h1 class="head">Otro hospital: aparece el domain shift</h1>
  <div class="row" style="margin-top:1.2vw">
    <div class="imgcard bare grow" style="flex:1.4"><img src="{IMG['chart_croatia.png']}"></div>
    <div class="col center" style="width:30vw;gap:1.3vw">
      <div class="card"><div class="kpi" style="font-size:1.5vw;color:#FBBF24">Cae la sensibilidad</div><div class="stat-l" style="margin-top:0.5vw">Cambian escáner, población y protocolo; el AUC aguanta.</div></div>
      <div class="card"><div class="stat-l">Tras fine-tuning (test Croacia)</div><div class="kpi" style="font-size:1.9vw;margin-top:0.3vw"><span style="color:#fff">0.924</span> <span style="color:var(--mut)">→</span> <span style="color:var(--teal)">0.936</span></div></div>
    </div>
  </div>
""", page=17))

# 18 APP
S.append(slide(f"""
  {kick('Aplicación')}
  <h1 class="head">Demostrador clínico interactivo</h1>
  <div class="row" style="margin-top:1.2vw">
    <div class="imgcard grow" style="flex:1.6;background:#0A1822;border-color:var(--cardb)"><img src="{IMG['app_dashboard.png']}"></div>
    <div class="col center" style="width:26vw;gap:1vw">
      <div class="card"><b style="color:#fff;font-size:1.1vw">Carga DICOM real</b><div class="stat-l">estudio comprimido → diagnóstico</div></div>
      <div class="card"><b style="color:#fff;font-size:1.1vw">Inferencia en vivo</b><div class="stat-l">selección de cortes + ensamble</div></div>
      <div class="card"><b style="color:#fff;font-size:1.1vw">Explicabilidad + 3 modelos</b><div class="stat-l">mapas de calor · intercambiables</div></div>
    </div>
  </div>
""", page=18))

# 19 CONCLUSIONES
con=[("Sistema fiable","AUC ~0.95 en test, sin segmentación manual.","var(--teal)"),
     ("La arquitectura importa menos que el pipeline","Cada familia gana en una métrica; ninguna domina.","var(--sky)"),
     ("Lo compacto gana","ViT-Small (22M) y Swin-Tiny (28M) baten a sus variantes grandes.","var(--teal)"),
     ("Decisiones en validación","Modelos y umbral fijados sin tocar el test; domain shift mitigado con fine-tuning.","var(--purp)")]
conhtml="".join(f'<div style="display:flex;gap:1.4vw;align-items:flex-start;margin-bottom:1.5vw"><div class="kpi" style="color:{c};font-size:1.9vw">{i+1:02d}</div><div><div style="font-size:1.45vw;font-weight:700;color:#fff;font-family:Space Grotesk,sans-serif">{t}</div><div class="stat-l" style="font-size:1.05vw;margin-top:0.2vw">{d}</div></div></div>' for i,(t,d,c) in enumerate(con))
S.append(slide(f"""
  {kick('Conclusiones')}
  <h1 class="head" style="margin-bottom:2vw">Lo que demuestra el trabajo</h1>
  <div style="z-index:2">{conhtml}</div>
""", page=19))

# 20 GRACIAS
S.append(slide(f"""
  <div style="z-index:2;margin-top:8vw">
    <h1 class="head" style="font-size:6vw">Gracias</h1>
    <div style="font-size:1.9vw;color:var(--teal);font-weight:700;margin-top:1vw;font-family:Space Grotesk,sans-serif">¿Preguntas?</div>
    <div style="font-size:1.1vw;color:var(--mut);margin-top:3vw">Pablo López Domínguez · Grado en Ciencia de Datos · ETSE-UV</div>
  </div>
""", foot=None))

HTML = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Defensa TFG · LCA</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>
{''.join(S)}
<script>
const sl=[...document.querySelectorAll('.slide')];let i=0;
function go(n){{i=Math.max(0,Math.min(sl.length-1,n));sl[i].scrollIntoView({{behavior:'smooth'}});}}
addEventListener('keydown',e=>{{
 if(['ArrowRight','ArrowDown',' ','PageDown'].includes(e.key)){{e.preventDefault();go(i+1);}}
 if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key)){{e.preventDefault();go(i-1);}}
 if(e.key==='Home')go(0);if(e.key==='End')go(sl.length-1);}});
sl.forEach(s=>new IntersectionObserver(es=>es.forEach(x=>{{if(x.isIntersecting)i=sl.indexOf(x.target);}}),{{threshold:.6}}).observe(s));
</script></body></html>"""
pathlib.Path("defensa_TFG.html").write_text(HTML,encoding="utf-8")
print("OK", f"{len(HTML)/1024/1024:.1f} MB", len(S),"slides")

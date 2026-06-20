#!/usr/bin/env python3
import base64, pathlib
import svg_charts as C

A=pathlib.Path("assets")
def b64(p):
    return f"data:image/png;base64,{base64.b64encode((A/p).read_bytes()).decode()}"
IMG={k:b64(v) for k,v in {
 "pipeline":"pipeline_general.png","app":"app_dashboard.png","escudo":"escudo.png",
 "train":"train_curves.png","gradcam":"app_gradcam_planes.png",
 "sag_pos":"mri/sag_pos.png","sag_neg":"mri/sag_neg.png","cor_pos":"mri/cor_pos.png","ax_pos":"mri/ax_pos.png",
 **{f"sel{i}":f"mri/sel_{i}.png" for i in range(5)},
 **{f"ms_{pl}":f"vit_ms_{pl}.png" for pl in ["sagittal","coronal","axial"]},
 **{f"best_{pl}":f"vit_best_{pl}.png" for pl in ["sagittal","coronal","axial"]},
}.items()}

INK="#102A37"; MUT="#3F586A"; FAINT="#6E8493"; LINE="#BCCAD3"; BG="#FFFFFF"; SOFT="#E7F2F0"
TEAL="#0E9488"; TEALD="#0B6E64"; BLUE="#2E7CC4"; SLATE="#64748B"; RED="#DC2626"; AMBER="#D97706"

CSS="""
*{margin:0;padding:0;box-sizing:border-box;-webkit-font-smoothing:antialiased}
html{scroll-snap-type:y mandatory;scroll-behavior:smooth}
body{font-family:'Inter',system-ui,sans-serif;background:#0A1620;color:#102A37}
.slide{position:relative;width:100vw;height:56.25vw;max-height:100vh;scroll-snap-align:center;
  overflow:hidden;display:flex;flex-direction:column;padding:3.8vw 5vw 3vw;background:#FFFFFF}
.slide.soft{background:#EFF6F5}
.kicker{font-family:'Space Grotesk',sans-serif;font-size:0.92vw;font-weight:600;letter-spacing:.2em;
  text-transform:uppercase;color:#0E9488;display:flex;align-items:center;gap:0.7vw;margin-bottom:1.1vw;z-index:2}
.kicker .dot{width:0.55vw;height:0.55vw;border-radius:50%;background:#0E9488}
h1.head{font-family:'Space Grotesk',sans-serif;font-size:2.7vw;font-weight:700;line-height:1.05;
  letter-spacing:-.02em;color:#102A37;max-width:84%;z-index:2}
.sub{font-size:1.18vw;color:#41586A;line-height:1.55;z-index:2}
.foot{position:absolute;left:5vw;bottom:2vw;font-size:0.8vw;font-weight:600;letter-spacing:.05em;color:#6E8493;z-index:2}
.pg{position:absolute;right:5vw;bottom:2vw;font-size:0.8vw;font-weight:700;color:#6E8493;z-index:2}
.row{display:flex;gap:2.4vw;flex:1;min-height:0;z-index:2}
.col{display:flex;flex-direction:column}.grow{flex:1}.center{justify-content:center}
.badge{font-family:'Space Grotesk',sans-serif;display:inline-flex;align-items:center;font-size:0.85vw;font-weight:600;
  color:#0B6E64;background:#D7EEEB;border-radius:2vw;padding:0.35vw 1vw;margin-bottom:0.9vw;width:fit-content;letter-spacing:.04em}
.why{border-left:0.2vw solid #0E9488;padding-left:1vw;font-size:1.18vw;line-height:1.5;color:#102A37}
.why b{color:#0B6E64}
ul.b{list-style:none}ul.b li{font-size:1.05vw;color:#2C4250;line-height:1.45;padding-left:1.4vw;position:relative;margin-bottom:0.85vw}
ul.b li:before{content:"";position:absolute;left:0;top:0.55vw;width:0.5vw;height:0.5vw;border-radius:50%;background:#0E9488}
ul.b li b{color:#102A37}
.stat{font-family:'Space Grotesk',sans-serif;font-size:2.9vw;font-weight:700;color:#0E9488;line-height:1;letter-spacing:-.02em}
.stat-l{font-size:0.92vw;color:#5A7383;margin-top:0.45vw;line-height:1.35}
.card{background:#FFFFFF;border:1.5px solid #B7C7D0;border-radius:0.9vw;padding:1.5vw;
  box-shadow:0 1vw 2.4vw rgba(16,42,55,.10)}
.card.soft{background:#F1F7F8;box-shadow:none;border-color:#BFD2D2}
.tile{border-radius:0.7vw;overflow:hidden;background:#0A1620;box-shadow:0 0.8vw 1.8vw rgba(16,42,55,.22);border:1.5px solid #0A1620;position:relative}
.tile img{width:100%;height:100%;object-fit:cover;display:block}
.tlabel{position:absolute;left:0.5vw;bottom:0.5vw;font-size:0.78vw;font-weight:600;color:#fff;
  background:rgba(10,22,32,.55);padding:0.15vw 0.55vw;border-radius:0.4vw}
.cap{font-size:0.85vw;color:#4D6577;font-weight:500;text-align:center;margin-top:0.5vw}
table.t{width:100%;border-collapse:collapse;font-size:1.05vw}
table.t th{text-align:left;color:#0B6E64;font-weight:700;font-size:0.82vw;letter-spacing:.05em;text-transform:uppercase;
  padding:0.6vw 0.7vw;border-bottom:2px solid #D7EEEB}
table.t td{padding:0.7vw 0.7vw;border-bottom:1px solid #D2DEE4;color:#23394A}
table.t td:first-child{color:#102A37;font-weight:700}
.legend{display:flex;gap:1.4vw;flex-wrap:wrap;margin-top:0.8vw}
.lg{display:flex;align-items:center;gap:0.45vw;font-size:0.95vw;color:#2C4250;font-weight:600}
.lg .sw{width:0.85vw;height:0.85vw;border-radius:0.2vw}
.win{display:flex;gap:0.85vw;align-items:flex-start;margin-bottom:1.2vw}
.win .d{width:0.8vw;height:0.8vw;border-radius:50%;margin-top:0.5vw;flex:none}
.win .l{font-size:0.9vw;color:#5A7383}.win .v{font-family:'Space Grotesk',sans-serif;font-size:1.25vw;font-weight:700;color:#102A37;line-height:1.15}
.kpi{font-family:'Space Grotesk',sans-serif;font-weight:700;letter-spacing:-.02em}
"""

MC={"ResNet50":SLATE,"ViT-Small":TEAL,"Swin-Tiny":BLUE}
def legend(names):
    return '<div class="legend">'+''.join(f'<span class="lg"><span class="sw" style="background:{MC[n]}"></span>{n}</span>' for n in names)+'</div>'

S=[]
def slide(inner,foot="ETSE · Universitat de València",soft=False,nopage=False,**_):
    return {"inner":inner,"foot":foot,"soft":soft,"nopage":nopage}
def kick(t): return f'<div class="kicker"><span class="dot"></span>{t}</div>'
def deci(n): return f'<div class="badge">DECISIÓN {n} · FIJADA EN VALIDACIÓN</div>'

# 1 HERO
S.append(slide(f"""
  <div style="position:absolute;top:-14vw;right:-10vw;width:34vw;height:34vw;border-radius:50%;background:#DCEFEC;z-index:0"></div>
  <div style="position:absolute;bottom:-16vw;right:14vw;width:20vw;height:20vw;border-radius:50%;background:#EAF4F2;z-index:0"></div>
  {kick('Trabajo Fin de Grado · Ciencia de Datos')}
  <h1 class="head" style="font-size:3.6vw;max-width:74%">Detección automática de rotura del LCA en resonancia de rodilla</h1>
  <p class="sub" style="margin-top:1.5vw;max-width:54vw;font-style:italic;color:#0B6E64">CNN · Vision Transformers · Swin Transformers — comparados a igualdad de pipeline, con todas las decisiones tomadas en validación.</p>
  <div style="margin-top:3vw;font-size:1.08vw;line-height:2;z-index:2">
    <div><span style="color:#0E9488;font-weight:700">Autor&nbsp;&nbsp;</span>Pablo López Domínguez</div>
    <div><span style="color:#0E9488;font-weight:700">Tutores&nbsp;&nbsp;</span>José David Martín Guerrero · Yolanda Vives Gilabert</div>
  </div>
  <img src="{IMG['escudo']}" style="position:absolute;right:5vw;bottom:2.6vw;height:10vw;z-index:2">
""", foot=None, soft=True, nopage=True))

# 2 PROBLEMA
S.append(slide(f"""
  {kick('El problema')}
  <h1 class="head">Leer el LCA en RM es lento y un fallo cuesta caro</h1>
  <div class="row" style="margin-top:1.8vw;align-items:center">
    <div class="col center" style="width:42vw;gap:1.6vw">
      <p class="sub" style="max-width:40vw">Una rotura puede verse en muy pocos cortes. Un <b style="color:#102A37">falso negativo</b> retrasa el tratamiento: la prioridad clínica es la <b style="color:#0E9488">sensibilidad</b>.</p>
      <div class="row" style="flex:none;gap:1.4vw">
        <div class="card soft col" style="flex:1"><div class="stat">~21%</div><div class="stat-l">prevalencia de rotura</div></div>
        <div class="card soft col" style="flex:1"><div class="stat">3</div><div class="stat-l">planos por estudio</div></div>
        <div class="card soft col" style="flex:1"><div class="stat">1</div><div class="stat-l">corte puede bastar</div></div>
      </div>
    </div>
    <div class="row" style="gap:1.2vw;flex:1">
      <div class="col" style="flex:1"><div class="tile" style="aspect-ratio:1"><img src="{IMG['sag_pos']}"><span class="tlabel" style="background:rgba(220,38,38,.85)">Rotura de LCA</span></div></div>
      <div class="col" style="flex:1"><div class="tile" style="aspect-ratio:1"><img src="{IMG['sag_neg']}"><span class="tlabel" style="background:rgba(14,148,136,.9)">Rodilla sana</span></div></div>
    </div>
  </div>
  <div class="cap" style="text-align:right">Ejemplos reales del conjunto (plano sagital)</div>
""", page=2))

# 3 DATOS
S.append(slide(f"""
  {kick('Los datos · MRNet')}
  <h1 class="head">1 250 estudios, tres vistas por rodilla</h1>
  <div class="row" style="margin-top:1.6vw;align-items:center">
    <div class="col" style="width:33vw;gap:1vw">
      <table class="t">
        <tr><th>Conjunto</th><th>Estudios</th><th>Rotura</th><th>Prev.</th></tr>
        <tr><td>Entrenamiento</td><td>875</td><td>183</td><td>20.9%</td></tr>
        <tr><td>Validación</td><td>188</td><td>36</td><td>19.1%</td></tr>
        <tr><td>Test</td><td>187</td><td>43</td><td>23.0%</td></tr>
      </table>
      <p class="sub" style="font-size:1.02vw;max-width:31vw;margin-top:0.4vw">Particiones <b style="color:#102A37">a nivel de paciente</b> para evitar fuga de información.</p>
    </div>
    <div class="row grow" style="gap:1.2vw">
      <div class="col" style="flex:1"><div class="tile" style="aspect-ratio:1"><img src="{IMG['sag_pos']}"><span class="tlabel">Sagital</span></div></div>
      <div class="col" style="flex:1"><div class="tile" style="aspect-ratio:1"><img src="{IMG['cor_pos']}"><span class="tlabel">Coronal</span></div></div>
      <div class="col" style="flex:1"><div class="tile" style="aspect-ratio:1"><img src="{IMG['ax_pos']}"><span class="tlabel">Axial</span></div></div>
    </div>
  </div>
  <div class="cap" style="text-align:right">Mismo estudio con rotura de LCA en los tres planos</div>
""", page=3, soft=True))

# 4 EVALUACIÓN HONESTA
cols4=[("Train","875","Aprende los pesos del modelo.",TEAL),
       ("Validación","188","Aquí se toman <b>todas</b> las decisiones y se calibra el umbral.",BLUE),
       ("Test","187","Se mira una sola vez. Estimación final, <b>cero</b> decisiones.",INK),
       ("Croacia","917","Dominio externo: generalización real.","#7C3AED")]
c4="".join(f'<div class="card col" style="flex:1;gap:0.5vw"><div class="kpi" style="font-size:1.6vw;color:{c}">{t}</div><div class="kpi" style="font-size:2.4vw;color:#102A37">{n}</div><div class="stat-l" style="font-size:1vw">{d}</div></div>' for t,n,d,c in cols4)
S.append(slide(f"""
  {kick('Metodología · evaluación honesta')}
  <h1 class="head">El test solo se toca al final</h1>
  <p class="sub" style="margin-top:1vw;max-width:62vw">Cada elección —arquitectura, cortes, regularización, umbral— se fija mirando <b style="color:#0E9488">solo validación</b>. Así el test es una estimación insesgada del rendimiento real.</p>
  <div class="row" style="margin-top:2vw;gap:1.5vw;flex:none;height:16vw">{c4}</div>
""", page=4))

# 5 PIPELINE
S.append(slide(f"""
  {kick('Visión general del sistema')}
  <h1 class="head">Un pipeline end-to-end multi-vista</h1>
  <div class="grow" style="margin-top:1vw;display:flex;align-items:center;justify-content:center;z-index:2">
    <img src="{IMG['pipeline']}" style="max-width:84%;max-height:100%;object-fit:contain">
  </div>
""", page=5))

# 6 DECISIÓN 1 cortes
tiles="".join(f'<div class="col" style="flex:1"><div class="tile" style="aspect-ratio:1"><img src="{IMG[f"sel{i}"]}"></div></div>' for i in range(5))
S.append(slide(f"""
  {kick('Selección de cortes')}{deci(1)}
  <h1 class="head">Un selector aprendido, no cortes al azar</h1>
  <div class="col center grow" style="margin-top:1.2vw;gap:1.4vw">
    <div class="row" style="flex:none;align-items:center;gap:2.4vw">
      <div class="why" style="width:40vw">Procesar todo el volumen añade ruido. Un <b>MobileNetV2</b> puntúa cada corte y selecciona los más informativos — en validación batió a <b>center</b> y <b>random</b>.</div>
      <div class="row" style="flex:none;gap:1vw">
        <div class="card soft col"><div class="kpi" style="font-size:1.4vw;color:#0E9488">K=5</div><div class="stat-l">sagital</div></div>
        <div class="card soft col"><div class="kpi" style="font-size:1.4vw;color:#0E9488">K=10</div><div class="stat-l">coronal</div></div>
        <div class="card soft col"><div class="kpi" style="font-size:1.4vw;color:#0E9488">K=10</div><div class="stat-l">axial</div></div>
      </div>
    </div>
    <div class="row" style="flex:none;gap:1vw;width:100%">{tiles}</div>
    <div class="cap" style="align-self:flex-start">Cortes sagitales seleccionados automáticamente, ordenados por relevancia</div>
  </div>
""", page=6))

# 7 DECISIÓN 2 pooling
S.append(slide(f"""
  {kick('Agregación entre cortes')}{deci(2)}
  <h1 class="head">El pooling se elige plano a plano</h1>
  <div class="row" style="margin-top:1.6vw;align-items:center">
    <div class="why" style="width:38vw">Cada plano tiene una estructura distinta; el mejor modo de combinar los cortes se decidió <b>por plano en validación</b>.</div>
    <div class="col grow center">
      <table class="t" style="font-size:1.25vw">
        <tr><th>Plano</th><th>Cortes</th><th>Agregación</th></tr>
        <tr><td>Sagital</td><td>5</td><td>Attention Pooling</td></tr>
        <tr><td>Coronal</td><td>10</td><td>Max Pooling</td></tr>
        <tr><td>Axial</td><td>10</td><td>Attention Pooling</td></tr>
      </table>
      <p class="cap" style="text-align:left;align-self:flex-start;margin-top:0.8vw">Attention aprende un peso por corte; Max destaca el corte más sospechoso.</p>
    </div>
  </div>
""", page=7, soft=True))

# 8 DECISIÓN 3 regularización (por modelo)
S.append(slide(f"""
  {kick('Regularización')}{deci(3)}
  <h1 class="head">Cada arquitectura, su propia receta</h1>
  <p class="sub" style="margin-top:0.7vw;max-width:64vw">No se aplicó la misma regularización a todos: cada modelo (y cada plano) se ajustó <b style="color:#0B6E64">de forma independiente en validación</b> según su tendencia al sobreajuste.</p>
  <div class="row" style="margin-top:1.4vw;align-items:center">
    <div class="col grow center">
      <table class="t" style="font-size:1.12vw">
        <tr><th>&nbsp;</th><th style="color:{SLATE}">ResNet50</th><th style="color:{TEAL}">ViT-Small</th><th style="color:{BLUE}">Swin-Tiny</th></tr>
        <tr><td>Aumentos</td><td>agresivo</td><td>conservador → agresivo</td><td>agresivo</td></tr>
        <tr><td>Dropout (in/dense)</td><td>0.42–0.47</td><td>0.20–0.25</td><td>0.00–0.42</td></tr>
        <tr><td>Planificador LR</td><td>ReduceLROnPlateau</td><td>Cosine + warmup</td><td>ReduceLROnPlateau</td></tr>
        <tr><td>Early stopping</td><td>10–12</td><td>15–25</td><td>12–40</td></tr>
      </table>
    </div>
    <div class="col center" style="width:24vw">
      <div class="why">Dentro de cada modelo, el plano <b>coronal</b> recibió más regularización (más inestable) y el <b>axial</b> menos.</div>
    </div>
  </div>
""", page=8))

# 9 DECISIÓN 4 early stopping
S.append(slide(f"""
  {kick('Criterio de parada')}{deci(4)}
  <h1 class="head">Se para por AUC de validación, no por la pérdida</h1>
  <div class="row" style="margin-top:1.6vw;align-items:center">
    <div class="col center grow" style="gap:1.4vw">
      <div class="why" style="max-width:44vw">Se conserva el checkpoint del <b>mejor AUC de validación</b>, no la última época. Que la pérdida de train aún baje es lo esperable: parar antes evita el sobreajuste.</div>
      <ul class="b">
        <li>Métrica: <b>AUC de validación</b> (maximización).</li>
        <li>Paciencia por plano: <b>25 / 15 / 15</b> épocas.</li>
        <li>El ruido de las curvas refleja el tamaño de validación (188 estudios).</li>
      </ul>
    </div>
    <div class="card col center" style="width:28vw;background:#EFF6F5;border-color:#D7EEEB">
      <div class="kpi" style="font-size:1.5vw;color:#0B6E64">mejor AUC de validación</div>
      <div class="stat-l" style="font-size:1.05vw;margin-top:0.5vw">= modelo conservado.<br>La cola de la curva es solo el margen de paciencia.</div>
    </div>
  </div>
""", page=9))

# 10 — Multi-semilla: curvas reales por plano (ViT-Small) + mejor checkpoint
def ms_slide(plane, head, note):
    return slide(f"""
      {kick('Estabilidad · multi-semilla')}{deci(5)}
      <h1 class="head">{head}</h1>
      <div class="col grow center" style="margin-top:0.6vw">
        <img src="{IMG['ms_'+plane]}" style="width:94%;max-height:30vw;object-fit:contain;align-self:center">
      </div>
      <p class="sub" style="font-size:1.08vw;max-width:82vw;margin-top:0.4vw">{note}</p>
    """, soft=True)

S.append(ms_slide("sagittal","Plano sagital — 10 semillas (ViT-Small)",
  "Convergencia limpia y AUC de validación alto (media <b style='color:#0B6E64'>0.962</b>). La pérdida de train sigue bajando cuando el early stopping ya ha guardado el mejor checkpoint por AUC."))
S.append(ms_slide("coronal","Plano coronal — la mayor variabilidad",
  "Aquí la <b style='color:#0B6E64'>inicialización importa mucho</b>: las semillas se dispersan (de ~0.80 a ~0.94). Es el plano más difícil y el que más regularización recibió. Justifica entrenar con 10 semillas."))
S.append(ms_slide("axial","Plano axial — el más estable",
  "Todas las semillas convergen casi solapadas (desviación mínima). El mapa transversal del LCA es el más informativo y reproducible."))

# 13 — Mejor checkpoint por plano
S.append(slide(f"""
  {kick('Estabilidad · selección final')}{deci(5)}
  <h1 class="head">El mejor checkpoint de cada plano forma el ensamble</h1>
  <div class="row" style="margin-top:0.5vw">
    <div class="col grow center" style="gap:0.6vw">
      <img src="{IMG['best_sagittal']}" style="max-height:12.5vw;width:auto;align-self:center">
      <img src="{IMG['best_coronal']}" style="max-height:12.5vw;width:auto;align-self:center">
      <img src="{IMG['best_axial']}" style="max-height:12.5vw;width:auto;align-self:center">
    </div>
    <div class="col center" style="width:25vw"><ul class="b">
      <li>Tras las 10 semillas se fija <b>un checkpoint por plano</b>, usando solo validación.</li>
      <li>Mejores semillas: sagital <b>43</b> · coronal <b>51</b> · axial <b>48</b>.</li>
      <li>Estos tres modelos son los que se promedian en el <b>ensamble multi-vista</b>.</li>
    </ul></div>
  </div>
""", soft=True))

# 11 DECISIÓN 6 umbral
S.append(slide(f"""
  {kick('Punto operativo clínico')}{deci(6)}
  <h1 class="head">Umbral calibrado para no perder roturas</h1>
  <div class="row" style="margin-top:1vw;align-items:center">
    <div class="col grow center" style="padding-right:1vw">{C.line_threshold()}
      <div class="legend"><span class="lg"><span class="sw" style="background:{TEAL}"></span>Sensibilidad</span><span class="lg"><span class="sw" style="background:{BLUE}"></span>Precisión</span></div>
    </div>
    <div class="col center" style="width:30vw;gap:1.2vw">
      <div class="why">τ* = máxima <b>sensibilidad</b> con <b>precisión ≥ 0.75</b>, fijado <b>solo en validación</b>.</div>
      <ul class="b"><li>Un falso negativo es el error más caro → priorizamos recall.</li><li>Cada arquitectura tiene su propio umbral.</li></ul>
    </div>
  </div>
""", page=11))

# 12 DECISIÓN 7 ensemble
S.append(slide(f"""
  {kick('Fusión multi-vista')}{deci(7)}
  <h1 class="head">Promedio simple de los tres planos</h1>
  <div class="row" style="margin-top:1.8vw;align-items:center;justify-content:center">
    <div class="why" style="width:44vw">Se promedian las probabilidades de sagital, coronal y axial. <b>Sin parámetros nuevos</b> ni decisiones dependientes del test, y mejora frente a cualquier plano individual.</div>
    <div class="card col center" style="width:34vw;background:#EAF4F2;border-color:#9FD3CD;text-align:center;padding:2.4vw 1.5vw">
      <div class="kpi" style="font-size:1.9vw;color:#102A37">p = ( p<sub style="font-size:0.62em">sag</sub> + p<sub style="font-size:0.62em">cor</sub> + p<sub style="font-size:0.62em">ax</sub> ) / 3</div>
    </div>
  </div>
""", page=12, soft=True))

# 13 RESULTADOS test (tabla)
def win(l,v,c): return f'<div class="win"><span class="d" style="background:{c}"></span><div><div class="l">{l}</div><div class="v">{v}</div></div></div>'
def hl(v,best): return f'<b style="color:#0B6E64">{v}</b>' if best else v
S.append(slide(f"""
  {kick('Resultados · MRNet')}
  <h1 class="head">No hay un ganador absoluto</h1>
  <div class="row" style="margin-top:1.2vw;align-items:center">
    <div class="col grow center">
      <table class="t" style="font-size:1.18vw">
        <tr><th>Modelo (ensamble)</th><th>AUC val</th><th>AUC test</th><th>Sensib.</th><th>Especif.</th><th>F1</th><th>FN</th></tr>
        <tr><td style="color:{SLATE}">ResNet50</td><td>0.9836</td><td>{hl('0.9545',1)}</td><td>0.884</td><td>0.847</td><td>0.738</td><td>5</td></tr>
        <tr><td style="color:{TEAL}">ViT-Small</td><td>0.9837</td><td>0.9433</td><td>0.884</td><td>{hl('0.861',1)}</td><td>{hl('0.753',1)}</td><td>5</td></tr>
        <tr><td style="color:{BLUE}">Swin-Tiny</td><td>0.9837</td><td>0.9536</td><td>{hl('0.907',1)}</td><td>0.840</td><td>0.743</td><td>{hl('4',1)}</td></tr>
      </table>
    </div>
    <div class="col center" style="width:24vw">
      {win('Mayor AUC','ResNet50 · 0.9545',SLATE)}
      {win('Mayor sensibilidad','Swin-Tiny · 4 FN',BLUE)}
      {win('Más equilibrado','ViT-Small · F1 0.753',TEAL)}
    </div>
  </div>
  <p class="sub" style="margin-top:1vw;font-size:1.02vw;color:#41586A">En validación las tres quedan <b style="color:#0B6E64">prácticamente empatadas</b> (0.9836–0.9837, validación de solo 188 casos); las diferencias reales emergen en test, donde cada una destaca en una métrica distinta.</p>
""", page=13))

def cm_slide(title_kick, head, subtxt, cms, page, soft):
    h="".join(f'<div class="col center" style="flex:1"><div style="font-family:Space Grotesk;font-weight:700;font-size:1.25vw;color:{MC[n]};text-align:center;margin-bottom:0.4vw">{n}</div>{C.confusion(tp,fn,fp,vn)}</div>' for n,tp,fn,fp,vn in cms)
    return slide(f"""{kick(title_kick)}<h1 class="head">{head}</h1>
      <p class="sub" style="margin-top:0.6vw;max-width:66vw">{subtxt}</p>
      <div class="row" style="margin-top:1vw;gap:2vw;align-items:center">{h}</div>""", page=page, soft=soft)

# 14 CONFUSIÓN — VALIDACIÓN
S.append(cm_slide("Resultados · validación",
  "Matrices de confusión en validación",
  "Punto operativo calibrado aquí. De 36 roturas, <b style=\"color:#0E9488\">ViT-Small las detecta todas</b> (0 falsos negativos); ResNet50 y Swin-Tiny dejan 1.",
  [("ResNet50",35,1,11,141),("ViT-Small",36,0,12,140),("Swin-Tiny",35,1,11,141)], 14, True))

# 15 CONFUSIÓN — TEST
S.append(cm_slide("Resultados · test MRNet",
  "Matrices de confusión en test",
  "De 43 roturas reales, las tres detectan 38–39. <b style=\"color:#0E9488\">Swin-Tiny deja solo 4 sin detectar</b>, el menor número de falsos negativos.",
  [("ResNet50",38,5,22,122),("ViT-Small",38,5,20,124),("Swin-Tiny",39,4,23,121)], 15, False))

# 15 EL PORQUÉ scatter
S.append(slide(f"""
  {kick('El porqué · hallazgo clave')}
  <h1 class="head">Más parámetros no dan más rendimiento</h1>
  <div class="row" style="margin-top:0.8vw;align-items:center">
    <div class="col grow center" style="padding-right:1vw">
      {C.scatter_params([
        ("",23.5,0.9545,SLATE,'o',0,0,'middle'),("",22,0.9433,TEAL,'o',0,0,'middle'),("",27.8,0.9536,BLUE,'o',0,0,'middle'),
        ("ViT-Base · 86M",86,0.9364,'#94A3B8','s',0,34,'middle'),("Swin-Small · 49M  (no converge)",49,0.50,RED,'x',0,-20,'middle')])}
      {legend(['ResNet50','ViT-Small','Swin-Tiny'])}
    </div>
    <div class="col center" style="width:27vw;gap:1.2vw">
      <div class="card" style="border-color:#D7EEEB"><div class="kpi" style="font-size:1.4vw;color:#0B6E64">ViT-Small &gt; ViT-Base</div><div class="stat-l" style="margin-top:0.4vw">22M iguala a 86M (4× menos).</div></div>
      <div class="card" style="border-color:#F4D0D0"><div class="kpi" style="font-size:1.4vw;color:#DC2626">Swin-Small colapsa</div><div class="stat-l" style="margin-top:0.4vw">49M no converge con datos limitados.</div></div>
    </div>
  </div>
""", page=16))

# 17 EXPLICABILIDAD (mapa de la memoria)
S.append(slide(f"""
  {kick('Interpretabilidad')}
  <h1 class="head">El modelo mira la región del LCA</h1>
  <div class="row" style="margin-top:1vw;align-items:center">
    <div class="grow" style="display:flex;align-items:center;justify-content:center">
      <img src="{IMG['gradcam']}" style="max-width:100%;max-height:38vw;width:auto;display:block;border-radius:0.7vw">
    </div>
    <div class="col center" style="width:23vw"><ul class="b">
      <li>Mapas de saliencia (Grad-CAM) por plano y corte.</li>
      <li>Activación en la <b>región intercondílea</b>, donde se localiza el LCA.</li>
      <li>Transparencia clínica en cada predicción.</li>
    </ul></div>
  </div>
""", page=17))

# 17 CROACIA
S.append(slide(f"""
  {kick('Validación externa · Croacia')}
  <h1 class="head">Otro hospital: aparece el domain shift</h1>
  <p class="sub" style="margin-top:0.7vw;max-width:64vw">Transferencia directa (zero-shot) a 917 estudios de otro hospital. El AUC se mantiene razonable, pero el punto operativo se degrada.</p>
  <div class="row" style="margin-top:1.6vw;align-items:stretch;gap:1.6vw">
    <div class="card col center" style="flex:1"><div class="stat" style="color:{SLATE};font-size:3vw">0.785</div><div class="stat-l" style="font-size:1.05vw">ResNet50 · AUC zero-shot</div></div>
    <div class="card col center" style="flex:1"><div class="stat" style="color:{TEAL};font-size:3vw">0.857</div><div class="stat-l" style="font-size:1.05vw">ViT-Small · AUC zero-shot</div></div>
    <div class="card col center" style="flex:1"><div class="stat" style="color:{BLUE};font-size:3vw">0.882</div><div class="stat-l" style="font-size:1.05vw">Swin-Tiny · AUC zero-shot</div></div>
  </div>
  <div class="row" style="margin-top:1.6vw;align-items:stretch;gap:1.6vw;flex:none">
    <div class="card col center" style="flex:1;border-color:#E6C99A"><div class="kpi" style="font-size:1.4vw;color:#B45309">Cae la sensibilidad</div><div class="stat-l" style="margin-top:0.3vw;font-size:1.02vw">Cambian escáner, población y protocolo de adquisición.</div></div>
    <div class="card col center" style="flex:1;border-color:#9FD3CD"><div class="stat-l" style="font-size:1.02vw">Tras fine-tuning (test de Croacia)</div><div class="kpi" style="font-size:2vw;margin-top:0.15vw"><span style="color:#102A37">0.924</span> <span style="color:#6E8493">→</span> <span style="color:#0E9488">0.936</span></div></div>
  </div>
""", page=18, soft=True))

# 18 APP
S.append(slide(f"""
  {kick('Aplicación')}
  <h1 class="head">Demostrador clínico interactivo</h1>
  <div class="row" style="margin-top:1.8vw;align-items:center">
    <div class="grow" style="display:flex;align-items:center;justify-content:center">
      <img src="{IMG['app']}" style="max-width:100%;max-height:42vw;width:auto;display:block;border-radius:0.9vw;box-shadow:0 1.6vw 3vw rgba(16,42,55,.22);border:1px solid #1E3647">
    </div>
    <div class="col center" style="width:26vw;gap:1vw">
      <div class="card soft"><b style="color:#102A37;font-size:1.1vw">Carga DICOM real</b><div class="stat-l">estudio comprimido → diagnóstico</div></div>
      <div class="card soft"><b style="color:#102A37;font-size:1.1vw">Inferencia en vivo</b><div class="stat-l">selección de cortes + ensamble</div></div>
      <div class="card soft"><b style="color:#102A37;font-size:1.1vw">Explicabilidad · 3 modelos</b><div class="stat-l">mapas de calor · intercambiables</div></div>
    </div>
  </div>
""", page=19))

# 19 CONCLUSIONES
con=[("Sistema fiable","AUC ~0.95 en test, sin segmentación manual.",TEAL),
     ("La arquitectura importa menos que el pipeline","Cada familia gana en una métrica; ninguna domina.",BLUE),
     ("Lo compacto gana","ViT-Small (22M) y Swin-Tiny (28M) baten a sus variantes grandes.",TEAL),
     ("Decisiones en validación","Modelos y umbral fijados sin tocar el test; domain shift mitigado con fine-tuning.","#7C3AED")]
conhtml="".join(f'<div style="display:flex;gap:1.4vw;align-items:flex-start;margin-bottom:1.4vw"><div class="kpi" style="color:{c};font-size:1.9vw">{i+1:02d}</div><div><div style="font-family:Space Grotesk;font-size:1.45vw;font-weight:700;color:#102A37">{t}</div><div class="stat-l" style="font-size:1.05vw;margin-top:0.15vw">{d}</div></div></div>' for i,(t,d,c) in enumerate(con))
S.append(slide(f"""
  {kick('Conclusiones')}
  <h1 class="head" style="margin-bottom:1.8vw">Lo que demuestra el trabajo</h1>
  <div style="z-index:2">{conhtml}</div>
""", page=20, soft=True))

# 20 GRACIAS
S.append(slide(f"""
  <div style="position:absolute;bottom:-16vw;left:-8vw;width:36vw;height:36vw;border-radius:50%;background:#E1F0ED;z-index:0"></div>
  <div style="z-index:2;margin-top:8vw">
    <h1 class="head" style="font-size:5.5vw">Gracias</h1>
    <div style="font-family:Space Grotesk;font-size:1.9vw;color:#0E9488;font-weight:700;margin-top:1vw">¿Preguntas?</div>
    <div style="font-size:1.1vw;color:#5A7383;margin-top:2.6vw">Pablo López Domínguez · Grado en Ciencia de Datos · ETSE-UV</div>
  </div>
""", foot=None, soft=True, nopage=True))

def render(S):
    out=[]; n=0
    for sl in S:
        n+=1
        pg="" if sl["nopage"] else f'<div class="pg">{n:02d}</div>'
        ft=f'<div class="foot">{sl["foot"]}</div>' if sl["foot"] else ""
        cls="slide soft" if sl["soft"] else "slide"
        out.append(f'<section class="{cls}">{sl["inner"]}{ft}{pg}</section>')
    return "".join(out)

HTML=f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Defensa TFG · LCA</title>
<style>__FONTS__
{CSS}</style></head><body>
{render(S)}
<script>
const sl=[...document.querySelectorAll('.slide')];let i=0;
function go(n){{i=Math.max(0,Math.min(sl.length-1,n));sl[i].scrollIntoView({{behavior:'smooth'}});}}
addEventListener('keydown',e=>{{if(['ArrowRight','ArrowDown',' ','PageDown'].includes(e.key)){{e.preventDefault();go(i+1);}}
 if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key)){{e.preventDefault();go(i-1);}}
 if(e.key==='Home')go(0);if(e.key==='End')go(sl.length-1);}});
sl.forEach(s=>new IntersectionObserver(es=>es.forEach(x=>{{if(x.isIntersecting)i=sl.indexOf(x.target);}}),{{threshold:.6}}).observe(s));
</script></body></html>"""
fonts=pathlib.Path("fonts_inline.css").read_text(encoding="utf-8") if pathlib.Path("fonts_inline.css").exists() else ""
HTML=HTML.replace("__FONTS__","\n"+fonts+"\n")
pathlib.Path("defensa_TFG.html").write_text(HTML,encoding="utf-8")
print("OK",f"{len(HTML)/1024/1024:.1f} MB",len(S),"slides")

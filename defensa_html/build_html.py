#!/usr/bin/env python3
import base64, pathlib
A = pathlib.Path("assets")
def b64(name):
    data = (A/name).read_bytes()
    ext = "png"
    return f"data:image/{ext};base64,{base64.b64encode(data).decode()}"

IMG = {n: b64(n) for n in [
    "pipeline_general.png","app_gradcam_planes.png","mri_slices_sagital.png","roc_test.png",
    "escudo.png","app_dashboard.png","chart_test.png","chart_size.png","chart_croatia.png","chart_multiseed.png"]}

# ---------- CSS ----------
CSS = """
:root{
  --ink:#0F2A3B; --navy:#0C2536; --teal:#14B8A6; --teal-d:#0E9384;
  --mut:#5B7385; --line:#E4EBF0; --bg:#FFFFFF; --panel:#F3F8F9; --ice:#CFE0EA;
}
*{margin:0;padding:0;box-sizing:border-box;-webkit-font-smoothing:antialiased}
html{scroll-snap-type:y mandatory;scroll-behavior:smooth}
body{font-family:'Inter',system-ui,sans-serif;background:#0A1822;color:var(--ink)}
.slide{position:relative;width:100vw;height:56.25vw;max-height:100vh;
  scroll-snap-align:center;overflow:hidden;display:flex;flex-direction:column;
  padding:4.2vw 5vw 3.4vw;background:var(--bg)}
.slide.dark{background:var(--navy);color:#fff}
/* ---- type ---- */
.kicker{font-size:0.95vw;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;
  color:var(--teal);display:flex;align-items:center;gap:0.7vw;margin-bottom:1.5vw}
.kicker .dot{width:0.62vw;height:0.62vw;border-radius:50%;background:var(--teal)}
.dark .kicker{color:var(--teal)}
h1.head{font-size:2.55vw;font-weight:800;line-height:1.08;letter-spacing:-0.02em;color:var(--ink);max-width:78%}
.dark h1.head{color:#fff}
.sub{font-size:1.2vw;font-weight:400;color:var(--mut);line-height:1.5;max-width:48vw}
.dark .sub{color:var(--ice)}
.foot{position:absolute;left:5vw;bottom:2.4vw;font-size:0.85vw;font-weight:600;letter-spacing:0.04em;color:var(--mut)}
.dark .foot{color:#7FA0B5}
.pg{position:absolute;right:5vw;bottom:2.4vw;font-size:0.85vw;font-weight:700;color:var(--mut)}
.dark .pg{color:#7FA0B5}
/* ---- components ---- */
.row{display:flex;gap:2vw;flex:1;min-height:0}
.col{display:flex;flex-direction:column}
.grow{flex:1}
.card{background:#fff;border:1px solid var(--line);border-radius:1vw;padding:1.8vw;
  box-shadow:0 1.2vw 2.6vw rgba(12,37,54,.06)}
.card.tint{background:var(--panel);box-shadow:none}
.card.navy{background:#103047;border:none;color:#fff}
.stat{font-size:3vw;font-weight:800;color:var(--teal);line-height:1;letter-spacing:-0.02em}
.stat-l{font-size:0.95vw;color:var(--mut);margin-top:0.6vw;line-height:1.35}
.dark .stat-l{color:var(--ice)}
.chip-num{font-size:2.4vw;font-weight:800;letter-spacing:-0.02em}
.lbl{font-size:1vw;color:var(--mut)}
.imgwrap{background:#fff;border:1px solid var(--line);border-radius:1vw;overflow:hidden;
  box-shadow:0 1.2vw 2.6vw rgba(12,37,54,.07);display:flex;align-items:center;justify-content:center}
.imgwrap img{width:100%;height:100%;object-fit:contain}
.cap{font-size:0.8vw;color:var(--mut);font-style:italic;margin-top:0.7vw;text-align:center}
.mcard{background:#fff;border:1px solid var(--line);border-radius:1vw;padding:1.6vw;flex:1;
  box-shadow:0 1vw 2.2vw rgba(12,37,54,.06);display:flex;flex-direction:column;gap:0.5vw}
.badge{width:2.6vw;height:2.6vw;border-radius:50%;display:flex;align-items:center;justify-content:center;
  color:#fff;font-weight:800;font-size:1.05vw;margin-bottom:0.6vw}
.mcard h3{font-size:1.5vw;font-weight:800;color:var(--ink)}
.mcard p{font-size:0.95vw;color:var(--mut);line-height:1.4}
.mcard .par{font-size:0.95vw;font-weight:700}
.win{display:flex;gap:0.9vw;align-items:flex-start}
.win .wd{width:0.85vw;height:0.85vw;border-radius:50%;margin-top:0.55vw;flex:none}
.win .wl{font-size:0.95vw;color:var(--mut)}
.win .wv{font-size:1.35vw;font-weight:800;color:var(--ink);line-height:1.15}
.bullets li{list-style:none;font-size:1.05vw;color:var(--ink);line-height:1.45;
  padding-left:1.5vw;position:relative;margin-bottom:0.9vw}
.bullets li:before{content:"";position:absolute;left:0;top:0.55vw;width:0.6vw;height:0.6vw;border-radius:50%;background:var(--teal)}
.dark .bullets li{color:#EAF2F6}
"""

# ---------- helpers ----------
def slide(inner, dark=False, page=None, foot="ETSE · Universitat de València"):
    cls = "slide dark" if dark else "slide"
    f = f'<div class="foot">{foot}</div>' if foot else ""
    pg = f'<div class="pg">{page:02d}</div>' if page else ""
    return f'<section class="{cls}">{inner}{f}{pg}</section>'

def kicker(t): return f'<div class="kicker"><span class="dot"></span>{t}</div>'

S = []

# 1 HERO
S.append(slide(f"""
  <div style="position:absolute;top:-12vw;right:-10vw;width:36vw;height:36vw;border-radius:50%;background:#103047"></div>
  <div style="position:absolute;bottom:-14vw;left:-8vw;width:30vw;height:30vw;border-radius:50%;background:#0E3A50"></div>
  <div style="position:relative;z-index:2">
    {kicker('Trabajo Fin de Grado · Ciencia de Datos')}
    <h1 class="head" style="font-size:3.6vw;max-width:74%;color:#fff">Detección automática de rotura del LCA en resonancia de rodilla</h1>
    <p class="sub" style="margin-top:1.6vw;max-width:52vw;font-style:italic">Comparación de CNN, Vision Transformers y Swin Transformers a igualdad de pipeline</p>
    <div style="margin-top:3.4vw;font-size:1.1vw;line-height:2">
      <div><span style="color:var(--teal);font-weight:700">Autor&nbsp;&nbsp;</span><span style="color:#fff">Pablo López Domínguez</span></div>
      <div><span style="color:var(--teal);font-weight:700">Tutores&nbsp;&nbsp;</span><span style="color:#fff">José David Martín Guerrero · Yolanda Vives Gilabert</span></div>
    </div>
  </div>
  <img src="{IMG['escudo.png']}" style="position:absolute;right:5vw;bottom:3vw;height:11vw;z-index:2;opacity:.95">
""", dark=True, foot=None))

# 2 PROBLEMA
stats = [("~21%","prevalencia de rotura del LCA"),("3 planos","sagital · coronal · axial"),("min.","de lectura experta / estudio")]
stat_html = "".join(f'<div class="card tint col" style="flex:1;justify-content:center;align-items:flex-start"><div class="stat">{a}</div><div class="stat-l">{b}</div></div>' for a,b in stats)
S.append(slide(f"""
  {kicker('Contexto clínico')}
  <h1 class="head">Diagnosticar el LCA en RM exige tiempo y experiencia</h1>
  <div class="row" style="margin-top:2.2vw">
    <div class="col grow" style="gap:1.4vw;justify-content:flex-start">
      <p class="sub" style="max-width:38vw">Un falso negativo retrasa el tratamiento. El objetivo: un apoyo automático, fiable y explicable.</p>
      <div class="row" style="flex:none;gap:1.4vw;height:11vw">{stat_html}</div>
    </div>
    <div class="col" style="width:34vw">
      <div class="imgwrap" style="flex:1"><img src="{IMG['mri_slices_sagital.png']}"></div>
      <div class="cap">Cortes sagitales de un estudio real de RM</div>
    </div>
  </div>
""", page=2))

# 3 OBJETIVO
ms=[("RN","ResNet50","CNN residual","23.5 M","#0EA5E9"),
    ("VS","ViT-Small","Transformer global","22 M","#14B8A6"),
    ("ST","Swin-Tiny","Transformer jerárquico","27.8 M","#7C3AED")]
mcards="".join(f'<div class="mcard"><div class="badge" style="background:{c}">{ab}</div><h3>{n}</h3><p>{d}</p><div class="par" style="color:{c}">{p}</div></div>' for ab,n,d,p,c in ms)
S.append(slide(f"""
  {kicker('Objetivo')}
  <h1 class="head">¿Qué arquitectura conviene a igualdad de condiciones?</h1>
  <p class="sub" style="margin-top:1.2vw;max-width:62vw">Mismo selector de cortes, misma cabeza de atención, mismo protocolo multi-semilla — para aislar el efecto de la arquitectura.</p>
  <div class="row" style="margin-top:2.4vw;gap:1.8vw">{mcards}</div>
""", page=3))

# 4 DATOS
S.append(slide(f"""
  {kicker('Datos')}
  <h1 class="head">Dos dominios: interno y validación externa</h1>
  <div class="row" style="margin-top:2.2vw;gap:2vw">
    <div class="card col grow" style="gap:1.1vw">
      <div style="font-size:1.6vw;font-weight:800;color:#0EA5E9">MRNet · Stanford</div>
      <div class="lbl">Dominio principal — train / val / test</div>
      <div style="display:flex;gap:2.4vw;margin-top:0.8vw">
        <div><div class="chip-num" style="color:var(--ink)">1 250</div><div class="lbl">estudios</div></div>
        <div><div class="chip-num" style="color:var(--ink)">875·188·187</div><div class="lbl">train·val·test</div></div>
        <div><div class="chip-num" style="color:var(--ink)">~21%</div><div class="lbl">prevalencia</div></div>
      </div>
    </div>
    <div class="card navy col grow" style="gap:1.1vw">
      <div style="font-size:1.6vw;font-weight:800;color:var(--teal)">Croacia · H. Rijeka</div>
      <div class="lbl" style="color:var(--ice)">Validación externa — otro escáner</div>
      <div style="display:flex;gap:2.4vw;margin-top:0.8vw">
        <div><div class="chip-num" style="color:#fff">917</div><div class="lbl" style="color:var(--ice)">estudios</div></div>
        <div><div class="chip-num" style="color:#fff">690·227</div><div class="lbl" style="color:var(--ice)">sano·roto</div></div>
        <div><div class="chip-num" style="color:#fff">zero-shot</div><div class="lbl" style="color:var(--ice)">+ fine-tuning</div></div>
      </div>
    </div>
  </div>
""", page=4))

# 5 PIPELINE
S.append(slide(f"""
  {kicker('Sistema propuesto')}
  <h1 class="head">Pipeline end-to-end multi-vista</h1>
  <div class="imgwrap" style="flex:1;margin-top:1.6vw;background:#fff"><img src="{IMG['pipeline_general.png']}" style="object-fit:contain;padding:1vw"></div>
""", page=5))

# 6 MULTISEED
S.append(slide(f"""
  {kicker('Resultados · robustez')}
  <h1 class="head">Análisis multi-semilla (10 inicializaciones)</h1>
  <div class="row" style="margin-top:1.6vw;gap:2.4vw">
    <div class="imgwrap col" style="flex:1.4;background:#fff;padding:1vw"><img src="{IMG['chart_multiseed.png']}"></div>
    <div class="col" style="width:28vw;justify-content:center">
      <ul class="bullets">
        <li>El plano <b>axial</b> es el más estable en las 3 familias.</li>
        <li>El <b>coronal</b> concentra la variabilidad.</li>
        <li>Una sola ejecución engaña: el multi-semilla es imprescindible.</li>
      </ul>
    </div>
  </div>
""", page=6))

# 7 COMPARATIVA TEST
wins=[("Mayor AUC","ResNet50 · 0.954","#0EA5E9"),
      ("Mayor sensibilidad","Swin-Tiny · 0.907 · 4 FN","#7C3AED"),
      ("Más equilibrado","ViT-Small · F1 0.753","#14B8A6")]
winhtml="".join(f'<div class="win"><span class="wd" style="background:{c}"></span><div><div class="wl">{l}</div><div class="wv">{v}</div></div></div>' for l,v,c in wins)
S.append(slide(f"""
  {kicker('Resultados · test MRNet')}
  <h1 class="head">No hay un ganador absoluto</h1>
  <div class="row" style="margin-top:1.6vw;gap:2.4vw">
    <div class="imgwrap col" style="flex:1.5;background:#fff;padding:1vw"><img src="{IMG['chart_test.png']}"></div>
    <div class="col" style="width:26vw;justify-content:center;gap:1.6vw">{winhtml}</div>
  </div>
""", page=7))

# 8 ROC
aucs=[("0.954","ResNet50","#0EA5E9"),("0.943","ViT-Small","#14B8A6"),("0.954","Swin-Tiny","#7C3AED")]
auchtml="".join(f'<div style="display:flex;align-items:baseline;gap:1vw;margin-bottom:1.1vw"><div style="font-size:2.6vw;font-weight:800;color:{c}">{a}</div><div class="lbl">AUC test · {n}</div></div>' for a,n,c in aucs)
S.append(slide(f"""
  {kicker('Resultados · discriminación')}
  <h1 class="head">Capacidad discriminativa: ROC en test</h1>
  <div class="row" style="margin-top:1.6vw;gap:2.6vw">
    <div class="col" style="width:34vw"><div class="imgwrap" style="flex:1;background:#fff;padding:1vw"><img src="{IMG['roc_test.png']}"></div></div>
    <div class="col grow" style="justify-content:center">
      <p class="sub" style="max-width:32vw;margin-bottom:2vw">Las tres arquitecturas rondan un AUC de <b style="color:var(--ink)">0.95</b> en test.</p>
      {auchtml}
    </div>
  </div>
""", page=8))

# 9 HALLAZGO (dark)
S.append(slide(f"""
  {kicker('Hallazgo clave')}
  <h1 class="head">Más parámetros no implica más rendimiento</h1>
  <div class="row" style="margin-top:1.6vw;gap:2.4vw">
    <div class="imgwrap col" style="flex:1.4;background:#fff;padding:1vw"><img src="{IMG['chart_size.png']}"></div>
    <div class="col" style="width:28vw;justify-content:center;gap:1.6vw">
      <div class="card navy"><div style="font-size:1.5vw;font-weight:800;color:var(--teal)">ViT-Small &gt; ViT-Base</div><p class="stat-l" style="font-size:1vw;margin-top:0.6vw">Iguala a un modelo 4× mayor (86M) con solo 22M.</p></div>
      <div class="card" style="background:#3A1E2B;border:none"><div style="font-size:1.5vw;font-weight:800;color:#FCA5A5">Swin-Small no converge</div><p class="stat-l" style="font-size:1vw;margin-top:0.6vw;color:#F3D4D4">La variante de 49M colapsa al azar con datos limitados.</p></div>
    </div>
  </div>
""", dark=True, page=9))

# 10 EXPLICABILIDAD
S.append(slide(f"""
  {kicker('Interpretabilidad')}
  <h1 class="head">El modelo mira donde debe</h1>
  <div class="row" style="margin-top:1.4vw;gap:2.4vw">
    <div class="imgwrap col" style="flex:1.7;background:#fff;padding:0.8vw"><img src="{IMG['app_gradcam_planes.png']}"></div>
    <div class="col" style="width:24vw;justify-content:center">
      <ul class="bullets">
        <li>Mapas de saliencia sobre los cortes de los 3 planos.</li>
        <li>La activación se concentra en la <b>región intercondílea</b> del LCA.</li>
        <li>Transparencia clínica en cada predicción.</li>
      </ul>
    </div>
  </div>
""", page=10))

# 11 CROACIA
S.append(slide(f"""
  {kicker('Resultados · generalización')}
  <h1 class="head">Validación externa: existe domain shift</h1>
  <div class="row" style="margin-top:1.6vw;gap:2.4vw">
    <div class="imgwrap col" style="flex:1.3;background:#fff;padding:1vw"><img src="{IMG['chart_croatia.png']}"></div>
    <div class="col" style="width:28vw;justify-content:center;gap:1.6vw">
      <div class="card tint"><div style="font-size:1.4vw;font-weight:800;color:#D97706">Cae la sensibilidad</div><p class="stat-l" style="font-size:1vw;margin-top:0.5vw">Cambian escáner, población y protocolo; el AUC aguanta.</p></div>
      <div class="card navy"><div class="lbl" style="color:var(--ice)">Tras fine-tuning (Croacia)</div><div style="font-size:2.2vw;font-weight:800;margin-top:0.4vw"><span style="color:#fff">0.924</span> <span style="color:var(--ice)">→</span> <span style="color:var(--teal)">0.936</span></div></div>
    </div>
  </div>
""", page=11))

# 12 APP
feats=[("Carga DICOM real","Estudio comprimido → diagnóstico"),
       ("Inferencia en vivo","Selección de cortes + ensamble"),
       ("Explicabilidad","Mapas de calor por corte"),
       ("3 modelos","intercambiables")]
fhtml="".join(f'<div class="card" style="padding:1.1vw 1.3vw"><div style="font-size:1.15vw;font-weight:800;color:var(--ink)">{t}</div><div class="lbl" style="margin-top:0.2vw">{d}</div></div>' for t,d in feats)
S.append(slide(f"""
  {kicker('Aplicación')}
  <h1 class="head">Demostrador clínico interactivo</h1>
  <div class="row" style="margin-top:1.4vw;gap:2.4vw">
    <div class="imgwrap col" style="flex:1.5;background:#0A1822;padding:0.6vw"><img src="{IMG['app_dashboard.png']}"></div>
    <div class="col" style="width:26vw;justify-content:center;gap:1.1vw">{fhtml}</div>
  </div>
""", page=12))

# 13 CONCLUSIONES (dark)
con=[("Sistema fiable","AUC ~0.95 en test, sin segmentación manual."),
     ("La arquitectura importa menos que el pipeline","Cada familia gana en una métrica."),
     ("Lo compacto gana","ViT-Small (22M) y Swin-Tiny (28M) baten a sus variantes grandes."),
     ("Generalización con matices","Domain shift real; el fine-tuning lo mitiga.")]
conhtml="".join(f'<div style="display:flex;gap:1.4vw;align-items:flex-start;margin-bottom:1.5vw"><div style="font-size:2vw;font-weight:800;color:var(--teal);line-height:1">{i+1:02d}</div><div><div style="font-size:1.5vw;font-weight:800;color:#fff">{t}</div><div class="stat-l" style="font-size:1.05vw;margin-top:0.2vw">{d}</div></div></div>' for i,(t,d) in enumerate(con))
S.append(slide(f"""
  <div style="position:absolute;top:-12vw;right:-8vw;width:30vw;height:30vw;border-radius:50%;background:#0E3A50"></div>
  {kicker('Conclusiones')}
  <h1 class="head" style="color:#fff;margin-bottom:2.2vw">Conclusiones</h1>
  <div style="position:relative;z-index:2">{conhtml}</div>
""", dark=True, page=13))

# 14 GRACIAS (dark)
S.append(slide(f"""
  <div style="position:absolute;bottom:-16vw;left:-8vw;width:38vw;height:38vw;border-radius:50%;background:#103047"></div>
  <div style="position:relative;z-index:2;margin-top:9vw">
    <h1 class="head" style="font-size:5.5vw;color:#fff">Gracias</h1>
    <div style="font-size:1.8vw;color:var(--teal);font-weight:700;margin-top:1vw">¿Preguntas?</div>
    <div style="font-size:1.1vw;color:var(--ice);margin-top:3vw">Pablo López Domínguez · Grado en Ciencia de Datos · ETSE-UV</div>
  </div>
""", dark=True, foot=None))

HTML = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Defensa TFG · Detección de LCA</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>
{''.join(S)}
<script>
const slides=[...document.querySelectorAll('.slide')];let i=0;
function go(n){{i=Math.max(0,Math.min(slides.length-1,n));slides[i].scrollIntoView({{behavior:'smooth'}});}}
addEventListener('keydown',e=>{{
  if(['ArrowRight','ArrowDown',' ','PageDown'].includes(e.key)){{e.preventDefault();go(i+1);}}
  if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key)){{e.preventDefault();go(i-1);}}
  if(e.key==='Home')go(0); if(e.key==='End')go(slides.length-1);
}});
new IntersectionObserver(es=>es.forEach(x=>{{if(x.isIntersecting)i=slides.indexOf(x.target);}}),{{threshold:.6}}).observe;
slides.forEach(s=>new IntersectionObserver(es=>es.forEach(x=>{{if(x.isIntersecting)i=slides.indexOf(x.target);}}),{{threshold:.6}}).observe(s));
</script></body></html>"""

out = pathlib.Path("defensa_TFG.html")
out.write_text(HTML, encoding="utf-8")
print("OK:", out, f"{len(HTML)/1024/1024:.1f} MB")

const pptx = require("pptxgenjs");
const p = new pptx();
p.layout = "LAYOUT_WIDE";            // 13.33 x 7.5
p.author = "Pablo López Domínguez";
p.title = "Detección de rotura del LCA en RM de rodilla";

const W = 13.33, H = 7.5;
// Paleta clínica premium
const NAVY = "0E2A3B", INK = "14293A", MUT = "64748B", LIGHT = "FFFFFF",
      PANEL = "F2F6F8", TEAL = "14B8A6", SKY = "0EA5E9", SLATE = "94A3B8",
      AMBER = "F59E0B", RED = "EF4444", GREEN = "10B981", LINE = "E2E8F0";
const FH = "Cambria", FB = "Calibri";   // serif headers + sans body (safe fonts)
const sh = () => ({ type: "outer", color: "0E2A3B", blur: 9, offset: 3, angle: 90, opacity: 0.18 });

// ---------- helpers ----------
function pageNum(s, n){
  s.addText(String(n).padStart(2,"0"), { x: W-1.0, y: H-0.5, w: 0.6, h: 0.3,
    fontFace: FB, fontSize: 9, color: MUT, align: "right" });
}
function kicker(s, t, color){
  s.addShape(p.shapes.OVAL, { x: 0.62, y: 0.66, w: 0.16, h: 0.16, fill: { color: color||TEAL } });
  s.addText(t.toUpperCase(), { x: 0.86, y: 0.5, w: 9, h: 0.45, fontFace: FB, fontSize: 12.5,
    bold: true, color: color||TEAL, charSpacing: 2, valign: "middle" });
}
function title(s, t){
  s.addText(t, { x: 0.6, y: 0.95, w: 12.1, h: 0.95, fontFace: FH, fontSize: 30, bold: true,
    color: INK, valign: "middle", margin: 0 });
}
function card(s, x, y, w, h, fill){
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.09,
    fill: { color: fill||LIGHT }, line: { color: LINE, width: 1 }, shadow: sh() });
}
function img(s, path, x, y, w, h){            // rounded-ish framed image on white card
  card(s, x-0.12, y-0.12, w+0.24, h+0.24, LIGHT);
  s.addImage({ path, x, y, w, h });
}

// =================================================================
// 1 · PORTADA (dark)
// =================================================================
let s = p.addSlide(); s.background = { color: NAVY };
s.addShape(p.shapes.OVAL, { x: -2.2, y: -2.6, w: 6, h: 6, fill: { color: "13405A" } });
s.addShape(p.shapes.OVAL, { x: W-3.0, y: H-3.0, w: 5.5, h: 5.5, fill: { color: "12506A" } });
s.addImage({ path: "img/etse_titulo.png", x: 0.7, y: 0.55, w: 2.2, h: 2.2*64/373 });
s.addText("TRABAJO FIN DE GRADO  ·  GRADO EN CIENCIA DE DATOS", { x: 0.75, y: 2.0, w: 11, h: 0.4,
  fontFace: FB, fontSize: 13, color: TEAL, bold: true, charSpacing: 2 });
s.addText("Detección automática de rotura del\nligamento cruzado anterior en RM de rodilla",
  { x: 0.7, y: 2.5, w: 10.6, h: 1.9, fontFace: FH, fontSize: 34, bold: true, color: "FFFFFF", lineSpacing: 38 });
s.addText("Comparación de CNN, Vision Transformers y Swin Transformers a igualdad de pipeline",
  { x: 0.75, y: 4.45, w: 9.5, h: 0.6, fontFace: FB, fontSize: 16, color: "CADCFC", italic: true });
s.addText([
  { text: "Autor   ", options: { color: TEAL, bold: true } }, { text: "Pablo López Domínguez", options: { color: "FFFFFF" } },
], { x: 0.75, y: 5.7, w: 11, h: 0.35, fontFace: FB, fontSize: 14 });
s.addText([
  { text: "Tutores   ", options: { color: TEAL, bold: true } },
  { text: "José David Martín Guerrero   ·   Yolanda Vives Gilabert", options: { color: "FFFFFF" } },
], { x: 0.75, y: 6.1, w: 12, h: 0.35, fontFace: FB, fontSize: 14 });
s.addImage({ path: "img/escudo_logo_UV_narrow.png", x: 11.6, y: 5.1, w: 1.2, h: 1.2*780/600 });

// =================================================================
// 2 · PROBLEMA (light) — stat callouts
// =================================================================
s = p.addSlide(); s.background = { color: LIGHT };
kicker(s, "Contexto clínico"); title(s, "El reto: diagnosticar el LCA en resonancia");
s.addText("La rotura del ligamento cruzado anterior es una de las lesiones de rodilla más frecuentes. Su lectura en RM exige experiencia y tiempo, y un falso negativo retrasa el tratamiento.",
  { x: 0.6, y: 1.95, w: 6.0, h: 1.4, fontFace: FB, fontSize: 16, color: INK, lineSpacing: 24 });
const stats = [["~21%","prevalencia de rotura\nen los estudios"],["3 planos","sagital · coronal · axial\npor estudio"],["min.","de lectura experta\npor resonancia"]];
stats.forEach((v,i)=>{
  const x = 0.6 + i*2.05;
  card(s, x, 3.6, 1.9, 1.7, PANEL);
  s.addText(v[0], { x, y: 3.75, w: 1.9, h: 0.8, fontFace: FH, fontSize: 30, bold: true, color: TEAL, align: "center" });
  s.addText(v[1], { x, y: 4.55, w: 1.9, h: 0.65, fontFace: FB, fontSize: 11, color: MUT, align: "center", lineSpacing: 13 });
});
img(s, "img/mri_slices_sagital.png", 7.0, 2.4, 5.9, 5.9*448/2125);
s.addText("Cortes sagitales de un estudio real de RM de rodilla",
  { x: 7.0, y: 4.0, w: 5.9, h: 0.3, fontFace: FB, fontSize: 11, italic: true, color: MUT, align: "center" });
pageNum(s,2);

// =================================================================
// 3 · OBJETIVO (light) — 3 model chips
// =================================================================
s = p.addSlide(); s.background = { color: LIGHT };
kicker(s, "Objetivo"); title(s, "¿Qué arquitectura conviene, a igualdad de condiciones?");
s.addText("Se comparan tres familias representativas de visión por computador bajo el MISMO pipeline (mismo selector de cortes, misma cabeza de atención, mismo protocolo multi-semilla), para aislar el efecto de la arquitectura.",
  { x: 0.6, y: 1.9, w: 12.1, h: 0.95, fontFace: FB, fontSize: 16, color: INK, lineSpacing: 24 });
const models = [
  ["ResNet50","Red convolucional residual","23.5 M parámetros", SKY],
  ["ViT-Small","Transformer de atención global","22 M parámetros", TEAL],
  ["Swin-Tiny","Transformer jerárquico (ventanas)","27.8 M parámetros", "7C3AED"],
];
models.forEach((m,i)=>{
  const x = 0.7 + i*4.05;
  card(s, x, 3.4, 3.7, 2.6, LIGHT);
  s.addShape(p.shapes.OVAL, { x: x+0.35, y: 3.75, w: 0.95, h: 0.95, fill: { color: m[3] } });
  s.addText(m[0].slice(0,2), { x: x+0.35, y: 3.75, w: 0.95, h: 0.95, fontFace: FH, fontSize: 20, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
  s.addText(m[0], { x: x+0.35, y: 4.85, w: 3.0, h: 0.5, fontFace: FH, fontSize: 22, bold: true, color: INK });
  s.addText(m[1], { x: x+0.35, y: 5.32, w: 3.1, h: 0.5, fontFace: FB, fontSize: 13, color: MUT });
  s.addText(m[2], { x: x+0.35, y: 5.66, w: 3.1, h: 0.3, fontFace: FB, fontSize: 12, bold: true, color: m[3] });
});
pageNum(s,3);

// =================================================================
// 4 · DATOS (light) — callouts + split
// =================================================================
s = p.addSlide(); s.background = { color: LIGHT };
kicker(s, "Datos"); title(s, "Dos dominios: interno y validación externa");
// MRNet card
card(s, 0.6, 1.95, 6.0, 4.7, LIGHT);
s.addText("MRNet  ·  Stanford", { x: 0.95, y: 2.2, w: 5.4, h: 0.5, fontFace: FH, fontSize: 22, bold: true, color: SKY });
s.addText("Dominio principal — entrenamiento, validación y test", { x: 0.95, y: 2.75, w: 5.4, h: 0.35, fontFace: FB, fontSize: 13, color: MUT });
[["1 250","estudios totales"],["875 / 188 / 187","train / val / test (por paciente)"],["~21%","prevalencia de rotura"]].forEach((v,i)=>{
  s.addText(v[0], { x: 0.95, y: 3.35+i*1.0, w: 2.7, h: 0.6, fontFace: FH, fontSize: 21, bold: true, color: INK, valign: "middle" });
  s.addText(v[1], { x: 3.7, y: 3.5+i*1.0, w: 2.75, h: 0.6, fontFace: FB, fontSize: 12.5, color: MUT, valign: "middle" });
});
// Croacia card
card(s, 6.9, 1.95, 6.0, 4.7, NAVY);
s.addText("Croacia  ·  H. Rijeka", { x: 7.25, y: 2.2, w: 5.4, h: 0.5, fontFace: FH, fontSize: 22, bold: true, color: TEAL });
s.addText("Validación externa — otro escáner y población", { x: 7.25, y: 2.75, w: 5.4, h: 0.35, fontFace: FB, fontSize: 13, color: "CADCFC" });
[["917","estudios externos"],["690 / 227","sin rotura / con rotura"],["zero-shot","+ adaptación (fine-tuning)"]].forEach((v,i)=>{
  s.addText(v[0], { x: 7.25, y: 3.35+i*1.0, w: 2.4, h: 0.6, fontFace: FH, fontSize: 21, bold: true, color: "FFFFFF", valign: "middle" });
  s.addText(v[1], { x: 9.7, y: 3.5+i*1.0, w: 2.9, h: 0.6, fontFace: FB, fontSize: 12.5, color: "CADCFC", valign: "middle" });
});
pageNum(s,4);

// =================================================================
// 5 · PIPELINE (light) — diagram big
// =================================================================
s = p.addSlide(); s.background = { color: LIGHT };
kicker(s, "Sistema propuesto"); title(s, "Pipeline end-to-end multi-vista");
img(s, "img/pipeline_general.png", 2.61, 2.15, 8.11, 4.95);
pageNum(s,5);

// =================================================================
// 6 · ESTABILIDAD MULTI-SEMILLA (light) — mean AUC bar + note
// =================================================================
s = p.addSlide(); s.background = { color: LIGHT };
kicker(s, "Resultados · robustez"); title(s, "Análisis multi-semilla (10 inicializaciones)");
s.addChart(p.charts.BAR, [
  { name: "Sagital",  labels: ["ResNet50","ViT-Small","Swin-Tiny"], values: [0.943,0.962,0.922] },
  { name: "Coronal",  labels: ["ResNet50","ViT-Small","Swin-Tiny"], values: [0.939,0.874,0.775] },
  { name: "Axial",    labels: ["ResNet50","ViT-Small","Swin-Tiny"], values: [0.961,0.953,0.956] },
], { x: 0.6, y: 2.0, w: 7.4, h: 4.6, barDir: "col", chartColors: [SKY, TEAL, "7C3AED"],
  valAxisMinVal: 0.5, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1,
  catAxisLabelColor: MUT, valAxisLabelColor: MUT, catAxisLabelFontSize: 12, valAxisLabelFontSize: 10,
  valGridLine: { color: LINE, size: 0.5 }, catGridLine: { style: "none" },
  showLegend: true, legendPos: "b", legendColor: INK, legendFontSize: 12,
  showTitle: true, title: "AUC medio de validación por plano", titleColor: INK, titleFontSize: 13, titleFontFace: FB,
  chartArea: { fill: { color: "FFFFFF" } } });
card(s, 8.4, 2.3, 4.4, 3.9, PANEL);
s.addText("Lectura", { x: 8.7, y: 2.5, w: 3.8, h: 0.4, fontFace: FH, fontSize: 16, bold: true, color: TEAL });
s.addText([
  { text: "El plano axial es el más estable en las 3 familias.", options: { bullet: true, breakLine: true } },
  { text: "El coronal concentra la variabilidad (peor en Swin-Tiny).", options: { bullet: true, breakLine: true } },
  { text: "Una sola ejecución puede engañar: el multi-semilla es imprescindible.", options: { bullet: true } },
], { x: 8.7, y: 3.0, w: 3.85, h: 3.0, fontFace: FB, fontSize: 13.5, color: INK, lineSpacing: 19, paraSpaceAfter: 8 });
pageNum(s,6);

// =================================================================
// 7 · COMPARATIVA EN TEST (light) — grouped chart + winners
// =================================================================
s = p.addSlide(); s.background = { color: LIGHT };
kicker(s, "Resultados · test MRNet"); title(s, "No hay un ganador absoluto");
s.addChart(p.charts.BAR, [
  { name: "ResNet50",  labels: ["AUC","Sensib.","Especif.","F1"], values: [0.954,0.884,0.847,0.738] },
  { name: "ViT-Small", labels: ["AUC","Sensib.","Especif.","F1"], values: [0.943,0.884,0.861,0.753] },
  { name: "Swin-Tiny", labels: ["AUC","Sensib.","Especif.","F1"], values: [0.954,0.907,0.840,0.743] },
], { x: 0.6, y: 2.0, w: 7.7, h: 4.7, barDir: "col", chartColors: [SLATE, TEAL, SKY],
  valAxisMinVal: 0.6, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1,
  catAxisLabelColor: INK, valAxisLabelColor: MUT, catAxisLabelFontSize: 13, valAxisLabelFontSize: 10,
  valGridLine: { color: LINE, size: 0.5 }, catGridLine: { style: "none" },
  showLegend: true, legendPos: "b", legendColor: INK, legendFontSize: 13,
  chartArea: { fill: { color: "FFFFFF" } } });
const winners = [
  ["Mayor AUC", "ResNet50 · 0.954", SKY],
  ["Mayor sensibilidad", "Swin-Tiny · 0.907  (solo 4 FN)", "7C3AED"],
  ["Más equilibrado (F1, precisión, especif.)", "ViT-Small", TEAL],
];
winners.forEach((wn,i)=>{
  const y = 2.15 + i*1.55;
  card(s, 8.65, y, 4.2, 1.35, LIGHT);
  s.addShape(p.shapes.OVAL, { x: 8.9, y: y+0.5, w: 0.32, h: 0.32, fill: { color: wn[2] } });
  s.addText(wn[0], { x: 9.35, y: y+0.2, w: 3.4, h: 0.55, fontFace: FB, fontSize: 12.5, color: MUT, valign: "middle" });
  s.addText(wn[1], { x: 9.35, y: y+0.68, w: 3.4, h: 0.5, fontFace: FH, fontSize: 16, bold: true, color: INK });
});
pageNum(s,7);

// =================================================================
// 8 · CURVAS ROC (light) — figure + AUC callouts
// =================================================================
s = p.addSlide(); s.background = { color: LIGHT };
kicker(s, "Resultados · discriminación"); title(s, "Capacidad discriminativa: curvas ROC (test)");
img(s, "img/SWIN_ROC_test.png", 0.8, 2.1, 5.6, 5.6*790/989);
s.addText("Ensamble multi-vista — Swin-Tiny", { x: 0.8, y: 6.7, w: 5.6, h: 0.3, fontFace: FB, fontSize: 11, italic: true, color: MUT, align: "center" });
s.addText("Las tres arquitecturas alcanzan un AUC de test en torno a 0.95: el sistema discrimina con fiabilidad la presencia de rotura.",
  { x: 6.9, y: 2.2, w: 5.9, h: 1.3, fontFace: FB, fontSize: 16, color: INK, lineSpacing: 24 });
[["0.954","ResNet50", SKY],["0.943","ViT-Small", TEAL],["0.954","Swin-Tiny","7C3AED"]].forEach((v,i)=>{
  const y = 3.9 + i*0.95;
  s.addText(v[0], { x: 6.9, y, w: 1.7, h: 0.8, fontFace: FH, fontSize: 30, bold: true, color: v[2] });
  s.addText("AUC de test  ·  "+v[1], { x: 8.7, y: y+0.05, w: 4.1, h: 0.7, fontFace: FB, fontSize: 14, color: INK, valign: "middle" });
});
pageNum(s,8);

// =================================================================
// 9 · HALLAZGO CLAVE: el tamaño no importa (dark)
// =================================================================
s = p.addSlide(); s.background = { color: NAVY };
kicker(s, "Hallazgo clave", TEAL);
s.addText("Más parámetros no implica más rendimiento", { x: 0.6, y: 0.95, w: 12.1, h: 0.95, fontFace: FH, fontSize: 30, bold: true, color: "FFFFFF", valign: "middle" });
s.addChart(p.charts.BAR, [
  { name: "AUC de test", labels: ["ViT-Small\n22M","ViT-Base\n86M","Swin-Tiny\n28M","Swin-Small\n49M"], values: [0.943, 0.936, 0.954, 0.50] },
], { x: 0.6, y: 2.0, w: 7.7, h: 4.7, barDir: "col", chartColors: [TEAL],
  valAxisMinVal: 0.45, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1,
  catAxisLabelColor: "CADCFC", valAxisLabelColor: "9FB3C8", catAxisLabelFontSize: 12, valAxisLabelFontSize: 10,
  valGridLine: { color: "1E4258", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelColor: "FFFFFF", dataLabelFontSize: 13, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
  showLegend: false,
  showTitle: true, title: "AUC de test  ·  familias Transformer (compacto vs. grande)", titleColor: "FFFFFF", titleFontSize: 13, titleFontFace: FB,
  chartArea: { fill: { color: NAVY } }, plotArea: { fill: { color: NAVY } } });
card(s, 8.6, 2.3, 4.25, 1.95, "13405A");
s.addText("ViT-Small  >  ViT-Base", { x: 8.85, y: 2.5, w: 3.8, h: 0.5, fontFace: FH, fontSize: 18, bold: true, color: TEAL });
s.addText("Iguala/supera a un modelo 4× mayor (86M) con solo 22M parámetros.",
  { x: 8.85, y: 3.05, w: 3.8, h: 1.1, fontFace: FB, fontSize: 14, color: "FFFFFF", lineSpacing: 20 });
card(s, 8.6, 4.45, 4.25, 1.95, "3A1E2B");
s.addText("Swin-Small  no converge", { x: 8.85, y: 4.65, w: 3.8, h: 0.5, fontFace: FH, fontSize: 18, bold: true, color: "FCA5A5" });
s.addText("La variante de 49M colapsa al azar bajo el mismo protocolo: más capacidad dificulta la optimización con datos limitados.",
  { x: 8.85, y: 5.2, w: 3.8, h: 1.1, fontFace: FB, fontSize: 13, color: "FFFFFF", lineSpacing: 18 });
pageNum(s,9);

// =================================================================
// 10 · EXPLICABILIDAD (light)
// =================================================================
s = p.addSlide(); s.background = { color: LIGHT };
kicker(s, "Interpretabilidad"); title(s, "El modelo mira donde debe: mapas de saliencia");
img(s, "img/app_gradcam_planes.png", 0.7, 2.05, 8.3, 8.3*1557/2482);
s.addText([
  { text: "Grad-CAM / atención", options: { bold: true, color: INK, breakLine: true } },
  { text: "superpuestos sobre los cortes seleccionados de los tres planos.", options: { color: MUT } },
], { x: 9.3, y: 2.4, w: 3.5, h: 1.2, fontFace: FB, fontSize: 15, lineSpacing: 22 });
s.addText([
  { text: "La activación se concentra en la región intercondílea, donde se localiza el LCA.", options: { bullet: true, breakLine: true } },
  { text: "Aporta transparencia clínica a cada predicción.", options: { bullet: true } },
], { x: 9.3, y: 4.0, w: 3.5, h: 2.3, fontFace: FB, fontSize: 14, color: INK, lineSpacing: 19, paraSpaceAfter: 10 });
pageNum(s,10);

// =================================================================
// 11 · VALIDACIÓN EXTERNA CROACIA (light)
// =================================================================
s = p.addSlide(); s.background = { color: LIGHT };
kicker(s, "Resultados · generalización"); title(s, "Validación externa: existe domain shift");
s.addChart(p.charts.BAR, [
  { name: "AUC zero-shot (Croacia)", labels: ["ResNet50","ViT-Small","Swin-Tiny"], values: [0.785, 0.857, 0.882] },
], { x: 0.6, y: 2.1, w: 6.9, h: 4.4, barDir: "col", chartColors: [SKY],
  valAxisMinVal: 0.5, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1,
  catAxisLabelColor: INK, valAxisLabelColor: MUT, catAxisLabelFontSize: 13, valAxisLabelFontSize: 10,
  valGridLine: { color: LINE, size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelColor: INK, dataLabelFontSize: 13, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
  showLegend: false, showTitle: true, title: "Transferencia directa al dominio externo (sin reentrenar)",
  titleColor: INK, titleFontSize: 13, titleFontFace: FB, chartArea: { fill: { color: "FFFFFF" } } });
card(s, 7.9, 2.25, 4.9, 2.0, PANEL);
s.addText("Cae la sensibilidad", { x: 8.2, y: 2.45, w: 4.3, h: 0.5, fontFace: FH, fontSize: 18, bold: true, color: AMBER });
s.addText("Cambian escáner, población y protocolo: el AUC se mantiene razonable pero el punto operativo se degrada.",
  { x: 8.2, y: 3.0, w: 4.35, h: 1.2, fontFace: FB, fontSize: 14, color: INK, lineSpacing: 20 });
card(s, 7.9, 4.45, 4.9, 2.05, NAVY);
s.addText("Adaptación (fine-tuning)", { x: 8.2, y: 4.65, w: 4.3, h: 0.5, fontFace: FH, fontSize: 18, bold: true, color: TEAL });
s.addText([
  { text: "0.924", options: { fontSize: 30, bold: true, color: "FFFFFF" } },
  { text: "  →  ", options: { fontSize: 22, color: "CADCFC" } },
  { text: "0.936", options: { fontSize: 30, bold: true, color: TEAL } },
], { x: 8.2, y: 5.2, w: 4.3, h: 0.8, fontFace: FH, valign: "middle" });
s.addText("AUC en test de Croacia tras adaptar el modelo al nuevo dominio.",
  { x: 8.2, y: 6.0, w: 4.35, h: 0.45, fontFace: FB, fontSize: 12, color: "CADCFC" });
pageNum(s,11);

// =================================================================
// 12 · APP / DEMOSTRADOR (light)
// =================================================================
s = p.addSlide(); s.background = { color: LIGHT };
kicker(s, "Aplicación"); title(s, "Demostrador clínico interactivo");
img(s, "img/app_dashboard.png", 0.7, 2.05, 6.27, 5.0);
const feats = [
  ["Carga DICOM real","Sube un estudio comprimido y obtén el diagnóstico"],
  ["Inferencia en vivo","Selección de cortes + ensamble multi-vista en segundos"],
  ["Explicabilidad","Mapas de calor por corte y plano"],
  ["3 modelos","ResNet50 · ViT-Small · Swin-Tiny intercambiables"],
];
feats.forEach((f,i)=>{
  const y = 2.2 + i*1.15;
  card(s, 8.9, y, 3.95, 1.0, LIGHT);
  s.addShape(p.shapes.OVAL, { x: 9.1, y: y+0.32, w: 0.36, h: 0.36, fill: { color: TEAL } });
  s.addText(f[0], { x: 9.6, y: y+0.12, w: 3.15, h: 0.4, fontFace: FH, fontSize: 14.5, bold: true, color: INK });
  s.addText(f[1], { x: 9.6, y: y+0.5, w: 3.15, h: 0.45, fontFace: FB, fontSize: 11, color: MUT, lineSpacing: 12 });
});
pageNum(s,12);

// =================================================================
// 13 · CONCLUSIONES (dark)
// =================================================================
s = p.addSlide(); s.background = { color: NAVY };
s.addShape(p.shapes.OVAL, { x: W-3.2, y: -2.2, w: 5.5, h: 5.5, fill: { color: "12506A" } });
kicker(s, "Conclusiones", TEAL);
s.addText("Conclusiones", { x: 0.6, y: 0.95, w: 11, h: 0.95, fontFace: FH, fontSize: 32, bold: true, color: "FFFFFF" });
const concl = [
  ["Sistema fiable","AUC ~0.95 en test con un pipeline end-to-end, sin segmentación manual."],
  ["La arquitectura importa menos que el pipeline","Cada familia gana en una métrica; ninguna domina en todas."],
  ["Lo compacto gana","ViT-Small (22M) y Swin-Tiny (28M) igualan o superan a sus variantes grandes."],
  ["Generalización con matices","Domain shift real hacia Croacia; el fine-tuning lo mitiga."],
];
concl.forEach((c,i)=>{
  const y = 2.15 + i*1.15;
  s.addText(String(i+1).padStart(2,"0"), { x: 0.7, y, w: 1.0, h: 0.9, fontFace: FH, fontSize: 30, bold: true, color: TEAL, valign: "middle" });
  s.addText(c[0], { x: 1.8, y: y+0.02, w: 10.8, h: 0.45, fontFace: FH, fontSize: 19, bold: true, color: "FFFFFF" });
  s.addText(c[1], { x: 1.8, y: y+0.5, w: 10.8, h: 0.5, fontFace: FB, fontSize: 14, color: "CADCFC", lineSpacing: 18 });
});
pageNum(s,13);

// =================================================================
// 14 · GRACIAS (dark)
// =================================================================
s = p.addSlide(); s.background = { color: NAVY };
s.addShape(p.shapes.OVAL, { x: -2.4, y: H-3.4, w: 6.5, h: 6.5, fill: { color: "13405A" } });
s.addShape(p.shapes.OVAL, { x: W-2.6, y: -2.6, w: 5.5, h: 5.5, fill: { color: "12506A" } });
s.addText("Gracias", { x: 0.8, y: 2.5, w: 11, h: 1.2, fontFace: FH, fontSize: 54, bold: true, color: "FFFFFF" });
s.addText("¿Preguntas?", { x: 0.85, y: 3.8, w: 11, h: 0.6, fontFace: FB, fontSize: 22, color: TEAL });
s.addText("Pablo López Domínguez   ·   Grado en Ciencia de Datos   ·   ETSE-UV",
  { x: 0.85, y: 5.9, w: 11.6, h: 0.4, fontFace: FB, fontSize: 14, color: "CADCFC" });

p.writeFile({ fileName: "defensa_TFG.pptx" }).then(f => console.log("OK:", f));

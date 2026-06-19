# SVG charts (tema claro, integrados en el deck). Devuelven strings <svg>.
INK="#102A37"; MUT="#41586A"; FAINT="#6E8493"; LINE="#BCCAD3"
TEAL="#0E9488"; BLUE="#2E7CC4"; SLATE="#64748B"; RED="#DC2626"; AMBER="#D97706"
FONT="Inter, system-ui, sans-serif"

def _txt(x,y,s,size,color=INK,anchor="middle",weight=400,style=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
            f'fill="{color}" text-anchor="{anchor}" font-weight="{weight}" {style}>{s}</text>')

def grouped_bars(cats, series, ymin, ymax, yticks, W=760, H=460, fmt="{:.3f}", showval=False):
    ml,mr,mt,mb=64,18,22,70
    pw,ph=W-ml-mr,H-mt-mb; y0=mt+ph
    def Y(v): return y0-(v-ymin)/(ymax-ymin)*ph
    s=[f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">']
    for t in yticks:
        s.append(f'<line x1="{ml}" y1="{Y(t):.1f}" x2="{ml+pw}" y2="{Y(t):.1f}" stroke="{LINE}" stroke-width="1"/>')
        s.append(_txt(ml-10,Y(t)+4,f'{t:g}',15,MUT,"end"))
    n=len(cats); ns=len(series); gw=pw/n; bw=gw*0.66/ns
    for ci,c in enumerate(cats):
        gx=ml+ci*gw+gw*0.17
        for si,(name,vals,col) in enumerate(series):
            v=vals[ci]
            if v is None: continue
            x=gx+si*bw; bh=y0-Y(v)
            s.append(f'<rect x="{x:.1f}" y="{Y(v):.1f}" width="{bw*0.86:.1f}" height="{bh:.1f}" rx="2" fill="{col}"/>')
            if showval:
                s.append(_txt(x+bw*0.43,Y(v)-7,fmt.format(v),13,INK,"middle",600))
        s.append(_txt(ml+ci*gw+gw/2,y0+26,c,16,INK,"middle",600))
    s.append(f'<line x1="{ml}" y1="{y0}" x2="{ml+pw}" y2="{y0}" stroke="{INK}" stroke-width="1.5"/>')
    s.append('</svg>')
    return "".join(s)

def simple_bars(labels, values, colors, ymin, ymax, yticks, W=720, H=460, fmt="{:.3f}"):
    ml,mr,mt,mb=64,18,30,64
    pw,ph=W-ml-mr,H-mt-mb; y0=mt+ph
    def Y(v): return y0-(v-ymin)/(ymax-ymin)*ph
    s=[f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">']
    for t in yticks:
        s.append(f'<line x1="{ml}" y1="{Y(t):.1f}" x2="{ml+pw}" y2="{Y(t):.1f}" stroke="{LINE}"/>')
        s.append(_txt(ml-10,Y(t)+4,f'{t:g}',15,MUT,"end"))
    n=len(labels); gw=pw/n; bw=gw*0.5
    for i,(lab,v,col) in enumerate(zip(labels,values,colors)):
        x=ml+i*gw+(gw-bw)/2
        s.append(f'<rect x="{x:.1f}" y="{Y(v):.1f}" width="{bw:.1f}" height="{y0-Y(v):.1f}" rx="3" fill="{col}"/>')
        s.append(_txt(x+bw/2,Y(v)-9,fmt.format(v),17,INK,"middle",700))
        s.append(_txt(ml+i*gw+gw/2,y0+26,lab,16,INK,"middle",600))
    s.append(f'<line x1="{ml}" y1="{y0}" x2="{ml+pw}" y2="{y0}" stroke="{INK}" stroke-width="1.5"/>')
    s.append('</svg>')
    return "".join(s)

def scatter_params(points, W=780, H=470):
    # points: (name,x,y,color,marker['o'|'s'|'x'], label_dx, label_dy, anchor)
    ml,mr,mt,mb=64,24,24,58
    pw,ph=W-ml-mr,H-mt-mb; y0=mt+ph; x0=ml
    xmin,xmax,ymin,ymax=10,95,0.45,1.0
    def X(v): return x0+(v-xmin)/(xmax-xmin)*pw
    def Y(v): return y0-(v-ymin)/(ymax-ymin)*ph
    s=[f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">']
    # zona compactos
    s.append(f'<rect x="{X(18):.1f}" y="{mt}" width="{X(32)-X(18):.1f}" height="{ph:.1f}" fill="{TEAL}" opacity="0.07"/>')
    for t in [0.5,0.6,0.7,0.8,0.9,1.0]:
        s.append(f'<line x1="{x0}" y1="{Y(t):.1f}" x2="{x0+pw}" y2="{Y(t):.1f}" stroke="{LINE}"/>')
        s.append(_txt(x0-10,Y(t)+4,f'{t:g}',14,MUT,"end"))
    for t in [10,30,50,70,90]:
        s.append(_txt(X(t),y0+24,str(t),14,MUT,"middle"))
    s.append(_txt(x0+pw/2,H-8,"Parámetros del modelo (millones)",15,INK,"middle",600))
    s.append(f'<text x="{18}" y="{mt+ph/2}" font-family="{FONT}" font-size="15" fill="{INK}" font-weight="600" text-anchor="middle" transform="rotate(-90 18 {mt+ph/2:.0f})">AUC de test</text>')
    for name,x,y,col,mk,ldx,ldy,anc in points:
        cx,cy=X(x),Y(y)
        if mk=='s': s.append(f'<rect x="{cx-11:.1f}" y="{cy-11:.1f}" width="22" height="22" rx="3" fill="{col}" stroke="#fff" stroke-width="2.5"/>')
        elif mk=='x':
            s.append(f'<line x1="{cx-11}" y1="{cy-11}" x2="{cx+11}" y2="{cy+11}" stroke="{col}" stroke-width="6" stroke-linecap="round"/>')
            s.append(f'<line x1="{cx-11}" y1="{cy+11}" x2="{cx+11}" y2="{cy-11}" stroke="{col}" stroke-width="6" stroke-linecap="round"/>')
        else: s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="13" fill="{col}" stroke="#fff" stroke-width="2.5"/>')
        if name:
            s.append(_txt(cx+ldx,cy+ldy,name,15,col,anc,700))
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x0+pw}" y2="{y0}" stroke="{INK}" stroke-width="1.5"/>')
    s.append('</svg>')
    return "".join(s)

def line_threshold(W=760, H=460):
    import math
    ml,mr,mt,mb=58,18,22,58
    pw,ph=W-ml-mr,H-mt-mb; y0=mt+ph; x0=ml
    def X(t): return x0+t*pw
    def Y(v): return y0-v*ph
    rec=lambda t:1/(1+math.exp((t-0.30)*12))
    pre=lambda t:1/(1+math.exp(-(t-0.18)*10))*0.5+0.45
    s=[f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">']
    for v in [0,0.5,1.0]:
        s.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x0+pw}" y2="{Y(v):.1f}" stroke="{LINE}"/>')
        s.append(_txt(x0-10,Y(v)+4,f'{v:g}',14,MUT,"end"))
    for t in [0,0.25,0.5,0.75,1.0]:
        s.append(_txt(X(t),y0+24,f'{t:g}',14,MUT,"middle"))
    s.append(_txt(x0+pw/2,H-6,"Umbral de decisión  τ",15,INK,"middle",600))
    def path(fn,col):
        pts=" ".join(f'{X(i/100):.1f},{Y(fn(i/100)):.1f}' for i in range(101))
        return f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="3.5"/>'
    s.append(path(rec,TEAL)); s.append(path(pre,BLUE))
    s.append(f'<line x1="{x0}" y1="{Y(0.75):.1f}" x2="{x0+pw}" y2="{Y(0.75):.1f}" stroke="{SLATE}" stroke-dasharray="5 4" stroke-width="1.5"/>')
    s.append(_txt(x0+pw-6,Y(0.75)+20,"precisión mínima 0.75",13,SLATE,"end"))
    s.append(f'<line x1="{X(0.373):.1f}" y1="{mt}" x2="{X(0.373):.1f}" y2="{y0}" stroke="{RED}" stroke-dasharray="6 4" stroke-width="2.5"/>')
    s.append(_txt(X(0.373)+10,mt+24,"τ* = 0.37",15,RED,"start",700))
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x0+pw}" y2="{y0}" stroke="{INK}" stroke-width="1.5"/>')
    s.append('</svg>')
    return "".join(s)

def confusion(tp,fn,fp,vn,W=300,H=300,title=""):
    # 2x2: filas Real (Sano,Roto), cols Predicho (Sano,Roto)
    ml,mt=66,40; cell=(W-ml-12)/2; ch=(H-mt-46)/2
    POS="#0E9488"; NEG="#16A34A"; ERR="#E11D48"
    cells=[("VN",vn,NEG,0,0),("FP",fp,ERR,1,0),("FN",fn,ERR,0,1),("VP",tp,POS,1,1)]
    mx=max(tp,fn,fp,vn,1)
    s=[f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">']
    if title: s.append(_txt(W/2,22,title,17,INK,"middle",700))
    for lab,val,col,cx,ry in cells:
        x=ml+cx*cell; y=mt+ry*ch
        op=0.28+0.60*(val/mx)
        s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell-6:.1f}" height="{ch-6:.1f}" rx="6" fill="{col}" opacity="{op:.2f}" stroke="{col}" stroke-width="2"/>')
        s.append(_txt(x+(cell-6)/2,y+(ch-6)/2+2,str(val),26,INK,"middle",700))
        s.append(_txt(x+(cell-6)/2,y+(ch-6)/2+22,lab,12,INK,"middle",500))
    # ejes
    s.append(_txt(ml+cell,mt-10,"Predicho",13,MUT,"middle",600))
    s.append(_txt(ml+cell*0.5,mt+2*ch+18,"Sano",13,MUT,"middle"))
    s.append(_txt(ml+cell*1.5,mt+2*ch+18,"Roto",13,MUT,"middle"))
    s.append(f'<text x="20" y="{mt+ch}" font-family="{FONT}" font-size="13" fill="{MUT}" font-weight="600" text-anchor="middle" transform="rotate(-90 20 {mt+ch:.0f})">Real</text>')
    s.append(_txt(ml-12,mt+ch*0.5,"Sano",12,MUT,"end"))
    s.append(_txt(ml-12,mt+ch*1.5,"Roto",12,MUT,"end"))
    s.append('</svg>')
    return "".join(s)

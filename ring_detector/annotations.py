from pathlib import Path
import json, shutil, hashlib
from collections import defaultdict
from PIL import Image,ImageOps,ImageDraw
from .io import read_csv,write_csv

LABELS={'positive','negative','uncertain','invalid'}

def review_export(proposals,output,per_fov=6):
    proposals=Path(proposals);output=Path(output)
    for folder in ['crops','crop_masks','contact_sheets','overlays']: (output/folder).mkdir(parents=True,exist_ok=True)
    rows=read_csv(proposals/'cells.csv');byfov=defaultdict(list)
    for r in rows: byfov[r['fov_id']].append(r)
    selected=[]
    for fid,group in byfov.items():
        # Balanced retrieval categories in EACH FOV; positives are not yet known.
        categories=[sorted([r for r in group if r['review_bucket']=='candidate'],key=lambda r:float(r['ring_score']),reverse=True),[r for r in group if r['review_bucket']=='bright_no_ring'],[r for r in group if r['review_bucket']=='weak_or_no_ring'],[r for r in group if r['segmentation_qc']!='pass']]
        chosen=[]
        for j in range(per_fov):
            category=categories[j%4]
            candidate=next((r for r in category if r not in chosen),None)
            if candidate is None: candidate=next((r for r in sorted(group,key=lambda r:float(r['ring_score']),reverse=True) if r not in chosen),None)
            if candidate: chosen.append(candidate)
        selected.extend(dict(r) for r in chosen)
    for r in selected:
        path=Path(r['crop_path']);dest=output/'crops'/path.name;shutil.copy2(path,dest)
        mp=Path(r['crop_mask_path']);shutil.copy2(mp,output/'crop_masks'/mp.name)
        r['crop_mask_path']='crop_masks/'+mp.name
        r['mask_path']=r['crop_mask_path']
        overlay=proposals/'overlays'/f'{r["fov_id"]}.png'
        shutil.copy2(overlay,output/'overlays'/overlay.name)
        r['overlay_path']='overlays/'+overlay.name
        r['crop_path']='crops/'+path.name;r['label']='';r['annotator']='';r['annotation_version']=''
        r['crop_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest()
    write_csv(output/'annotations.csv',selected)
    write_csv(output/'all_cells_for_review.csv',rows)
    for start in range(0,len(selected),24):
        batch=selected[start:start+24];sheet=Image.new('RGB',(1200,6*210),'white');d=ImageDraw.Draw(sheet)
        for j,r in enumerate(batch):
            x=j%4*300;y=j//4*210
            im=Image.open(output/r['crop_path']);im.thumbnail((294,160));sheet.paste(im,(x+(300-im.width)//2,y+40))
            d.text((x+3,y+2),f'{r["fov_id"]} cell {r["cell_id"]} score {float(r["ring_score"]):.2f}',fill='black')
            d.text((x+3,y+16),r['review_bucket']+' / '+r['segmentation_qc'],fill='black')
        sheet.save(output/'contact_sheets'/f'sheet_{start//24+1:02d}.jpg')
    payload=json.dumps(selected,ensure_ascii=True).replace('</','<\\/')
    storage_prefix='ring-review-'+hashlib.sha256(payload.encode()).hexdigest()[:20]+'-'
    template='''<!doctype html><meta charset="utf-8"><title>Ring candidate review</title>
<style>body{font:16px system-ui;margin:25px;background:#eee}article{background:white;padding:12px;margin:10px;display:inline-block;width:440px;vertical-align:top}img{width:100%;image-rendering:auto}select,input,button{font:inherit;margin:4px}header{position:sticky;top:0;background:#fff;padding:12px;z-index:1}.meta{font-size:13px}</style>
<header><b>P0/P1 人工审核：候选不等于真阳性</b><p>每张图：左 morphology，中荧光强度，右 mask 与候选（紫色）。无候选不代表阴性；错误分割标 invalid，模糊或不确定标 uncertain。</p>
<label>Annotator <input id="annotator"></label><label>Version <input id="version" value="v1"></label><button id="save">下载 annotations.csv</button><span id="count"></span></header><main id="main"></main>
<script>const rows=__DATA__;const prefix=__PREFIX__;const root=document.querySelector('#main');
function stored(k){try{return localStorage.getItem(k)||''}catch{return ''}}function store(k,v){try{localStorage.setItem(k,v)}catch{}}
rows.forEach((r,i)=>{const a=document.createElement('article');const meta=document.createElement('p');meta.className='meta';meta.textContent=`${r.fov_id} cell ${r.cell_id} | ${r.review_bucket} | QC ${r.segmentation_qc} | score ${Number(r.ring_score).toFixed(3)}`;a.append(meta);const link=document.createElement('a');link.href=r.crop_path;link.target='_blank';const img=document.createElement('img');img.src=r.crop_path;link.append(img);a.append(link);const full=document.createElement('a');full.href=r.overlay_path;full.target='_blank';full.textContent='打开完整 FOV';a.append(full);const s=document.createElement('select');['','positive','negative','uncertain','invalid'].forEach(v=>s.add(new Option(v||'未审核',v)));s.value=stored(prefix+r.fov_id+'-'+r.cell_id);r.label=s.value;s.onchange=()=>{r.label=s.value;store(prefix+r.fov_id+'-'+r.cell_id,s.value);update()};a.append(s);root.append(a)});
function update(){document.querySelector('#count').textContent=` 已审核 ${rows.filter(r=>r.label).length}/${rows.length}`};update();
document.querySelector('#save').onclick=()=>{const q=v=>'"'+String(v??'').replaceAll('"','""')+'"';const keys=Object.keys(rows[0]);const out=[keys.map(q).join(','),...rows.map(r=>{r.annotator=document.querySelector('#annotator').value;r.annotation_version=document.querySelector('#version').value;return keys.map(k=>q(r[k])).join(',')})].join('\\r\\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\\ufeff'+out],{type:'text/csv;charset=utf-8'}));a.download='annotations.csv';a.click();URL.revokeObjectURL(a.href)};</script>'''
    (output/'review.html').write_text(template.replace('__DATA__',payload).replace('__PREFIX__',json.dumps(storage_prefix)),encoding='utf-8')
    (output/'export.json').write_text(json.dumps(dict(n_exported=len(selected),n_total_cells=len(rows),n_fovs=len(byfov),source_cells_sha256=hashlib.sha256((proposals/'cells.csv').read_bytes()).hexdigest(),annotation_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),labels='all blank; researcher review required'),indent=2),encoding='utf-8')
    print(f'REVIEW: {len(selected)} balanced crops across {len(byfov)} FOV; {len(rows)} total cells available')
    return selected

def validate_annotations(annotations,cells):
    rows=read_csv(annotations);reference={(r['fov_id'],r['cell_id']) for r in read_csv(cells)};seen=set()
    for r in rows:
        key=(r['fov_id'],r['cell_id'])
        if key not in reference or key in seen: raise ValueError(f'Unmapped/duplicate annotation: {key}')
        seen.add(key)
        if r['label'] and r['label'] not in LABELS: raise ValueError(f'Invalid label: {r["label"]}')
    counts={label:sum(r['label']==label for r in rows) for label in LABELS}
    print(json.dumps(dict(counts=counts,n_unreviewed=sum(not r['label'] for r in rows),grouping='All cells from same FOV must stay together; no training performed')))
    return rows

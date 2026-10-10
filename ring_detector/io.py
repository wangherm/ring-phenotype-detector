"""Role-based TIFF IO. Never mix fluorescence channels or project Z/T silently."""
from pathlib import Path
import csv, hashlib, json, re, zipfile
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation
from PIL import Image, ImageDraw

MARKER = re.compile(r'_(c1-2|c1|c2)_')

def write_csv(path, rows, fields=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:
        fields=list(rows[0]) if rows else []
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def normalize(a, lower=1, upper=99.5):
    a=np.asarray(a,dtype=np.float32)
    if not np.isfinite(a).all(): raise ValueError('Non-finite image values')
    lo,hi=np.percentile(a,[lower,upper])
    return np.clip((a-lo)/max(float(hi-lo),1e-6),0,1)

def load_planes(path):
    with tifffile.TiffFile(path) as t:
        s=t.series[0]; a=s.asarray(); axes=s.axes
        meta={'axes':axes,'shape':list(a.shape),'dtype':str(a.dtype),'series_count':len(t.series)}
    if meta['series_count']!=1: raise ValueError(f'Multiple TIFF series require explicit selection: {path}')
    if a.ndim==2 and axes=='YX': return a[...,None], ['gray'], meta
    if a.ndim==3 and axes in ('YXS','YXC','SYX','CYX'):
        if axes[0] in 'SC': a=np.moveaxis(a,0,-1)
        names=['R','G','B'] if 'S' in axes and a.shape[-1] in (3,4) else [str(i) for i in range(a.shape[-1])]
        if a.shape[-1]==4 and 'S' in axes: a=a[...,:3]
        return a,names,meta
    raise ValueError(f'Unsupported axes {axes}, shape {a.shape}: select Z/T plane explicitly; no projection performed')

def fluorescence(path, channels=('auto',)):
    a,names,meta=load_planes(path)
    evidence=[]
    for k,n in enumerate(names):
        p=a[...,k].astype(np.float32); dynamic=float(np.percentile(p,99.9)-np.percentile(p,5))
        contrast=float(np.percentile(np.abs(p-gaussian_filter(p,3)),99.5))
        evidence.append({'channel':n,'dynamic_range':dynamic,'local_contrast':contrast,'selection_score':dynamic*contrast})
    out={}
    for requested in channels:
        if requested=='auto': i=int(np.argmax([x['selection_score'] for x in evidence]))
        elif requested=='gray':
            if len(names)>1 and not all(np.array_equal(a[...,0],a[...,j]) for j in range(1,len(names))):
                raise ValueError('gray cannot mix distinct channels; select R/G/B or indexes explicitly')
            i=0
        elif str(requested) in names: i=names.index(str(requested))
        else:
            try: i=int(requested)
            except ValueError: raise ValueError(f'Unknown channel {requested}; available: {names}')
            if not 0<=i<a.shape[-1]: raise ValueError(f'Channel index out of range: {requested}')
        out[names[i]]=a[...,i]
    return out,dict(meta,channel_evidence=evidence,selected_channels=list(out))

def morphology(path):
    a,names,meta=load_planes(path)
    if a.shape[-1]>1 and not all(np.array_equal(a[...,0],a[...,j]) for j in range(1,a.shape[-1])):
        if names==['R','G','B']: return a.astype(np.float32).mean(axis=-1),meta
        raise ValueError('Select one morphology channel explicitly for multichannel TIFF')
    return a[...,0],meta

def pair_paths(paths):
    groups={}; issues=[]
    for p in sorted(map(Path,paths)):
        m=MARKER.search(p.name)
        if not m: issues.append(f'Unrecognized role: {p.name}'); continue
        # Retain the complete stem on BOTH sides of the marker.
        key=p.stem[:m.start()]+'_'+p.stem[m.end():]
        role={'c1':'c1_path','c2':'c2_path','c1-2':'merged_path'}[m.group(1)]
        g=groups.setdefault(key,{})
        if role in g: raise ValueError(f'Duplicate role {role} for {key}')
        g[role]=str(p.resolve())
    return groups,issues

def extract_zip(source,dest):
    dest=Path(dest); dest.mkdir(parents=True,exist_ok=True); kept=[]; ignored=0
    with zipfile.ZipFile(source) as z:
        for member in z.infolist():
            p=Path(member.filename)
            if '__MACOSX' in p.parts or p.name.startswith('._') or p.suffix.lower() not in ('.tif','.tiff'):
                ignored+=1; continue
            if member.file_size>512*1024**2: raise ValueError('Unreasonably large archive member')
            if any(part=='..' for part in p.parts) or p.is_absolute(): raise ValueError('Unsafe zip path')
            out=dest/p.name
            if out in kept: raise ValueError(f'Basename collision: {p.name}')
            out.write_bytes(z.read(member)); kept.append(out)
    return kept,ignored

def audit(source,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    source=Path(source)
    if source.suffix.lower()=='.zip': paths,ignored=extract_zip(source,output/'images')
    else: paths=sorted(p for p in source.rglob('*') if p.suffix.lower() in ('.tif','.tiff') and not p.name.startswith('._') and '__MACOSX' not in p.parts); ignored=0
    groups,warnings=pair_paths(paths); rows=[]; details=[]
    thumbs=Image.new('RGB',(1000,((len(groups)+4)//5)*235),'white'); draw=ImageDraw.Draw(thumbs)
    for i,(key,g) in enumerate(sorted(groups.items()),1):
        fov=f'FOV{i:03d}'; status='complete' if len(g)==3 else 'incomplete'
        date=re.match(r'(\d{8})',key); treatment=re.search(r'(\d+(?:\.\d+)?mM)DTT',key,re.I); duration=re.search(r'-(\d+(?:\.\d+)?h)-',key)
        row=dict(fov_id=fov,source_stem=key,c1_path=g.get('c1_path',''),c2_path=g.get('c2_path',''),merged_path=g.get('merged_path',''),date=date.group(1) if date else 'unknown',treatment=treatment.group(1)+' DTT' if treatment else 'unknown',duration=duration.group(1) if duration else 'unknown',image_shape='',dtype='',status=status,alignment_status='not_checked',alignment_shift_y='',alignment_shift_x='')
        metas={}; planes={}
        for role,p in g.items():
            a,n,m=load_planes(p); metas[role]=dict(m,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()); planes[role]=a
        if len({tuple(a.shape[:2]) for a in planes.values()})>1: row['status']='shape_mismatch'
        if 'c2_path' in g:
            gray,_=morphology(g['c2_path']); row['image_shape']=json.dumps(list(gray.shape)); row['dtype']=metas['c2_path']['dtype']
            if 'merged_path' in planes and row['status']=='complete':
                # In a display merge, select the plane most similar to morphology. No registration correction.
                mg=planes['merged_path']; err=[float(np.mean((normalize(mg[...,j])-normalize(gray))**2)) for j in range(mg.shape[-1])]; j=int(np.argmin(err))
                shift,error,_=phase_cross_correlation(normalize(gray)[::2,::2],normalize(mg[...,j])[::2,::2],normalization=None)
                shift=shift*2
                row.update(alignment_shift_y=float(shift[0]),alignment_shift_x=float(shift[1]),alignment_status='merge_morphology_consistent' if np.max(np.abs(shift))<=2 else 'review_alignment')
                # Also compare fluorescence to the merge difference, which supports c1 alignment but does not prove biological co-registration.
                fluor,fmeta=fluorescence(g['c1_path']); fg=normalize(next(iter(fluor.values())))
                residual=np.max(mg.astype(np.float32)-gray[...,None],axis=-1)
                fshift,_,_=phase_cross_correlation(fg[::2,::2],normalize(residual)[::2,::2],normalization=None)
                metas['merge_fluorescence_shift_yx']=(fshift*2).tolist()
                if np.max(np.abs(fshift*2))>2: row['alignment_status']='review_alignment'
                metas['fluorescence']=fmeta
            tile=Image.fromarray((normalize(gray)*255).astype('uint8')); tile.thumbnail((195,195)); x=((i-1)%5)*200; y=((i-1)//5)*235
            thumbs.paste(tile,(x,y+35)); draw.text((x+3,y+3),f'{fov} {row["date"]}',fill='black'); draw.text((x+3,y+18),f'{row["treatment"]} {row["duration"]}',fill='black')
        if row['status']!='complete' or row['alignment_status']=='review_alignment': warnings.append(f'{fov}: {row["status"]}, {row["alignment_status"]}')
        rows.append(row); details.append(dict(fov_id=fov,metadata=metas))
    write_csv(output/'manifest.csv',rows); thumbs.save(output/'thumbnails.jpg')
    report=dict(n_tiffs=len(paths),n_fovs=len(rows),complete_fovs=sum(r['status']=='complete' for r in rows),ignored_entries=ignored,warnings=warnings,files=details,quantitation='Display RGB intensity is relative only',alignment_note='Merge consistency checks and visual review; not proof of acquisition registration')
    (output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); (output/'warnings.log').write_text('\n'.join(warnings) or 'No pairing/shape/merge-shift warnings. Missing metadata remains unknown.\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['n_tiffs','n_fovs','complete_fovs','warnings']})); return rows

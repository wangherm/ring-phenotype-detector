from pathlib import Path
import json, sys, platform, importlib.metadata, hashlib, time
import numpy as np, tifffile, yaml
from PIL import Image,ImageDraw
from skimage.segmentation import find_boundaries
from .io import read_csv,write_csv,morphology,fluorescence,normalize
from .segment import segment_cells
from .candidates import detect_rings
from . import __version__

def load_config(path): return yaml.safe_load(Path(path).read_text(encoding='utf-8'))

def provenance(output,config,command):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'config.yaml').write_text(yaml.safe_dump(config),encoding='utf-8')
    versions={m:importlib.metadata.version(m) for m in ['numpy','scipy','scikit-image','tifffile','opencv-python-headless','Pillow','PyYAML','imagecodecs']}
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    log=dict(version=__version__,model='M0 geometry candidate baseline (untrained)',p_positive='not_available',python=sys.version,platform=platform.platform(),dependencies=versions,source_sha256=source_hashes,command=command,seed=config.get('seed',42),threshold_note='Unvalidated retrieval defaults; do not treat proposals as ground truth or tune on held-out test labels')
    (output/'run.json').write_text(json.dumps(log,indent=2),encoding='utf-8')

def run_segment(manifest,config,output):
    output=Path(output);(output/'masks').mkdir(parents=True,exist_ok=True);allcells=[];summary=[]
    provenance(output,config,'segment');
    for f in read_csv(manifest):
        if f['status']!='complete': continue
        a,meta=morphology(f['c2_path']);labels,cells=segment_cells(a,config['segmentation'])
        tifffile.imwrite(output/'masks'/f'{f["fov_id"]}.tif',labels,metadata={'axes':'YX'})
        for c in cells: c.update(fov_id=f['fov_id'],mask_path=str((output/'masks'/f'{f["fov_id"]}.tif').resolve()))
        allcells.extend(cells)
        row=dict(fov_id=f['fov_id'],date=f['date'],treatment=f['treatment'],duration=f['duration'],n_detected=len(cells),n_qc_pass=sum(c['segmentation_qc']=='pass' for c in cells),n_flagged=sum(c['segmentation_qc']!='pass' for c in cells))
        for flag in ['edge_truncated','large_merged','irregular_or_overlap','low_focus']: row['n_'+flag]=sum(flag in c['segmentation_qc'] for c in cells)
        summary.append(row);print(f'SEGMENT {f["fov_id"]}: {len(cells)} regions, {row["n_flagged"]} QC flagged',flush=True)
    write_csv(output/'cells.csv',allcells);write_csv(output/'segmentation_summary.csv',summary)
    return allcells

def draw_overlay(gray,labels,cells,rings,path,fluor=None):
    # Display is normalized morphology plus independent overlay colors.
    rgb=np.repeat((normalize(gray)*200).astype('uint8')[...,None],3,axis=-1)
    if fluor is not None:
        signal=normalize(fluor,0,99.9)
        rgb=(rgb.astype(np.float32)*.65).astype('uint8')
        rgb[...,0]=np.clip(rgb[...,0].astype(np.float32)+signal*180,0,255).astype('uint8')
    for c in cells:
        x1,y1,x2,y2=[int(c[k]) for k in ['bbox_x1','bbox_y1','bbox_x2','bbox_y2']]
        local_mask=labels[y1:y2,x1:x2]==int(c['cell_id'])
        boundary=find_boundaries(np.pad(local_mask,1),mode='inner')[1:-1,1:-1]
        color=[40,190,240] if c['segmentation_qc']=='pass' else [255,165,50]
        rgb[y1:y2,x1:x2][boundary]=color
    im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
    for c in cells: d.text((float(c['centroid_x']),float(c['centroid_y'])),str(c['cell_id']),fill='white')
    for r in rings:
        from .candidates import ellipse_points
        y,x=ellipse_points(r['x'],r['y'],r['radius_x'],r['radius_y'],r['angle'],1,n=96)
        d.line(list(zip(x.tolist(),y.tolist()))+[ (float(x[0]),float(y[0])) ],fill=(255,50,220),width=3)
        d.text((r['x']+3,r['y']+3),f'r{r["candidate_id"]}',fill=(255,80,220))
    im.save(path)

def run_propose(manifest,segments,config,output):
    output=Path(output)
    for folder in ['overlays','crops','crop_masks','raw_crops','channel_qc']: (output/folder).mkdir(parents=True,exist_ok=True)
    provenance(output,config,'propose / baseline-only predict')
    allrows=[];allrings=[];fovs=[];channel_logs=[]
    segments=Path(segments);sourcecells=read_csv(segments/'cells.csv')
    manifest_rows=read_csv(manifest)
    write_csv(output/'fov_groups.csv',[dict(fov_id=f['fov_id'],session_group=f['date'],split='not_assigned_before_human_review') for f in manifest_rows])
    for f in manifest_rows:
        if f['status']!='complete': continue
        fid=f['fov_id'];gray,_=morphology(f['c2_path']);planes,meta=fluorescence(f['c1_path'],config.get('fluorescence_channels',['auto']))
        labels=tifffile.imread(segments/'masks'/f'{fid}.tif');cells=[c for c in sourcecells if c['fov_id']==fid];rings=[]
        for c in cells:
            cid=int(c['cell_id']);x1,y1,x2,y2=[int(c[k]) for k in ['bbox_x1','bbox_y1','bbox_x2','bbox_y2']]
            pad=12;cx1=max(0,x1-pad);cy1=max(0,y1-pad);cx2=min(gray.shape[1],x2+pad);cy2=min(gray.shape[0],y2+pad)
            mask=labels[cy1:cy2,cx1:cx2]==cid;selected=[];cr=[]
            for name,plane in planes.items():
                proposals=detect_rings(plane[cy1:cy2,cx1:cx2],mask,config['candidate'])
                if proposals: selected.append(name)
                for r in proposals:
                    r.update(x=r['x']+cx1,y=r['y']+cy1,channel=name,fov_id=fid,cell_id=cid,candidate_id=len(rings)+1)
                    rings.append(r);cr.append(r)
            # Every cell requires human label; invalid here is ONLY automatic segmentation QC, not a biological verdict.
            severe=any(s in c['segmentation_qc'] for s in ['edge_truncated','large_merged'])
            score=max((r['ring_candidate_score'] for r in cr),default=0)
            tile=Image.new('RGB',((cx2-cx1)*3,cy2-cy1));mdis=normalize(gray[cy1:cy2,cx1:cx2]);tile.paste(Image.fromarray((mdis*255).astype('uint8')).convert('RGB'),(0,0))
            # Preview fluorescence shows intensity (not source display color), with no channel sum.
            fluor=next(iter(planes.values()))[cy1:cy2,cx1:cx2];fdis=normalize(fluor,0,99.5)
            tile.paste(Image.fromarray((fdis*255).astype('uint8')).convert('RGB'),(cx2-cx1,0))
            comp=np.zeros((*mask.shape,3),np.uint8);comp[:]=np.stack([mdis*120+fdis*130,mdis*120,mdis*120],axis=-1).clip(0,255).astype('uint8');comp[find_boundaries(mask)]=[40,190,240]
            ri=Image.fromarray(comp);dr=ImageDraw.Draw(ri)
            from .candidates import ellipse_points
            for r in cr:
                yy,xx=ellipse_points(r['x']-cx1,r['y']-cy1,r['radius_x'],r['radius_y'],r['angle'],1,96);dr.line(list(zip(xx.tolist(),yy.tolist()))+[(float(xx[0]),float(yy[0]))],fill=(255,50,220),width=2)
            tile.paste(ri,(2*(cx2-cx1),0));stem=f'{fid}_cell{cid:03d}'
            crop_path=output/'crops'/f'{stem}.png';mask_path=output/'crop_masks'/f'{stem}.png';tile.save(crop_path);Image.fromarray(mask.astype('uint8')*255).save(mask_path)
            rawpath=output/'raw_crops'/f'{stem}.npz';np.savez_compressed(rawpath,morphology=gray[cy1:cy2,cx1:cx2],mask=mask,**{f'fluorescence_{n}':p[cy1:cy2,cx1:cx2] for n,p in planes.items()})
            row=dict(c,fluorescence_channel=';'.join(planes),trigger_channels=';'.join(selected),ring_score=score,p_positive='',label_pred='invalid' if severe else 'uncertain',candidate_suggestion='ring_candidate' if cr else 'no_clear_ring_candidate',ring_candidate_count=len(cr),review_required=True,crop_path=str(crop_path.resolve()),crop_mask_path=str(mask_path.resolve()),raw_crop_path=str(rawpath.resolve()),crop_x1=cx1,crop_y1=cy1,crop_x2=cx2,crop_y2=cy2,label='',annotator='',annotation_version='',review_bucket='candidate' if cr else ('bright_no_ring' if np.percentile(fluor,99)>.5*np.max(next(iter(planes.values()))) else 'weak_or_no_ring'))
            allrows.append(row)
        draw_overlay(gray,labels,cells,rings,output/'overlays'/f'{fid}.png',next(iter(planes.values())))
        allrings.extend(rings);fovs.append(dict(fov_id=fid,date=f['date'],treatment=f['treatment'],duration=f['duration'],n_regions=len(cells),n_candidate_cells=sum(r['candidate_suggestion']=='ring_candidate' for r in allrows if r['fov_id']==fid),n_candidate_rings=len(rings),n_qc_flagged=sum(c['segmentation_qc']!='pass' for c in cells),n_review_required=len(cells)))
        channel_logs.append(dict(fov_id=fid,**meta));print(f'PROPOSE {fid}: {len(cells)} regions, {len(rings)} rings',flush=True)
    write_csv(output/'cells.csv',allrows);write_csv(output/'rings.csv',allrings,fields=['fov_id','cell_id','candidate_id','channel','x','y','radius_x','radius_y','angle','boundary_to_center_contrast','boundary_mean','center_mean','inner_outer_contrast','continuity','ellipse_axis_ratio','ring_band_occupancy','ring_candidate_score','contour_level'])
    write_csv(output/'fov_qc.csv',fovs);(output/'channel_qc'/'selection.json').write_text(json.dumps(channel_logs,indent=2),encoding='utf-8')
    # Annotation-free summaries contain no positive-fraction estimate.
    return allrows

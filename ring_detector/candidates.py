"""Ellipse/near-closed contour proposals scored by angular intensity profiles.

Scores are retrieval heuristics, NOT calibrated probabilities. No training.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi

def preprocess(image, config,return_info=False):
    a=np.asarray(image,np.float32)
    bg=ndi.gaussian_filter(a,config.get('background_sigma',24))
    corrected=np.maximum(a-bg,0)
    corrected=ndi.gaussian_filter(corrected,config.get('smoothing_sigma',1.1))
    scale=float(np.percentile(corrected,99.5))
    # Do not amplify one/two display quantization levels into convincing rings.
    dtype=np.asarray(image).dtype
    quantum=1.0 if np.issubdtype(dtype,np.integer) else np.finfo(np.float32).eps*max(float(np.max(np.abs(a))),1)
    minimum=config.get('min_signal_steps',4)*quantum
    if scale<minimum:
        strong=corrected[corrected>=minimum]
        scale=float(np.percentile(strong,99)) if strong.size else 0.
    residual=a-ndi.gaussian_filter(a,1)
    bg=residual[a<=np.percentile(a,50)]
    noise=float(1.4826*np.median(np.abs(bg-np.median(bg)))) if bg.size else 0.
    result=np.clip(corrected/max(scale,1e-6),0,1.5) if scale>=minimum else np.zeros_like(a)
    info=dict(scale=scale,noise_sigma=noise,min_raw_contrast=max(config.get('min_boundary_steps',4)*quantum,config.get('min_signal_snr',3)*noise))
    return (result,info) if return_info else result

def ellipse_points(cx,cy,rx,ry,angle,radial,n=72):
    theta=np.linspace(0,2*np.pi,n,endpoint=False); rot=np.deg2rad(angle)
    x=rx*np.cos(theta)*radial; y=ry*np.sin(theta)*radial
    return cy+x*np.sin(rot)+y*np.cos(rot),cx+x*np.cos(rot)-y*np.sin(rot)

def evaluate_ellipse(a,mask,distance,ellipse,config):
    (cx,cy),(dx,dy),angle=ellipse; rx,ry=dx/2,dy/2
    if min(rx,ry)<config.get('min_radius',3) or max(rx,ry)>config.get('max_radius',38): return None
    if min(rx,ry)/max(rx,ry)<.4: return None
    ys,xs=ellipse_points(cx,cy,rx,ry,angle,1.0)
    margins=ndi.map_coordinates(distance,[ys,xs],order=1,mode='constant',cval=0)
    if np.mean(margins>=config.get('boundary_margin',5))<.95: return None
    profiles=[]
    for radius in [.85,1.0,1.12]:
        y,x=ellipse_points(cx,cy,rx,ry,angle,radius)
        profiles.append(ndi.map_coordinates(a,[y,x],order=1,mode='constant',cval=0))
    rim=np.max(profiles,axis=0)
    centers=[]; outs=[]
    for radius in [.0,.25,.45]:
        y,x=ellipse_points(cx,cy,rx,ry,angle,radius); centers.extend(ndi.map_coordinates(a,[y,x],order=1,mode='constant',cval=0))
    for radius in [1.4,1.6]:
        y,x=ellipse_points(cx,cy,rx,ry,angle,radius); outs.extend(ndi.map_coordinates(a,[y,x],order=1,mode='constant',cval=0))
    center=float(np.mean(centers)); boundary=float(np.mean(rim)); outside=float(np.mean(outs))
    contrast=boundary-center; outer=boundary-outside
    continuity=float(np.mean(rim>center+max(.06,config.get('min_contrast',.1)/2)))
    angular_valid=rim>center+max(.06,config.get('min_contrast',.1)/2)
    doubled=np.concatenate([~angular_valid,~angular_valid]);longest=run=0
    for missing in doubled:
        run=run+1 if missing else 0;longest=max(longest,run)
    gap_fraction=min(longest,len(rim))/len(rim)
    if gap_fraction>config.get('max_gap_fraction',.28): return None
    if contrast<config.get('min_contrast',.1) or continuity<config.get('min_continuity',.6) or outer<.025: return None
    score=float(np.clip(contrast/(boundary+.1),0,1)*continuity*np.clip(outer/(boundary+.1),0,1)**.5)
    thickness=float(np.mean(np.asarray(profiles)>.5*max(boundary,.01)))
    return dict(x=float(cx),y=float(cy),radius_x=float(rx),radius_y=float(ry),angle=float(angle),boundary_to_center_contrast=contrast,boundary_mean=boundary,center_mean=center,inner_outer_contrast=outer,continuity=continuity,max_gap_fraction=gap_fraction,ellipse_axis_ratio=float(min(rx,ry)/max(rx,ry)),ring_band_occupancy=thickness,ring_candidate_score=score)

def detect_rings(image,mask=None,config=None):
    config=config or {}; a,info=preprocess(image,config,return_info=True)
    if mask is None: mask=np.ones(a.shape,bool)
    distance=ndi.distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]
    proposals=[]
    if a.max()==0: return proposals
    for level in config.get('levels',[.15,.25,.38,.52,.68]):
        binary=((a>level)&mask).astype('uint8')
        contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if len(contour)<12: continue
            ellipse=cv2.fitEllipse(contour)
            candidate=evaluate_ellipse(a,mask,distance,ellipse,config)
            if candidate:
                candidate['raw_boundary_to_center_contrast']=candidate['boundary_to_center_contrast']*info['scale']
                candidate['noise_sigma']=info['noise_sigma'];candidate['min_raw_contrast']=info['min_raw_contrast']
                if candidate['raw_boundary_to_center_contrast']<info['min_raw_contrast']:continue
                candidate['contour_level']=level
                if candidate['ring_candidate_score']>=config.get('proposal_score',.18): proposals.append(candidate)
    proposals.sort(key=lambda c:c['ring_candidate_score'],reverse=True)
    keep=[]
    for c in proposals:
        if any(same_ring(c,k) for k in keep): continue
        keep.append(c)
    return keep

def same_ring(a,b):
    """Conservative spatial/scale NMS; do not merge separated concentric rings."""
    ra=np.sqrt(a['radius_x']*a['radius_y']);rb=np.sqrt(b['radius_x']*b['radius_y'])
    return np.hypot(a['x']-b['x'],a['y']-b['y'])<.45*min(ra,rb) and max(ra,rb)/min(ra,rb)<1.55

def detect_fov_rings(image,config=None):
    """Overlapping fluorescence tiles, independent of any cell mask."""
    config=config or {};size=int(config.get('tile_size',384));overlap=int(config.get('tile_overlap',112))
    if size<=2*overlap: raise ValueError('tile_size must exceed twice tile_overlap')
    h,w=image.shape;step=size-overlap;found=[]
    for y in range(0,h,step):
        for x in range(0,w,step):
            crop=image[y:min(h,y+size),x:min(w,x+size)]
            for r in detect_rings(crop,config=config):
                r=dict(r,x=r['x']+x,y=r['y']+y);found.append(r)
    found.sort(key=lambda r:r['ring_candidate_score'],reverse=True);keep=[]
    for r in found:
        if not any(same_ring(r,k) for k in keep):keep.append(r)
    return keep

def assign_rings(rings,labels,margin=5):
    """Preserve unassigned/edge proposals; never force them into a cell."""
    # Distance to EACH label boundary, including contact boundaries.
    from skimage.segmentation import find_boundaries
    distance=ndi.distance_transform_edt(np.pad((labels>0)&~find_boundaries(labels,mode='inner'),1))[1:-1,1:-1]
    for r in rings:
        y,x=ellipse_points(r['x'],r['y'],r['radius_x'],r['radius_y'],r['angle'],1)
        yy=np.clip(np.rint(y).astype(int),0,labels.shape[0]-1);xx=np.clip(np.rint(x).astype(int),0,labels.shape[1]-1)
        ids=labels[yy,xx];cid=int(labels[int(round(r['y'])),int(round(r['x']))])
        r['cell_id']='';r['association_qc']='unassigned';r['cell_overlap_fraction']=0.
        if cid:
            coverage=float(np.mean(ids==cid));clear=float(np.mean(distance[yy,xx]>=margin))
            r['cell_overlap_fraction']=coverage
            if coverage>=.95 and clear>=.90:r['cell_id']=cid;r['association_qc']='associated'
            else:r['association_qc']='boundary_or_overlap'
    return rings

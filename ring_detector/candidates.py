"""Ellipse/near-closed contour proposals scored by angular intensity profiles.

Scores are retrieval heuristics, NOT calibrated probabilities. No training.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi

def preprocess(image, config):
    a=np.asarray(image,np.float32)
    bg=ndi.gaussian_filter(a,config.get('background_sigma',24))
    corrected=np.maximum(a-bg,0)
    corrected=ndi.gaussian_filter(corrected,config.get('smoothing_sigma',1.1))
    scale=float(np.percentile(corrected,99.5))
    # Do not amplify one/two display quantization levels into convincing rings.
    dtype=np.asarray(image).dtype
    quantum=1.0 if np.issubdtype(dtype,np.integer) else np.finfo(np.float32).eps*max(float(np.max(np.abs(a))),1)
    if scale<config.get('min_signal_steps',4)*quantum: return np.zeros_like(a)
    return np.clip(corrected/scale,0,1.5)

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
    if contrast<config.get('min_contrast',.1) or continuity<config.get('min_continuity',.6) or outer<.025: return None
    score=float(np.clip(contrast/(boundary+.1),0,1)*continuity*np.clip(outer/(boundary+.1),0,1)**.5)
    thickness=float(np.mean(np.asarray(profiles)>.5*max(boundary,.01)))
    return dict(x=float(cx),y=float(cy),radius_x=float(rx),radius_y=float(ry),angle=float(angle),boundary_to_center_contrast=contrast,boundary_mean=boundary,center_mean=center,inner_outer_contrast=outer,continuity=continuity,ellipse_axis_ratio=float(min(rx,ry)/max(rx,ry)),ring_band_occupancy=thickness,ring_candidate_score=score)

def detect_rings(image,mask=None,config=None):
    config=config or {}; a=preprocess(image,config)
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
                candidate['contour_level']=level
                if candidate['ring_candidate_score']>=config.get('proposal_score',.18): proposals.append(candidate)
    proposals.sort(key=lambda c:c['ring_candidate_score'],reverse=True)
    keep=[]
    for c in proposals:
        if any(np.hypot(c['x']-k['x'],c['y']-k['y'])<.7*min(c['radius_x'],c['radius_y'],k['radius_x'],k['radius_y']) for k in keep): continue
        keep.append(c)
    return keep

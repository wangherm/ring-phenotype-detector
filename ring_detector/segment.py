"""Traditional morphology-only segmentation; conservative QC is mandatory."""
import numpy as np
from scipy import ndimage as ndi
from skimage import feature, morphology as morph, measure, segmentation
from .io import normalize

def segment_classical(image, config):
    a=normalize(image)
    smooth=ndi.gaussian_filter(a,config.get('edge_sigma',2))
    edges=feature.canny(smooth,sigma=1,low_threshold=config.get('edge_low',.05),high_threshold=config.get('edge_high',.12))
    closed=morph.closing(edges,morph.disk(config.get('closing_radius',4)))
    filled=ndi.binary_fill_holes(closed)
    components,n=ndi.label(filled); sizes=np.bincount(components.ravel()); sizes[0]=0
    filled=sizes[components]>=config.get('min_area',900)
    filled=morph.opening(filled,morph.disk(2))
    distance=ndi.distance_transform_edt(filled)
    # Closed dark cell rims enclose separate interiors even when outer cell
    # silhouettes touch. Use those interiors first; distance-only seeds merge
    # dense rod-shaped cells into large clusters.
    interiors=filled & ~closed
    inner_labels,n=ndi.label(interiors); inner_sizes=np.bincount(inner_labels.ravel());inner_sizes[0]=0
    valid=(inner_sizes>=config.get('min_seed_area',2000))&(inner_sizes<=config.get('max_area',40000))
    markers=measure.label(valid[inner_labels])
    # Blurred edge objects can have no closed interior: fall back to distance
    # maxima, but only in connected objects with no interior marker.
    outer_labels,n=ndi.label(filled)
    covered=np.unique(outer_labels[markers>0]);uncovered=filled & ~np.isin(outer_labels,covered)
    peaks=morph.h_maxima(ndi.gaussian_filter(distance,2),config.get('seed_prominence',12)) & uncovered
    extra=measure.label(peaks);markers[extra>0]=extra[extra>0]+markers.max()
    labels=segmentation.watershed(-distance,markers,mask=filled)
    return extract_cells(labels,image,config)

def extract_cells(labels,image,config):
    a=normalize(image);cells=[]; clean=np.zeros(labels.shape,dtype=np.uint16)
    for r in measure.regionprops(labels,intensity_image=a):
        if r.area<config.get('min_area',900): continue
        cid=len(cells)+1
        y1,x1,y2,x2=r.bbox; mask=labels[y1:y2,x1:x2]==r.label
        clean[y1:y2,x1:x2][mask]=cid
        grad=np.hypot(ndi.sobel(a[y1:y2,x1:x2],axis=0),ndi.sobel(a[y1:y2,x1:x2],axis=1))
        focus=float(np.mean(grad[mask])); flags=[]
        if min(x1,y1)==0 or x2==a.shape[1] or y2==a.shape[0]: flags.append('edge_truncated')
        if r.area>config.get('max_area',40000): flags.append('large_merged')
        if r.solidity<.80: flags.append('irregular_or_overlap')
        if focus<config.get('min_focus',.025): flags.append('low_focus')
        cells.append(dict(cell_id=cid,bbox_x1=x1,bbox_y1=y1,bbox_x2=x2,bbox_y2=y2,centroid_x=float(r.centroid[1]),centroid_y=float(r.centroid[0]),area_px=int(r.area),solidity=float(r.solidity),focus_score=focus,segmentation_qc=';'.join(flags) or 'pass',mask_label=cid))
    areas=[c['area_px'] for c in cells if 'edge_truncated' not in c['segmentation_qc']]
    typical=np.median(areas) if areas else 0
    for c in cells:
        if typical and c['area_px']<typical*config.get('small_fragment_fraction',.30):c['segmentation_qc']=('' if c['segmentation_qc']=='pass' else c['segmentation_qc']+';')+'small_fragment'
    return clean,cells

def segment_cells(image,config,model=None):
    backend=config.get('backend','classical')
    if backend=='classical':return segment_classical(image,config)
    if backend!='cellpose':raise ValueError(f'Unknown segmentation backend: {backend}')
    if model is None:model=load_cellpose(config)
    import cv2
    scale=float(config.get('inference_scale',.5))
    if not 0<scale<=1:raise ValueError('inference_scale must be in (0, 1]')
    small=cv2.resize(image.astype('float32'),None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else image
    masks,_,_=model.eval(small,diameter=config.get('diameter',100)*scale,channels=[0,0],normalize=True,flow_threshold=config.get('flow_threshold',.4),cellprob_threshold=config.get('cellprob_threshold',0),min_size=max(15,int(config.get('min_area',900)*scale**2)))
    if scale<1:masks=cv2.resize(masks.astype('float32'),(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST).astype('uint16')
    return extract_cells(masks,image,config)

def load_cellpose(config):
    import os
    from pathlib import Path
    os.environ.setdefault('CELLPOSE_LOCAL_MODELS_PATH',str(Path(config.get('model_cache','outputs/model_cache')).resolve()))
    try:
        import torch
        from cellpose import models
    except ImportError as exc:
        raise RuntimeError('Cellpose backend requires: pip install -r requirements-cellpose.txt. Select configs/classical.yaml to use the old baseline; no silent fallback.') from exc
    torch.set_num_threads(int(config.get('cpu_threads',2)))
    if config.get('model_type','yeast_PhC_cp3') not in models.MODEL_NAMES:raise ValueError('Unknown Cellpose pretrained model name; no silent model substitution')
    return models.CellposeModel(gpu=config.get('gpu',False),model_type=config.get('model_type','yeast_PhC_cp3'))

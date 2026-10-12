import numpy as np,pytest
import tifffile
from ring_detector.candidates import detect_rings,detect_fov_rings,assign_rings,evaluate_ellipse
from ring_detector.annotations import validate_annotations
from ring_detector.io import write_csv
from ring_detector.segment import extract_cells
from ring_detector.pipeline import run_propose

def test_global_ring_survives_bad_cell_mask_and_tile_nms():
    y,x=np.mgrid[:520,:520];r=np.hypot(x-285,y-270);a=(np.exp(-((r-18)/2)**2)*200).astype('uint8')
    rings=detect_fov_rings(a);assert len(rings)==1
    assign_rings(rings,np.zeros(a.shape,dtype='uint16'))
    assert len(rings)==1 and rings[0]['cell_id']=='' and rings[0]['association_qc']=='unassigned'

def test_small_isolated_ring_not_lost_to_large_tile_normalization():
    y,x=np.mgrid[:520,:520];r=np.hypot(x-285,y-270)
    a=(np.exp(-((r-6)/1.1)**2)*200).astype('uint8')
    rings=detect_fov_rings(a)
    assert rings and min(np.hypot(c['x']-285,c['y']-270) for c in rings)<3

def test_raw_contrast_gate_survives_low_normalization_override():
    y,x=np.mgrid[:128,:128];r=np.hypot(x-64,y-64)
    weak=(np.exp(-((r-18)/2)**2)*3).astype('uint8')
    assert not detect_rings(weak,config={'min_signal_steps':.1,'min_boundary_steps':4})

def test_association_inside_and_contact_edge():
    labels=np.zeros((128,128),np.uint16);labels[10:118,10:64]=1;labels[10:118,64:118]=2
    rings=[dict(x=35.,y=64.,radius_x=10.,radius_y=10.,angle=0.),dict(x=63.,y=64.,radius_x=10.,radius_y=10.,angle=0.)]
    assign_rings(rings,labels)
    assert rings[0]['cell_id']==1
    assert rings[1]['cell_id']=='' and rings[1]['association_qc']=='boundary_or_overlap'

def test_two_bright_arcs_not_a_near_closed_ring():
    y,x=np.mgrid[:128,:128];rad=np.hypot(x-64,y-64);theta=np.arctan2(y-64,x-64)
    a=np.exp(-((rad-18)/2)**2);a[np.abs(theta)<1.2]=0
    mask=np.ones(a.shape,bool);dist=np.ones(a.shape)*50
    assert evaluate_ellipse(a,mask,dist,((64.,64.),(36.,36.),0.),{'max_gap_fraction':.28}) is None

def test_old_annotations_rejected_after_segmentation_change(tmp_path):
    cells=tmp_path/'cells.csv';ann=tmp_path/'ann.csv'
    write_csv(cells,[dict(fov_id='f1',cell_id='1',cell_uid='new-mask')]);write_csv(ann,[dict(fov_id='f1',cell_id='1',cell_uid='old-mask',label='positive')])
    with pytest.raises(ValueError,match='Stale'):validate_annotations(ann,cells)

def test_changed_mask_file_rejected_before_candidate_assignment(tmp_path):
    image=tmp_path/'source.tif';tifffile.imwrite(image,np.zeros((64,64),np.uint8))
    segments=tmp_path/'segments';(segments/'masks').mkdir(parents=True)
    tifffile.imwrite(segments/'masks/f1.tif',np.zeros((64,64),np.uint16))
    write_csv(segments/'cells.csv',[dict(fov_id='f1',cell_id='1',cell_uid='old-mask')])
    manifest=tmp_path/'manifest.csv';write_csv(manifest,[dict(fov_id='f1',date='unknown',status='complete',c1_path=str(image),c2_path=str(image))])
    with pytest.raises(ValueError,match='masks/source changed'):run_propose(manifest,segments,dict(segmentation={'backend':'classical'},candidate={}),tmp_path/'out')

def test_small_gap_objects_flagged_against_cell_size():
    labels=np.zeros((150,150),np.uint16);labels[5:45,5:45]=1;labels[55:95,5:45]=2;labels[10:20,90:100]=3
    masks,cells=extract_cells(labels,np.zeros(labels.shape),dict(min_area=50,min_focus=0))
    assert 'small_fragment' in cells[2]['segmentation_qc']
    assert cells[0]['segmentation_qc']=='pass'

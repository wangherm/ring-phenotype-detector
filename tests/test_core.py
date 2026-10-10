from pathlib import Path
import numpy as np
import pytest,tifffile
from ring_detector.io import fluorescence,pair_paths,load_planes,write_csv
from ring_detector.candidates import detect_rings
from ring_detector.annotations import validate_annotations

def synthetic(kind='ring'):
    y,x=np.mgrid[:128,:128];r=np.sqrt(((x-64)/1.3)**2+(y-64)**2)
    if kind=='circle': r=np.hypot(x-64,y-64);a=np.exp(-((r-17)/2)**2)
    elif kind=='ring': a=np.exp(-((r-17)/2)**2)
    elif kind=='partial': a=np.exp(-((r-17)/2)**2);a[(x>70)&(y<53)]=0
    elif kind=='disk': a=(r<17).astype(float)
    elif kind=='diffuse': a=np.exp(-(r/22)**2)
    else:a=np.zeros(r.shape)
    return (a*50000).astype('uint16')

@pytest.mark.parametrize('kind',['circle','ring','partial'])
def test_ring_and_ellipse(kind):
    rs=detect_rings(synthetic(kind));assert rs;assert max(r['continuity'] for r in rs)>=.6

@pytest.mark.parametrize('kind',['disk','diffuse','blank'])
def test_non_ring_shapes(kind): assert not detect_rings(synthetic(kind))

@pytest.mark.parametrize('dtype',['uint8','uint16'])
def test_color_permutation_and_uint16(tmp_path,dtype):
    intensity=synthetic();intensity=(intensity//256).astype('uint8') if dtype=='uint8' else intensity;results=[]
    for k in range(3):
        rgb=np.zeros((*intensity.shape,3),dtype=dtype);rgb[...,k]=intensity
        p=tmp_path/f'color{k}.tif';tifffile.imwrite(p,rgb,photometric='rgb')
        planes,meta=fluorescence(p);assert meta['selected_channels']==[['R','G','B'][k]]
        assert np.array_equal(next(iter(planes.values())),intensity)
        results.append(detect_rings(next(iter(planes.values()))))
    p=tmp_path/'gray.tif';tifffile.imwrite(p,intensity);planes,_=fluorescence(p);assert planes['gray'].dtype==np.dtype(dtype)
    results.append(detect_rings(planes['gray']))
    assert results[0]==results[1]==results[2]==results[3]

def test_multichannel_no_mix(tmp_path):
    a=np.stack([synthetic(),synthetic('disk')]);p=tmp_path/'channels.tif';tifffile.imwrite(p,a,metadata={'axes':'CYX'},photometric='minisblack')
    planes,_=fluorescence(p,['0','1']);assert detect_rings(planes['0']);assert not detect_rings(planes['1'])
    with pytest.raises(ValueError,match='cannot mix'):fluorescence(p,['gray'])

def test_z_stack_rejected(tmp_path):
    p=tmp_path/'z.tif';tifffile.imwrite(p,np.zeros((3,64,64),np.uint16),metadata={'axes':'ZYX'},photometric='minisblack')
    with pytest.raises(ValueError,match='Unsupported axes'):load_planes(p)

def test_pairing_complete_stem():
    paths=[f'{prefix}_{c}_suffix.tif' for prefix in ['date-a','date-b'] for c in ['c1','c2','c1-2']]
    groups,issues=pair_paths(paths);assert len(groups)==2 and not issues
    assert all(len(g)==3 for g in groups.values());assert 'date-a_suffix' in groups
    with pytest.raises(ValueError,match='Duplicate'):pair_paths(paths+[paths[0]])

def test_cell_boundary_ring_exclusion():
    image=synthetic();y,x=np.mgrid[:128,:128];mask=((x-64)/1.3)**2+(y-64)**2<19**2
    assert not detect_rings(image,mask)

def test_quantization_noise_not_amplified():
    weak=(synthetic().astype(float)/50000*2).astype('uint8')
    assert not detect_rings(weak)

def test_annotation_mapping(tmp_path):
    cells=tmp_path/'cells.csv';ann=tmp_path/'ann.csv';write_csv(cells,[dict(fov_id='f1',cell_id='1')]);write_csv(ann,[dict(fov_id='f1',cell_id='1',label='uncertain')]);assert validate_annotations(ann,cells)[0]['label']=='uncertain'
    write_csv(ann,[dict(fov_id='f2',cell_id='1',label='positive')])
    with pytest.raises(ValueError,match='Unmapped'):validate_annotations(ann,cells)

"""Exact tensor mapping for this frozen Corrector checkpoint across Keras2.15 builds.

No missing or extra tensors are accepted. The original checkpoint stays untouched.
"""
from pathlib import Path
import shutil
import h5py
import numpy as np

def mapping(model):
    result=[]
    assert len(model.blocks)==4 and len(model.hidden.layers)==2
    for i,block in enumerate(model.blocks):
        prefix='layers/sequential'+('' if i==0 else '_'+str(i))+'/layers/'
        for layer,key in [(block.layers[0],'conv1d'),(block.layers[1],'layer_normalization'),(block.layers[3],'conv1d_1')]:
            result.append((layer,prefix+key+'/vars'))
    result.extend([(model.hidden.layers[0],'layers/sequential_4/layers/dense/vars'),
                   (model.hidden.layers[1],'layers/sequential_4/layers/dense_1/vars'),(model.final,'layers/dense/vars')])
    return result

def load_corrector_exact(model,path):
    seen=set()
    with h5py.File(path,'r') as f:
        actual=set()
        f.visititems(lambda name,item:actual.add(name) if isinstance(item,h5py.Dataset) else None)
        for layer,prefix in mapping(model):
            arrays=[]
            for i,variable in enumerate(layer.weights):
                name=prefix+'/'+str(i);assert name in actual,name
                value=np.asarray(f[name]);assert tuple(value.shape)==tuple(variable.shape),(name,value.shape,variable.shape)
                arrays.append(value);seen.add(name)
            layer.set_weights(arrays)
            for a,b in zip(arrays,layer.get_weights()):assert np.array_equal(a,b)
        assert actual==seen,actual-seen
    return dict(tensors=len(seen),all_tensors_exact=True)

def save_corrector_exact(model,path,template):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    shutil.copyfile(template,path)
    with h5py.File(path,'r+') as f:
        for layer,prefix in mapping(model):
            for i,value in enumerate(layer.get_weights()):
                name=prefix+'/'+str(i);assert name in f
                del f[name];f.create_dataset(name,data=value)
    # Check every stored tensor again; do not silently serialize a partial model.
    with h5py.File(path,'r') as f:
        for layer,prefix in mapping(model):
            for i,value in enumerate(layer.get_weights()):assert np.array_equal(f[prefix+'/'+str(i)][:],value)

"""Strict frozen cnn_multi proposal tensor mapping; never alters the checkpoint."""
import h5py
import numpy as np

def mapping(model):
    assert model.architecture=='cnn_multi' and model.hypotheses==6
    counts={};result=[]
    kinds={'Conv1D':'conv1d','LayerNormalization':'layer_normalization','Dense':'dense'}
    for layer in model.encoder.layers:
        if not layer.weights:continue
        kind=kinds[type(layer).__name__];index=counts.get(kind,0);counts[kind]=index+1
        name=kind+('' if index==0 else '_'+str(index))
        result.append((layer,'layers/functional/layers/'+name+'/vars'))
    assert counts==dict(conv1d=13,layer_normalization=4,dense=1),counts
    result.extend([(model.classifier,'layers/dense/vars'),(model.embedding,'layers/embedding/vars')])
    assert len(model.decoder.layers)==3
    for i,layer in enumerate(model.decoder.layers):result.append((layer,'layers/sequential/layers/dense'+('' if i==0 else '_'+str(i))+'/vars'))
    return result

def load_proposal_exact(model,path):
    seen=set()
    with h5py.File(path,'r') as f:
        actual=set();f.visititems(lambda n,v:actual.add(n) if isinstance(v,h5py.Dataset) else None)
        for layer,prefix in mapping(model):
            arrays=[]
            for i,variable in enumerate(layer.weights):
                name=prefix+'/'+str(i);assert name in actual,name
                value=np.asarray(f[name]);assert tuple(value.shape)==tuple(variable.shape),(name,value.shape,variable.shape)
                arrays.append(value);seen.add(name)
            layer.set_weights(arrays);assert all(np.array_equal(a,b) for a,b in zip(arrays,layer.get_weights()))
        assert seen==actual,(seen-actual,actual-seen)
    return dict(tensors=len(seen),all_tensors_exact=True)

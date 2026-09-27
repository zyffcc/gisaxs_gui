"""Exact, user-supplied component multiset; no implied partial-prior semantics."""
import numpy as np

TYPE_NAMES = {1:'sphere', 2:'random_cylinder', 3:'vertical_cylinder'}


def resolve_components(components, combos):
    if isinstance(components,(str,bytes)):
        raise ValueError('Provide a list, e.g. ["sphere", "random_cylinder"]')
    tokens=list(components)
    if not 1<=len(tokens)<=4:
        raise ValueError('Specify the complete multiset of 1 to 4 components, including repeats')
    names={**{str(k):k for k in TYPE_NAMES},**{v:k for k,v in TYPE_NAMES.items()}}
    types=[]
    for token in tokens:
        if isinstance(token,(bool,np.bool_)) or str(token) not in names:
            raise ValueError(f'Unsupported component {token!r}; use 1/sphere, 2/random_cylinder, 3/vertical_cylinder')
        types.append(names[str(token)])
    types=sorted(types);padded=types+[0]*(4-len(types))
    found=np.flatnonzero(np.all(np.asarray(combos)==padded,axis=1))
    if len(found)!=1:raise ValueError('Requested component combination is not supported by this model')
    return int(found[0]),types

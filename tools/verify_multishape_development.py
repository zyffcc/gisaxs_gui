"""Audit the portable development snapshot in isolated interpreter processes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = r'''
import sys,json,hashlib
from pathlib import Path
root=Path(sys.argv[1]).resolve();sys.path.insert(0,str(root))
import numpy as np
import tensorflow as tf
from research.multishape_v2 import feature_controls,new_model,weight_io
from research.multishape_global_context import model as context_model,weight_io as context_io
with np.load(root/'research/multishape_v1/results/data/dev.npz',allow_pickle=False) as archive:
    data={k:archive[k][:2] for k in ('q','observed','sigma','count','mask')}
records={}; arrays={}
for name,relative,context in (
    ('r3','research/multishape_online/results/models/point_stream',False),
    ('r5_baseline','research/multishape_global_context/results/models/baseline',True),
    ('r5_global','research/multishape_global_context/results/models/global',True)):
    directory=root/relative
    config=json.loads((directory/'model_config.json').read_text())
    model=(context_model.build_model if context else new_model.build_model)(**config)
    (context_io.load if context else weight_io.load)(model,directory/'curve_best.npz')
    x=feature_controls.prepare_from_arrays(data,encoder=config['encoder'],normalization=config['normalization'])
    out=model({**x,'combo':np.array([0,33],np.int32)},training=False)
    for key in ('u','logits'):arrays[name+'_'+key]=out[key].numpy()
    records[name]={'checkpoint_sha256':hashlib.sha256((directory/'curve_best.npz').read_bytes()).hexdigest(),'parameters':model.count_params()}
modules={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
         if name.startswith('research.') and getattr(module,'__file__',None)}
assert all(Path(path).is_relative_to(root) for path in modules.values())
np.savez_compressed(sys.argv[2],**arrays)
Path(sys.argv[3]).write_text(json.dumps({'models':records,'loaded_research_modules':modules,'tf':tf.__version__},indent=2))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.snapshot/'MANIFEST.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'development_only_not_stable'
    assert not manifest['test_included'] and not manifest['gui_model_promoted']
    for relative, expected in manifest['copied_files'].items():
        path = (args.snapshot/relative).resolve()
        assert path.is_relative_to(args.snapshot.resolve())
        assert path.stat().st_size == expected['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256']
    environment = dict(os.environ, OMP_NUM_THREADS='2', TF_NUM_INTRAOP_THREADS='2',
                       TF_NUM_INTEROP_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       TF_CPP_MIN_LOG_LEVEL='2', TF_DETERMINISTIC_OPS='1',
                       PYTHONDONTWRITEBYTECODE='1', TF_ENABLE_ONEDNN_OPTS='0')
    for name, source in (('original', ROOT), ('relocated', args.snapshot)):
        result = subprocess.run([sys.executable, '-I', '-B', '-c', PROGRAM, str(source.resolve()),
                                 str((args.out/(name+'.npz')).resolve()),
                                 str((args.out/(name+'.json')).resolve())],
                                cwd=source, env=environment, capture_output=True, text=True)
        (args.out/(name+'.log')).write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{name} isolated load failed; see its log')
    differences = {}
    with np.load(args.out/'original.npz', allow_pickle=False) as left, np.load(args.out/'relocated.npz', allow_pickle=False) as right:
        assert left.files == right.files
        for key in left.files:
            np.testing.assert_array_equal(left[key], right[key])
            differences[key] = float(np.max(np.abs(left[key]-right[key])))
    first, second = [json.loads((args.out/(name+'.json')).read_text()) for name in ('original', 'relocated')]
    assert first['models'] == second['models']
    # -B prevents further bytecode writes. A previous -I-only audit can leave
    # interpreter caches; these are not source/model artifacts in the manifest.
    caches = [str(p.relative_to(args.snapshot)) for p in args.snapshot.rglob('*.pyc')]
    for relative, expected in manifest['copied_files'].items():
        assert hashlib.sha256((args.snapshot/relative).read_bytes()).hexdigest() == expected['sha256']
    output = dict(passed=True, manifest_files=len(manifest['copied_files']),
                  model_outputs_max_abs_difference=differences, models=second['models'],
                  isolated_imports=True, incidental_python_bytecode=caches,
                  no_training_or_physics_forward=True,
                  no_test_access=True, gui_model_promoted=False)
    (args.out/'AUDIT.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output))


if __name__ == '__main__':
    main()

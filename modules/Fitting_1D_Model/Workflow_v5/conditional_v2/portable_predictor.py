"""Local compatibility adapter for the unchanged frozen prediction bundle."""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from predict_1d import Predictor
from benchmark_model import ProposalModel
from correction_model import Corrector
from diagnose_v5 import proposals
from input_representation import prepare
from portable_proposal_weights import load_proposal_exact
from portable_corrector_weights import load_corrector_exact

class PortablePredictor(Predictor):
    def __init__(self,bundle,data):
        self.bundle=Path(bundle);self.config=json.loads((self.bundle/'model.json').read_text())
        self.output_policy=json.loads((self.bundle/'output_policy.json').read_text())
        directory=self.bundle/self.config['base_dir'];bc=json.loads((directory/'config.json').read_text())
        self.base=ProposalModel(bc['architecture'],bc['hypotheses'])
        self.base({'curve':data['curve'][:1],'context':data['context'][:1],'combo':np.zeros(1,'int32')})
        self.portable_load_audit={'base':load_proposal_exact(self.base,directory/'best.weights.h5')}
        self.corrector=None
        if self.config['passes']:
            self.corrector=Corrector(True)
            warm={k:v[:1] for k,v in data.items()};cand=proposals(self.base,warm,topk=self.config['topk'])
            self.corrector({k:tf.convert_to_tensor(v[:1]) for k,v in prepare(warm,cand,self.config.get('representation','log512')).items()})
            self.portable_load_audit['corrector']=load_corrector_exact(self.corrector,self.bundle/self.config['corrector_dir']/'best.weights.h5')

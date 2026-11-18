"""The proposal network of a 1D cut and the posterior it starts (numpy).

``encoding`` turns a curve, compositions and parameters into what the network sees and back; ``flow`` evaluates an
exported network (no PyTorch needed); ``infer`` weights the network's samples against cut_model (importance sampling,
evidence, modes); ``refine`` turns them into Fitting's models and solutions (``predict``). The training is in
``tools/ml1d``.
"""

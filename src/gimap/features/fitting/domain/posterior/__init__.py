"""The proposal network of a 1D cut: an amortised posterior of a population's parameters given the curve (numpy).

``encoding`` turns a curve and parameters into what the network sees and back; ``flow`` evaluates an exported
network (no PyTorch needed). The training is in ``tools/ml1d``.
"""

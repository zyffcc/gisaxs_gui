# Fitting presentation bindings

This package splits the legacy all-in-one Qt binding by user-facing concern.
Modules may coordinate widgets and the injected `FittingViewModel`; they must
not introduce scientific algorithms, concrete storage, TensorFlow, or
BornAgain calls.

`FittingViewBinding` composes these focused mixins so existing Qt signals and
legacy entry points keep the same public surface during incremental migration.

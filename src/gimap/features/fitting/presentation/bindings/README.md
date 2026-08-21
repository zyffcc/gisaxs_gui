# Fitting presentation bindings

This package owns focused Qt bindings grouped by user-facing concern. Modules
may coordinate widgets and the injected `FittingViewModel`; they must not
introduce scientific algorithms, concrete storage, TensorFlow, or BornAgain
calls.

`FittingViewBinding` composes these focused mixins and keeps the public Qt
signal surface stable.

# Calibration infrastructure

Adapters 在这里实现 application ports：复用现有 detector image loader 和经过回归
测试的 calibration engine，读写 calibration JSON / detector catalog，并将
`global_params` 隔离在 geometry parameter adapter 中。

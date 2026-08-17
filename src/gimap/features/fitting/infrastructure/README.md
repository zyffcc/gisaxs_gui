# Fitting infrastructure

本层实现 application ports，包括现有 CBF/NXS/TIFF、1D 文本、fitting TXT/CSV、AI
candidate JSON，以及由 JobRunner 隔离的 AI pipeline。它可以依赖 fabio、h5py、Pillow
和本地文件系统，但不能被 domain 导入。

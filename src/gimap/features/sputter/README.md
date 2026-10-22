# `sputter`

Workspace **Sputter** (sidebar). Particle size R and centre-to-centre spacing D of an in-situ sputter series on a
polymer, window by window, and how far each value can be trusted.

- **Open Results Folder…** shows a toolkit result folder (`structure.csv`, `structure_summary.json`).
- **Analyze a Scan** runs the external *sputter toolkit* (`sputter_struct_v1`) on a folder of CBF frames: exact
  0.5 s count windows (0.1 s for the first 5 s), direct features, polymer/TS networks. The toolkit runs in its own
  Python with PyTorch (chosen under **Toolkit**), so GIMaP needs no PyTorch; progress arrives as
  `SPUTTER_PROGRESS` lines, Stop terminates the process and finished steps are reused.
- **Add Refined Windows…** merges physical-refinement results (one JSON per window, e.g. computed on a GPU
  cluster with `scripts/refine_polymer_review_v1.py` of the toolkit).

Every R and D value carries a grade — **A** converged physical refinement with a unique solution, **B** refinement
with an equally good alternative, **C** network estimate only — the simulation-based typical error of that source,
and the share of window pairs put in the right order. Missing values give their reason (no resolved peak, late TS
window). R is the base radius of a flat-based hemisphere (h = R); burial depth and polymer thickness are not
reliable and are not plotted.

Layers: `domain` (rows, series, reliability report), `application` (`SputterService`, `AnalyzeRequest`, progress
parsing, ports), `infrastructure` (`SubprocessToolkit`, `StructureFolderReader`), `presentation` (`SputterPage`),
`bootstrap.py`. Tests: `tests/test_sputter_feature.py`.

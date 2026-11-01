# Public test data

Real detector frames downloaded on 2026-09-29 for GIMaP's tests (`tests/test_public_data.py`).
They are not GIMaP's data: keep the licences and the attribution below when you redistribute them.
`THIRD_PARTY_NOTICES.md` at the top of the repository lists them too.

| Folder | Files | Source | Licence | What it is |
|---|---|---|---|---|
| `gisaxs_galaxi/` | `galaxi_data.tif` (decompressed from `galaxi_data.tif.gz`) | BornAgain r23 test data, `https://jugit.fz-juelich.de/mlz/bornagain/-/raw/r23/testdata/scatter2d/galaxi_data.tif.gz` | GPL-3.0 (BornAgain) | GISAXS, GALAXI (Jülich), Pilatus 1M, 981 × 1043 px |
| `saxs_agbh_pyfai/` | `Pilatus1M.edf`, `Pilatus1M.poni` | pyFAI test images, `http://www.silx.org/pub/pyFAI/testimages/` | MIT (pyFAI); the licence of the test-image server's files is not stated there: to be confirmed | Silver behenate calibration image (ESRF BM26, Pilatus 1M) |
| `giwaxs_p08_mapi/` | `S121_MAI_A2_00841.tif`, `LaB6_2021_12_DESY_P08.poni`, `S121_MAI_A2_metadata.yaml` | pygid usage examples, Zenodo 10.5281/zenodo.17466183 (attribution below) | CC BY 4.0 | GIWAXS of a MAPbI₃ film, DESY P08, PerkinElmer 2048², αi = 0.075° |
| `giwaxs_xeuss_pbte/` | `PbTe_3010_r1_PT_1_10_0_00028.edf`, `mask_PbTe_3010.edf`, `AgBe_250403_XENOCS.poni` | same Zenodo record | CC BY 4.0 | GIWAXS of PbTe nanoplatelets, Xeuss 2.0 lab source, Pilatus 300K, αi = 0.235° |

`galaxi_bornagain.poni` and `manifest.json` are GIMaP's own files (see below).

## Attribution of the pygid examples (CC BY 4.0)

- **Creators**: Ainur Abukaev, Constantin Völter, Mikhail Romodin, Sebastian Schwartzkopff,
  Florian Bertram, Oleg Konovalov, Alexander Hinderhofer, Dmitry Lapkin, Frank Schreiber.
- **Title**: "pygid: Fast Preprocessing of Grazing Incidence Diffraction Data (GID). Usage examples."
  Zenodo, 2025-10-28.
- **DOI**: https://doi.org/10.5281/zenodo.17466183
- **Licence**: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/
- **Changes**: none. Six of the record's 25 files are copied unchanged; their MD5 checksums are the ones the
  Zenodo record lists (checked 2026-11-01). Only the folder names `giwaxs_p08_mapi/` and
  `giwaxs_xeuss_pbte/` are GIMaP's.

## Geometry (as documented by the sources)

- **GALAXI** (BornAgain example `fit/scatter2d/experiment_at_galaxi`): λ = 1.34 Å, αi = 0.463°, sample–detector
  1730 mm, pixel 0.172 mm, direct beam at (597.1, 323.4) px counted from the **bottom-left**, i.e.
  (597.1, 719.6) in GIMaP's canonical pixels (row 0 at the top). Pilatus module gaps are −1.
  `galaxi_bornagain.poni` is written by GIMaP from these documented values (no tilt), for
  `tools/gimap_agent.py auto … --technique gisaxs --calibration …`. BornAgain's sample model: Ag spheres
  (R 5.75 nm, log-normal σ 0.4) with a radial paracrystal of peak distance 53.6 nm.
- **AgBh** (pyFAI): the `.poni` states λ = 1.0 Å and L = 1583.2 mm with 0.57° tilt. Measured ring radii of
  orders 1–5 around the `.poni` direct beam (180.0, 263.9) px give L = 1635 mm at 1.0 Å, so the stated
  wavelength is not the one of the calibration (≈ 1.033 Å would give 1583 mm). Use the image for the beam
  centre and for ring self-consistency, not for an absolute distance.
- **P08 MAPbI₃**: `.poni` → direct beam (545.5, 1826.9) px, L = 811.8 mm, λ = 0.6888 Å, tilt 0.31°.
  Radial peaks at q ≈ 0.899 (PbI₂ 001), 0.999 (MAPbI₃ 110), 1.41, 1.99, 2.23 Å⁻¹. The raw frame is
  dark-subtracted floating-point data: negative values are real (noise), not detector gaps.
- **Xeuss PbTe**: `.poni` → direct beam (7.0, 589.5) px, L = 147.6 mm, λ = 1.54 Å, **6° detector tilt**:
  GIMaP's flat-detector geometry is only approximate for it (kept on purpose as a tilt case).
  The EDF header says `Center_1/2` = (8.3, 590.39), `SampleDistance` = 0.37 m, `Dummy` = −1.5 ± 0.6.

`manifest.json` holds the same values for the tests.

# First use of GIMaP on four public detector frames

Three simulated first-time users followed the English User Manual in the desktop application on
22 November 2026. The baseline was commit `7fc06ff`. Each pass started with a fresh user directory
and temporary copies of the public frames. The GIWAXS user opened the second sample in the same
session. The evidence consists of offscreen Qt widget captures, action logs and the files saved
through the GUI. Screenshot names below are relative to the temporary evidence folder.

An action is one control activation (click, menu choice, shortcut or confirmation) or one entered
value (a complete number or path). A file chooser opened and confirmed therefore contributes an
activation for opening it, another for confirmation, and one entry if the user typed a name.
Dialog counts are shown separately. The main windows measured 1536 × 960 (GISAXS), 1440 × 920
(AgBh) and 1440 × 900 (GIWAXS). Qt reported an 800 × 600 virtual screen for the calibration tool,
which opened at its 900 × 560 minimum. Waiting for analysis, reading the manual, inspecting a plot
and capturing a screenshot add no actions. Counts describe these particular routes. They do not
measure reading time or predict the fastest route an experienced user could take.

The student normally measures Au on Si. The GALAXI example represents Ag particles and was fitted
as the public example. The AgBh folder has one exposure, so the beamline user opened a renamed
copy in a different folder to test geometry transfer. P08 and Xeuss are separate experiments.
The CIFs used with them are candidate comparison models.

## 1. Radius and spacing from a GISAXS cut

The student followed manual §§1.3, 5.2–5.4, 5.8 and 6.2–6.4.
Screens below are in `gisaxs/before` and `gisaxs/after`.

| Step | What the student saw and did | Screenshot evidence |
|---|---|---|
| Open and check geometry | Opened the TIFF with Ctrl+O. The matching PONI beside it supplied 1730 mm and λ = 1.34 Å immediately. The manual instructed the student to look for a missing-geometry banner. The status directed the student to Geometry to set αi, although the editable box was above the image. Entered the documented 0.463° there. | `01-start.png`, `02-open-dialog.png`, `03-open.png`, `04-geometry.png`, `05-incidence.png` in both passes |
| Mask and take the horizontal cut | Inspected and kept the applied exclusions, 13,832 pixels (1.4%). The horizontal band at rows 605–609 crossed the Yoneda maximum. Entering αi changed the reported exit angle from 0.639° to 0.176°. Compared the two sides of I(qy) and sent the signed cut to Fitting. | `06-mask.png`, `07-cuts.png`, `08-handoff.png` in both |
| Predict and refine | The handoff showed both halves (968 points). Before the fixes the toolbar said “Fit”, the rail still asked for a method, and the student reselected the recommended method's existing particle choice. Predict switched to the mean (598 points) when it started and fitted 596 points within its q window. Refine ran on all 598. The replay showed the averaging rule before the run and named the method on the toolbar. | Before: `09-fit-method.png` through `12-refined.png`. After: `09-predicting.png`, `10-predicted.png`, `11-refined.png` |
| Interpret and save | The one-sphere model gave R = 4.943 nm and D = 522.2 nm, with refined χ²ᵣ = 2.3922 in both passes. Predict flagged D as unconstrained and Refine placed it at a bound. A dash in the original error column hid that distinction. D cannot establish a particle spacing here. The replay showed the warning at D and provided parameter definitions on hover. Saved the curve and JSON with one command and a PDF with the fitted curve and residuals. The baseline also saved a redundant reusable-model JSON. | Before: `13-data.png`, `14-model.png`, `15-figure.png`. After: `12-data.png`, `13-figure.png` and their `-dialog.png` captures |

The two exported curve CSVs were byte-identical. The fit-input DAT uses Å⁻¹ and records the
source of the cut. The Fitting CSV labels both q units. The after-pass Results controls were
also checked in `qa-zh-light.png`, `qa-zh-dark.png` and `qa-en-dark.png`.

## 2. Calibrate a transmission frame and transfer the PONI

The beamline user followed §§4.3, 5.3 and 5.4. The public AgBh README supplies λ = 1.0 Å.
Its companion PONI is a pixel-to-q reference for a different distance. The transfer file was
in a separate folder so its geometry had to be loaded explicitly. Representative AgBh screenshots
below use `agbh-before-` or `agbh-after-` before the indicated filename.

| Step | What the beamline user saw and did | Screenshot evidence |
|---|---|---|
| Open Geometry Calibration and load the EDF | Opened the tool and entered the frame path. Silver behenate was selected. The EDF has no recorded energy. The interface offered only energy in keV, although the example supplied wavelength. Its pixel pitch was inferred from the frame dimensions but labelled “metadata”, and a detector-model prompt suggested further input was required. | `01-start.png`, `02-calibration.png`, `03-loaded.png` |
| Set the beam and fit the rings | Initially converted 1.0 Å to 12.39842 keV. The job prompt still requested energy after entry. On replay entered 1.0 Å directly and checked the displayed conversion and pitch of 172 µm. Ran Auto Calibration, examined the fitted overlays and toggled Clean image off and on to compare them with the standard. | `04-energy.png`, `05-calibrated.png`, `06-clean.png`, `07-overlay.png` |
| Review and save | Both passes matched five rings. Centre (179.796, 264.005) px in display coordinates, distance 1633.840 mm, ring-point RMS 0.584 px. The medium-confidence result flagged limited angular coverage. At 900 × 560 the original results pane showed the centre and distance first and required scrolling for the warning. The replay put confidence and the warning beside the job status. Used the dedicated PONI export and accepted its filename. | `08-poni-dialog.png`, `09-saved.png`. Wrapped warning: `agbh/layout/agbh-after-05-settled-200.png` |
| Apply to the second file and integrate | Closed Calibration, opened the renamed EDF, then chose **Load .poni…** and selected the file just written. Analyze's transmission reduction produced I(q) with 1,072 points and I(χ) with 304 measured points over q = 0.105–0.109 Å⁻¹. The numerical PONI fields and point counts agreed between passes. The identical exposure tests transfer and repeatability. An independent exposure would be needed to validate the setup on another measurement. | `10-second-frame-dialog.png`, `11-second-frame.png`, `12-use-poni-dialog.png`, `13-integrated.png` |

## 3. Indexing and ring orientation on P08 and Xeuss

The scientist used §§1.2, 5.2, 5.5 and 5.9. Each screenshot below begins with the pass and sample,
for example `before-p08-01-open.png` or `after-xeuss-03-mask.png`.

| Step | What the scientist saw and did | Screenshot suffix |
|---|---|---|
| Open and check geometry | Opened P08, verified its companion PONI and entered 0.075°. For Xeuss, chose Replace, checked the new PONI, changed αi to 0.235° and loaded the supplied EDF mask. The PONI supplies the detector geometry. αi is a separate setting which can carry across a sample change. | `00-replace.png` (Xeuss), `01-open.png`, `02-geometry.png`, `03-mask.png` (Xeuss) |
| Convert and save a figure | Switched to **q map**, inspected the q∥ and qz axes and saved the calibrated image through the image toolbar. The map view persisted when switching samples. Both standalone PNGs measured 2007 × 1706 pixels. | `04-qmap.png`, `05-figure.png` |
| Find candidate phases | **Add from CIF…** reached an empty file list with the original example data. The P08 YAML named a bromide CIF which was absent and would describe a different material. This blocked the task using only the original inputs. To exercise the rest of the baseline, the observer supplied local comparison CIFs from GIMaP's documented structures. On replay the same structures were available beside each frame. Selected the two P08 files together. On Xeuss, hid the prior sample's phases and loaded bulk PbTe. An exploratory centre click in the original Show cell missed its narrow checkbox and left old assignments visible. The counted baseline used verified indicator clicks. | `06-phases.png`, `missing-CIF.png` (before), `07-phases-loaded.png`, `08-reflections.png` |
| Index and find the strongest ring | Ran **Run Automatic Analysis** with the chosen phases visible. Previously the three orientation summaries appeared in q order. Finding the largest fitted area took 12 activations on P08 and six on Xeuss. The replay identified that ring with its coverage assessment at the top of Results. | `09-indexed.png`, baseline `10-area-1.png` through `10-area-3.png`, `10-orientation.png` |
| Read and export the result | Read the automatic report for each frame. Originally the table could be saved inside the Markdown report. On replay **Save Peak Table…** beside Peaks exported CSV with units, fit errors, candidate assignments, ambiguities and the analysis record. Every exported q and phase agreed with the corresponding report. | `11-report.png`, `12-peaks-export.png` |

**P08.** Eleven of 12 peaks had candidate reflections. The peak at 0.89894 Å⁻¹ matched PbI₂
(001). The strongest reliable peak by fitted area was at 0.99723 Å⁻¹. MAPbI₃ (110) was the
nearest candidate and the assignment was ambiguous. The measured Hermans factor was 0.142
± 0.00275 (statistical), with a same-coverage isotropic reference of 0.0563. The systematic
range was 0.0711–0.157 and the pole fit exceeded the reduced χ² threshold. The report kept the
orientation undetermined. For this peak the Gaussian fit had reduced χ² = 11. Fit details now
shows the poor-fit flag that was already in the saved report (`after-p08-13-fit-warning.png`).
P08 formerly said “NEEDS INPUT” and requested a rerun when all its
outstanding items concerned angular coverage or fit quality. It now says “COMPLETED WITH CHECKS”.

**Xeuss.** None of the nine fitted peaks matched the selected bulk PbTe reference within the
existing tolerances. The report asked the user to reconcile the header's 370 mm with the PONI's
147.6 mm. With the supplied mask the strongest reliable peak by area was at 0.22116 Å⁻¹.
Only 56% of the sin χ weighted orientation range was measured. GIMaP withheld a Hermans factor.
The peak positions, selected rings and these limitations persisted across the two passes.

## Action counts

| Task | Before: activations + entries | After: activations + entries | Dialogs before → after | Actions saved |
|---|---:|---:|---:|---:|
| GISAXS, through curve, parameters and PDF | 24 + 5 = 29 | 14 + 4 = 18 | 4 → 3 | 11 |
| AgBh, through both integrations | 18 + 5 = 23 | 15 + 4 = 19 | 4 → 4 | 4 |
| GIWAXS, P08 | 40 + 8 = 48 | 23 + 6 = 29 | 8 → 5 | 19 |
| GIWAXS, Xeuss in the same session | 41 + 8 = 49 | 29 + 6 = 35 | 9 → 7 | 14 |
| **All three users** | **123 + 26 = 149** | **81 + 20 = 101** | **25 → 19** | **48** |

For GISAXS, open/geometry took 7 → 7 actions, mask/cuts/handoff 5 → 3, predict/refine 5 → 2,
and export 12 → 6. Two baseline clicks attempted to choose the mean in Analyze. The signed cut
still arrived in Fitting. The replay used the stated averaging rule. The extra baseline model
save reflects uncertainty about what the CSV command saved. For AgBh all four saved actions were
at PONI export. The manual energy conversion and time spent searching for the quality warning add
no clicks. One offscreen file-chooser retry (four activations, one entry, one dialog) was excluded
from the baseline. Physical scrolling to focus some calibration fields was not measured.

The two GIWAXS tasks together took **97 → 64** actions. Baseline totals include checking
the empty CIF picker and the assisted continuation. The strict baseline could not finish
unassisted. The time to obtain missing CIFs and extract a standalone table from a report is
excluded. The replay delivered the CSV directly. These differences matter when interpreting the totals.

## Findings ranked by likely time lost

The order is a judgement about the next thing a scientist would have to do: obtain missing
inputs, repeat or reinterpret an analysis, look up a method or unit, or make extra selections.
The action counts above are observations. No human completion times were measured.

| Rank | Observed cost | Change and check |
|---:|---|---|
| 1 | Neither GIWAXS example supplied the CIFs needed to index peaks from the written instructions. | Shipped the two documented P08 comparison structures and an ideal bulk PbTe reference beside the frames. Verified the menu's example copy, P08 assignments and the bulk reference reflections. |
| 2 | Clicking the centre of a phase's Show cell could leave old phases active after replacing P08 with Xeuss. The resulting assignments looked plausible enough to require checking the whole run. | Made the full checkbox cell clickable. Replayed the switch and verified that only bulk PbTe contributed on Xeuss. |
| 3 | P08 Fit details said there were no warnings even when the saved report flagged a poor Gaussian fit. A reader could quote a width or fit error without seeing its qualification. | Fit details now shows recorded shape-fit, overlap, edge and resolution flags alongside the existing peak caveat. Compared the P08 warning on screen with the report. |
| 4 | The student could copy a numerical spacing from a fit that could not determine it. Finding the constraint warning required opening Fit details. | Results marks a current bound-limited fit value in the error column. The details button says **Review fit warnings…** when a warning is flagged. The fit and its model limits still come from the solver. |
| 5 | P08's report requested “input” and a rerun even when the only problems were limited angular coverage and an undetermined pole fit. | Such reports say **Completed with checks** and list the orientation checks. The Xeuss distance discrepancy and missing-input test cases still request attention. |
| 6 | Identifying the strongest fitted ring required comparing areas across the list before interpreting its orientation. | Results brings the strongest reliable ring by fitted area, its measured factor and its coverage assessment to the top. The choice uses the same ordered peaks as the existing analysis. |
| 7 | The peak table could leave the application only as part of a report or via copying cells. Recovering a reusable indexed table also required gathering its metadata. | **Save Peak Table…** writes a standalone CSV beside the result, through a save dialog. Re-read both files and compared the q positions and phases to the reports. |
| 8 | The quick starts described a missing geometry and a calibration search after Analyze had already applied the local PONI. A user could unnecessarily rerun analysis to supply the known αi. | Rewrote both quick starts around checking the geometry source and entering the experimental angle once, before analysis. The zero-incidence status now points to the αi field above the image. |
| 9 | The AgBh pixel pitch looked like an explicit header value. A second message requested a detector model despite the available pitch. | Calibration identifies shape-inferred pitch and asks the user to check it. A known model or a manual pixel size is still available to correct it. |
| 10 | At minimum dialog height the most relevant calibration warning was below the visible fitted values. | The pinned Run status gives matched rings, confidence and the first warning. Checked the wrapped warning in English and Chinese at 900 × 560. |
| 11 | The beamline user had a wavelength but the dialog demanded energy. The missing input scrolled out of view and its prompt could remain after entry. | Calibration accepts either unit, brings the required field into view and updates the prompt on entry. The two routes produced the same numerical PONI. |
| 12 | Predict changed the signed curve to its mean on starting a fit. The changed number of points looked like data loss. | The Fit step explains the mean beforehand and points to **Curve** for a single side. The regression checks that inspecting the hint leaves both halves selected. |
| 13 | Curve, model and paper-figure saves were hidden below the parameter table or inside a menu. “Data and Fit” gave little indication it saved the parameters. | Placed **Save Curve and Parameters…** and **Save Plot…** above the table. The CSV and accompanying fit JSON serve the first purpose; **Save Model…** remains available for reuse. |
| 14 | The CIF picker started away from the frame and accepted one file per visit. The peak tooltip directed the user to ask AI or compare q by hand, even though automatic indexing was available. | The picker opens beside the frame and supports multiple files. The tooltip directs the user to run the analysis with the candidate phases shown. |
| 15 | Export Calibration proposed JSON. Saving a PONI required finding the filter and editing the name. | **Export .poni…** offers the fitted-image stem and PONI type together. The AgBh replay accepted the suggested name. |
| 16 | “Fit” and “Choose a method” made an already configured prediction look unfinished. The results table offered R and D without definitions. | Both run buttons name the selected method, the rail shows readiness, and hovering over a parameter gives its model-specific meaning. |

## Remaining friction and scope

- At 900 × 560 the calibration candidate table still scrolls horizontally, and parts of the
  preview axes can be clipped. **Focus image** enlarges the image. Some Matplotlib toolbar icons
  are hard to distinguish in the dark theme. Before loading any image, the initial prompt can
  already say that the file lacks a beam setting.
- On narrower panes the fit table elides long parameter names and the indexed peak table needs
  horizontal scrolling to read an ambiguous hkl. Tooltips provide the full text. In Fitting the
  single-solution card can still promise errors that a fit cannot estimate. A toast on screen
  during a language switch keeps the language it had when it opened.
- The full GIWAXS report sits below several screens of checks. Replacing a sample keeps the
  previous visibility choices and αi override. The scientist still has to review both. The saved
  q-map figure contains the calibrated image, axes and colour bar. The interactive phase and region
  overlays visible on screen are omitted. With the phase arcs visible, a second export on each sample
  was byte-identical to the figure saved before indexing. The canvas captures
  `after-p08-13-indexed-qmap.png` and `after-xeuss-13-indexed-qmap.png` and corresponding
  `p08_indexed_qmap.png` and `xeuss_indexed_qmap.png` exports are in `giwaxs/overlay-check`.
  The phase dialog’s last file-read message can retain
  its original language until another file is loaded.
- This was an offscreen, manual-guided simulation using QTest, fresh profiles, nonnative file
  choosers and temporary public-data copies. Native OS pickers, a real beamline operator's timing,
  and calibration transfer to an independent second exposure were outside the evidence collected.

## Verification

The offscreen test run covered Analyze, Fitting, Geometry Calibration, automatic analysis, the
posterior benchmarks, architecture, translations, release documentation and UI source-of-truth
checks (`296 passed in 331.48s`). The calibration and tool-translation release tests ran
separately (`149 passed in 17.29s`). Ruff reported `All checks passed!` on the changed Python
files. The English and Chinese user manuals were checked against the walkthroughs. Section 3.1
of both physics manuals now identifies the existing energy–wavelength conversion used by the
calibration dialog.

The [second usability round](2026-11_second_round.md) follows up the publication and sample
replacement findings and adds three walkthroughs of series fitting, joint fitting and record reuse.

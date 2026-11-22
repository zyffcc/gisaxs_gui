# Publishing a q map, fitting a series and returning to a fit

Three simulated users followed the English manual through the desktop GUI on 22 November 2026.
The baseline was `42f0860`, which includes the first usability round. Each pass used Qt QTest,
a fresh GIMaP profile, temporary output folders and public or generated inputs. The observer
captured the widget after each step. Action counts use the first round's definition. A control
activation or confirmation counts once, as does entering a complete value or path. Reading a
plot, computation and screenshots add no actions. The evidence covers offscreen Qt dialogs.

| Task | Baseline actions | Replay actions | Baseline outcome |
|---|---:|---:|---|
| Fixed cut and R/D against time | 30 | 25 | Two separate figures. Combining them and recovering the times required external work. |
| Joint radius and height | 27 | 22 | The handover stopped at action 9. The total includes an assisted continuation. |
| Resume, compare and hand off | 24 | 20 | The recipient's copy of the input could not be located. |
| **Total** | **81** | **67** | All three replays reached export. |

The temporary evidence folders on the test machine are listed below. Each contains the QTest
captures and the files written through the GUI. The manuals include selected English and Chinese captures.

| Workflow | Folder under the system temporary directory | Passes |
|---|---|---|
| Series | `gimap_round2_series` | `before-verified`, `after-final` |
| Joint fit | `gimap-round2-joint-b72c` | `before`, `after` |
| Return and handoff | `gimap_resume_b0765269-94d7-47d1-be49-2b1d5e0e6428` | `before`, `after` |
| Indexed q map, results and calibration | `gimap_round2_publication_im34o_3q` | `indexed.png`, `indexed.svg`, `indexed.pdf` and widget captures |

## Publication and the other first-round findings

On the public P08 GIWAXS frame the operator supplied αi = 0.075°, loaded the adjacent candidate
CIFs in **Phases**, selected the q map and saved it through the image toolbar. The annotated PNG,
PDF and SVG include the visible region boundaries and phase legend. The publication controls
set the physical dimensions, typeface, point size and resolution. The vector files contain the
intensity image and separate vector overlays. An export regression switches off a phase and the
regions, checks the SVG labels, then compares the rendered PNG pixels and their dimensions.

The automatic analysis indexed 11 of 12 peaks to candidate phases, five ambiguously. Results
opens with these counts, orientation evidence grouped by candidate phase, warnings and the full
report button. The rings, peak fits and checks remain available below. The report distinguishes
measured ring orientation, directional evidence from a sector comparison and uncertain
assignments. Replacing the frame lists the incidence override and visibility choices. **Replace
and reset** takes the incoming incidence, hides the old phases and restores the cuts and marks.

The single-solution card explains when Refine can report uncertainties. A visible save notice
and the last CIF import message now follow the selected language. The AgBh calibration candidates
adapt to the available width at 900 × 560. The adjacent selected-solution card carries the
details omitted from the narrow table. We checked the results and calibration windows in both
languages and both themes.

## Findings ranked by time lost

The rank reflects whether a failure blocked completion, required reconstruction outside GIMaP,
or added a repeatable GUI detour. Seconds of human effort were not measured.

| Rank | Observed cost | Change exercised on replay |
|---:|---|---|
| 1 | The enabled Analyze command failed before either GALAXI cut reached Fitting. | The command sends the two cuts to their joint-fit dialog. |
| 2 | A colleague had the original data bytes but could not reproduce the fit after the source moved. | Replay verifies the same-named copy beside the record and records the digest and resolved location. |
| 3 | Returning to yesterday's fit required recovering the model, point selection and method separately. | Open Saved Fit restores these from the exported JSON and verified curve. |
| 4 | The series CSV lacked the time coordinates. Rebuilding its trend required matching frames to detector timestamps. | CSV and JSON include elapsed seconds from the same clock used on the plot. |
| 5 | Combining the radius, spacing and uncertainties required two figure exports and an external plotting step. | A paired trend saves both parameters for the same particle with their available intervals. |
| 6 | Comparing yesterday's fit with today's required reading separate records and matching the fitted points. | A comparison shows both point selections, parameter values, available errors and bounds. |
| 7 | The sphere in the joint dialog supplied no independent height. Changing it meant closing and reopening the dialog. | The scientist selects a height-bearing particle in the dialog. |
| 8 | The joint fit could be started before the required substrate had been set. | The dialog identifies the missing substrate and enables fitting once it is supplied. |
| 9 | The series quality summary gave a frame number while its flags were beyond the visible table columns. | Review selects that frame and shows its reasons beside the curve. |
| 10 | The joint-fit workflow showed little evidence about the transferred cuts before fitting, then required searching the result for the dimensions. | Preview the cut before fitting and read R and h with available errors in the result. |
| 11 | Each series save required finding the table or figure in a six-command menu. | Dedicated buttons save the table and plot from Results. |

## User 4. Radius and particle spacing during growth

The operator used User Manual §6.5. Both passes started with an instrument profile for
the synthetic detector and the same one-sphere starting model in Single analysis. These
represent a setup already checked on one frame. Setup of the instrument and model is
outside this count. The Qt window measured 1440 × 920. Step names below refer to the
temporary `before-verified` and `after-final` screenshot folders for this walkthrough.

The input was one four-frame NeXus file. The frames used the detector-band generator
from `tools/benchmarks/insitu_synthetic.py` with a 640 × 640 detector, R increasing
from 5.0 to 5.6 nm and D fixed at 40 nm. In the last frame an added signal in every
fifth column of the band made a single sphere a poor description. The detector time
dataset contained 40, 42, 44.5 and 47 seconds. Each pass used the same file.

| Step | What the operator saw and did | Screenshots |
|---|---|---|
| Open the series | Opened the NeXus file in Analyze, accepted **All Frames** and inspected the cut and geometry. Opened Fitting, switched to **Batch / In-situ series** and chose the detector folder. The Curves card showed the cut taken from the first frame. | `00-start` through `05-fixed-cut` |
| Fit every cut | Pressed **Start** with Refine and the previous result as the starting point. All four fits used the same horizontal band, detector row boundaries 400–405, and 320 points per frame after averaging the two halves. | `07-fitted` |
| Read R and D | Selected **Time (s)**. Originally the operator saved one SVG for R, switched the parameter to D and saved another SVG. The replay used **Radius and spacing (nm)** and saved one SVG with both sets of error bars. | Before `08-time` through `13-spacing-saved`. After `08-time`, `09-both`, `13-plot-dialog`, `14-saved-figure` |
| Inspect quality and save the table | Results identified frame 4. The baseline required looking across the result table for its flags. In the replay **Review frame 4** selected its row and displayed the reasons beside its measured curve. **Save Table…** wrote the CSV and JSON. | Before `13b-flags` through `15-saved-table`. After `10-warning` through `12-saved-table` |

The first three radii were 4.991, 5.187 and 5.398 nm with standard errors below
0.01 nm. Their fitted spacings ranged from 39.97 to 40.00 nm. At 7 s the fit
returned R = 5.46 ± 0.73 nm and D = 39.89 ± 1.13 nm, but reduced χ² rose from
1.07 to 3167. The curve and model separate visibly. The nominal errors from that
fit do not establish the final radius or spacing. All pre-existing CSV fields,
including all fitted values, errors and flags, matched between passes.

| Cost, in descending order | Change |
|---|---|
| The saved CSV omitted the time coordinates of the graph. Rebuilding the plot elsewhere required returning to the detector data to match acquisition times to fit rows. | The CSV now gives `elapsed_time_s` when times share a clock. Per-frame JSON entries keep the recorded time source and the elapsed seconds. The four CSV/JSON times agreed with the graph at 0, 2, 4.5 and 7 s. |
| A single R–D figure required combining two exported files in another program. Different axes were easy to miss. | The parameter chooser can show R and D for the same particle on a common axis, with the error bars of each estimate. Refine uses its recorded standard errors. Predict uses the existing conditional posterior intervals where available. |
| The fit quality summary identified frame 4, but its full flags lay beyond the visible columns. | The Results button selects the earliest row marked for review. Its plot note gives the poor-fit or parameter warning. |
| Each ordinary save meant opening the Save menu and finding the table or the plot among six commands. | Added direct table and trend saves beside the results while retaining the full Save menu. |

**Actions.** Opening the source, accepting the four-frame import, listing the folder
in Fitting and starting the fit took 11 actions in each pass. Choosing the time axis
took three more (open the chooser, select the item, confirm). The baseline used four
actions on each of three saves, three to choose D and one to read the Flags column.
The replay used four to choose the paired plot (open, Home, Down, confirm), one to
review frame 4 and three on each of two saves. Total **25 activations + 5 entries = 30 → 21 + 4 = 25**.
Dialogs fell from six to five. These counts exclude external work to merge the baseline
figures and recover detector times.

## User 5. Height and radius from two cuts of the GALAXI frame

The scientist copied the public GALAXI TIFF and its PONI to a temporary folder. Both passes used a clean profile,
a 1536 × 960 main window, a 1080 × 740 joint-fit dialog and Qt's nonnative file choosers. The baseline used
`42f0860`. Screenshots and action ledgers are under `%TEMP%/gimap-round2-joint-b72c/{before,after}/`.
Actions follow the first round's definition. An observer intervention is identified separately.

| Step | What the scientist saw | Evidence in each pass |
|---|---|---|
| Open the frame and inspect the cuts | Opened the TIFF with Ctrl+O. The adjacent PONI supplied the geometry. Entered αi = 0.463° from the public example and selected Cuts. The horizontal band covered rows 605–609 and the off-specular vertical band covered columns 613–622. | `01-start` through `04-cuts` |
| Send both cuts | The enabled **Analyze ▸ Send Both Cuts to Fitting** command raised an exception. The baseline stopped in Analyze. To exercise the remaining steps, the observer invoked the existing handover on the component that owns it. The replay reached the joint dialog through the same menu. | `05-blocked-menu` and `05-assisted-joint` (before), `05-joint` (after) |
| Set the model and substrate | With the initial sphere model there was no independent height. The baseline required closing the dialog, selecting Model, changing the particle, then reopening the dialog. Trying the enabled Fit Together button with the substrate still unknown failed. On replay the scientist chose **Vertical cylinder** in the joint dialog and selected Si at its substrate prompt. The transferred points were visible before starting. | Before `06-substrate-needed`, `07-model`; after `06-model`; both `08-ready` |
| Interpret and export | Ran the existing joint Refine, selected the vertical cut in Show, then saved the JSON record and Markdown report. The replay displayed R and h together with available fit errors above the selected curve. | `09-running` through `15-report`; `result.json`, `report.md` |

The effective Si substrate follows the simplified single-interface model used in GIMaP's GALAXI fit comparison.
The public specimen is described by Ag spheres in HMDSO on PTFE on Si (Physics and Methods §9.10). The upright
cylinder fit here exercises an independent height parameter. Its fitted dimensions are conditional on this model.

Both JSON records gave R = 2.9192 ± 0.652 nm and h = 19.769 ± 0.189 nm. The errors are local fit errors. The
vertical reduced χ² was 4.821, the relative radius spread reached its upper bound and the full-range height check
failed. The model misses the high-q upturn of the vertical cut. The two records agreed on both curves' geometry,
point counts (968 horizontal, 570 vertical), fitted models and uncertainties, covariance, bounds and warnings.
The two Markdown reports differed only in the reported elapsed time for the fit.

| Actions | Before, with assisted continuation | After |
|---|---:|---:|
| Open frame, enter incidence, inspect cuts | 7 | 7 |
| Hand over, select cylinder and substrate, run | 12 | 7 |
| Select vertical curve, save record and report | 8 | 8 |
| **Total** | **23 activations + 4 entries = 27** | **18 activations + 4 entries = 22** |

Both passes opened three file choosers. The joint dialog was shown twice on the assisted baseline and once on the
replay. The strict baseline was blocked at the ninth action. The five saved actions were a failed run, closing the
dialog, selecting Model and reopening the dialog through the Fitting menu. Neither the observer intervention nor
the time spent finding the missing command handler was assigned an action.

## User 6. Continue the GALAXI fit the next day

The starting point was user 1's actual exported `particle.json` and its horizontal cut of the
public GALAXI exposure. Both passes used a fresh user profile, a 1536 × 960 main window, QTest
and nonnative pickers. Screenshots came from `widget.grab()` after each step. The baseline used
the worktree's original source. It needed the separately saved `particle-model.json` from user
1's first pass. Its model fields matched the model in the JSON from user 1's second pass.

| Step | Baseline | Replay | Screenshot stems in temporary evidence |
|---|---|---|---|
| Reopen yesterday's result | Read the curve path from the record. Opened the DAT, then opened the arrow beside **Open Curve…** and chose **Load Model…** to restore the model. Opening the cut set the method to Predict. | **Open Curve… ▸ Open Saved Fit…** opened the exported JSON, checked the source digest and restored the model, the mean of the two halves and the saved Refine method. | Before `01-open` to `06-model`. After `01-open` to `04-open-record`. |
| Change one setting and fit | Changed the lower q limit to 0.25 nm⁻¹ and selected Refine. The fit used 543 points. | Entered the same limit and clicked **Refine**. The point count and fitted model agreed. | `07-curve-settings`, `08-new-range`, `10-ready`, `11-refitted`; also `09-fit-method` before. |
| Compare and save | Read yesterday's exported numbers separately and compared them with Results. Saved today's CSV and JSON using a new name. | **Compare with Saved Fit…** showed both point selections, values, available fit errors and parameters at their bounds. Saved the new CSV and JSON. The latter identifies its parent record by path and SHA-256. | After `12-compared`; both `13-export-picker`, `14-exported`. |
| Hand the new record to a colleague | Gave the recipient today's JSON and the identical cut beside it. **Analyze ▸ Tools ▸ Reproduce from Record…** could not use the copy after the original path became unavailable. Acknowledged the missing-input warning. | The same command checked the copy's SHA-256. All three reproduced tables (fit curve, parameters and covariance) matched bit for bit. The comparison identified the relocated file and its digest. | `15a-analyze-menu` to `17-output-folder` in both; `19-missing-file` before, `20-outcome` after. |

Yesterday's 598-point fit gave R = 4.943 nm and D = 522.2 nm, with D at a bound. The 543-point
refit gave R = 4.185 nm and D = 375.7 nm with reduced χ² = 1.558. The displayed standard error on
the new R was 2.5 nm. The difference between the point selections matters when interpreting those
numbers. Agreement between the two passes validates the handoff and refit workflow.

| Finding ranked by likely time lost | Fix and regression check |
|---|---|
| 1. A colleague who had the exact input still had to edit paths in the JSON or recreate the original folder structure. | Open and replay check a same-named copy beside the record when the original path is absent. A substituted file with a different SHA-256 fails. Tests verify the digest and location in the comparison and that a failed replay publishes nothing. |
| 2. Recreating yesterday's fit needed the original cut, model file, halves, range, excluded points and method. The cut alone selected another method. | Open the saved fit JSON from Fitting. A regression restores the source, selected half, q range, excluded point and fitting method, then refits. |
| 3. Comparing two fits required transcribing values and discovering whether they came from the same set of points. | The comparison table places both selections above the parameter values. Its regression opens the comparison after changing the range, verifies the reference survives the fit, and checks it clears when another curve opens. |

**Actions.** Reopen, refit, compare and export took 16 → 13. On the colleague's computer the
route took 8 → 7. Overall, **18 activations + 6 entries = 24 → 15 + 5 = 20**, with 6 → 5 decision
dialogs. The baseline stopped at the missing-file warning. Its total excludes extracting the curve path,
obtaining the separate model file, reading the old JSON and any external JSON edits. The replay
includes opening and closing the comparison dialog. These are GUI actions, not human timings.
The transfer used a temporary copy and made only the original source path unavailable to the
recipient process. The baseline warning used GIMaP's offscreen-compatible question dialog with the
message from the reproduction worker. The comparison dialog was also checked in English and Chinese
in both themes. The final replay also captured the cancellable input verification
(`03b-verification.png`). It completed without an extra action. No second physical computer was involved.

## Verification

The focused offscreen runs covered the series workflow (85 passed), saved fits and existing
curve loading (31 passed), figure rendering, joint fits and reproduction (102 passed), and the
Analyze, Fitting and calibration changes with the unchanged posterior benchmark (153 passed).
After checking the final report, notices and screenshots, the publication regressions and four
architecture, translation, manual and UI gates ran again (99 passed). The phase dialog and release
documentation checks passed separately (27 passed). Some files ran more than once. Ruff reported
`All checks passed!` on all changed Python files.

The English and Chinese user manuals were checked in §§4–6 and 12. The physics manuals document
the time origin and uncertainty bars in §11.13 and the relocated inputs and parent record in
§14.7. The fit objectives, scattering equations and Predict parameters implementation are unchanged.

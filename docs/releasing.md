# Releasing GIMaP

The maintainer's checklist, from choosing the version to the published release. Work through it in order.
Most steps change git or GitHub, so do them yourself. An agent prepares files and runs checks only when
asked (`AGENTS.md`). `docs/development.md` ("Version and release") describes the tooling. In the steps,
`<v>` stands for the new version, for example `0.2.0`.

## 1. Decide the version

- Choose the number: `0.2.0` for new features, `0.1.1` for fixes only. Decide whether the version is a
  pre-release. The About box, `README.md`, both manuals and `CHANGELOG.md` say so when it is.
- Read the version's block of `CHANGELOG.md` (`## Unreleased`, or `## <v> — unreleased`) from top to bottom. Every entry must belong to this version, and so
  must every known limitation under it. A limitation fixed since the last version is removed there and in
  the manuals' "Known limitations" chapter.
- Settle the open questions that `THIRD_PARTY_NOTICES.md` marks *to be confirmed*, or leave them marked
  as they are. Never fill them in by guessing.

## 2. Write the version everywhere it is named

- `src/gimap/shared/version.py`: `__version__ = "<v>"`, and `PRE_RELEASE` (the About box's
  "(Pre-release)"). The About box, the error log, the export records (`software_version`) and the name of
  the portable package read it from there.
- `CHANGELOG.md`: the heading of the version's block (`## Unreleased`, or `## <v> — unreleased (pre-release)`
  while it is prepared) becomes `## <v> — YYYY-MM-DD`, with ` (pre-release)` added when it is one. The date
  is the day you publish. Change it if publishing slips.
- `README.md`: the first paragraph ("The current version is `<v>`…").
- `docs/User_Manual.md` and `docs/User_Manual_zh.md`: the preface, the first sentence of chapter 13 (Known
  limitations) and sections 14.11 and 14.12. Change both editions together, with the same content.
- `docs/development.md`: the sentence that says whether the version is a pre-release.
- `CITATION.cff`: `version: <v>` and `date-released: YYYY-MM-DD`. The DOI comes later (step 11).
- `LICENSE`: the copyright years (`2025–<year of the release>`).
- The tests hold these statements together:
  - `tests/test_release_docs_contents.py::test_the_documents_name_the_version_of_the_program`;
  - `tests/test_release2_docs_decisions.py::test_the_documents_say_whether_this_version_is_a_pre_release`
    reads `__version__` and `PRE_RELEASE`. With `PRE_RELEASE = True` every place above must call the
    version a pre-release. With `False` none may, and the CHANGELOG heading must carry the date.

  A failure names the sentence to change. Change the document, not the test.

## 3. Check the program and commit

- Run `python tools/check.py` and wait for all three gates (main suite, offscreen smoke, ruff) to pass
  (about 15 minutes).
- Check the manuals against what changed (`AGENTS.md`, "每次更新都要检查手册"). Retake screenshots of
  windows that changed.
- Commit everything that belongs to the release. `git status` must show no modified file and no untracked
  file in the program folders (`src`, `modules`, `utils`, `assets`, `config`, `main.py`) or in the
  documents the program and the manuals use (`docs`, `CITATION.cff`, `README.md`, `CHANGELOG.md`,
  `tests/data/external`). New files such as `THIRD_PARTY_NOTICES.md`, `tools/collect_licenses.py`,
  `CITATION.cff`, `docs/Physics_Manual*.md` and new screenshots in `docs/manual_images/` must be tracked
  too: the package holds only what git tracks.
- `tests/test_release_history.py` must pass on this branch. When it fails, the branch carries the history
  removed on 2026-10-13. Do not push then: follow `tools/update_source.ps1` (README, "Updating a source
  checkout").
- Push `main`. Push nothing from a `backup/…` branch, ever.

## 4. Create the build environment `gimap-release`

The package packs a Conda environment of its own. It is kept apart from the environment you develop in:
conda-pack refuses an environment where pip replaced files of conda packages, and every extra package
would reach the users.

- Open a prompt where `conda` works: the "Anaconda PowerShell Prompt" of your Miniconda, or a
  PowerShell after running Miniconda's `shell\condabin\conda-hook.ps1`. On the development computer,
  conda is not on `PATH` in a normal shell.
- Create the environment once, from conda-forge only. Packages from Anaconda's `defaults` channel come
  under Anaconda's terms of service, and `THIRD_PARTY_NOTICES.md` says the runtime's conda packages come
  from conda-forge: the build stops when one does not. Then install `requirements.txt`:

  ```powershell
  conda create -n gimap-release -c conda-forge --override-channels python=3.10 -y
  conda run -n gimap-release python -m pip install -r requirements.txt
  ```

- Do not upgrade conda's pip or setuptools in it. Do not install torch (Sputter installs it on its first
  Predict), pytest or anything else.
- After `requirements.txt` changes, run the pip line again. When conda-pack reports files
  "deleted/overwritten", remove the environment (`conda env remove -n gimap-release`) and create it again.
- Do not run `conda clean --all` before a build. The licences of the conda packages (Python, OpenSSL,
  Tcl/Tk …) are read from the package cache, `pkgs\<package>\info\licenses`.
- The Python on `PATH` that runs the build needs conda-pack: `python -m pip install --user conda-pack`, or
  `conda install -n base -c conda-forge conda-pack`.
- Check the environment: `conda run -n gimap-release python -c "import PyQt5, bornagain, tensorflow, h5py, fabio"`.
- Record what will be packed, for the release page and for readers who want to rebuild it:

  ```powershell
  New-Item -ItemType Directory -Force release | Out-Null
  conda run -n gimap-release python -m pip freeze | Out-File -Encoding ascii release\GIMaP-<v>-pip-freeze.txt
  conda list -n gimap-release --explicit | Out-File -Encoding ascii release\GIMaP-<v>-conda-explicit.txt
  ```

  (`>` in Windows PowerShell 5.1 would write UTF-16, and `release\` does not exist before the first build.
  The build in step 6 keeps other files in `release\`; it replaces only the ZIP, its parts and
  `SHA256SUMS.txt`.)

- The optional fit1d kit (`utils/gisaxs1d_gui_kit/`) goes into the package when it is present at build
  time. Remove it from the build folder when its licence is not settled (`THIRD_PARTY_NOTICES.md`,
  section 4).

## 5. Check the file list (`-ListOnly`)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_portable_release.ps1 -ListOnly
```

- It packs nothing and writes nothing. The list goes to standard output; the summary, the warnings and
  the problems go to the error stream.
- The exit code must be `0`, with no `PROBLEM:` line.
- Read the warnings. A tracked file that differs from the last commit, or a licence input that is not
  tracked, means step 3 is not finished. An untracked file in the program folders or in the documents
  that Help or the manuals use is a `PROBLEM:` (a warning only with `-AllowUntracked`), and so is a file
  Help opens that is missing (the manuals, the Physics manuals, `CITATION.cff`, the examples'
  `manifest.json`).
- Read the counts by folder. Look out for personal files, new large files, and files that should not
  reach users. `docs/research/`, `work/` and `GISAXS_HANDOFF_*.md` are left out by design. Paths under
  `_handoff/`, `research/` or `validation/`, notebooks, caches and analysis outputs stop the build.
- Build from a folder with a short path. Windows PowerShell cannot open a path of 260 characters or more.

## 6. Build the package

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_portable_release.ps1
```

- The default environment is `gimap-release` (`-EnvironmentName <name>` for another one). The default
  version is the one in `version.py`. The default tag is `v<v>` (`-ReleaseTag`).
- Before anything is packed, `tools/collect_licenses.py` collects the licence texts of the environment
  into `licenses\` of the package. Read its lines:
  - "licence text added": a package that carries no text of its declared licence got it from another
    package. PyQt5 gets the GPL-3.0 text this way.
  - "WARNING: no licence file": a package carries no licence text at all. When it is new, look up its
    licence and add it to the table in `THIRD_PARTY_NOTICES.md`.
  - "WARNING: a GPL/LGPL version the build does not check for its full text": a package under a GPL or
    LGPL other than version 3 (GPL-2.0, LGPL-2.1 …) that carries no GNU licence text. Check it by hand
    against `THIRD_PARTY_NOTICES.md`.
  - A GPL-3.0 or LGPL-3.0 package without its text, a GPL or LGPL package with no licence file at all, or
    a conda package that is not from conda-forge stops the build: "The licence texts could not be
    collected".
- What lands in `release\`:
  - `GIMaP-<v>-windows-x64-portable.zip`;
  - when the ZIP is larger than 1900 MiB: `GIMaP-<v>-windows-x64-portable.zip.part01` …, plus
    `Install_GIMaP_parts.bat` (joins local parts) and `00_Download_and_Install_GIMaP-<v>.bat` (the online
    installer, with the SHA-256 of each part written into it);
  - `SHA256SUMS.txt`: one `<sha256>  <file name>` line for each file to upload: the ZIP, or, when it is
    split, its parts and the two `.bat` files (not the ZIP, which is then not uploaded). `sha256sum -c
    SHA256SUMS.txt` checks them, as does `Get-FileHash -Algorithm SHA256 <file>` in PowerShell; a user
    who downloaded only some of them uses `sha256sum -c --ignore-missing SHA256SUMS.txt`.
- Compare `licenses\README.txt` in the ZIP with the table in `THIRD_PARTY_NOTICES.md`. A new component
  under a copyleft licence (GPL, LGPL, MPL) belongs in the notices.

## 7. Test the package on a clean Windows computer

Use a computer, a virtual machine or Windows Sandbox that has never had GIMaP, Python or conda. Measure
as you go: the size of the download, the disk space after the first start, the memory GIMaP uses, and the
time of the first start. The README and the paper give them as the machine requirements.

- Extract the whole ZIP (or join the parts with `Install_GIMaP_parts.bat`) into a short folder whose path
  has only ASCII characters, for example `C:\GIMaP`.
- Before the first start, open a console in the package folder and run
  `runtime\python.exe tools\portable_unpack.py --check`. It changes nothing. Its last line counts the files
  it lists, the files it would rewrite and the files it cannot find (`… not in this package`). Files it
  cannot find mean the ZIP was not extracted completely.
- Double-click `Start_GIMaP.bat`. The first start prepares the packed environment; later starts must be
  quick. Windows SmartScreen warns because the files are not signed; the manual explains this.
- Check:
  - Help ▸ About names `<v>`, and says pre-release only when it is one. Help ▸ User Manual and Help ▸
    Physics and Methods open in both languages, with their pictures. Help ▸ How to Cite shows the authors
    from `CITATION.cff`, not the plain "GIMaP (version …)" fallback.
  - Copy `tests\data\external` out of the program folder (Analyze writes `gimap_analysis\` next to the
    data). Then run the manual's quick start on the GALAXI GISAXS frame and the P08 GIWAXS frame, with
    Run Automatic Analysis, Export and Send to Fitting.
  - Run a Fitting fit, Tools ▸ 1D Predict on an exported curve (its weights are checked against their
    SHA-256), a Compare of two series, and Batch Export.
  - Switch the language to Chinese and the theme to dark, and close and reopen GIMaP.
  - `licenses\README.txt`, `licenses\THIRD_PARTY_NOTICES.md` and
    `licenses\python\PyQt5-<version>\GPL-3.0.txt` are there.
  - Optional, when the computer has network access: the first Sputter Predict installs PyTorch.
- Fix what fails, then go back to step 3. A new build means new checksums.

## 8. Switch Zenodo on (before the GitHub release)

- Log in at [zenodo.org](https://zenodo.org) with the GitHub account that owns the repository. Under
  *GitHub* in the account settings, switch on `zyffcc/gisaxs_gui`.
- Do this **before** step 10. Zenodo archives only releases published after the switch.
- Zenodo takes the metadata of the record (title, authors, licence) from `CITATION.cff`, or from
  `.zenodo.json` when that file exists. Check them before the release.
- Zenodo archives the source ZIP of the tag, not the release assets. To preserve the portable package
  too, deposit it on Zenodo yourself, as its own upload or as files of the record.

## 9. Tag the release

```powershell
git tag -a v<v> -m "GIMaP <v>"
git push origin v<v>
```

- Tag the commit that the package was built from: `git rev-parse HEAD` in the build folder. Make it an
  annotated tag whose name is the one the online installer downloads from (`v<v>`).

## 10. Publish the GitHub release

- On GitHub: Releases ▸ Draft a new release ▸ tag `v<v>`. Title it `GIMaP <v>`, the same as the tag. Tick
  *pre-release* when it is one.
- Notes:
  - this version's section of `CHANGELOG.md`, including its known limitations;
  - how to install: extract the ZIP and run `Start_GIMaP.bat`; or, when it is split, download and run
    `00_Download_and_Install_GIMaP-<v>.bat` (or download every part and `Install_GIMaP_parts.bat`);
  - the SmartScreen warning;
  - the measured machine requirements;
  - licences: GIMaP is MIT, and the Windows package is distributed under the GPL-3.0 terms for its GPL
    parts (`THIRD_PARTY_NOTICES.md`).
- Upload the parts (or the ZIP when it is not split), both `.bat` files, `SHA256SUMS.txt`, and the two
  environment records of step 4. GitHub accepts at most 2 GiB per file, so a part is 1900 MiB.
- Publish. Then download `00_Download_and_Install_GIMaP-<v>.bat` from the release page on a clean
  computer and run it once. It can only be tested after publishing, because it downloads from the
  release.

## 11. After Zenodo: DOI, citation, badge

- Zenodo gives two DOIs: one for this version, and a concept DOI that always resolves to the latest
  version. Open the record, check its authors, affiliations and licence (MIT), and note both DOIs.
- `CITATION.cff`: add `doi: <concept DOI>`, and the version DOI under `identifiers` with a description.
  Later add a `preferred-citation` for the paper.
- `README.md`: a DOI badge at the top,
  `[![DOI](https://zenodo.org/badge/DOI/<concept DOI>.svg)](https://doi.org/<concept DOI>)`, and the DOI
  in the "How to cite" text. Do the same in both manuals, and in the About box where it names a citation.
- Open a new `## Unreleased` section at the top of `CHANGELOG.md`. Commit and push to `main`. The tag
  stays where it is.

## 12. Submitting the paper (Journal of Applied Crystallography)

Read the current JAC *Notes for Authors* before submitting; the points below are from 2026-11-01.

- **Article type**: Computer Programs. The authors are normally the program's developers. Settle who
  wrote the bundled research code and networks (`THIRD_PARTY_NOTICES.md`, section 5) before you fix the
  author list.
- **Two independent users**: JAC asks that the program's usefulness be shown by at least two users who
  are independent of the authors. Name them, with their consent, in the acknowledgements. Keep a short
  record of what each one did with GIMaP.
- **Availability**: referees must be able to get the program without identifying themselves. The public
  GitHub release and its installer need no account. Give the version and its Zenodo DOI in the paper,
  and the web page (README) with documentation, download and contact.
- **Data**: data shown in the paper's figures should be archived with a DOI where possible.
  - The pygid examples have one: [10.5281/zenodo.17466183](https://doi.org/10.5281/zenodo.17466183).
  - Cite BornAgain for the GALAXI frame.
  - Deposit everything else the figures use, for example as a Zenodo dataset with its own licence.
  - Do not use unpublished data of other people without their permission.
- **AI tools**: GIMaP is developed with AI coding assistants (see `AGENTS.md`; commits carry
  `Co-Authored-By` lines). AI tools cannot be authors. Disclose their use in the methods or the
  acknowledgements, and say what they were used for.
- **Licences**: state that GIMaP is MIT-licensed, and that the Windows package is distributed under
  GPL-3.0 terms for PyQt5 and BornAgain.
- **Machine requirements**: give the measured numbers of step 7 and the platforms: Windows tested; macOS
  from source, partly checked; Linux not tested.
- **Open access**: JAC's open-access option uses CC BY. Decide when you submit.

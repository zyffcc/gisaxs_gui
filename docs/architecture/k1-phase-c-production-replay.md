# K1 Phase-C production replay

Status: Lossless writer/consumer implemented; complete live Phase-C capture not yet run

Scope: Lossless, filesystem-backed replay of K1 Phase-C evidence. This is the
boundary that may run the frozen formal plan; it does not produce evidence and
does not make Phase-C a final paper-model acceptance gate.

## Artifact contract

The entrypoint requires a canonical mode-0400 JSON manifest, a completion-last
writer receipt, plus an out-of-band SHA-256 of that manifest file. The manifest binds every input by a
unique ID, normalized relative path, role, and SHA-256. Every listed file must
be referenced and every reference must be listed. Absolute paths, parent
traversal, duplicate paths/IDs, symlinks, writable files, unsupported schemas,
and unreferenced manifest entries are rejected. Every file must have link count
one; every output directory is mode 0500. Adding or removing a hardlink, or
replacing a path with the same bytes, changes the verified filesystem identity
and is rejected.

`artifact_sha256` values in this boundary mean the SHA-256 of the exact
canonical JSON bytes of the new lossless artifact. They are not aliases for an
upstream audit digest or for a file that the adapter did not open. This avoids
recursive file identities: a typed representative's `source_artifact_sha256`
is assigned from its enclosing file hash and is not stored inside that same
file.

The complete manifest contains:

- source/model/selected-checkpoint and proposal-policy provenance;
- the exact split/disjointness receipt and evaluator configuration;
- for every clean parent, provenance and a lossless query-distance context;
- exactly twelve branch-search sidecars;
- the contextual reference bank, its complete contiguous exact-call trace,
  and separate typed representative payload files;
- product, Sobol-only, and retrieval-only complete contiguous exact-call
  traces, including separate emitted representative payload files.

Candidate payload files include the full exact-intensity vector, frozen as a
non-empty one-dimensional little-endian float64 C-order array with explicit
shape and a recomputed array SHA-256. Every decoded element must be a JSON
numeric scalar (never a boolean or string), finite, and strictly positive.
Audit-only payloads that retain merely an intensity digest cannot satisfy this
contract.

The writer does not accept an audit JSON or a reconstructed aggregate record as
a production source. A production capture requires exact live types from the
frozen search task, checked search sidecar and evidence receipt, contextual
reference bank, full reference trace, all three method traces, and their
lossless representative source files. It also requires physical provenance for
the source archive/manifest, cross-platform gate, Phase-A launch, selected model
artifact/weights/training result, and split receipt. Those files are rehashed
before writing, before the manifest, and immediately before the writer receipt.

The retained-checkpoint runtime now persists each actual typed emission before
its in-memory intensity is reduced to an audit digest. These per-emission files
use the same lossless codec as Phase-C and are inventoried and replayed by the
tuning completion reader. Older audit-only trace JSON cannot be promoted or
converted losslessly.

## Replay and formal capability

Loading opens and hashes every file and reconstructs
`V5K1PhaseCReplayBundle`; it never accepts aggregate parent records. The runner
then asks the adapter to reopen, rehash, reparse, and compare every file and
filesystem identity before and after numerical replay.

For a formal plan, the runner accepts only the exact production filesystem
adapter type. The authorization path verifies the completion-last writer
receipt, the exact physical file inventory, all mode/link/inode identities,
the canonical frozen formal plan, and a full typed reconstruction. Verification
mints one private, process-local, single-use capability that must be consumed by
one exact adapter instance before formal loading. Both full filesystem
revalidations still follow. Only then may the adapter mint a private, process-local sealed
capability bound to the adapter instance, plan, contract, bundle, manifest,
distance context, and revalidation count. Adapter IDs, subclasses, fixture
ports, serialized tokens, manually assembled records, and test-only raw JSON
cannot substitute for these capabilities. A formal replay receipt records
`formal=true`; its `claim_eligible` field is
exactly the formal gate decision in the freshly recomputed assessment.

Nonformal replay remains fixture-only and non-claiming. The stable fail-closed
condition is now
`formal_phase_c_requires_verified_lossless_writer_receipt_capability`.
The repository still has no concrete orchestration that produces the complete
300-parent Phase-C population, contextual reference traces, and product/Sobol/
retrieval exact-call traces. Therefore no current aggregate audit or historical
trace can mint a formal capability: formal runtime remains closed until those
real upstream outputs are run and captured by the writer.

Related code:

- `utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_c_raw_codecs_v5.py`
- `utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_c_raw_artifacts_v5.py`
- `utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_c_filesystem_replay_v5.py`
- `utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_c_writer_contract_v5.py`
- `utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_c_writer_payloads_v5.py`
- `utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_c_production_writer_v5.py`
- `utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_c_writer_capability_v5.py`
- `utils/ML_Fitting_1D_GISAXS/PosteriorV8/lossless_representative_artifact_v5.py`
- `utils/ML_Fitting_1D_GISAXS/PosteriorV8/read_only_json_publication_v5.py`
- `utils/ML_Fitting_1D_GISAXS/PosteriorV8/tuning_lossless_emission_store_v5.py`
- `utils/ML_Fitting_1D_GISAXS/PosteriorV8/tuning_trace_artifact_v5.py`
- `utils/ML_Fitting_1D_GISAXS/PosteriorV8/tuning_checkpoint_runtime_v5.py`
- `utils/ML_Fitting_1D_GISAXS/PosteriorV8/tuning_checkpoint_summary_io_v5.py`
- `utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_c_replay_runner_v5.py`
- `utils/ML_Fitting_1D_GISAXS/PosteriorV8/run_k1_phase_c_filesystem_replay_v5.py`

Related tests:

- `utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_k1_phase_c_filesystem_replay_v5.py`
- `utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_k1_phase_c_raw_codecs_v5.py`
- `utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_k1_phase_c_writer_v5.py`
- `utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_tuning_checkpoint_runtime_v5.py`
- `utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_k1_phase_c_replay_v5.py`

Last verified: 2026-09-03

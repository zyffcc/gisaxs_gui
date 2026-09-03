# K1 Phase-C production replay

Status: Consumer implemented; production raw-artifact producer not implemented

Scope: Lossless, filesystem-backed replay of K1 Phase-C evidence. This is the
boundary that may run the frozen formal plan; it does not produce evidence and
does not make Phase-C a final paper-model acceptance gate.

## Artifact contract

The entrypoint requires a canonical read-only JSON manifest plus an
out-of-band SHA-256 of that manifest file. The manifest binds every input by a
unique ID, normalized relative path, role, and SHA-256. Every listed file must
be referenced and every reference must be listed. Absolute paths, parent
traversal, duplicate paths/IDs, symlinks, writable files, unsupported schemas,
and unreferenced manifest entries are rejected.

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

## Replay and formal capability

Loading opens and hashes every file and reconstructs
`V5K1PhaseCReplayBundle`; it never accepts aggregate parent records. The runner
then asks the adapter to reopen, rehash, reparse, and compare every file and
filesystem identity before and after numerical replay.

For a formal plan, the runner accepts only the exact production filesystem
adapter type. The designed authorization path additionally requires a future
audited production-writer receipt/capability, followed by both full filesystem
revalidations. Only then may the adapter mint a private, process-local sealed
capability bound to the adapter instance, plan, contract, bundle, manifest,
distance context, and revalidation count. Adapter IDs, subclasses, fixture
ports, serialized tokens, manually assembled records, and test-only raw JSON
cannot substitute for these capabilities. Once that writer integration exists,
a formal receipt will record `formal=true`; its `claim_eligible` field will be
exactly the formal gate decision in the freshly recomputed assessment.

Nonformal replay remains fixture-only and non-claiming. The current stable
formal blocker is
`production_phase_c_lossless_artifact_writer_not_implemented_or_audited`.
Until a production writer and its receipt verifier are implemented and audited,
this repository is consumer-ready but every formal Phase-C launch fails closed.

Related code:

- `utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_c_raw_codecs_v5.py`
- `utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_c_raw_artifacts_v5.py`
- `utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_c_filesystem_replay_v5.py`
- `utils/ML_Fitting_1D_GISAXS/PosteriorV8/k1_phase_c_replay_runner_v5.py`
- `utils/ML_Fitting_1D_GISAXS/PosteriorV8/run_k1_phase_c_filesystem_replay_v5.py`

Related tests:

- `utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_k1_phase_c_filesystem_replay_v5.py`
- `utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_k1_phase_c_raw_codecs_v5.py`
- `utils/ML_Fitting_1D_GISAXS/tests/test_posterior_v8_k1_phase_c_replay_v5.py`

Last verified: 2026-09-03

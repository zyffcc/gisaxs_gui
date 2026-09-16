"""Actual small-budget search tests, not formal scientific acceptance."""

from hashlib import sha256
from dataclasses import replace

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import tuning_reference_executor_v5 as reader
from utils.ML_Fitting_1D_GISAXS.tests.test_posterior_v8_exact_search_executor_v5 import (
    _schedule, _optimizer, _task, V5FrozenExactSearchExecutor, v5_exact_search_artifact_path,
)


def _case(tmp_path, *, threshold=1e12):
    schedule, optimizer = _schedule(count=4), _optimizer(scouts=2, per_seed=2)
    task = _task(schedule=schedule, optimizer=optimizer, sigma_available=True,
                 threshold=threshold, fixed=True)
    result = V5FrozenExactSearchExecutor(output_directory=tmp_path,
        seed_schedule=schedule, optimizer_schedule=optimizer)(task)
    path = v5_exact_search_artifact_path(tmp_path, task, schedule, optimizer)
    path.chmod(0o400)
    return task, path, sha256(path.read_bytes()).hexdigest(), result


def test_actual_completed_search_preserves_all_reference_parameters(tmp_path):
    task, path, digest, result = _case(tmp_path)
    values = reader.read_v5_executor_reference_payloads(path, task=task, expected_artifact_sha256=digest)
    assert len(values) == len(result.representatives) == 1
    assert values[0].source_artifact_sha256 == digest
    assert values[0].query_context_sha256 == task.universal_context.audit_sha256
    assert values[0].global_branch_key == task.branch.global_key.wire_key
    assert values[0].parameter.components == task.codec.decode(result.representatives[0].target_local)[0]


def test_completed_negative_branch_is_empty_not_fabricated_truth(tmp_path):
    task, path, digest, result = _case(tmp_path, threshold=1e-15)
    assert result.completed and result.outcome != 'compatible_found'
    assert reader.read_v5_executor_reference_payloads(path, task=task, expected_artifact_sha256=digest) == ()


@pytest.mark.parametrize('drift', ['hash', 'mode', 'replace'])
def test_reference_extraction_rejects_actual_file_drift(tmp_path, monkeypatch, drift):
    task, path, digest, _ = _case(tmp_path)
    if drift == 'hash':
        digest = '0' * 64
    elif drift == 'mode':
        path.chmod(0o600)
    else:
        original = reader.read_v5_exact_search_executor_artifact
        def replace(*args, **kwargs):
            result = original(*args, **kwargs)
            replacement = path.with_suffix('.replacement')
            replacement.write_bytes(path.read_bytes())
            replacement.chmod(0o400)
            replacement.replace(path)
            return result
        monkeypatch.setattr(reader, 'read_v5_exact_search_executor_artifact', replace)
    with pytest.raises((ValueError, RuntimeError)):
        reader.read_v5_executor_reference_payloads(path, task=task, expected_artifact_sha256=digest)


def _query_case(tmp_path):
    schedule, optimizer = _schedule(count=4), _optimizer(scouts=2, per_seed=2)
    base = _task(schedule=schedule, optimizer=optimizer, sigma_available=True,
                 threshold=1e12, fixed=False)
    entries = []
    for index in range(base.universal_context.branch_count):
        task = replace(base, branch_index=index)
        V5FrozenExactSearchExecutor(output_directory=tmp_path,
            seed_schedule=schedule, optimizer_schedule=optimizer)(task)
        path = v5_exact_search_artifact_path(tmp_path, task, schedule, optimizer)
        path.chmod(0o400)
        entries.append((task, path, sha256(path.read_bytes()).hexdigest()))
    assert len(entries) > 1
    return base, entries


def test_actual_all_branch_query_references_preserve_branch_coverage(tmp_path):
    task, entries = _query_case(tmp_path)
    values = reader.read_v5_query_executor_reference_payloads(
        query_task=task, branch_artifacts=reversed(entries))
    expected = tuple(value for branch, path, digest in entries
        for value in reader.read_v5_executor_reference_payloads(
            path, task=branch, expected_artifact_sha256=digest))
    assert tuple(value.sha256 for value in values) == tuple(value.sha256 for value in expected)


@pytest.mark.parametrize('drift', ['missing', 'duplicate', 'observation', 'path'])
def test_query_references_reject_incomplete_or_mixed_branch_inputs(tmp_path, drift):
    task, entries = _query_case(tmp_path)
    if drift == 'missing':
        entries.pop()
    elif drift == 'duplicate':
        entries[-1] = entries[0]
    elif drift == 'observation':
        branch, path, digest = entries[-1]
        entries[-1] = (replace(branch, observation_id='another-observation'), path, digest)
    else:
        entries[-1] = (entries[-1][0], entries[0][1], entries[0][2])
    with pytest.raises(ValueError):
        reader.read_v5_query_executor_reference_payloads(query_task=task, branch_artifacts=entries)


def test_all_negative_query_cannot_be_omitted_from_tuning(tmp_path):
    task, path, digest, _ = _case(tmp_path, threshold=1e-15)
    assert task.universal_context.branch_count == 1
    with pytest.raises(ValueError, match='no compatible reference'):
        reader.read_v5_query_executor_reference_payloads(
            query_task=task, branch_artifacts=[(task, path, digest)])


def test_query_replay_detects_previous_branch_replaced_while_reading_next(tmp_path, monkeypatch):
    task, entries = _query_case(tmp_path)
    original = reader.read_v5_executor_reference_payloads

    def replace_previous(path, **kwargs):
        values = original(path, **kwargs)
        if kwargs['task'].branch_index == entries[-1][0].branch_index:
            previous = entries[0][1]
            replacement = previous.with_suffix('.replacement')
            replacement.write_bytes(previous.read_bytes())
            replacement.chmod(0o400)
            replacement.replace(previous)
        return values

    monkeypatch.setattr(reader, 'read_v5_executor_reference_payloads', replace_previous)
    with pytest.raises(RuntimeError, match='all-branch replay'):
        reader.read_v5_query_executor_reference_payloads(query_task=task, branch_artifacts=entries)


def test_actual_query_publication_round_trip_into_evaluator_reference(tmp_path, monkeypatch):
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import k1_balanced_full_search_collector_v5 as collector
    monkeypatch.setattr(collector, '_worker_guard', lambda **kwargs: None)
    monkeypatch.setattr(reader, 'MAXWELL_DUST_ROOT', tmp_path)
    task, entries = _query_case(tmp_path)
    path = tmp_path / 'reference.json'
    kwargs = dict(query_task=task, branch_artifacts=entries, reference_set_id='query-reference',
                  comparison_protocol_id='test-comparison', comparison_protocol_sha256='a' * 64)
    reference = reader.publish_v5_query_executor_reference(path, **kwargs)
    actual_sha = sha256(path.read_bytes()).hexdigest()
    restored = reader.read_v5_query_executor_reference(path, expected_file_sha256=actual_sha, **kwargs)
    assert reference.reference_set_sha256 == restored.reference_set_sha256 == actual_sha
    assert restored.query_id == task.observation_id
    assert restored.pairing_unit_id == task.clean_group_id
    assert restored.representative_payload_set_sha256 == reference.representative_payload_set_sha256
    assert path.stat().st_mode & 0o777 == 0o400 and path.stat().st_nlink == 1
    with pytest.raises(FileExistsError):
        reader.publish_v5_query_executor_reference(path, **kwargs)
    with pytest.raises(ValueError, match='does not reproduce'):
        reader.read_v5_query_executor_reference(path, expected_file_sha256=actual_sha,
            **{**kwargs, 'comparison_protocol_id': 'different-comparison'})


def test_query_publication_rejects_login_node_before_any_search(tmp_path, monkeypatch):
    monkeypatch.setattr(reader.socket, 'gethostname', lambda: 'max-wgs001')
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    with pytest.raises((ValueError, RuntimeError)):
        reader.publish_v5_query_executor_reference(tmp_path / 'not-created.json',
            query_task=None, branch_artifacts=[], reference_set_id='x',
            comparison_protocol_id='x', comparison_protocol_sha256='a' * 64)
    assert not (tmp_path / 'not-created.json').exists()

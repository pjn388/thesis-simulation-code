import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark_pressure_restraints as benchmark


def setup_report(tmp_path, monkeypatch, state):
    target = tmp_path / 'bottom'
    target.mkdir()
    (target / 'manifest.json').write_text(json.dumps({'files': {}}))
    (target / 'status.json').write_text(json.dumps({'state': state}))
    monkeypatch.setattr(benchmark, 'ROOT', tmp_path)
    monkeypatch.setattr(benchmark, 'audit', lambda _: {'candidate_node_dof_coordinates': []})
    return target


def test_report_rejects_live_outputs_before_extract(tmp_path, monkeypatch):
    setup_report(tmp_path, monkeypatch, 'running')
    monkeypatch.setattr(benchmark, 'extract', lambda _: pytest.fail('Read moving native output'))
    with pytest.raises(ValueError, match='completed immutable'):
        benchmark.report()


def test_report_rejects_incomplete_final_stress_frame(tmp_path, monkeypatch):
    target = setup_report(tmp_path, monkeypatch, 'completed')
    (target / 'results').mkdir()
    (target / 'results/integration_points.csv').write_text('time,pressure_mpa\n0.928125,0.928125\n')
    monkeypatch.setattr(benchmark, 'extract', lambda _: None)
    monkeypatch.setattr(benchmark, 'elastic_benchmark', lambda _: pytest.fail('Compared wrong pressure'))
    with pytest.raises(AssertionError, match='Final stress frame missing'):
        benchmark.report()

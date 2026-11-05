import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_elastic_restraint_ips as audit


def test_ip_audit_rejects_live_native_output(tmp_path, monkeypatch):
    (tmp_path / 'bottom').mkdir()
    (tmp_path / 'bottom/status.json').write_text(json.dumps({'state': 'running'}))
    monkeypatch.setattr(audit, 'audit', lambda _: {'candidate_node_dof_coordinates': []})
    with pytest.raises(AssertionError, match='completed immutable'):
        audit.audit_ips(tmp_path)


def test_ip_audit_rejects_unaccepted_final_frame(tmp_path, monkeypatch):
    run = tmp_path / 'bottom'
    run.mkdir()
    (run / 'status.json').write_text(json.dumps({'state': 'completed'}))
    (run / 'manifest.json').write_text(json.dumps({'all_pipe_stress': True, 'files': {}}))
    (run / 'job.sta').write_text('1 1 1 2 0.9 0.9 0.9\n1 2 1U 4 1.0 1.0 0.1\n')
    for name in ['job.dat', 'job.frd', 'solver.log']:
        (run / name).write_text('')
    monkeypatch.setattr(audit, 'audit', lambda _: {'candidate_node_dof_coordinates': []})
    with pytest.raises(AssertionError, match='not accepted in STA'):
        audit.audit_ips(tmp_path)

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import review_live_case


@pytest.mark.parametrize('kind', ['pressure_strain', 'load_displacement'])
def test_live_review_respects_reference_quantity(tmp_path, monkeypatch, kind):
    config = tmp_path / 'results/config'
    config.mkdir(parents=True)
    (config / 'full_campaign.json').write_text(json.dumps({
        'jobs': [{'id': 'example', 'entry': {'comparison_kind': kind}}]}))
    result = dict(run='sym/runs/campaign/example', solver_state='running',
                  latest_frame={}, maximum_peeq=.2, peak_tool_force_n=100,
                  withdrawal_completed=True, residual_crown_mm=5,
                  tensile_force_detected=False, material_supported_peeq=.1,
                  history=[{'max_peeq': .2}])
    monkeypatch.setattr(review_live_case, 'FINAL', tmp_path)
    monkeypatch.setattr(review_live_case, 'snapshot', lambda run: result)
    calls = []

    def compare(value, **kwargs):
        calls.append(kwargs)
        value['branch_comparison'] = {'experiment_indent': {'rmse': 1}}
        return value

    monkeypatch.setattr(review_live_case, 'compare_snapshot', compare)
    monkeypatch.setattr(sys, 'argv', ['review_live_case.py', 'example'])
    review_live_case.main()
    saved = json.loads((tmp_path / 'results/output/live_cases/example/snapshot.json').read_text())
    assert saved['residual_crown_mm'] == 5
    assert saved['first_sampled_material_exceedance'] == {'max_peeq': .2}
    if kind == 'pressure_strain':
        assert not calls
        assert saved['branch_comparison']['status'] == 'pending_pressure_gauge_overlay'
    else:
        assert len(calls) == 1
        assert saved['branch_comparison']['experiment_indent']['rmse'] == 1


def test_pressure_only_snapshot_without_tool(tmp_path, monkeypatch):
    import review_live_withdrawal as live
    from types import SimpleNamespace
    run = tmp_path / 'sym/runs/campaign/intact'
    run.mkdir(parents=True)
    (run / 'case.json').write_text('{}')
    (run / 'model.json').write_text(json.dumps({'nsets': {'CROWN': [9]}, 'elsets': {'PIPE': [12]}}))
    (run / 'manifest.json').write_text('{"files": {}}')
    (run / 'status.json').write_text('{"state": "failed", "solver_sha256": "example"}')
    (run / 'job.sta').write_text('1 1 1 4 .5 .5 .5\n')
    (run / 'job.frd').write_text('')
    (run / 'job.dat').write_text(
        'displacements for set HISTORY and time .5\n 9 0 2 0\n'
        'equivalent plastic strain for set PIPE and time .5\n' +
        ''.join(f' 12 {ip} .05\n' for ip in range(1, 9)) +
        'stresses for set DEFECT and time .5\n')
    case = SimpleNamespace(dent=SimpleNamespace(enabled=False),
                           material=SimpleNamespace(source='{"source_supported_plastic_strain_max": 0.7}'))
    monkeypatch.setattr(live, 'FINAL', tmp_path)
    monkeypatch.setattr(live.Case, 'load', lambda _: case)
    monkeypatch.setattr(live, 'stage_at', lambda c, t: ('pressurise', 17.5))
    result = live.snapshot(run)
    assert result['latest_frame']['crown_inward_mm'] == -2
    assert result['latest_frame']['pressure_mpa'] == 17.5
    assert result['maximum_peeq'] == .05
    assert result['peak_tool_force_n'] == 0
    assert result['withdrawal_completed'] is None
    assert result['residual_crown_mm'] is None


def test_numerical_baseline_has_no_loop_reference(tmp_path, monkeypatch):
    config = tmp_path / 'results/config'
    config.mkdir(parents=True)
    (config / 'full_campaign.json').write_text('{"jobs": [{"id": "example", "entry": null}]}')
    result = dict(run='sym/runs/campaign/example', solver_state='failed', latest_frame={},
                  maximum_peeq=.05, peak_tool_force_n=0, withdrawal_completed=None,
                  residual_crown_mm=None, tensile_force_detected=False,
                  material_supported_peeq=.7, history=[{'max_peeq': .05}])
    monkeypatch.setattr(review_live_case, 'FINAL', tmp_path)
    monkeypatch.setattr(review_live_case, 'snapshot', lambda _: result)
    monkeypatch.setattr(review_live_case, 'compare_snapshot', lambda *a, **k: pytest.fail('No reference exists'))
    monkeypatch.setattr(sys, 'argv', ['review_live_case.py', 'example'])
    review_live_case.main()
    assert result['branch_comparison']['status'] == 'no_literature_loop_reference'

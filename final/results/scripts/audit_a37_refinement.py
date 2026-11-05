"""Compare frozen coarse A37 extraction with an accepted refined snapshot.

Run review_live_case.py a37_refined first. This audit never reads a moving
native result for metrics or writes to a solver directory.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from audit_pressure_failures import digest, converged_times, matches_sta
from comparison_utils import error_metrics, loop_branches
from plot_investigation import reference

FINAL = Path(__file__).resolve().parents[2]
COARSE = 'full_allouti2012_3_percent_od_figure_curve'
REFINED = 'a37_refined'


def main():
    runs = {name: FINAL / 'sym/runs/campaign' / name for name in (COARSE, REFINED)}
    snapshot_path = FINAL / 'results/output/live_cases' / REFINED / 'snapshot.json'
    snap = json.loads(snapshot_path.read_text())
    assert snap['withdrawal_completed'], 'Complete withdrawal required'
    cases = {name: json.loads((run / 'case.json').read_text()) for name, run in runs.items()}
    evidence = {}
    for name, run in runs.items():
        manifest = json.loads((run / 'manifest.json').read_text())
        assert all(digest(run / f) == sha for f, sha in manifest['files'].items())
        evidence[name] = {'manifest_sha256': digest(run / 'manifest.json'),
                          'case_sha256': digest(run / 'case.json'),
                          'solver_sha256': json.loads((run / 'status.json').read_text())['solver_sha256']}
    assert evidence[COARSE]['solver_sha256'] == snap['solver_sha256']
    for key in ('pipe', 'gouge', 'dent', 'material'):
        assert cases[COARSE][key] == cases[REFINED][key], key + ' differs'
    differing_analysis = {key: [cases[COARSE]['analysis'].get(key), value]
                          for key, value in cases[REFINED]['analysis'].items()
                          if cases[COARSE]['analysis'].get(key) != value}
    assert set(differing_analysis) == {'max_pressure', 'max_increment'}
    coarse_history = runs[COARSE] / 'results/history.csv'
    evidence[COARSE]['history_sha256'] = digest(coarse_history)
    evidence[REFINED].update(snapshot_sha256=digest(snapshot_path),
                             snapshot_created_utc=snap['created_utc'],
                             native_prefix_evidence=snap['prefix_evidence'])
    rows = {COARSE: pd.read_csv(coarse_history),
            REFINED: pd.DataFrame(snap['history']).rename(columns={
                'tool_force_n': 'indenter_force_n', 'tool_travel_mm': 'indenter_travel_mm',
                'crown_inward_mm': 'crown_inward_displacement_mm'})}
    for name, history in rows.items():
        sta = converged_times(runs[name] / 'job.sta')
        assert all(matches_sta(sta, t) for t in history.time)
    plan = json.loads((FINAL / 'results/config/full_campaign.json').read_text())
    entry = next(j['entry'] for j in plan['jobs'] if j['id'] == COARSE)
    refs = reference(entry)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'evidence': evidence, 'shared_physics_verified': ['pipe', 'gouge', 'dent', 'material'],
              'reference_sha256': digest(FINAL / entry['reference_file']),
              'differing_analysis_coarse_refined': differing_analysis,
              'mesh_coarse_refined': [cases[COARSE]['mesh'], cases[REFINED]['mesh']],
              'metrics': {}, 'pressure_overlap': [],
              'limitations': 'Two meshes with different increment caps; not an isolated mesh or increment convergence study. '
                             'Pressure targets differ: compare accepted overlap only. Local maxima can switch integration points. '
                             'Material extrapolation and fixture/strain conventions remain unresolved. No burst or validation claim.'}
    for name, history in rows.items():
        indent_end = history[np.isclose(history.time, 1.)].iloc[0]
        withdrawal_end = history[np.isclose(history.time, 2.)].iloc[0]
        metrics = dict(peak_force_kn=float(history.indenter_force_n.max() / 1000),
                       end_indent_peeq=float(indent_end.max_peeq),
                       end_withdraw_peeq=float(withdrawal_end.max_peeq),
                       residual_crown_mm=float(withdrawal_end.crown_inward_displacement_mm))
        for role, ref in refs.items():
            for branch, target in loop_branches(ref).items():
                part = history[history.stage == branch].copy()
                if branch == 'withdraw':
                    part = pd.concat([history[history.stage == 'indent'].tail(1), part])
                part['x'] = part.indenter_travel_mm - cases[name]['dent']['clearance']
                part['y'] = part.indenter_force_n / 1000
                metrics[role + '_' + branch], _ = error_metrics(target, part)
        result['metrics'][name] = metrics
    result['refined_change_percent'] = {
        key: 100 * (result['metrics'][REFINED][key] / result['metrics'][COARSE][key] - 1)
        for key in ('peak_force_kn', 'end_indent_peeq', 'end_withdraw_peeq', 'residual_crown_mm')}
    fine = rows[REFINED][rows[REFINED].time >= 2]
    coarse = rows[COARSE][rows[COARSE].time >= 2]
    for _, row in coarse.iterrows():
        p = float(row.pressure_mpa)
        if p > fine.pressure_mpa.max():
            break
        comparison = {'pressure_mpa': p, 'refined_values': 'linear interpolation within accepted pressure samples'}
        for key in ('max_peeq', 'crown_inward_displacement_mm'):
            comparison[key + '_coarse'] = float(row[key])
            comparison[key + '_refined'] = float(np.interp(p, fine.pressure_mpa, fine[key]))
        result['pressure_overlap'].append(comparison)
    out = FINAL / 'results/output/live_cases' / REFINED / 'refinement_audit.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['refined_change_percent']))


if __name__ == '__main__':
    main()

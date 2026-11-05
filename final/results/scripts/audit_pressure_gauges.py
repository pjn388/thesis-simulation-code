"""Audit completed pressure gauges against both digitized experiment and paper FE."""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from compare import FINAL, check_run, digest
from comparison_utils import canonical_curve, error_metrics, pressure_gauge_curve
from audit_pressure_failures import converged_times, matches_sta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_id')
    args = parser.parse_args()
    plan = json.loads((FINAL / 'results/config/full_campaign.json').read_text())
    entry = next(job['entry'] for job in plan['jobs'] if job['id'] == args.case_id)
    if entry.get('comparison_kind') != 'pressure_strain':
        raise ValueError('Requires a registered pressure-strain case')
    run = check_run({'run': 'sym/runs/campaign/' + args.case_id}, entry)
    status = json.loads((run / 'status.json').read_text())
    if status['state'] != 'completed':
        raise ValueError('Requires a terminal completed run; use bounded live diagnostics otherwise')
    out = FINAL / 'results/output/campaign' / args.case_id
    mappings = json.loads((out / 'gauge_mappings.json').read_text())
    probes = pd.read_csv(run / 'results/probes.csv')
    history = pd.read_csv(run / 'results/history.csv')
    sta = converged_times(run / 'job.sta')
    if not all(matches_sta(sta, t) for t in history.time):
        raise ValueError('History contains a frame without converged STA evidence')
    if not all(matches_sta(sta, t) for t in probes.time.unique()):
        raise ValueError('Probe contains a frame without converged STA evidence')
    reference = pd.read_csv(FINAL / entry['reference_file'])
    rows = []
    for mapping in mappings:
        component = 'hoop_strain' if mapping['direction'] == 'circumferential' else 'axial_strain'
        curve = pressure_gauge_curve(probes, mapping['node'], component)
        sim = pd.DataFrame({'x': curve.pressure_mpa, 'y': curve.gauge_engineering_strain * 1e6})
        saved = pd.read_csv(out / f'gauge_{mapping["gauge"]}_{mapping["direction"]}.csv')
        if len(sim) != len(saved) or not np.allclose(sim.x, saved.x, atol=1e-8, rtol=1e-12) or not np.allclose(sim.y, saved.y, atol=1e-8, rtol=1e-12):
            raise ValueError('Saved overlay differs from independent post-dent conversion')
        baseline = probes[(probes.requested_node == mapping['node']) & np.isclose(probes.time, 2.)]
        row = dict(mapping, baseline_time=float(baseline.time.iloc[0]),
                   baseline_green_lagrange_strain=float(baseline[component].iloc[0]),
                   pressure_min_mpa=float(sim.x.min()), pressure_max_mpa=float(sim.x.max()),
                   final_engineering_microstrain=float(sim.y.iloc[-1]),
                   strain_min_microstrain=float(sim.y.min()), strain_max_microstrain=float(sim.y.max()))
        for role in ('Exp', 'FE'):
            ref = reference[(reference.panel == mapping['panel']) &
                            (reference.series == mapping['direction'] + '_' + role)]
            metrics, _ = error_metrics(canonical_curve(ref, 'MPa', 'microstrain'), sim)
            row[role] = metrics
        rows.append(row)
    paths = [run / 'results/probes.csv', run / 'results/history.csv', run / 'job.sta',
             run / 'status.json', run / 'manifest.json', FINAL / entry['reference_file'],
             out / 'gauge_mappings.json']
    result = dict(case_id=args.case_id, utc=datetime.now(timezone.utc).isoformat(),
                  validation_status='preliminary_unvalidated', solver_status=status,
                  checks={'frozen_inputs_and_sources': True, 'history_matches_sta': True,
                          'probes_match_sta': True,
                          'saved_overlay_matches_independent_conversion': True},
                  provenance={str(path.relative_to(FINAL)): digest(path) for path in paths},
                  gauges=rows,
                  limitations=['Nearest integration points, not surface gauge-length averages.',
                               'Initial material directions and post-dent Green-Lagrange stretch reference.',
                               'Metrics use digitized overlap only, with no reference extrapolation.',
                               'Point 18 remains excluded because coordinates are unresolved.',
                               'Material extrapolation and pressure support discrepancy remain unresolved.'])
    (out / 'gauge_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'checks': result['checks'], 'gauges': rows}, indent=2))


if __name__ == '__main__':
    main()

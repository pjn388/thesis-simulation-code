"""Read-only source uncertainty and matched completed support diagnostics.

No constitutive fitting, source edits or simulation launches. Run from any cwd.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_pressure_failures import digest
from comparison_utils import error_metrics, loop_branches
from plot_investigation import reference

FINAL = Path(__file__).resolve().parents[2]
OUT = FINAL / 'results/output/pressure_failure_audit'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = FINAL / 'data/figures/zhao2021/fig11_material_curve.csv'
    raw = pd.read_csv(source).sort_values('point_order')
    run = FINAL / 'sym/runs/campaign/orientation_gouge_00_adaptive'
    case = json.loads((run / 'case.json').read_text())
    history = pd.read_csv(run / 'results/history.csv')
    endpoint = history.iloc[-1]
    hotspot = []
    for chunk in pd.read_csv(run / 'results/integration_points.csv', chunksize=100000,
                             usecols=['time', 'stage', 'pressure_mpa', 'element', 'ip',
                                      'peeq', 'sxx', 'syy', 'szz', 'sxy', 'sxz', 'syz']):
        hotspot.append(chunk[(chunk.element == endpoint.max_peeq_element) &
                             (chunk.ip == endpoint.max_peeq_ip)])
    hot = pd.concat(hotspot).sort_values('time').copy()
    assert np.isclose(hot.peeq.iloc[-1], endpoint.max_peeq)
    hot['mises_mpa'] = np.sqrt(.5*((hot.sxx-hot.syy)**2 + (hot.syy-hot.szz)**2 +
                                  (hot.szz-hot.sxx)**2) + 3*(hot.sxy**2+hot.sxz**2+hot.syz**2))
    hot['triaxiality'] = (hot.sxx+hot.syy+hot.szz)/(3*hot.mises_mpa.replace(0,np.nan))
    hot.to_csv(OUT / 'zhao_fixed_hotspot.csv', index=False)
    young = case['material']['young']
    plastic = np.asarray(case['material']['plastic'])
    # Evaluate both PUBLISHED coefficient conventions, without optimizing them.
    # Section 3.2 uses E=208000; Figure 11 is explicitly true stress/strain.
    power = .002 * (raw.y.to_numpy() / 365.) ** 9.98
    normalized = raw.y.to_numpy() / 208000. + (365. / 208000.) * power
    offset = raw.y.to_numpy() / 208000. + power
    zero = np.flatnonzero(np.diff(plastic[:, 0]) == 0)
    groups = []
    for i in zero:
        if groups and i == groups[-1][-1] + 1:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    plateaus = []
    for group in groups:
        a, b = group[0], group[-1] + 1
        stress, lo, hi = plastic[a, 0], plastic[a, 1], plastic[b, 1]
        # Source coordinate uncertainty is a digitization bound, not independent
        # statistical error. Adjacent samples share raster/axis uncertainties.
        total_lo, total_hi = lo + stress / young, hi + stress / young
        select = raw[(raw.x >= total_lo - 1e-6) & (raw.x <= total_hi + 1e-6)]
        observed = history[(history.max_peeq >= lo) & (history.max_peeq <= hi)]
        plateaus.append(dict(stress_mpa=float(stress), plastic_lo=float(lo),
                             plastic_hi=float(hi), width=float(hi-lo),
                             source_points=len(select),
                             source_x_uncertainty_max=None if select.empty else float(select.x_digitization_uncertainty.max()),
                             source_y_uncertainty_max=None if select.empty else float(select.y_digitization_uncertainty.max()),
                             sampled_global_max_frames=len(observed),
                             endpoint_inside=bool(lo <= history.max_peeq.iloc[-1] <= hi)))
    material = dict(source=str(source.relative_to(FINAL)), source_sha256=digest(source),
                    frozen_case_sha256=digest(run / 'case.json'),
                    integration_points_sha256=digest(run / 'results/integration_points.csv'),
                    source_page_6_sha256=digest(FINAL / 'data/evidence/zhao2021/page_06.png'),
                    source_page_8_sha256=digest(FINAL / 'data/evidence/zhao2021/page_08.png'),
                    case_young_mpa=young, section_3_2_young_mpa=208000.,
                    source_point_count=len(raw),
                    decreasing_stress_source_intervals=int((raw.y.diff() < 0).sum()),
                    zero_hardening_intervals=len(zero), contiguous_plateaus=plateaus,
                    normalized_total_strain_rmse=float(np.sqrt(np.mean((normalized-raw.x)**2))),
                    offset_total_strain_rmse=float(np.sqrt(np.mean((offset-raw.x)**2))),
                    endpoint_peeq=float(history.max_peeq.iloc[-1]),
                    endpoint_fixed_hotspot=hot.iloc[-1].to_dict(),
                    interpretation='Figure agreement does not resolve the printed equation/coefficient conflict. Plateau association is not failure causation; sampled maxima can change integration point. No curve replacement justified.')
    plan = json.loads((FINAL / 'results/config/full_campaign.json').read_text())
    job = next(j for j in plan['jobs'] if j['id'] == 'a37_finite_support')
    refs = reference(job['entry'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    axes[0].errorbar(raw.x, raw.y, xerr=raw.x_digitization_uncertainty,
                     yerr=raw.y_digitization_uncertainty, fmt='.', alpha=.25,
                     color='black', label='Figure 11 and digitization bounds')
    axes[0].plot(normalized, raw.y, '--', label='Printed normalized convention')
    axes[0].plot(offset, raw.y, label='Offset convention, same coefficients')
    axes[0].set(xlabel='True total strain', ylabel='True stress (MPa)', xlim=(0,.82))
    supports = []
    for path, label in [('campaign/full_allouti2012_3_percent_od_figure_curve', 'Full line, increment cap .05'),
                        ('investigation/a37_contact_fixed', 'Full line, increment cap .10'),
                        ('campaign/a37_finite_support', 'One-OD span, increment cap .10')]:
        directory = FINAL / 'sym/runs' / path
        name = directory.name
        c = json.loads((directory / 'case.json').read_text())
        manifest = json.loads((directory / 'manifest.json').read_text())
        assert all(digest(directory / f) == sha for f, sha in manifest['files'].items())
        h = pd.read_csv(directory / 'results/history.csv')
        s = json.loads((directory / 'results/summary.json').read_text())
        assert s['withdrawal_completed'] and s['contact_checks_passed']
        sim = h[h.stage.isin(['indent', 'withdraw'])].copy()
        sim['x'] = sim.indenter_travel_mm - c['dent']['clearance']
        sim['y'] = sim.indenter_force_n / 1000.
        metrics = {}
        for role, ref in refs.items():
            for branch, part in loop_branches(ref).items():
                curve = sim[sim.stage == branch]
                if branch == 'withdraw':
                    curve = pd.concat([sim[sim.stage == 'indent'].tail(1), curve])
                metrics[role + '_' + branch], _ = error_metrics(part, curve)
        peak = float(sim.y.max())
        supports.append(dict(case_id=name, run=str(directory.relative_to(FINAL)),
                             frozen_case_sha256=digest(directory / 'case.json'),
                             history_sha256=digest(directory / 'results/history.csv'),
                             peak_kn=peak, peak_error_percent=100*(peak/refs['experiment'].y.max()-1),
                             residual_crown_mm=s['residual_crown_displacement_mm'],
                             maximum_peeq=s['maximum_observed_peeq'],
                             maximum_denting_peeq=float(sim.max_peeq.max()),
                             contact_penetration_mm=s['max_nodal_contact_penetration_mm'],
                             max_increment=c['analysis']['max_increment'],
                             linear_solver=c['analysis'].get('linear_solver', 'default'),
                             solver_state=s['solver_state'], metrics=metrics))
        axes[1].plot(sim.x, sim.y, label=label)
    for role, style in [('experiment','o-'), ('paper_fe','--')]:
        axes[1].plot(refs[role].x, refs[role].y, style, color='black' if role == 'experiment' else 'grey', label=role)
    axes[1].set(xlabel='Tool indentation (mm)', ylabel='Force (kN)')
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('Source conventions and fixture sensitivity: no fitted coefficients')
    fig.savefig(OUT / 'material_support_diagnostic.png', dpi=180)
    plt.close(fig)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  material=material, support_comparison=supports,
                  reference_sha256=digest(FINAL / job['entry']['reference_file']),
                  limitations='Full-pressure baseline has increment cap .05; other two have .10. Legacy full-line baseline uses the earlier checked solver; campaign cases use verified PARDISO. Compare denting branches only, not different pressure targets. Residual crown displacement is not experimental zero-force tool position. Finite bilateral span is a sensitivity, not a recovered physical fixture.')
    (OUT / 'material_support_diagnostic.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"Material: {len(groups)} plateau groups; offset/normalized strain RMSE "
          f"{material['offset_total_strain_rmse']:.6g}/{material['normalized_total_strain_rmse']:.6g}")
    for row in supports:
        print(row['case_id'], row['peak_kn'], row['residual_crown_mm'],
              {k: v['rmse'] for k,v in row['metrics'].items()})


if __name__ == '__main__':
    main()

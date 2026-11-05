#!/usr/bin/env python3
"""Reproduce the presentation fit and derive reference-dependent pressure loss.

Uses saved numerical histories only; never runs a solver or edits the presentation.
Run with final/sym/.venv/bin/python from any working directory.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THESIS = Path(__file__).resolve().parent
FINAL = THESIS.parent
SOURCE = FINAL / 'presentation/exploratory_orientation'
OUT = THESIS / 'evidence/orientation_reduction_20261026'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pressure(angle_deg, strain, coefficients, angular_coefficient):
    """Case-specific interpolation only, conservatively bounded by coverage."""
    if not np.isfinite(strain) or not .3 <= strain <= .8:
        raise ValueError('Target maximum PEEQ must be in [0.30, 0.80]')
    upper = 90 if strain <= .5 else 75 if strain <= .6 else 60
    angles = np.asarray(angle_deg)
    if not np.all(np.isfinite(angles)) or np.any((angles < 0) | (angles > upper)):
        raise ValueError('Angle outside observed crossing coverage for this strain')
    return np.polynomial.polynomial.polyval(strain, coefficients) + angular_coefficient * np.sin(np.deg2rad(angles))**2


def reduction(angle_deg, reference_deg, strain, coefficients, angular_coefficient):
    """Fractional pressure loss relative to another damaged orientation, not intact."""
    reference = pressure(reference_deg, strain, coefficients, angular_coefficient)
    if reference <= 0:
        raise ValueError('Reference pressure must be positive')
    return 1 - pressure(angle_deg, strain, coefficients, angular_coefficient) / reference


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((SOURCE / 'baseline_strain_provenance.json').read_text())
    sources = {}
    for relative, expected in provenance['sources_relative_to_final'].items():
        assert sha(FINAL / relative) == expected, relative
        sources[relative] = expected
    # Verify the final coefficients and points against the retained provenance chain.
    for name in ('baseline_strain_model.json', 'baseline_strain_points.csv'):
        assert sha(SOURCE / name) == provenance['outputs'][name], name
    pprov = json.loads((SOURCE / 'slide_orientation_fit_provenance.json').read_text())
    points_path = SOURCE / 'slide_orientation_fit_points.csv'
    assert sha(points_path) == pprov['outputs'][points_path.name]
    points = pd.read_csv(points_path)
    history_path = FINAL / next(k for k in sources if k.endswith('/orientation_histories.csv'))
    history = pd.read_csv(history_path)
    # Independently recover each first crossing from frozen parent histories.
    for row in points.itertuples():
        q = history[history.case_id == row.case_id].sort_values('pressure_mpa').reset_index(drop=True)
        assert (np.diff(q.pressure_mpa) > 0).all()
        hit = np.flatnonzero(q.max_peeq.to_numpy() >= row.target_peeq)
        if not len(hit):
            assert row.status == 'not reached in retained history' and np.isnan(row.pressure)
            continue
        assert hit[0] > 0 and row.status == 'bracketed'
        lo, hi = q.iloc[hit[0]-1], q.iloc[hit[0]]
        value = lo.pressure_mpa + (row.target_peeq-lo.max_peeq)/(hi.max_peeq-lo.max_peeq)*(hi.pressure_mpa-lo.pressure_mpa)
        np.testing.assert_allclose([value, lo.pressure_mpa, hi.pressure_mpa],
                                   [row.pressure, row.pressure_low, row.pressure_high], atol=1e-11, rtol=0)
    observed = points[points.status == 'bracketed'].copy()
    assert len(points) == 42 and len(observed) == 37
    separate = []
    for e, group in observed.groupby('target_peeq'):
        x = np.sin(np.deg2rad(group.angle_deg))**2
        design = np.column_stack((np.ones(len(x)), x))
        ab = np.linalg.lstsq(design, group.pressure, rcond=None)[0]
        separate.append(dict(strain=e, a_mpa=ab[0], b_mpa=ab[1], n=len(group)))
    levels = pd.DataFrame(separate)
    previous = pd.read_csv(SOURCE / 'slide_orientation_fit_metrics.csv')
    np.testing.assert_allclose(levels[['a_mpa','b_mpa']], previous[['intercept_a_mpa','sin_squared_coefficient_b_mpa']], atol=1e-11, rtol=0)
    offsets = observed.target_peeq.map(levels.set_index('strain').a_mpa).to_numpy()
    x = np.sin(np.deg2rad(observed.angle_deg.to_numpy()))**2
    b = float(np.dot(x, observed.pressure-offsets) / np.dot(x,x))
    design = np.column_stack((np.ones(len(levels)), levels.strain, levels.strain**2))
    c = np.linalg.lstsq(design, levels.a_mpa, rcond=None)[0]
    original = json.loads((SOURCE / 'baseline_strain_model.json').read_text())
    np.testing.assert_allclose(c, original['models']['quadratic']['coefficients_in_increasing_powers_of_E'], atol=1e-10, rtol=0)
    np.testing.assert_allclose(b, original['pressure_checks_with_original_shared_B']['quadratic']['shared_b_kept_mpa'], atol=1e-11, rtol=0)
    y = observed.pressure.to_numpy()
    pred = np.array([pressure(r.angle_deg, r.target_peeq, c, b) for r in observed.itertuples()])
    residual = y-pred
    metrics = dict(n=37, r_squared=float(1-np.sum(residual**2)/np.sum((y-y.mean())**2)),
                   rmse_mpa=float(np.sqrt(np.mean(residual**2))), max_abs_residual_mpa=float(np.max(abs(residual))))
    np.testing.assert_allclose(metrics['rmse_mpa'], original['pressure_checks_with_original_shared_B']['quadratic']['raw_pressure_rmse_mpa'], atol=1e-11, rtol=0)
    observed['composite_pressure_mpa'] = pred
    observed['sample_minus_fit_mpa'] = residual
    observed.to_csv(OUT / 'verified_crossings.csv', index=False)
    examples=[]
    for e, reference in ((.5,90),(.8,60)):
        for angle in range(0,reference+1,15):
            p = float(pressure(angle,e,c,b)); r=float(reduction(angle,reference,e,c,b))
            # Check algebra independently and reference identity.
            a=np.polynomial.polynomial.polyval(e,c)
            np.testing.assert_allclose(r, b*(np.sin(np.deg2rad(reference))**2-np.sin(np.deg2rad(angle))**2)/(a+b*np.sin(np.deg2rad(reference))**2), atol=1e-14, rtol=0)
            examples.append(dict(strain=e,angle_deg=angle,reference_deg=reference,pressure_mpa=p,reduction_fraction=r))
        assert abs(reduction(reference,reference,e,c,b)) < 1e-14
    for e, angle in ((.8,90),(.6,90),(.29,0),(.81,0),(.5,float('nan')),(float('nan'),0)):
        try: pressure(angle,e,c,b)
        except ValueError: pass
        else: raise AssertionError('Coverage guard did not reject extrapolation')
    pd.DataFrame(examples).to_csv(OUT / 'relative_reduction_examples.csv',index=False)
    e8=levels[levels.strain==.8].iloc[0]
    model=dict(coefficients_mpa=c.tolist(),shared_b_mpa=b,raw_pressure_metrics=metrics,
               separate_level_fits=separate,examples=examples,
               criterion_specific_080=dict(a_mpa=float(e8.a_mpa),b_mpa=float(e8.b_mpa),
                    axial_reduction_relative_to_60=float(.75*e8.b_mpa/(e8.a_mpa+.75*e8.b_mpa))),
               domain='Same X52 combined parent family; strain0.30-0.50:0-90deg; (0.50,0.60]:0-75deg; (0.60,0.80]:0-60deg. Intermediate strain/angle values are surrogate interpolations.',
               meaning='Fitted pressure at global maximum PEEQ, not physical burst. Reduction uses another damaged orientation as reference. No accepted intact strength is available.',
               checks='37 first crossings reproduced from frozen history;5 absent retained; separate and pooled coefficients reproduced; ratio identities and extrapolation guards passed.')
    (OUT / 'model.json').write_text(json.dumps(model,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes=plt.subplots(1,2,figsize=(10.8,4.3),layout='constrained')
    for e, reference, color in ((.5,90,'#176493'),(.8,60,'#be561b')):
        grid=np.linspace(0,reference,181)
        axes[0].plot(grid,pressure(grid,e,c,b),color=color,label=f'max PEEQ {e:.2f}: composite fit')
        q=observed[observed.target_peeq==e]
        axes[0].errorbar(q.angle_deg,q.pressure,yerr=[q.pressure-q.pressure_low,q.pressure_high-q.pressure],fmt='o',color=color,ms=4,capsize=2)
        axes[1].plot(grid,100*reduction(grid,reference,e,c,b),color=color,label=f'max PEEQ {e:.2f}; reference {reference}°')
    for ax in axes:
        ax.set(xlabel='Angle from pipe axis (degrees)',xticks=range(0,91,15),xlim=(-2,92));ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('Pressure at selected maximum PEEQ (MPa)')
    axes[1].set_ylabel('Pressure reduction relative to reference (%)')
    fig.savefig(OUT / 'orientation_reduction.pdf')
    fig.savefig(OUT / 'orientation_reduction.png',dpi=180)
    plt.close(fig)
    for path in (points_path,SOURCE/'baseline_strain_provenance.json',SOURCE/'baseline_strain_model.json',
                 FINAL/'presentation/thesis presentation.pptx',FINAL/'presentation/02_questions_and_knowledge_guide.md',
                 SOURCE/'BASELINE_STRAIN_FIT.md',SOURCE/'INTERCEPT_ALIGNED_FIT.md',Path(__file__)):
        sources[str(path.relative_to(FINAL))]=sha(path)
    outputs={p.name:sha(p) for p in OUT.iterdir() if p.name!='provenance.json'}
    (OUT/'provenance.json').write_text(json.dumps(dict(sources_relative_to_final=sources,outputs=outputs),indent=2)+'\n')
    print(json.dumps(model,indent=2))


if __name__ == '__main__':
    main()

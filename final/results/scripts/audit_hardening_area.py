"""Retained coupon flow-curve areas, not a native stored-energy identification."""
import json
import sys
from pathlib import Path

import numpy as np
from audit_plastic_return_map import F, sha256
from audit_coupon_state_work import audit


def primitive(q, curve):
    """Integral of native piecewise linear/clamped Cauchy flow curve from zero."""
    x, y = curve[:, 1], curve[:, 0]
    assert x[0] == 0 and q >= 0
    total = 0.
    for i in range(len(x)-1):
        width = max(0., min(q, x[i+1])-x[i])
        total += y[i]*width + .5*(y[i+1]-y[i])/(x[i+1]-x[i])*width**2
    return total + y[-1]*max(0., q-x[-1])


def trajectory(lane):
    identity = audit(lane)
    assert identity['checks_passed']
    run = F/identity['run']
    lines = (run/'job.inp').read_text().splitlines()
    curve = []
    for line in lines[lines.index('*PLASTIC')+1:]:
        if line.startswith('*'): break
        curve.append(list(map(float, line.split(','))))
    curve = np.array(curve)
    assert np.all(np.diff(curve[:, 1]) > 0) and np.all(np.diff(curve[:, 0]) >= 0)
    assert curve[0, 1] == 0
    raw = np.memmap(run/'fields.bin', dtype='<f8', mode='r')
    records = json.loads((run/'fields.json').read_text())['records']
    previous = np.zeros((1, 8))
    accumulated = np.zeros_like(previous)
    maxima = dict(integral_identity_mpa=0., partition_identity_mpa=0., peeq_decrease=0.)
    history = []
    for rec in records:
        meta = rec['fields']['state']
        q = raw[meta['offset']//8:meta['offset']//8+meta['count']].reshape((1, 8, rec['states']))[..., 0]
        maxima['peeq_decrease'] = max(maxima['peeq_decrease'], float((previous-q).max()))
        for index in np.ndindex(q.shape):
            lo, hi = previous[index], q[index]
            assert hi >= lo-1e-12
            # Independent knot-split trapezoids are exact for each linear interval.
            knots = np.r_[lo, curve[(curve[:, 1] > lo) & (curve[:, 1] < hi), 1], hi]
            values = np.interp(knots, curve[:, 1], curve[:, 0])
            accumulated[index] += float(np.sum(np.diff(knots)*(values[:-1]+values[1:])/2))
        area = np.array([primitive(float(v), curve) for v in q.flat]).reshape(q.shape)
        baseline_area = curve[0, 0]*q
        hardening_area = area-baseline_area
        maxima['integral_identity_mpa'] = max(maxima['integral_identity_mpa'], float(abs(area-accumulated).max()))
        # Multiple scalar storage/resistance splits give exactly the same flow.
        for fraction in (0., .5, 1.):
            storage = fraction*hardening_area
            resistance_area = baseline_area+(1-fraction)*hardening_area
            maxima['partition_identity_mpa'] = max(maxima['partition_identity_mpa'], float(abs(area-storage-resistance_area).max()))
            assert storage.min() >= -1e-12 and resistance_area.min() >= -1e-12
        history.append(dict(time=rec['time'], peeq_range=[float(q.min()), float(q.max())],
                            flow_curve_area_mpa=float(area.mean()), baseline_area_mpa=float(baseline_area.mean()),
                            hardening_excess_area_mpa=float(hardening_area.mean())))
        previous = q.copy()
    assert all(sha256(run/n) == h for n, h in identity['input_output_hashes'].items())
    limits = dict(integral_identity_mpa=1e-10, partition_identity_mpa=1e-10, peeq_decrease=1e-12)
    passed = all(maxima[k] < value for k, value in limits.items())
    return dict(lane=lane, run=identity['run'], accepted_frames=len(records), checks_passed=passed,
                fixed_limits=limits, maximum_errors=maxima, endpoint=history[-1], history=history,
                native_endpoint_work_n_mm=identity['totals_n_mm']['native_work'],
                native_endpoint_potential_change_n_mm=identity['endpoint_potential_change_n_mm'],
                input_output_hashes=identity['input_output_hashes'],
                interpretation='Scalar flow-curve areas have units MPa. Unit reference coupon volume is 1 mm^3, but no finite-strain conjugacy or J-weighted within-increment plastic work is established. These areas are not subtracted from native work or identified as native storage/dissipation.')


if __name__ == '__main__':
    out = Path(sys.argv[1])
    assert not out.exists()
    result = dict(trajectories=[trajectory(lane) for lane in ('standard', 'half')],
                  script_sha256=sha256(__file__),
                  dependencies={n:sha256(F/'results/scripts'/n) for n in ('audit_coupon_state_work.py', 'audit_plastic_return_map.py', 'audit_frozen_metric_work.py')},
                  source_hashes={n:sha256(F/'sym/vendor/CalculiX/ccx_2.23/src'/n) for n in ('incplas.f', 'resultsmech.f')},
                  sources='incplas.f:264-310,620-675,754-790; resultsmech.f:1064-1074',
                  verdict='Flow interpolation fixes yield resistance, but the audited source provides no separate hardening-energy identification. Three scalar storage/resistance partitions reproduce the same area. Native finite-strain thermodynamic identification remains unresolved; no production correction justified.',
                  limitations='Partition family is a scalar identifiability diagnostic, not three validated finite-strain models. No altered metric, flow, native output, pipe load, cap tangent, contact/MPC or Abaqus acceptance.')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({**result, 'trajectories':[{k:v for k,v in t.items() if k not in ('history', 'input_output_hashes')} for t in result['trajectories']]}, indent=2))
    assert all(t['checks_passed'] for t in result['trajectories'])

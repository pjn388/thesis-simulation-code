"""Propagate independent coupon history without resetting to native state.

Identity audit only: no stored-energy or physical dissipation identification.
"""
import json
import sys
from pathlib import Path

import numpy as np
from audit_coupon_state_work import audit
from audit_plastic_return_map import F, tensors, sha256, update


def replay(lane):
    control = audit(lane)
    assert control['checks_passed']
    run = F / control['run']
    lines = (run / 'job.inp').read_text().splitlines()
    curve = []
    for line in lines[lines.index('*PLASTIC')+1:]:
        if line.startswith('*'):
            break
        curve.append(list(map(float, line.split(','))))
    curve = np.array(curve)
    raw = np.memmap(run / 'fields.bin', dtype='<f8', mode='r')
    records = json.loads((run / 'fields.json').read_text())['records']
    xyz = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                    [0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    metric = np.broadcast_to(np.eye(3), (1,8,3,3)).copy()
    peeq = np.zeros((1,8))
    mu, bulk = 202500/(2*1.3), 202500/(3*.4)
    limits = dict(stress_mpa=1e-4, metric=1e-8, peeq=1e-8)
    maxima = dict.fromkeys(limits, 0.)
    history = []
    previous_q = peeq.copy()
    for rec in records:
        def field(name, width):
            meta = rec['fields'][name]
            values = raw[meta['offset']//8:meta['offset']//8+meta['count']]
            return values.reshape((-1,width)) if name == 'u' else values.reshape((1,8,width))
        coef = np.linalg.lstsq(np.c_[np.ones(8),xyz], field('u',4)[:,1:], rcond=None)[0]
        deformation = np.eye(3)+coef[1:].T
        assert np.linalg.det(deformation) > 0
        c = np.broadcast_to(deformation.T@deformation, metric.shape)
        np.testing.assert_allclose(c, np.eye(3)+2*tensors(field('strain',6)), atol=1e-12, rtol=1e-12)
        inverse = np.linalg.inv(deformation)
        native_stress = np.linalg.det(deformation)*(inverse@tensors(field('stress',6))@inverse.T)
        native_state = field('state',rec['states'])
        native_metric = np.eye(3)-2*tensors(native_state[...,1:7])
        # Crucially: only independent predicted state feeds the next increment.
        stress, metric, peeq, active = update(c, metric, peeq, curve, mu, bulk)
        assert np.linalg.eigvalsh(metric).min() > 0
        assert np.min(peeq-previous_q) >= -1e-12
        errors = dict(stress_mpa=float(abs(stress-native_stress).max()),
                      metric=float(abs(metric-native_metric).max()),
                      peeq=float(abs(peeq-native_state[...,0]).max()))
        for key, value in errors.items():
            maxima[key] = max(maxima[key], value)
        history.append(dict(time=rec['time'], plastic_update=bool(active.any()),
                            peeq_range=[float(peeq.min()),float(peeq.max())], **errors))
        previous_q = peeq.copy()
    assert all(sha256(run/name) == value for name,value in control['input_output_hashes'].items())
    return dict(lane=lane, run=control['run'], accepted_frames=len(records),
                plastic_frames=sum(frame['plastic_update'] for frame in history),
                elastic_frames=sum(not frame['plastic_update'] for frame in history),
                fixed_limits=limits, maximum_errors=maxima,
                checks_passed=all(maxima[key] < value for key,value in limits.items()),
                endpoint=history[-1], history=history,
                input_output_hashes=control['input_output_hashes'])


if __name__ == '__main__':
    output = Path(sys.argv[1])
    assert not output.exists()
    trajectories = [replay(lane) for lane in ('standard','half')]
    result = dict(trajectories=trajectories,
                  provenance={str(path.relative_to(F)):sha256(path) for path in
                              [Path(__file__), Path(__file__).with_name('audit_plastic_return_map.py'),
                               Path(__file__).with_name('audit_coupon_state_work.py'),
                               F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f']},
                  limitations='Prescribed retained homogeneous coupon histories, zero virgin state, isotropic hardening only. Independent bisection follows the native constitutive equations. No reset to native state after initialization. Does not establish temporal convergence, thermodynamic energy partition, tangent consistency, pipe/contact/MPC/cap/Abaqus transfer or experimental validation.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{key:value for key,value in row.items() if key not in ('history','input_output_hashes')} for row in trajectories],indent=2))
    assert all(row['checks_passed'] for row in trajectories), 'Preserve failed audit; do not relax limits.'

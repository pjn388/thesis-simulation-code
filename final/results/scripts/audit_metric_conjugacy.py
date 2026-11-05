"""Verify native elastic metric conjugacy; endpoint split is not a plastic path."""
import json
import sys
from pathlib import Path

import numpy as np
from audit_coupon_state_work import audit
from audit_frozen_metric_work import elastic
from audit_plastic_return_map import F, tensors, sha256


def trajectory(lane):
    gate = audit(lane)
    assert gate['checks_passed']
    run = F/gate['run']
    raw = np.memmap(run/'fields.bin', dtype='<f8', mode='r')
    records = json.loads((run/'fields.json').read_text())['records']
    xyz = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    mu, bulk = 202500/(2*1.3), 202500/(3*.4)
    previous = np.broadcast_to(np.eye(3), (1,8,3,3)).copy()
    maxima = dict(metric_drop_n_mm=0., stress_correction_mpa=0.)
    history = []
    for rec in records:
        def field(name, width):
            meta = rec['fields'][name]
            values = raw[meta['offset']//8:meta['offset']//8+meta['count']]
            return values.reshape((-1,width)) if name == 'u' else values.reshape((1,8,width))
        coef = np.linalg.lstsq(np.c_[np.ones(8),xyz], field('u',4)[:,1:], rcond=None)[0]
        deformation = np.eye(3)+coef[1:].T
        c = np.broadcast_to(deformation.T@deformation, previous.shape)
        j = np.sqrt(np.linalg.det(c))
        assert np.all(j > 0)
        state = field('state', rec['states'])
        metric = np.eye(3)-2*tensors(state[...,1:7])
        assert np.linalg.eigvalsh(metric).min() > 0
        trial_p, trial_s = elastic(c, previous, mu, bulk)
        new_p, new_s = elastic(c, metric, mu, bulk)
        # W(C,M) is linear in M at fixed C. Q=dW/dM retains native J.
        conjugate = mu/2*c/j[...,None,None]**(2/3)
        dm = metric-previous
        exact_drop = -np.einsum('...ij,...ij->...', conjugate, dm)
        correction = mu/j[...,None,None]**(2/3)*(dm-np.einsum('...ij,...ij->...', c, dm)[...,None,None]*np.linalg.inv(c)/3)
        maxima['metric_drop_n_mm'] = max(maxima['metric_drop_n_mm'], float(abs(trial_p-new_p-exact_drop).max()))
        maxima['stress_correction_mpa'] = max(maxima['stress_correction_mpa'], float(abs(new_s-trial_s-correction).max()))
        history.append(dict(time=rec['time'], endpoint_metric_drop_n_mm=float(exact_drop.mean()),
                            j_range=[float(j.min()),float(j.max())], metric_increment_norm=float(np.linalg.norm(dm))))
        previous = metric.copy()
    limits = dict(metric_drop_n_mm=1e-9, stress_correction_mpa=1e-8)
    assert all(sha256(run/n) == h for n,h in gate['input_output_hashes'].items())
    return dict(lane=lane, run=gate['run'], accepted_frames=len(records), maximum_errors=maxima,
                fixed_limits=limits, checks_passed=all(maxima[k] < v for k,v in limits.items()),
                summed_endpoint_metric_drop_n_mm=sum(h['endpoint_metric_drop_n_mm'] for h in history),
                prior_endpoint_drop_n_mm=gate['totals_n_mm']['endpoint_metric_drop'], history=history,
                input_output_hashes=gate['input_output_hashes'])


def derivative_check():
    mu, bulk = 202500/(2*1.3), 202500/(3*.4)
    c = np.array([[1.03,.01,.02],[.01,.98,.005],[.02,.005,1.01]])
    metric = np.diag([.99,1.02,1.])
    direction = np.array([[.3,.1,.2],[.1,-.2,.05],[.2,.05,.4]])
    eps = 1e-5
    plus,_ = elastic(c,metric+eps*direction,mu,bulk)
    minus,_ = elastic(c,metric-eps*direction,mu,bulk)
    exact = mu/2*np.sum(c*direction)/np.linalg.det(c)**(1/3)
    error = float(abs((plus-minus)/(2*eps)-exact))
    return dict(error_mpa=error, fixed_limit_mpa=1e-5, checks_passed=error < 1e-5)


if __name__ == '__main__':
    out = Path(sys.argv[1])
    assert not out.exists()
    result = dict(derivative_check=derivative_check(), trajectories=[trajectory(lane) for lane in ('standard','half')],
                  script_sha256=sha256(__file__),
                  dependencies={n:sha256(F/'results/scripts'/n) for n in ('audit_coupon_state_work.py','audit_frozen_metric_work.py','audit_plastic_return_map.py')},
                  source_hashes={n:sha256(F/'sym/vendor/CalculiX/ccx_2.23/src'/n) for n in ('incplas.f','resultsmech.f')},
                  derivation='dW = (S/2):dC + Q:dM; Q=mu C/(2 J^(2/3)), J=sqrt(det(C)). Fixed-C metric change is exactly Q:deltaM. incplas.f:754-790 changes M and stress at final C; resultsmech.f:1064-1074 uses endpoint-average S:dC/2.',
                  limitations='Verifies conjugacy of the retained original-metric elastic potential, not thermodynamic identification. Endpoint metric split is a counterfactual fixed-C segment, not the actual within-increment plastic path. Endpoint fields do not supply the integral of Q:dM along the coupled C/M path. Hardening storage is unidentified; no dissipation, correction, burst, pipe/cap/contact/MPC or Abaqus validation.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**result,'trajectories':[{k:v for k,v in t.items() if k not in ('history','input_output_hashes')} for t in result['trajectories']]},indent=2))
    assert result['derivative_check']['checks_passed'] and all(t['checks_passed'] for t in result['trajectories'])

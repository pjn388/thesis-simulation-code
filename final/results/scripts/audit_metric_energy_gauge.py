"""Test a stress-free energy gauge without changing native stress or metric."""
import json
import sys
from pathlib import Path
import numpy as np
from audit_coupon_state_work import audit
from audit_frozen_metric_work import elastic
from audit_plastic_return_map import F, sha256, tensors

MU = 202500/(2*1.3)
BULK = 202500/(3*.4)


def candidate(c, metric):
    potential, stress = elastic(c, metric, MU, BULK)
    # AM-GM: tr(C M)/det(C)^(1/3) >= 3 det(M)^(1/3).
    # Equality at C = det(M)^(1/3) M^-1, J=1, where native S=0.
    gauge = 1.5*MU*np.linalg.det(metric)**(1/3)
    return potential-gauge, stress


def manufactured():
    metric = np.array([[.99,.01,.002],[.01,1.02,.003],[.002,.003,.98]])
    root = np.linalg.det(metric)**(1/3)
    relaxed = root*np.linalg.inv(metric)
    energy, stress = candidate(relaxed, metric)
    c = np.array([[1.03,.01,.02],[.01,.98,.005],[.02,.005,1.01]])
    direction = np.array([[.3,.1,.2],[.1,-.2,.05],[.2,.05,.4]])
    eps = 1e-5
    wp,_ = candidate(c+eps*direction,metric)
    wm,_ = candidate(c-eps*direction,metric)
    _, s = elastic(c,metric,MU,BULK)
    stress_derivative_error = float(abs((wp-wm)/(2*eps)-.5*np.sum(s*direction)))
    wp,_ = candidate(c,metric+eps*direction)
    wm,_ = candidate(c,metric-eps*direction)
    q = MU/2*(c/np.linalg.det(c)**(1/3)-root*np.linalg.inv(metric))
    metric_derivative_error = float(abs((wp-wm)/(2*eps)-np.sum(q*direction)))
    # Source's C-deviatoric isotropic update is orthogonal to C.
    trial = metric/np.linalg.det(c)**(1/3)
    trial -= np.sum(c*trial)*np.linalg.inv(c)/3
    original_power = float(np.sum(MU/2*c/np.linalg.det(c)**(1/3)*trial))
    errors = dict(relaxed_energy=abs(float(energy)),relaxed_stress=float(abs(stress).max()),
                  stress_derivative=stress_derivative_error,metric_derivative=metric_derivative_error,
                  original_flow_power=abs(original_power))
    limits = dict(relaxed_energy=1e-9,relaxed_stress=1e-8,stress_derivative=1e-5,
                  metric_derivative=1e-5,original_flow_power=1e-8)
    return dict(errors=errors,fixed_limits=limits,checks_passed=all(errors[k]<v for k,v in limits.items()))


def trajectory(lane):
    gate = audit(lane)
    assert gate['checks_passed']
    run = F/gate['run']
    records = json.loads((run/'fields.json').read_text())['records']
    raw = np.memmap(run/'fields.bin', dtype='<f8', mode='r')
    xyz = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    initial_gauge = 1.5*MU
    previous_gauge = initial_gauge
    previous_peeq = 0.
    history = []
    for rec in records:
        def field(name, width):
            meta = rec['fields'][name]
            values = raw[meta['offset']//8:meta['offset']//8+meta['count']]
            return values.reshape((-1,width)) if name == 'u' else values.reshape((1,8,width))
        coef = np.linalg.lstsq(np.c_[np.ones(8),xyz], field('u',4)[:,1:], rcond=None)[0]
        deformation = np.eye(3)+coef[1:].T
        c = np.broadcast_to(deformation.T@deformation,(1,8,3,3))
        state = field('state',rec['states'])
        metric = np.eye(3)-2*tensors(state[...,1:7])
        assert np.linalg.eigvalsh(metric).min() > 0
        energy,_ = candidate(c,metric)
        gauge = float((1.5*MU*np.linalg.det(metric)**(1/3)).mean())
        peeq = float(state[...,0].mean())
        history.append(dict(time=rec['time'],candidate_energy_n_mm=float(energy.mean()),
                            fixed_c_candidate_release_n_mm=gauge-previous_gauge,
                            peeq_increment=peeq-previous_peeq))
        previous_gauge,previous_peeq = gauge,peeq
    assert all(sha256(run/n)==h for n,h in gate['input_output_hashes'].items())
    determinant = np.mean(gate['final_metric_determinant_range'])
    gauge_change = 1.5*MU*determinant**(1/3)-initial_gauge
    energy = gate['endpoint_potential_change_n_mm']-gauge_change
    return dict(lane=lane,run=gate['run'],accepted_frames=gate['accepted_frames'],
                prior_native_replay_checks_passed=gate['checks_passed'],
                final_metric_determinant_range=gate['final_metric_determinant_range'],
                original_potential_change_n_mm=gate['endpoint_potential_change_n_mm'],
                gauge_change_n_mm=gauge_change,candidate_endpoint_energy_n_mm=energy,
                native_work_n_mm=gate['totals_n_mm']['native_work'],
                work_minus_candidate_energy_n_mm=gate['totals_n_mm']['native_work']-energy,
                minimum_candidate_energy_n_mm=min(h['candidate_energy_n_mm'] for h in history),
                plastic_frames=sum(h['peeq_increment']>1e-12 for h in history),
                minimum_plastic_fixed_c_release_n_mm=min(h['fixed_c_candidate_release_n_mm'] for h in history if h['peeq_increment']>1e-12),
                history=history,input_output_hashes=gate['input_output_hashes'])


if __name__ == '__main__':
    out=Path(sys.argv[1]); assert not out.exists()
    result=dict(manufactured=manufactured(),trajectories=[trajectory(lane) for lane in ('standard','half')],
                script_sha256=sha256(__file__),
                dependencies={n:sha256(F/'results/scripts'/n) for n in ('audit_coupon_state_work.py','audit_frozen_metric_work.py','audit_plastic_return_map.py')},
                source_hashes={n:sha256(F/'sym/vendor/CalculiX/ccx_2.23/src'/n) for n in ('incplas.f','resultsmech.f')},
                derivation='W*=W-3mu/2 det(M)^(1/3). S=2dW*/dC is unchanged. Q*=mu/2[C/J^(2/3)-det(M)^(1/3) M^-1]. For SPD C,M, W*>=0 with equality at C=det(M)^(1/3) M^-1, J=1. Original Q:dotM=0 for C-deviatoric flow, but Q*:dotM need not vanish.',
                limitations='A stress-free minimum chooses a metric-only energy gauge; endpoint stress data alone do not identify that gauge as physical storage. Isotropic zero-backstress/no-thermal/no-creep branch only. Endpoints are homogeneous unit-reference-volume coupons; determinant midpoint uses their uniform state. Work-minus-candidate-energy includes temporal quadrature and unidentified hardening storage, is not measured dissipation. No native temporal convergence, production correction, pipe/cap/contact/MPC/Abaqus or experimental validation. Do not subtract this from production work gaps.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**result,'trajectories':[{k:v for k,v in t.items() if k not in ('input_output_hashes','history')} for t in result['trajectories']]},indent=2))
    assert result['manufactured']['checks_passed']

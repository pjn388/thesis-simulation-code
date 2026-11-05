"""Exact endpoint work/potential accounting; no dissipation identification."""
import json
import sys
from pathlib import Path

import numpy as np
from audit_plastic_return_map import F, tensors, sha256, update
from audit_frozen_metric_work import elastic


def audit(lane):
    suffix = 'plastic_state_coupon_20261011T095613Z' if lane == 'standard' else 'plastic_state_coupon_half_20261011T100722Z'
    run = F/'sym/runs/investigation'/suffix
    gate = F/'results/output'/('plastic_state_stress_coupon_20261011T100320Z.json' if lane == 'standard' else 'plastic_state_coupon_half_20261011T100722Z_audit.json')
    evidence = json.loads(gate.read_text())
    assert evidence['checks_passed']
    assert all(sha256(run/n) == h for n, h in evidence['hashes'].items())
    lines = (run/'job.inp').read_text().splitlines()
    curve = []
    for line in lines[lines.index('*PLASTIC')+1:]:
        if line.startswith('*'): break
        curve.append(list(map(float, line.split(','))))
    curve = np.array(curve)
    mu, bulk = 202500/(2*1.3), 202500/(3*.4)
    xyz = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    raw = np.memmap(run/'fields.bin', dtype='<f8', mode='r')
    records = json.loads((run/'fields.json').read_text())['records']
    metric = np.broadcast_to(np.eye(3), (1,8,3,3)).copy()
    c0 = metric.copy(); s0 = np.zeros_like(metric); peeq = np.zeros((1,8))
    initial, _ = elastic(c0, metric, mu, bulk)
    previous_p = initial
    totals = dict(native_work=0., trial_quadrature_error=0., endpoint_metric_drop=0., endpoint_stress_correction=0.)
    maxima = dict(replay_stress_mpa=0., replay_metric=0., replay_peeq=0., potential_stress_mpa=0., accounting_n_mm=0., native_energy_n_mm=0.)
    history = []
    normalized_history = []
    for rec in records:
        def field(name, width):
            m = rec['fields'][name]
            a = raw[m['offset']//8:m['offset']//8+m['count']]
            return a.reshape((-1,width)) if name == 'u' else a.reshape((1,8,width))
        coef = np.linalg.lstsq(np.c_[np.ones(8),xyz], field('u',4)[:,1:], rcond=None)[0]
        deformation = np.eye(3)+coef[1:].T
        assert np.linalg.det(deformation) > 0
        inv = np.linalg.inv(deformation)
        c = np.broadcast_to(deformation.T@deformation, c0.shape)
        state = field('state', rec['states'])
        new_metric = np.eye(3)-2*tensors(state[...,1:7])
        assert np.linalg.eigvalsh(new_metric).min() > 0
        stress = np.linalg.det(deformation)*(inv@tensors(field('stress',6))@inv.T)
        predicted, pm, pp, active = update(c, metric, peeq, curve, mu, bulk)
        potential, potential_stress = elastic(c, new_metric, mu, bulk)
        determinant = np.linalg.det(new_metric)
        root = determinant**(1/3)
        normalized_p, normalized_s = elastic(c, new_metric/root[...,None,None], mu, bulk)
        j = np.sqrt(np.linalg.det(c))
        volumetric_s = bulk/2*(j*j-1)[...,None,None]*np.linalg.inv(c)
        expected_s = (potential_stress-volumetric_s)/root[...,None,None]+volumetric_s
        # Algebraic scaling identity is a verification gate; agreement with
        # native stress is a separately reported counterfactual verdict.
        assert abs(expected_s-normalized_s).max() < 1e-8
        normalized_history.append(dict(time=rec['time'], plastic_update=bool(active.any()),
            metric_determinant_range=[float(determinant.min()),float(determinant.max())],
            potential_change_n_mm=float((normalized_p-initial).mean()),
            potential_shift_n_mm=float((normalized_p-potential).mean()),
            stress_change_mpa=float(abs(normalized_s-stress).max()),
            stress_within_native_identity_tolerance=bool(abs(normalized_s-stress).max() < 1e-4),
            scaling_identity_error_mpa=float(abs(expected_s-normalized_s).max())))
        trial_p, trial_s = elastic(c, metric, mu, bulk)
        dc = c-c0
        work = .25*np.einsum('...ij,...ij->...', stress+s0, dc)
        trial_work = .25*np.einsum('...ij,...ij->...', trial_s+s0, dc)
        pieces = dict(native_work=work, trial_quadrature_error=trial_work-(trial_p-previous_p),
                      endpoint_metric_drop=trial_p-potential,
                      endpoint_stress_correction=.25*np.einsum('...ij,...ij->...', stress-trial_s, dc))
        delta_p = potential-previous_p
        accounting = work-delta_p-pieces['trial_quadrature_error']-pieces['endpoint_metric_drop']-pieces['endpoint_stress_correction']
        for k,v in pieces.items(): totals[k] += float(v.mean())
        errors = dict(replay_stress_mpa=float(abs(predicted-stress).max()), replay_metric=float(abs(pm-new_metric).max()),
                      replay_peeq=float(abs(pp-state[...,0]).max()), potential_stress_mpa=float(abs(potential_stress-stress).max()),
                      accounting_n_mm=float(abs(accounting).max()), native_energy_n_mm=abs(totals['native_work']-float(field('energy',2)[...,0].mean())))
        for k,v in errors.items(): maxima[k] = max(maxima[k],v)
        history.append(dict(time=rec['time'], plastic_update=bool(active.any()),
                            work_minus_potential_increment_n_mm=float((work-delta_p).mean()),
                            **{k:float(v.mean()) for k,v in pieces.items()}))
        metric, peeq, c0, s0, previous_p = new_metric, state[...,0], c, stress, potential
    limits = dict(replay_stress_mpa=1e-4, replay_metric=1e-8, replay_peeq=1e-8,
                  potential_stress_mpa=1e-4, accounting_n_mm=1e-10, native_energy_n_mm=1e-8)
    passed = all(maxima[k] < v for k,v in limits.items())
    assert all(sha256(run/n) == h for n,h in evidence['hashes'].items())
    delta = float((previous_p-initial).mean())
    determinant = np.linalg.det(metric)
    normalized = metric/determinant[...,None,None]**(1/3)
    normalized_p, normalized_s = elastic(c0, normalized, mu, bulk)
    return dict(lane=lane, run=str(run.relative_to(F)), checks_passed=passed, accepted_frames=len(records),
                fixed_limits=limits, maximum_errors=maxima, totals_n_mm=totals, endpoint_potential_change_n_mm=delta,
                native_work_minus_endpoint_potential_n_mm=totals['native_work']-delta,
                minimum_increment_residual_n_mm=min(h['work_minus_potential_increment_n_mm'] for h in history),
                final_metric_determinant_range=[float(determinant.min()),float(determinant.max())],
                counterfactual_unit_determinant_potential_change_n_mm=float((normalized_p-initial).mean()),
                counterfactual_endpoint_stress_change_mpa=float(abs(normalized_s-s0).max()),
                counterfactual_all_endpoints=dict(stress_tolerance_mpa=1e-4,
                    maximum_stress_change_mpa=max(h['stress_change_mpa'] for h in normalized_history),
                    frames_exceeding_stress_tolerance=sum(not h['stress_within_native_identity_tolerance'] for h in normalized_history),
                    maximum_scaling_identity_error_mpa=max(h['scaling_identity_error_mpa'] for h in normalized_history),
                    history=normalized_history),
                history=history, input_output_hashes=evidence['hashes'], gate_sha256=sha256(gate))


if __name__ == '__main__':
    out = Path(sys.argv[1]); assert not out.exists()
    result = dict(trajectories=[audit(lane) for lane in ('standard','half')],
                  script_sha256=sha256(__file__), potential_script_sha256=sha256(F/'results/scripts/audit_frozen_metric_work.py'),
                  return_map_script_sha256=sha256(F/'results/scripts/audit_plastic_return_map.py'),
                  source_hashes={n:sha256(F/'sym/vendor/CalculiX/ccx_2.23/src'/n) for n in ('incplas.f','resultsmech.f')},
                  sources='incplas.f:150-190,315-345,754-790; resultsmech.f:1064-1074. Unit reference-volume homogeneous coupon, mean over eight integration points.',
                  limitations='Exact endpoint algebra for elastic potential at retained evolving metric, excluding hardening storage. Work-minus-potential is not identified physical dissipation. Unit-determinant normalization is a counterfactual endpoint diagnostic, changes stress and is not a corrected native model or free energy identification. Endpoint metric update is an accounting split, not an actual within-increment path. No native temporal convergence, work-gap correction, pipe/cap/contact/MPC/Abaqus or experimental validation.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**result,'trajectories':[{k:v for k,v in t.items() if k not in ('history','input_output_hashes','counterfactual_all_endpoints')} for t in result['trajectories']]},indent=2))
    assert all(t['checks_passed'] for t in result['trajectories'])

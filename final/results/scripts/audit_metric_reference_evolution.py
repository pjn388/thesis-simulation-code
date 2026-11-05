"""Retained metric-reference increments; descriptive, no physical acceptance."""
import json
import sys
import numpy as np
from compare_native_plastic_time import F, STANDARD, QUARTER, load, field, compare
from audit_internal_pipe_fields import spatial, tensors
from pipe_sim.model import Model
from pipe_sim.runner import sha256


def audit(gate, reference):
    evidence, run, records, raw = load(gate)
    model = Model.load(run/'model.json')
    inside = False
    for line in (run/'mesh.inp').read_text().splitlines():
        if line.startswith('**'): continue
        if line.startswith('*'): inside = line.upper().strip() == '*NODE'
        elif inside and line.strip():
            row = line.split(','); model.nodes[int(row[0])] = list(map(float,row[1:4]))
    ids, _, _, weights, checks = spatial(model)
    selection = np.array(ids)-1
    material = json.loads((run/'case.json').read_text())['material']
    mu = material['young']/(2*(1+material['poisson']))
    previous = np.broadcast_to(np.eye(3), weights.shape+(3,3)).copy()
    previous_peeq = np.zeros(weights.shape)
    history = []
    for rec, retained in zip(records, reference['history']):
        assert rec['time'] == retained['time']
        state = field(raw,rec,'state',rec['states'])[selection]
        metric = np.eye(3)-2*tensors(state[...,1:7])
        c = np.eye(3)+2*tensors(field(raw,rec,'strain',6)[selection])
        delta = metric-previous
        det = np.linalg.det(metric)
        assert np.linalg.eigvalsh(metric).min()>0
        # Exact 3x3 determinant polynomial, independent of direct det(Mnew).
        a = np.linalg.solve(previous,delta)
        trace = np.trace(a,axis1=-2,axis2=-1)
        second = (trace**2-np.trace(a@a,axis1=-2,axis2=-1))/2
        cubic = np.linalg.det(a)
        ratio = 1+trace+second+cubic
        determinant_error = float(abs(det/np.linalg.det(previous)-ratio).max())
        gauge_density_change = 1.5*mu*(np.cbrt(det)-1)
        stable_gauge = float((gauge_density_change*weights).sum())
        gauge = float((1.5*mu*det**(1/3)*weights).sum())-1.5*mu*float(weights.sum())
        gauge_error = abs(gauge-retained['gauge_change_n_mm'])
        active = state[...,0] > previous_peeq
        history.append(dict(time=rec['time'],pressure_mpa=24*rec['time'],
            gauge_change_n_mm=gauge,gauge_reconstruction_error_n_mm=gauge_error,
            stable_subtraction_difference_n_mm=stable_gauge-gauge,
            maximum_determinant_polynomial_error=determinant_error,
            minimum_metric_determinant=float(det.min()),maximum_metric_determinant=float(det.max()),
            weighted_mean_metric_determinant=float((det*weights).sum()/weights.sum()),
            maximum_c_contracted_metric_increment=float(abs(np.einsum('...ij,...ij->...',c,delta)).max()),
            maximum_metric_increment=float(abs(delta).max()),
            maximum_inactive_metric_increment=float(abs(delta[~active]).max()) if (~active).any() else 0.,
            plastic_increment_volume_fraction=float(weights[active].sum()/weights.sum()),
            maximum_peeq=float(state[...,0].max())))
        assert determinant_error<1e-12 and gauge_error<1e-4, (rec["time"],determinant_error,gauge_error)
        previous = metric.copy(); previous_peeq=state[...,0].copy()
    assert len(history)==len(reference['history'])
    assert all(sha256(run/n)==h for n,h in evidence['input_output_hashes'].items())
    return dict(identity_checks_passed=True,run=evidence['run'],history=history,
        reference_volume_mm3=float(weights.sum()),initial_gauge_n_mm=1.5*mu*float(weights.sum()),
        spatial_checks=checks,gate_sha256=sha256(gate),input_output_hashes=evidence['input_output_hashes'])


if __name__=='__main__':
    out=F/'results/output'/sys.argv[1]; assert not out.exists()
    reference_path=F/'results/output/pipe_metric_energy_source_20261011T143001Z.json'
    reference=json.loads(reference_path.read_text()); assert reference['checks_passed']
    comparison=compare(); assert comparison['provenance_checks_passed']
    lanes=[audit(g,r) for g,r in zip((STANDARD,QUARTER),reference['lanes'])]
    result=dict(identity_checks_passed=True,lanes=lanes,
        reference_report_sha256=sha256(reference_path),script_sha256=sha256(__file__),
        dependencies={n:sha256(F/'results/scripts'/n) for n in ('compare_native_plastic_time.py','audit_internal_pipe_fields.py')},
        source_sha256=sha256(F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'),
        source_lines='156-189 trial C-deviatoric tensor; 754-790 additive metric update',
        limitations='Exact determinant/reference identities only. C-contracted additive increments and determinant changes are descriptive; no altered native metric, unit-determinant normalization, energy-gap correction, physical storage/dissipation, asymptotic convergence, cap/contact/MPC/Abaqus or experimental acceptance. Two resolutions isolate temporal sensitivity but not intermediate native plastic power.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(identity_checks_passed=True,endpoints=[l['history'][-1] for l in lanes],
        maxima=[{k:max(h[k] for h in l['history']) for k in ('maximum_c_contracted_metric_increment','maximum_inactive_metric_increment','gauge_reconstruction_error_n_mm','maximum_determinant_polynomial_error')} for l in lanes],initial_gauge_n_mm=lanes[0]['initial_gauge_n_mm']),indent=2))

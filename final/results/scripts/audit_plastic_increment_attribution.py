"""Attribute retained reference changes to actual accepted plastic increments.

Descriptive identities only; no constitutive state interpolation or energy claim.
"""
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
            row = line.split(','); model.nodes[int(row[0])] = list(map(float, row[1:4]))
    ids, _, _, weights, checks = spatial(model)
    selection = np.array(ids)-1
    material = json.loads((run/'case.json').read_text())['material']
    mu = material['young']/(2*(1+material['poisson']))
    previous_peeq = np.zeros(weights.shape)
    previous_density = np.zeros(weights.shape)
    previous_time = 0.
    previous_gauge = 0.
    history = []
    for rec, retained in zip(records, reference['history']):
        assert rec['time'] == retained['time']
        state = field(raw, rec, 'state', rec['states'])[selection]
        increment = state[..., 0]-previous_peeq
        assert increment.min() >= -1e-12
        active = increment > 0
        metric = np.eye(3)-2*tensors(state[..., 1:7])
        density = 1.5*mu*(np.cbrt(np.linalg.det(metric))-1)
        contribution = (density-previous_density)*weights
        total = float(contribution.sum())
        inactive = float(abs(contribution[~active]).sum())
        assert inactive < 1e-4
        gauge_increment = retained['gauge_change_n_mm']-previous_gauge
        assert abs(total-gauge_increment) < .01  # retained integrated subtraction rounding
        history.append(dict(time=rec['time'], pressure_mpa=24*rec['time'],
            dt=rec['time']-previous_time, plastic=bool(active.any()),
            active_volume_fraction=float(weights[active].sum()/weights.sum()),
            maximum_peeq_increment=float(increment.max()),
            volume_integrated_peeq_increment_mm3=float((increment*weights).sum()),
            volume_integrated_squared_peeq_increment_mm3=float((increment**2*weights).sum()),
            reference_increment_n_mm=total, inactive_absolute_reference_increment_n_mm=inactive,
            retained_increment_identity_error_n_mm=abs(total-gauge_increment)))
        previous_time=rec['time']; previous_peeq=state[..., 0].copy()
        previous_density=density; previous_gauge=retained['gauge_change_n_mm']
    assert len(history)==len(reference['history'])
    plastic = [h for h in history if h['plastic']]
    endpoint = dict(plastic_frames=len(plastic), first_plastic_pressure_mpa=plastic[0]['pressure_mpa'],
        plastic_dt_min=min(h['dt'] for h in plastic), plastic_dt_max=max(h['dt'] for h in plastic),
        maximum_peeq_increment=max(h['maximum_peeq_increment'] for h in plastic),
        accumulated_squared_peeq_increment_mm3=sum(h['volume_integrated_squared_peeq_increment_mm3'] for h in history),
        integrated_final_peeq_mm3=sum(h['volume_integrated_peeq_increment_mm3'] for h in history),
        accumulated_reference_n_mm=sum(h['reference_increment_n_mm'] for h in history))
    assert abs(endpoint['accumulated_reference_n_mm']-previous_gauge)<.01
    assert all(sha256(run/n)==h for n,h in evidence['input_output_hashes'].items())
    return dict(run=evidence['run'], gate_sha256=sha256(gate), input_output_hashes=evidence['input_output_hashes'],
        spatial_checks=checks, history=history, endpoint=endpoint)


if __name__ == '__main__':
    out=F/'results/output'/sys.argv[1]; assert not out.exists()
    reference_path=F/'results/output/metric_reference_evolution_20261011T150003Z.json'
    reference=json.loads(reference_path.read_text()); assert reference['identity_checks_passed']
    comparison=compare(); assert comparison['provenance_checks_passed']
    lanes=[audit(g,r) for g,r in zip((STANDARD,QUARTER),reference['lanes'])]
    shared=[]
    for a in lanes[0]['history']:
        matches=[b for b in lanes[1]['history'] if abs(a['time']-b['time'])<=1e-10]
        if not matches: continue
        b=matches[0]
        shared.append(dict(pressure_mpa=a['pressure_mpa'],
            accumulated_reference_difference_n_mm=sum(h['reference_increment_n_mm'] for h in lanes[1]['history'] if h['time']<=b['time'])-sum(h['reference_increment_n_mm'] for h in lanes[0]['history'] if h['time']<=a['time'])))
    result=dict(identity_checks_passed=True,physical_acceptance=False,lanes=lanes,shared_history=shared,
        reference_sha256=sha256(reference_path),script_sha256=sha256(__file__),
        dependencies={n:sha256(F/'results/scripts'/n) for n in ('compare_native_plastic_time.py','audit_internal_pipe_fields.py')},
        source_sha256=sha256(F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'),
        limits=dict(peeq_monotonic_slack=1e-12,inactive_reference_n_mm=1e-4,integrated_arithmetic_n_mm=.01),
        limitations='Attribution of retained reference, not stored energy/dissipation or causation. Squared PEEQ increments are descriptive, not native return-map work. No interpolated states, new solve, production correction, convergence, cap/contact/MPC/Abaqus or physical burst acceptance.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(identity_checks_passed=True,endpoints=[l['endpoint'] for l in lanes],shared_times=len(shared)),indent=2))

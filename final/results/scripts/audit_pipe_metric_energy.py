"""Gated reference-volume candidate storage; not identified physical energy."""
import json
import sys
import numpy as np
from compare_native_plastic_time import F, STANDARD, QUARTER, load, field, retained_work, compare
from audit_internal_pipe_fields import spatial, tensors
from audit_frozen_metric_work import elastic
from pipe_sim.model import Model
from pipe_sim.runner import sha256


def audit(gate):
    evidence, run, records, raw = load(gate)
    retained = retained_work(evidence, run)
    model = Model.load(run/'model.json')
    inside = False
    for line in (run/'mesh.inp').read_text().splitlines():
        if line.startswith('**'): continue
        if line.startswith('*'): inside = line.upper().strip() == '*NODE'
        elif inside and line.strip():
            row = line.split(','); model.nodes[int(row[0])] = list(map(float,row[1:4]))
    ids, nodes, grad, weights, spatial_checks = spatial(model)
    selection = np.array(ids)-1
    material = json.loads((run/'case.json').read_text())['material']
    mu = material['young']/(2*(1+material['poisson']))
    bulk = material['young']/(3*(1-2*material['poisson']))
    curve_extent = json.loads(material['source'])['source_supported_plastic_strain_max']
    initial_gauge = 1.5*mu*float(weights.sum())
    previous_gauge = initial_gauge
    minimum_density = float('inf'); stress_error = green_error = 0.
    history = []
    for rec in records:
        u = field(raw,rec,'u',4)
        deformation = np.eye(3)+np.einsum('eni,epnj->epij',u[nodes-1],grad)
        c = np.swapaxes(deformation,-1,-2)@deformation
        strain = tensors(field(raw,rec,'strain',6)[selection])
        green_error = max(green_error,float(abs(c-np.eye(3)-2*strain).max()))
        state = field(raw,rec,'state',rec['states'])[selection]
        metric = np.eye(3)-2*tensors(state[...,1:7])
        assert np.linalg.eigvalsh(metric).min()>0 and abs(state[...,7:13]).max()==0
        potential, stress = elastic(c,metric,mu,bulk)
        inverse = np.linalg.inv(deformation); j = np.linalg.det(deformation)
        sigma = tensors(field(raw,rec,'stress',6)[selection])
        native = j[...,None,None]*(inverse@sigma@np.swapaxes(inverse,-1,-2))
        stress_error = max(stress_error,float(abs(stress-native).max()))
        gauge_density = 1.5*mu*np.linalg.det(metric)**(1/3)
        candidate = potential-gauge_density
        minimum_density = min(minimum_density,float(candidate.min()))
        gauge_value = float((gauge_density*weights).sum())
        history.append(dict(time=rec['time'],pressure_mpa=24*rec['time'],
            candidate_energy_n_mm=float((candidate*weights).sum()),
            gauge_change_n_mm=gauge_value-initial_gauge,
            endpoint_gauge_increment_n_mm=gauge_value-previous_gauge,
            native_work_n_mm=float((field(raw,rec,'energy',2)[selection,...,0]*weights).sum()),
            maximum_peeq=float(state[...,0].max())))
        previous_gauge = gauge_value
    assert stress_error<1e-7 and green_error<1e-10 and minimum_density>=-1e-8
    assert all(sha256(run/n)==h for n,h in evidence['input_output_hashes'].items())
    return dict(checks_passed=True,run=evidence['run'],accepted_frames=len(records),
        maximum_pk2_error_mpa=stress_error,maximum_c_green_error=green_error,
        minimum_candidate_density_n_mm_per_mm3=minimum_density,
        recovered_material_peeq_extent=curve_extent,
        within_recovered_material_extent=max(h['maximum_peeq'] for h in history)<=curve_extent,
        spatial_checks=spatial_checks,retained_work=retained,history=history,
        gate_sha256=sha256(gate),input_output_hashes=evidence['input_output_hashes'])


if __name__=='__main__':
    out=F/'results/output'/sys.argv[1]; assert not out.exists()
    identity=compare()
    lanes=[audit(gate) for gate in (STANDARD,QUARTER)]
    a,b=lanes
    common=[]
    for ha in a['history']:
        hb=min(b['history'],key=lambda h:abs(h['time']-ha['time']))
        if abs(hb['time']-ha['time'])<1e-10:
            common.append(dict(time=ha['time'],candidate_difference_n_mm=hb['candidate_energy_n_mm']-ha['candidate_energy_n_mm'],gauge_difference_n_mm=hb['gauge_change_n_mm']-ha['gauge_change_n_mm']))
    assert common
    result=dict(checks_passed=True,lanes=lanes,common_frames=common,
        deck_temporal_comparison_checks_passed=identity['provenance_checks_passed'],
        script_sha256=sha256(__file__),dependencies={n:sha256(F/'results/scripts'/n) for n in ('compare_native_plastic_time.py','audit_internal_pipe_fields.py','audit_frozen_metric_work.py')},
        sources={n:sha256(F/'sym/vendor/CalculiX/ccx_2.23/src'/n) for n in ('incplas.f','shape20h.f','resultsmech.f')},
        limitations='W*=W-1.5mu det(M)^(1/3), integrated over literal C3D20R reference Jacobians. A mathematical stress-free gauge, not identified physical storage or dissipation. Endpoint gauge increments are not within-increment plastic power. Standard/quarter temporal differences are sensitivity, not asymptotic convergence. No production work-gap subtraction, cap/contact/MPC/Abaqus or experimental validation.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(checks_passed=True,lanes=[{k:v for k,v in t.items() if k not in ('history','input_output_hashes','retained_work')} for t in lanes],endpoints=[t['history'][-1] for t in lanes],matched_frames=len(common),maximum_common_candidate_difference_n_mm=max(abs(h['candidate_difference_n_mm']) for h in common)),indent=2))

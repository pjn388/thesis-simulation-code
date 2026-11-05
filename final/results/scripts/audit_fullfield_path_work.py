"""Bounded full-field assumed affine-F total material work reconstruction."""
import json
import sys
from pathlib import Path

import numpy as np
from compare_native_plastic_time import STANDARD, QUARTER, load, field
from audit_plastic_return_map import update, F, tensors, sha256
from audit_internal_pipe_fields import spatial
from pipe_sim.model import Model


def audit(gate):
    evidence, run, records, raw = load(gate)
    model = Model.load(run/'model.json')
    inside = False
    for line in (run/'mesh.inp').read_text().splitlines():
        if line.startswith('**'): continue
        if line.startswith('*'): inside = line.upper().strip() == '*NODE'
        elif inside and line.strip():
            row = line.split(',')
            model.nodes[int(row[0])] = list(map(float, row[1:4]))
    ids, nodes, gradients, weights, spatial_checks = spatial(model)
    select = np.array(ids)-1
    grad = gradients
    node_ids = nodes
    material = json.loads((run/'case.json').read_text())['material']
    mu = material['young']/(2*(1+material['poisson']))
    bulk = material['young']/(3*(1-2*material['poisson']))
    lines = (run/'job.inp').read_text().splitlines(); curve = []
    for line in lines[lines.index('*PLASTIC')+1:]:
        if line.startswith('*'): break
        curve.append(list(map(float, line.split(','))))
    curve = np.array(curve)
    resolutions = []
    for count in (1, 2, 4):
        metric = np.broadcast_to(np.eye(3), (len(ids), 8, 3, 3)).copy(); peeq = np.zeros((len(ids), 8))
        previous_f = metric.copy(); previous_e = np.zeros_like(metric); previous_s = previous_e.copy()
        work = 0.; minimum_j = 1.
        errors = dict(stress_mpa=0., metric=0., peeq=0.)
        for rec in records:
            deformation = np.eye(3)+np.einsum('eni,epnj->epij', field(raw, rec, 'u', 4)[node_ids-1], grad)
            inverse = np.linalg.inv(deformation)
            actual_stress = np.linalg.det(deformation)[...,None,None]*(inverse@tensors(field(raw, rec, 'stress', 6)[select])@np.swapaxes(inverse,-1,-2))
            state = field(raw, rec, 'state', rec['states'])[select]
            frame = dict(F=deformation, stress=actual_stress, metric=np.eye(3)-2*tensors(state[...,1:7]), peeq=state[...,0])
            for step in range(1, count+1):
                deformation = previous_f+(frame['F']-previous_f)*step/count
                minimum_j = min(minimum_j, float(np.linalg.det(deformation).min()))
                assert minimum_j > 0
                c = np.swapaxes(deformation,-1,-2)@deformation; strain = (c-np.eye(3))/2
                stress, metric, peeq, _ = update(c, metric, peeq, curve, mu, bulk)
                work += float((.5*np.einsum('epij,epij->ep', stress+previous_s, strain-previous_e)*weights).sum())
                previous_s, previous_e = stress, strain
            for key, value in dict(stress_mpa=float(abs(stress-frame['stress']).max()), metric=float(abs(metric-frame['metric']).max()), peeq=float(abs(peeq-frame['peeq']).max())).items():
                errors[key] = max(errors[key], value)
            previous_f = frame['F']
        resolutions.append(dict(substeps=count, pipe_work_n_mm=work, maximum_final_peeq=float(peeq.max()), maximum_endpoint_differences=errors, minimum_assumed_path_jacobian=minimum_j))
    limits = dict(stress_mpa=1e-4, metric=1e-8, peeq=1e-8)
    native_energy = evidence['history'][-1]['native_energy_n_mm']
    energy_error = float(abs(resolutions[0]['pipe_work_n_mm']-native_energy))
    passed = all(resolutions[0]['maximum_endpoint_differences'][k] < v for k,v in limits.items()) and energy_error < 1e-5
    assert all(sha256(run/n) == h for n,h in evidence['input_output_hashes'].items())
    return dict(endpoint_identity_passed=bool(passed), fixed_limits=limits, native_energy_n_mm=native_energy, baseline_energy_error_n_mm=energy_error, accepted_frames=len(records), elements=len(ids), resolutions=resolutions, gate_sha256=sha256(gate), run=evidence['run'], spatial_checks=spatial_checks)


if __name__ == '__main__':
    out = Path(sys.argv[1]); assert not out.exists()
    result = dict(trajectories=[audit(g) for g in (STANDARD, QUARTER)], script_sha256=sha256(__file__), return_map_sha256=sha256(F/'results/scripts/audit_plastic_return_map.py'), limitations='All pipe integration points, reference-volume quadrature. Affine F between output endpoints is assumed; finer substeps change accumulated plastic states. Work is assumed-path total material work, not plastic dissipation, recovered native trajectory or an error bound. No native temporal convergence, coupled cap/contact/MPC/Abaqus or physical validation.')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    assert all(t['endpoint_identity_passed'] for t in result['trajectories'])

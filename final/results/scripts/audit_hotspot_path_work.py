"""Bounded assumed affine-F plastic work at the retained endpoint hotspot."""
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
    ids, nodes, gradients, _, spatial_checks = spatial(model)
    last = field(raw, records[-1], 'state', records[-1]['states'])[np.array(ids)-1, :, 0]
    element_index, ip = np.unravel_index(last.argmax(), last.shape)
    element = ids[element_index]
    grad = gradients[element_index, ip]
    node_ids = nodes[element_index]
    material = json.loads((run/'case.json').read_text())['material']
    mu = material['young']/(2*(1+material['poisson']))
    bulk = material['young']/(3*(1-2*material['poisson']))
    lines = (run/'job.inp').read_text().splitlines(); curve = []
    for line in lines[lines.index('*PLASTIC')+1:]:
        if line.startswith('*'): break
        curve.append(list(map(float, line.split(','))))
    curve = np.array(curve)
    frames = []
    for rec in records:
        deformation = np.eye(3)+field(raw, rec, 'u', 4)[node_ids-1].T@grad
        inverse = np.linalg.inv(deformation)
        stress = np.linalg.det(deformation)*inverse@tensors(field(raw, rec, 'stress', 6)[element-1, ip])@inverse.T
        state = field(raw, rec, 'state', rec['states'])[element-1, ip]
        frames.append(dict(F=deformation, stress=stress, metric=np.eye(3)-2*tensors(state[1:7]), peeq=state[0], energy=field(raw, rec, 'energy', 2)[element-1, ip, 0]))
    resolutions = []
    for count in (1, 2, 4, 8, 16):
        metric = np.eye(3); peeq = np.array(0.)
        previous_f = np.eye(3); previous_e = np.zeros((3,3)); previous_s = previous_e.copy()
        work = 0.; minimum_j = 1.
        errors = dict(stress_mpa=0., metric=0., peeq=0.)
        for frame in frames:
            for step in range(1, count+1):
                deformation = previous_f+(frame['F']-previous_f)*step/count
                minimum_j = min(minimum_j, float(np.linalg.det(deformation)))
                assert minimum_j > 0
                c = deformation.T@deformation; strain = (c-np.eye(3))/2
                stress, metric, peeq, _ = update(c, metric, peeq, curve, mu, bulk)
                work += float(.5*np.einsum('ij,ij', stress+previous_s, strain-previous_e))
                previous_s, previous_e = stress, strain
            for key, value in dict(stress_mpa=float(abs(stress-frame['stress']).max()), metric=float(abs(metric-frame['metric']).max()), peeq=float(abs(peeq-frame['peeq']))).items():
                errors[key] = max(errors[key], value)
            previous_f = frame['F']
        resolutions.append(dict(substeps=count, work_density_n_mm_per_mm3=work, final_peeq=float(peeq), maximum_endpoint_differences=errors, minimum_assumed_path_jacobian=minimum_j))
    limits = dict(stress_mpa=1e-4, metric=1e-8, peeq=1e-8)
    energy_error = float(abs(resolutions[0]['work_density_n_mm_per_mm3']-frames[-1]['energy']))
    passed = all(resolutions[0]['maximum_endpoint_differences'][k] < v for k,v in limits.items()) and energy_error < 1e-8
    assert all(sha256(run/n) == h for n,h in evidence['input_output_hashes'].items())
    return dict(endpoint_identity_passed=bool(passed), fixed_limits=limits, native_energy_density=float(frames[-1]['energy']), baseline_energy_density_error=energy_error, hotspot=dict(element=int(element), integration_point=int(ip)+1), accepted_frames=len(frames), resolutions=resolutions, gate_sha256=sha256(gate), run=evidence['run'], spatial_checks=spatial_checks)


if __name__ == '__main__':
    out = Path(sys.argv[1]); assert not out.exists()
    result = dict(trajectories=[audit(g) for g in (STANDARD, QUARTER)], script_sha256=sha256(__file__), return_map_sha256=sha256(F/'results/scripts/audit_plastic_return_map.py'), limitations='Selected endpoint hotspot only. Affine F between output endpoints is assumed; finer substeps change accumulated plastic states. Work is total material work density, not plastic dissipation, total pipe energy, recovered native trajectory or an error bound. No native temporal convergence, coupled cap/contact/MPC/Abaqus or physical validation.')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    assert all(t['endpoint_identity_passed'] for t in result['trajectories'])

"""Separate an assumed stress interpolation from the native endpoint work rule."""
import json
import sys
import numpy as np
from audit_internal_pipe_fields import F, spatial, tensors
from pipe_sim.model import Model
from pipe_sim.runner import sha256


def audit(lane):
    evidence = F/'results/output'/f'internal_pipe_work_verified_{lane}_20261011T093005Z.json'
    report = json.loads(evidence.read_text())
    assert report['field_checks_passed']
    run = F/report['run']
    hashes = report['input_output_hashes']
    assert all(sha256(run/name) == value for name, value in hashes.items())
    model = Model.load(run/'model.json')
    inside = False
    for line in (run/'mesh.inp').read_text().splitlines():
        if line.startswith('**'):
            continue
        if line.startswith('*'):
            inside = line.upper().strip() == '*NODE'
        elif inside and line.strip():
            values = line.split(',')
            model.nodes[int(values[0])] = list(map(float, values[1:4]))
    ids, nodes, grad, weights, checks = spatial(model)
    raw = np.memmap(run/'fields.bin', dtype='<f8', mode='r')
    records = json.loads((run/'fields.json').read_text())['records']
    previous_f = np.broadcast_to(np.eye(3), (len(ids), 8, 3, 3)).copy()
    previous_s = np.zeros_like(previous_f)
    cumulative = trapezoid = 0.
    maximum_identity = maximum_native = 0.
    history = []
    assert len(records) == len(report['history'])
    for record, retained in zip(records, report['history']):
        assert record['time'] == retained['time']
        def field(name, width):
            meta = record['fields'][name]
            values = raw[meta['offset']//8:meta['offset']//8+meta['count']]
            if name == 'u':
                return values.reshape((-1, width))[:, 1:]
            return values.reshape((record['elements'], record['ips'], width))[np.array(ids)-1, :8]
        u = field('u', 4)
        current_f = np.eye(3)+np.einsum('eni,epnj->epij', u[nodes-1], grad)
        inverse = np.linalg.inv(current_f)
        current_s = np.linalg.det(current_f)[..., None, None]*(inverse@tensors(field('stress', 6))@np.swapaxes(inverse, -1, -2))
        delta_f = current_f-previous_f
        # E(a)=E0+a*A+a^2*B, B=deltaF^T deltaF/2. For linear S(a),
        # integral S:dE minus endpoint-average S:deltaE = deltaS:B/6.
        correction = np.einsum('epij,epij->ep', current_s-previous_s, np.swapaxes(delta_f, -1, -2)@delta_f)/12
        cumulative += float((correction*weights).sum())
        delta_e = (np.swapaxes(current_f, -1, -2)@current_f-np.swapaxes(previous_f, -1, -2)@previous_f)/2
        trapezoid += float((.5*np.einsum('epij,epij->ep', current_s+previous_s, delta_e)*weights).sum())
        native_difference = trapezoid-retained['internal_energy_n_mm']
        identity_difference = cumulative-(retained['linear_output_path_material_work_n_mm']-trapezoid)
        maximum_native = max(maximum_native, abs(native_difference))
        maximum_identity = max(maximum_identity, abs(identity_difference))
        history.append(dict(time=record['time'],analytical_interpolation_correction_n_mm=cumulative,endpoint_trapezoid_work_n_mm=trapezoid,trapezoid_minus_native_n_mm=native_difference,analytical_identity_error_n_mm=identity_difference,external_minus_endpoint_trapezoid_n_mm=sum(retained['external_work_by_surface_n_mm'].values())-trapezoid))
        previous_f, previous_s = current_f, current_s
    # Numerical identity tolerance only; no physical work acceptance fitted.
    assert maximum_identity < 1e-5 and maximum_native < 1e-5
    assert all(sha256(run/name) == value for name, value in hashes.items())
    source = F/'sym/vendor/CalculiX/ccx_2.23/src/resultsmech.f'
    return dict(lane=lane,run=report['run'],checks_passed=True,accepted_frames=len(history),maximum_identity_error_n_mm=maximum_identity,maximum_trapezoid_minus_native_n_mm=maximum_native,history=history,input_output_hashes=hashes,evidence_sha256=sha256(evidence),script_sha256=sha256(__file__),source_sha256=sha256(source),source='resultsmech.f:1044-1074 endpoint-average mechanical energy update',spatial_checks=checks,limitations='Algebraic decomposition of assumed linear F and S paths, not the actual plastic constitutive trajectory. Native endpoint work reproduction does not prove physical energy accuracy, plastic convergence, cap/contact/MPC/Abaqus equivalence or burst.')


if __name__ == '__main__':
    result = audit(sys.argv[1])
    out = F/'results/output'/f'stress_path_decomposition_{sys.argv[1]}_20261011T095613Z.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('history','input_output_hashes')}|{'final':result['history'][-1]}, indent=2))

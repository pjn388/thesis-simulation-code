"""Algebraic energy decomposition on shared observed states; no native launch."""
import json
import sys
from pathlib import Path

import numpy as np
from compare_native_plastic_time import STANDARD, QUARTER, compare, load, field, F
from audit_internal_pipe_fields import spatial, tensors
from pipe_sim.model import Model
from pipe_sim.runner import sha256


def partition_work(gate, times):
    evidence, run, records, raw = load(gate)
    model = Model.load(run/'model.json')
    inside = False
    for line in (run/'mesh.inp').read_text().splitlines():
        if line.startswith('**'):
            continue
        if line.startswith('*'):
            inside = line.upper().strip() == '*NODE'
        elif inside and line.strip():
            row = line.split(',')
            model.nodes[int(row[0])] = list(map(float, row[1:4]))
    ids, nodes, grad, weights, checks = spatial(model)
    select = np.array(ids)-1
    previous_s = np.zeros((len(ids), 8, 3, 3))
    previous_e = previous_s.copy()
    work = 0.
    history = []
    for time in times:
        rec = min(records, key=lambda r: abs(r['time']-time))
        assert abs(rec['time']-time) < 1e-10
        u = field(raw, rec, 'u', 4)
        deformation = np.eye(3)+np.einsum('eni,epnj->epij', u[nodes-1], grad)
        inverse = np.linalg.inv(deformation)
        stress = np.linalg.det(deformation)[..., None, None]*(inverse@tensors(
            field(raw, rec, 'stress', 6)[select])@np.swapaxes(inverse, -1, -2))
        strain = tensors(field(raw, rec, 'strain', 6)[select])
        work += float((.5*np.einsum('epij,epij->ep', stress+previous_s,
                                  strain-previous_e)*weights).sum())
        native = evidence['history'][records.index(rec)]['native_energy_n_mm']
        history.append(dict(time=rec['time'], shared_partition_work_n_mm=work,
                            native_energy_n_mm=native,
                            native_minus_shared_partition_n_mm=native-work))
        previous_s, previous_e = stress, strain
    assert all(sha256(run/n) == h for n, h in evidence['input_output_hashes'].items())
    return dict(gate=str(gate.relative_to(F)), gate_sha256=sha256(gate),
                run=str(run.relative_to(F)), spatial_checks=checks, history=history)


def audit(standard=STANDARD, quarter=QUARTER):
    paired = compare(standard, quarter)
    times = [r['time_standard'] for r in paired['history']]
    a, b = [partition_work(g, times) for g in (standard, quarter)]
    history = []
    for x, y in zip(a['history'], b['history']):
        response = y['shared_partition_work_n_mm']-x['shared_partition_work_n_mm']
        partition = y['native_minus_shared_partition_n_mm']-x['native_minus_shared_partition_n_mm']
        native = y['native_energy_n_mm']-x['native_energy_n_mm']
        history.append(dict(time=x['time'], pressure_mpa=24*x['time'],
                            shared_partition_response_difference_n_mm=response,
                            partition_difference_n_mm=partition,
                            native_energy_difference_n_mm=native,
                            algebraic_closure_error_n_mm=native-response-partition))
    error = max(abs(r['algebraic_closure_error_n_mm']) for r in history)
    assert error < 1e-5
    # Same trajectory and all accepted states must reproduce the native rule.
    if standard == quarter:
        assert max(abs(r['native_minus_shared_partition_n_mm'])
                   for r in a['history']) < 1e-5
        assert all(r['native_energy_difference_n_mm'] == 0 for r in history)
    return dict(provenance_and_algebra_checks_passed=True,
                physical_or_convergence_acceptance=False,
                shared_accepted_times=len(times), maximum_closure_error_n_mm=error,
                trajectories=[a, b], history=history, script_sha256=sha256(__file__),
                sources=['resultsmech.f:1064-1074 endpoint-average energy rule',
                         'audit_plastic_pipe_replay.py verified actual state/energy identities'],
                limitations='Shared partition discards unmatched observed states. Response and partition contributions are descriptive and partition dependent; neither is physical dissipation or an integration-error bound. No interpolated plastic states, convergence tolerance, cap/contact/MPC/Abaqus or burst acceptance. Initial unloaded zero-stress state follows the gated deck without initial conditions.')


if __name__ == '__main__':
    out = Path(sys.argv[1])
    assert not out.exists()
    result = audit()
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(shared_times=result['shared_accepted_times'],
                          final=result['history'][-1],
                          closure=result['maximum_closure_error_n_mm']), indent=2))

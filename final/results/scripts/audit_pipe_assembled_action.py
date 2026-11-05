"""Reconstruct a retained pipe directional derivative without changing native runs.

Integrate every element touched by the probe, including geometric stiffness and
the exact unsymmetrized constitutive derivative; integrate follower load action
independently. Compare both to the retained native two-size central differences.
"""
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_plastic_return_map import F, sha256, tensors, update
from audit_material_linearization import derivative, vector
from audit_internal_pipe_fields import shape
from audit_quadratic_pressure_derivative import assemble as pressure_matrix
from audit_end_surface_traction import face_nodes
from verify_heterogeneous_coupon import points
from verify_pipe_direction import action


def assemble_actions(base, direction, curve, young, nu, deck):
    assert base['nmpc'] == 0
    xyz = np.array(base['coordinates']).reshape(-1, 3)
    u = np.array(base['displacement_after']).reshape(-1, 4)[:, 1:]
    mapping = np.array(base['mapping']).reshape(-1, 4)[:, 1:]
    du = np.zeros_like(u)
    du[mapping > 0] = direction[mapping[mapping > 0] - 1]
    connect = np.array(base['connectivity'])
    nodes = np.array([connect[k:k+20] for k in base['ipkon']]) - 1
    affected = np.flatnonzero(np.any(du[nodes] != 0, axis=(1, 2)))
    prior = np.array(base['previous_state']).reshape(base['ne'], 8, base['nstate'])
    native_stress = tensors(np.array(base['stress']).reshape(base['ne'], 8, 6))
    # No thermal, creep, residual stress or kinematic hardening in this probe.
    assert not any(key in deck for key in ('*TEMPERATURE', '*CREEP', '*INITIAL CONDITIONS', 'HARDENING='))
    assert np.max(abs(prior[affected, :, 7:])) < 1e-14
    mu = young/(2*(1+nu)); bulk = young/(3*(1-2*nu))
    exact = np.zeros_like(u); surrogate = np.zeros_like(u)
    stress_error = 0.; active_count = 0
    shear = np.array([1, 1, 1, 2, 2, 2])
    derivatives = [np.array([shape(q.astype(complex)+1e-30j*np.eye(3)[j]).imag/1e-30
                             for j in range(3)]).T for q in points()]
    for e in affected:
        en = nodes[e]
        for ip, natural in enumerate(derivatives):
            jac = xyz[en].T @ natural
            weight = np.linalg.det(jac)
            assert weight > 0
            grad = natural @ np.linalg.inv(jac)
            deform = np.eye(3) + u[en].T @ grad
            df = du[en].T @ grad
            c = deform.T @ deform
            dc = df.T @ deform + deform.T @ df
            old = prior[e, ip]
            metric = np.eye(3)-2*tensors(old[None, 1:7])[0]
            stress, _, _, active = update(c, metric, np.array(old[0]), curve, mu, bulk)
            stress_error = max(stress_error, float(abs(stress-native_stress[e, ip]).max()))
            active_count += int(active)
            ds = derivative(c, metric, old[0], curve, mu, bulk, dc)
            # Source-backed hypothesis: native packed tangent is symmetric,
            # and holds J fixed in the yield stress's determinant factor.
            frozen = np.array([vector(derivative(c, metric, old[0], curve, mu, bulk,
                                2*tensors((np.eye(6)[col]/shear[col])[None])[0], True))
                               for col in range(6)]).T
            ds_native = tensors((((frozen+frozen.T)/2) @ (vector(dc/2)*shear))[None])[0]
            geometric = df @ stress
            np.add.at(exact, en, weight*grad @ (geometric+deform @ ds).T)
            np.add.at(surrogate, en, weight*grad @ (geometric+deform @ ds_native).T)
    external = np.zeros_like(u)
    model = SimpleNamespace(elements={i+1: ('C3D20R', list(n+1)) for i, n in enumerate(nodes)})
    blocks = deck.split('*DLOAD\n')
    assert len(blocks) == 2
    faces = 0
    for line in blocks[1].split('*')[0].strip().splitlines():
        e, face, pressure = line.split(',')
        en = np.array(face_nodes(model, int(e), int(face.strip()[1:]))) - 1
        if not np.any(du[en]):
            continue
        _, matrix = pressure_matrix((xyz+u)[en], 2)
        # Same native positive/inward DLOAD convention and linear step amplitude.
        matrix *= -float(pressure)*base['time']/2.3
        np.add.at(external, en, (matrix @ du[en].ravel()).reshape(-1, 3))
        faces += 1
    def project(nodal):
        result = np.zeros(base['neq'])
        np.add.at(result, mapping[mapping > 0]-1, nodal[mapping > 0])
        return result
    return project(exact), project(surrogate), project(external), dict(
        affected_elements=(affected+1).tolist(), integration_points=len(affected)*8,
        active_plastic_points=active_count, loaded_faces_touched=faces,
        maximum_stress_replay_error_mpa=stress_error)


def main(tag, parent_report='results/output/pipe_direction_sampled_early_plastic_20261018T034621Z.json'):
    gate = F/parent_report
    evidence = json.loads(gate.read_text())
    parent_path = F/evidence['parent_gate']
    assert sha256(parent_path) == evidence['parent_sha256']
    parent = json.loads(parent_path.read_text())
    assert parent['checks_passed']
    run = F/parent['run']
    for name, digest in parent['hashes'].items():
        assert sha256(run/name) == digest, name
    for name, digest in evidence['provenance'].items():
        assert sha256(F/name) == digest, name
    base = json.loads((run/'assembly.json').read_text())
    direction_path = F/evidence['cases'][0]['run']/'direction.json'
    direction = np.array(json.loads(direction_path.read_text()))
    case_path = F/parent['parent_run']/'case.json'
    material = json.loads(case_path.read_text())['material']
    deck = (run/'job.inp').read_text()
    curve = np.array([list(map(float, line.split(','))) for line in
                      deck.split('*PLASTIC\n')[1].split('*')[0].strip().splitlines()])
    exact, surrogate, external, coverage = assemble_actions(
        base, direction, curve, material['young'], material['poisson'], deck)
    native = action(base['global_matrix'], direction)
    del base
    gc.collect()
    states = {}
    for case in evidence['cases']:
        probe = F/case['run']
        for name, digest in case['hashes'].items():
            assert sha256(probe/name) == digest, name
        data = json.loads((probe/'assembly.json').read_text())
        states[case['h']] = (np.array(data['internal']), np.array(data['residual']['fext']))
        del data
        gc.collect()
    comparisons = []
    for h in sorted((value for value in states if value > 0), reverse=True):
        internal_fd = (states[h][0]-states[-h][0])/(2*h)
        external_fd = (states[h][1]-states[-h][1])/(2*h)
        net = internal_fd-external_fd
        norm = np.linalg.norm(net)
        comparisons.append(dict(h=h,
            original_net_relative=float(np.linalg.norm(native-net)/norm),
            exact_internal_relative=float(np.linalg.norm(exact-internal_fd)/np.linalg.norm(internal_fd)),
            reconstructed_native_relative=float(np.linalg.norm(surrogate-native)/np.linalg.norm(native)),
            corrected_net_relative=float(np.linalg.norm(native+(exact-surrogate)-external-net)/norm),
            independent_net_relative=float(np.linalg.norm(exact-external-net)/norm),
            external_relative=float(np.linalg.norm(external-external_fd)/np.linalg.norm(external_fd)),
            external_error_over_net=float(np.linalg.norm(external-external_fd)/norm)))
    limit = 2e-7
    passed = coverage['maximum_stress_replay_error_mpa'] < 1e-4 and all(
        max(r[k] for k in ('exact_internal_relative', 'reconstructed_native_relative',
                          'corrected_net_relative', 'independent_net_relative')) < limit
        for r in comparisons)
    sources = [Path(__file__), gate, parent_path, case_path, direction_path]
    sources += [Path(__file__).with_name(n) for n in ('audit_material_linearization.py',
                'audit_plastic_return_map.py', 'audit_internal_pipe_fields.py',
                'audit_quadratic_pressure_derivative.py', 'audit_end_surface_traction.py',
                'verify_heterogeneous_coupon.py', 'verify_pipe_direction.py')]
    sources += [F/'sym/vendor/CalculiX/ccx_2.23/src'/n for n in ('incplas.f', 'e_c3d.f')]
    report = dict(checks_passed=passed, relative_limit=limit, coverage=coverage,
                  comparisons=comparisons, provenance={str(p.relative_to(F)):sha256(p) for p in sources},
                  parent_hashes=parent['hashes'], probe_cases=evidence['cases'],
                  limitations='One retained early-plastic state/direction with no contact or MPC. Full action over all touched elements; source-derived native tangent surrogate independently checked against global native action. Unsymmetrized correction is diagnostic only, not installed. External FD has small-step rounding uncertainty; its separate relative error is reported, not claimed passed. No convergence causation, late-pressure stability, physical burst, energy or Abaqus validation.')
    output = F/'results/output'/('pipe_assembled_action_'+tag+'.json')
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('provenance','parent_hashes','probe_cases')}, indent=2))
    assert passed, 'Retain failed report; no acceptance gate relaxation.'


if __name__ == '__main__':
    main(*sys.argv[1:])

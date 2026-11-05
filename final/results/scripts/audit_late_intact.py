"""Retained-data correction and local tangent comparison; no native replay."""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from audit_plastic_return_map import F, sha256
from audit_pipe_assembled_action import assemble_actions
from verify_pipe_direction import action


def main():
    gate = F/'results/output/late_intact_capture_20261019.json'
    capture = json.loads(gate.read_text())
    assert capture['checks_passed']
    for name, digest in capture['sources'].items():
        assert sha256(F/name) == digest, name
    run = F/capture['run']
    base = json.loads((run/'assembly.json').read_text())
    g = base['global_matrix']; n = g['neq']
    rows = np.array(g['irow'])-1
    cols = np.repeat(np.arange(n), np.diff(g['jq']))
    assert len(rows) == len(g['au']) and rows.min() >= 0 and rows.max() < n
    off = coo_matrix((g['au'], (rows, cols)), shape=(n,n)).tocsc()
    matrix = off+off.T+diags(g['ad'])
    rhs = np.array(g['rhs'])
    correction = spsolve(matrix, rhs)
    assert np.isfinite(correction).all()
    residual = float(np.linalg.norm(action(g, correction)-rhs)/np.linalg.norm(rhs))
    assert residual < 1e-7
    mapping = np.array(base['mapping']).reshape(-1,4)[:,1:]
    nodal = np.zeros(mapping.shape)
    nodal[mapping>0] = correction[mapping[mapping>0]-1]
    norms = np.linalg.norm(nodal,axis=1)
    selected = int(np.argmax(norms))
    # Localize the actual correction at its largest node. Include all touched
    # elements/faces, never compare a partial assembly to a global action.
    direction = np.zeros(n)
    active = mapping[selected]>0
    direction[mapping[selected,active]-1] = nodal[selected,active]
    direction /= np.linalg.norm(direction)
    deck = (run/'job.inp').read_text()
    curve = np.array([list(map(float,l.split(','))) for l in
                      deck.split('*PLASTIC\n')[1].split('*')[0].strip().splitlines()])
    parent = F/'sym/runs/campaign/orientation_intact_00_adaptive/case.json'
    material = json.loads(parent.read_text())['material']
    exact, surrogate, external, coverage = assemble_actions(
        base,direction,curve,material['young'],material['poisson'],deck)
    # The inherited early-plastic helper uses a 2.3-second step. This intact
    # deck has one linear static step; obtain its duration rather than reuse it.
    duration=float(deck.split('*STATIC, SOLVER=PARDISO\n')[1].splitlines()[0].split(',')[1])
    external *= 2.3/duration
    native = action(g,direction)
    denom = np.linalg.norm(native)
    reconstruction = float(np.linalg.norm(surrogate-native)/denom)
    passed = coverage['maximum_stress_replay_error_mpa'] < 1e-4 and reconstruction < 2e-7
    report = dict(checks_passed=passed,capture_report=str(gate.relative_to(F)),
        time=base['time'],cycle=base['cycle'],neq=n,
        sparse_solve_relative_residual=residual,
        correction_norm_mm=float(np.linalg.norm(correction)),
        maximum_nodal_correction_mm=float(norms[selected]),selected_node=selected+1,
        selected_coordinates_mm=np.array(base['coordinates']).reshape(-1,3)[selected].tolist(),
        correction_rayleigh_n_per_mm=float(correction@rhs/(correction@correction)),
        diagonal_rayleigh_n_per_mm=float(np.dot(correction*np.array(g['ad']),correction)/(correction@correction)),
        local_comparison=dict(coverage=coverage,native_action_norm= float(denom),
            reconstructed_native_relative=reconstruction,
            constitutive_correction_over_native=float(np.linalg.norm(exact-surrogate)/denom),
            follower_action_over_native=float(np.linalg.norm(external)/denom),
            corrected_net_change_over_native=float(np.linalg.norm(exact-surrogate-external)/denom)),
        sources={str(p.relative_to(F)):sha256(p) for p in
            (Path(__file__),gate,parent,Path(__file__).with_name('audit_pipe_assembled_action.py'))},
        limitations='One first iteration at time .7875, not the terminal failed iteration. Sparse solve checks algebra only. Rayleigh quotient is directional, not a minimum eigenvalue or stability test. Localized largest-node correction is not the full Newton correction. Source-derived component comparison has no independent late-state finite difference. No production correction, physical burst or Abaqus acceptance.',
        next='Resolve local stress/native-tangent reconstruction failures before full-correction or native replay. See late_local_state_20261019.json; no convergence causation established.')
    output=F/'results/output/late_intact_components_duration_fixed_20261019.json'
    with output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report,indent=2))
    assert passed, 'Retained failed report; no acceptance gate relaxation.'


if __name__=='__main__':main()

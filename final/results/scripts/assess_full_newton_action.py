"""Full retained Newton-ray assembly; no new native solve or pressure acceptance."""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from audit_plastic_return_map import F, sha256
from audit_pipe_assembled_action import assemble_actions
from verify_pipe_direction import action


def main():
    capture_path=F/'results/output/late_intact_capture_20261019.json'
    capture=json.loads(capture_path.read_text()); assert capture['checks_passed']
    identity_path=F/'results/output/late_native_input_audit_20261019.json'
    identity=json.loads(identity_path.read_text()); assert identity['checks_passed']
    run=F/capture['run']; p=run/'assembly.json'
    base=json.loads(p.read_text()); g=base['global_matrix']; n=g['neq']
    rows=np.array(g['irow'])-1; cols=np.repeat(np.arange(n),np.diff(g['jq']))
    off=coo_matrix((g['au'],(rows,cols)),shape=(n,n)).tocsc()
    matrix=off+off.T+diags(g['ad']); rhs=np.array(g['rhs'])
    correction=spsolve(matrix,rhs); assert np.isfinite(correction).all()
    algebra=float(np.linalg.norm(action(g,correction)-rhs)/np.linalg.norm(rhs))
    assert algebra<1e-7
    direction=correction/np.linalg.norm(correction)
    native_inputs=F/'sym/runs/investigation/late_local_inputs_20261019/assembly.json'
    inputs=json.loads(native_inputs.read_text())
    assert inputs['time']==base['time'] and inputs['cycle']==base['cycle']
    table=np.array(inputs['ips'][0]['plconloc'][:400])
    assert all(np.array_equal(table,np.array(r['plconloc'][:400])) for r in inputs['ips'])
    curve=table.reshape(-1,2)[:,::-1]; del inputs
    deck_path=run/'job.inp'; deck=deck_path.read_text()
    print('Starting all-touched full Newton-ray integration',flush=True)
    exact,surrogate,external,coverage=assemble_actions(base,direction,curve,208000,.3,deck)
    duration=float(deck.split('*STATIC, SOLVER=PARDISO\n')[1].splitlines()[0].split(',')[1])
    external/=duration
    native=action(g,direction); scale=np.linalg.norm(native)
    reconstruction=float(np.linalg.norm(surrogate-native)/scale)
    passed=coverage['maximum_stress_replay_error_mpa']<1e-4 and reconstruction<2e-7
    report=dict(checks_passed=passed,relative_limit=2e-7,stress_limit_mpa=1e-4,
        sparse_solve_relative_residual=algebra,time=base['time'],cycle=base['cycle'],
        direction='Normalized full retained symmetric Newton correction, no localization',
        coverage=coverage,native_reconstruction_relative=reconstruction,
        material_correction_over_native=float(np.linalg.norm(exact-surrogate)/scale),
        follower_action_over_native=float(np.linalg.norm(external)/scale),
        full_net_change_over_native=float(np.linalg.norm(exact-surrogate-external)/scale),
        sources={str(q.relative_to(F)):sha256(q) for q in (p,deck_path,capture_path,identity_path,native_inputs,Path(__file__),Path(__file__).with_name('audit_pipe_assembled_action.py'))},
        limitations='Retained analytic assembly only. Native input table captured at16 selected IPs and uniform material assumed for this unchanged intact deck. First iteration time.7875, not final failed iteration. No independent full-ray native FD or nonsymmetric sparse solve, no convergence causation or stability/burst acceptance.',
        next='If reconstruction fails, localize full-coverage replay mismatch before native implementation. If it passes, implement isolated additive material correction with independent native scatter/FD checks before any pressure run; preserve follower-load uncertainty separately.')
    out=F/'results/output/full_newton_action_20261019.json'
    with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2),flush=True)
    assert passed, 'Full-direction reconstruction failed; report preserved.'

if __name__=='__main__':main()

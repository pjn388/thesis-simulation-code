"""Matched revised-state full-ray reconstruction; no pressure authorization."""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from audit_plastic_return_map import F, sha256
from audit_pipe_assembled_action import assemble_actions


def main():
    gate=F/'results/output/volume_material_capture_v5_20261019.json'
    accepted=json.loads(gate.read_text());assert accepted['checks_passed']
    run=F/accepted['run']; frame=run/'assembly.json'
    assert sha256(frame)==accepted['assembly_sha256']
    for name,h in accepted['sources'].items():assert sha256(F/name)==h,name
    base=json.loads(frame.read_text());g=base['global_matrix'];n=g['neq']
    assert g['inputformat']==1 and g['symmetryflag']==2
    rows=np.array(g['irow'])-1;cols=np.repeat(np.arange(n),np.diff(g['jq']))
    lower=coo_matrix((g['au'],(rows,cols)),shape=(n,n)).tocsc()
    upper=coo_matrix((g['au_upper'],(cols,rows)),shape=(n,n)).tocsc()
    matrix=lower+upper+diags(g['ad']);rhs=np.array(g['rhs'])
    correction=spsolve(matrix,rhs);assert np.isfinite(correction).all()
    algebra=float(np.linalg.norm(matrix@correction-rhs)/np.linalg.norm(rhs))
    assert algebra<2e-7
    direction=correction/np.linalg.norm(correction)
    # Uniform, temperature-independent table is transferable; history/geometry
    # are exclusively the revised capture. Require identical frozen input bytes.
    table_frame=F/'sym/runs/investigation/late_local_inputs_20261019/assembly.json'
    table_run=table_frame.parent
    for name in ('job.inp','mesh.inp'):assert sha256(run/name)==sha256(table_run/name),name
    inputs=json.loads(table_frame.read_text());table=np.array(inputs['ips'][0]['plconloc'][:400])
    assert all(np.array_equal(table,np.array(p['plconloc'][:400])) for p in inputs['ips'])
    curve=table.reshape(-1,2)[:,::-1];del inputs
    deck=(run/'job.inp').read_text()
    print('Integrating revised-state full Newton ray',flush=True)
    exact,surrogate,external,coverage=assemble_actions(base,direction,curve,208000,.3,deck)
    duration=float(deck.split('*STATIC, SOLVER=PARDISO\n')[1].splitlines()[0].split(',')[1]);external*=2.3/duration
    native=matrix@direction
    reconstruction=float(np.linalg.norm(exact-native)/np.linalg.norm(native))
    passed=reconstruction<2e-7 and coverage['maximum_stress_replay_error_mpa']<1e-4
    (run/'full_direction.json').write_text(json.dumps(direction.tolist())+'\n')
    report=dict(checks_passed=passed,relative_limit=2e-7,stress_limit_mpa=1e-4,
        time=base['time'],cycle=base['cycle'],coverage=coverage,
        sparse_solve_relative_residual=algebra,native_full_material_relative=reconstruction,
        follower_action_over_internal=float(np.linalg.norm(external)/np.linalg.norm(native)),
        sources={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),gate,frame,table_frame,run/'job.inp',run/'mesh.inp',run/'full_direction.json',Path(__file__).with_name('audit_pipe_assembled_action.py'))},
        limitations='Separate upper/lower live matrix compared to exact internal action at matching revised geometry/history. Native uniform hardening table transferred only after frozen input identity. Native follower derivative, two-size full-ray FD and residual/state rollback remain unverified. No pressure continuation or physical acceptance.',
        next='If passed, four serial revised-solver fixed-history native probes with this full direction, two amplitudes, explicit residual/state rollback. If failed, localize exact reconstruction discrepancy; no pressure successor.')
    out=F/'results/output/volume_material_full_ray_20261019.json'
    with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('coverage','sources')},indent=2),flush=True)
    assert passed,'Preserve rejected reconstruction; unchanged gates.'

if __name__=='__main__':main()

"""Independently integrate actual retained rounded secant directions."""
import json
import numpy as np
from scipy.sparse import coo_matrix, diags
from audit_plastic_return_map import F, sha256
from audit_pipe_assembled_action import assemble_actions


def main():
    gate=F/'results/output/volume_material_full_ray_rounded_direction_20261019.json'
    accepted=json.loads(gate.read_text());assert accepted['checks_passed']
    for name,digest in accepted['sources'].items():assert sha256(F/name)==digest,name
    root=F/'sym/runs/investigation/volume_material_full_ray_fd_20261019'
    basepath=F/'sym/runs/investigation/volume_material_capture_v5_20261019/assembly.json'
    base=json.loads(basepath.read_text());g=base['global_matrix'];n=g['neq']
    rows=np.array(g['irow'])-1;cols=np.repeat(np.arange(n),np.diff(g['jq']))
    matrix=coo_matrix((g['au'],(rows,cols)),shape=(n,n))+coo_matrix((g['au_upper'],(cols,rows)),shape=(n,n))+diags(g['ad'])
    tablepath=F/'sym/runs/investigation/late_local_inputs_20261019/assembly.json'
    inputs=json.loads(tablepath.read_text());table=np.array(inputs['ips'][0]['plconloc'][:400])
    assert all(np.array_equal(table,np.array(p['plconloc'][:400])) for p in inputs['ips'])
    for name in ('job.inp','mesh.inp'):assert sha256(basepath.parent/name)==sha256(tablepath.parent/name)
    curve=table.reshape(-1,2)[:,::-1];del inputs
    mapping=np.array(base['mapping']);active=mapping>0;comparisons=[]
    for h in (1e-6,5e-7):
        frames=[json.loads((root/f'h{sign*h}'/'assembly.json').read_text()) for sign in (1,-1)]
        nodal=(np.array(frames[0]['displacement_after'])-frames[1]['displacement_after'])/(2*h)
        actual=np.zeros(n);actual[mapping[active]-1]=nodal[active]
        print(f'Integrating rounded direction h={h}',flush=True)
        exact,_,_,coverage=assemble_actions(base,actual,curve,208000,.3,(basepath.parent/'job.inp').read_text())
        sparse=matrix@actual;fd=(np.array(frames[0]['internal'])-frames[1]['internal'])/(2*h)
        error=float(np.linalg.norm(exact-sparse)/np.linalg.norm(sparse))
        comparisons.append(dict(h=h,assembly_sparse_relative=error,native_fd_relative=float(np.linalg.norm(fd-exact)/np.linalg.norm(fd)),coverage=coverage,assembly_passed=error<2e-7 and coverage['maximum_stress_replay_error_mpa']<1e-4))
    report=dict(checks_passed=all(c['assembly_passed'] for c in comparisons),comparisons=comparisons,relative_limit=2e-7,sources={str(p.relative_to(F)):sha256(p) for p in (gate,basepath,tablepath,__import__('pathlib').Path(__file__))},limitations='Retained-data diagnostic only. No new native probes, rollback verification, convergence improvement or physical acceptance.',next='Inspect terminal independent rounded-direction assembly; remaining native residual mismatch must be explained before pressure or rollback successor.')
    out=F/'results/output/volume_material_full_ray_rounded_assembly_20261019.json'
    with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2),flush=True)
    assert report['checks_passed']

if __name__=='__main__':main()

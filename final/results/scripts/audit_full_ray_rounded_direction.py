"""Separate retained displacement rounding from native residual FD error."""
import json
import numpy as np
from scipy.sparse import coo_matrix, diags
from audit_plastic_return_map import F, sha256


def main():
    parent=F/'results/output/volume_material_full_ray_fd_20261019.json'
    evidence=json.loads(parent.read_text());assert evidence['state']=='failed'
    root=F/'sym/runs/investigation/volume_material_full_ray_fd_20261019'
    basepath=F/'sym/runs/investigation/volume_material_capture_v5_20261019/assembly.json'
    base=json.loads(basepath.read_text());g=base['global_matrix'];n=g['neq']
    rows=np.array(g['irow'])-1;cols=np.repeat(np.arange(n),np.diff(g['jq']))
    matrix=coo_matrix((g['au'],(rows,cols)),shape=(n,n))+coo_matrix((g['au_upper'],(cols,rows)),shape=(n,n))+diags(g['ad'])
    directionpath=basepath.parent/'full_direction.json'
    direction=np.array(json.loads(directionpath.read_text()));intended=matrix@direction
    mapping=np.array(base['mapping']);active=mapping>0
    assert len(np.unique(mapping[active]))==n and active.sum()==n
    sources=[__file__,parent,basepath,directionpath];comparisons=[]
    for h in (1e-6,5e-7):
        paths=[root/f'h{sign*h}'/'assembly.json' for sign in (1,-1)]
        frames=[json.loads(p.read_text()) for p in paths];sources.extend(paths)
        for p,d in zip(paths,frames):
            case=next(c for c in evidence['cases'] if c['run']==str(p.parent.relative_to(F)))
            assert sha256(p)==case['assembly_sha256']
            for key in ('previous_state','previous_stress','mapping','coordinates','time','cycle'):
                assert d[key]==base[key],key
            assert d['displacement_imposed']==d['displacement_after']==d['residual']['displacement']
        nodal=(np.array(frames[0]['displacement_after'])-frames[1]['displacement_after'])/(2*h)
        actual=np.zeros(n);actual[mapping[active]-1]=nodal[active]
        predicted=matrix@actual
        fd=(np.array(frames[0]['internal'])-frames[1]['internal'])/(2*h)
        norm=np.linalg.norm
        comparisons.append(dict(h=h,actual_direction_relative_error=float(norm(actual-direction)/norm(direction)),
            sparse_action_rounding_relative=float(norm(predicted-intended)/norm(intended)),
            original_native_fd_relative=float(norm(fd-intended)/norm(fd)),
            rounded_direction_native_fd_relative=float(norm(fd-predicted)/norm(fd)),
            constrained_secant_max=float(abs(nodal[~active]).max()),
            unchanged_gate_passed=bool(norm(fd-predicted)/norm(fd)<2e-7)))
    report=dict(kind='retained_rounded_secant_sparse_action',checks_passed=True,global_fd_accepted=all(c['unchanged_gate_passed'] for c in comparisons),relative_limit=2e-7,comparisons=comparisons,
        sources={str(__import__('pathlib').Path(p).relative_to(F)):sha256(p) for p in sources},
        limitations='Sparse action uses previously independently reconstructed retained full matrix. Does not independently reintegrate rounded directions, verify rollback, establish convergence benefit or physical burst. Original native FD failure preserved.',
        next='If rounding is negligible against remaining FD error, investigate retained residual/stress evaluation precision using a full-domain higher-precision replay; no new native amplitude sweep or pressure successor.')
    out=F/'results/output/volume_material_full_ray_rounded_direction_20261019.json'
    with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__':main()

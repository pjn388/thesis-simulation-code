"""Compile isolated native additive scatter and compare to retained native FD.
No live solver integration, state edits, pressure solve, or production promotion.
"""
import gc, json, os, struct, subprocess
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F, sha256, tensors
from audit_material_linearization import derivative, vector
from audit_internal_pipe_fields import shape
from verify_heterogeneous_coupon import points
from verify_pipe_direction import action


def main():
    full=F/'results/output/full_newton_action_20261019.json'
    assert json.loads(full.read_text())['checks_passed']
    root=F/'sym/runs/investigation/volume_material_scatter_20261019'
    root.mkdir(exist_ok=False)
    src=F/'sym/vendor/CalculiX/ccx_2.23/src'
    c=Path(__file__).with_name('volume_material_scatter.c'); exe=root/'scatter'
    objects=[src/'add_sm_st_as.o',src/'nident.o']
    cmd=['gcc','-O2','-Wall','-Wextra',str(c),*[str(p) for p in objects],'-l:libgfortran.so.5','-o',str(exe)]
    with (root/'build.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    native=F/'sym/runs/investigation/late_local_inputs_20261019/assembly.json'
    b=json.loads(native.read_text());g=b['global_matrix'];n=g['neq'];nz=len(g['au'])
    xyz=np.array(b['coordinates']).reshape(-1,3)
    u=np.array(b['displacement_after']).reshape(-1,4)[:,1:]
    mapping=np.array(b['mapping']).reshape(-1,4)[:,1:]
    conn=np.array(b['connectivity']);nodes=np.array([conn[k:k+20] for k in b['ipkon']])-1
    old=np.array(b['previous_state']).reshape(b['ne'],8,b['nstate'])
    curve=np.array(b['ips'][0]['plconloc'][:400]).reshape(-1,2)[:,::-1]
    evidence=F/'results/output/late_native_action_20261019.json';ev=json.loads(evidence.read_text())
    probe=F/ev['cases'][0]['run'];direction=np.array(json.loads((probe/'direction.json').read_text()))
    du=np.zeros_like(u);du[mapping>0]=direction[mapping[mapping>0]-1]
    affected=np.flatnonzero(np.any(du[nodes]!=0,axis=(1,2)))
    grads=[np.array([shape(q.astype(complex)+1e-30j*np.eye(3)[j]).imag/1e-30 for j in range(3)]).T for q in points()]
    shear=np.array([1,1,1,2,2,2]);mu=208000/(2*1.3);bulk=208000/(3*.4)
    triplets=[];correction=np.zeros(n)
    for e in affected:
        en=nodes[e];k=np.zeros((60,60));eq=mapping[en].ravel()
        for ip,natural in enumerate(grads):
            jac=xyz[en].T@natural;weight=np.linalg.det(jac);assert weight>0
            grad=natural@np.linalg.inv(jac);deform=np.eye(3)+u[en].T@grad;cmetric=deform.T@deform
            metric=np.eye(3)-2*tensors(old[e,ip,1:7][None])[0]
            ds=[];frozen=[]
            for col in range(6):
                dc=2*tensors((np.eye(6)[col]/shear[col])[None])[0]
                ds.append(vector(derivative(cmetric,metric,old[e,ip,0],curve,mu,bulk,dc)))
                frozen.append(vector(derivative(cmetric,metric,old[e,ip,0],curve,mu,bulk,dc,True)))
            exact=np.array(ds).T;frozen=np.array(frozen).T;delta=exact-(frozen+frozen.T)/2
            strain=np.zeros((6,60))
            for col in range(60):
                df=np.zeros((3,3));df[col%3]=grad[col//3]
                strain[:,col]=vector((df.T@deform+deform.T@df)/2)*shear
            k+=weight*strain.T@delta@strain
        local=np.zeros(60);local[eq>0]=direction[eq[eq>0]-1]
        np.add.at(correction,eq[eq>0]-1,(k@local)[eq>0])
        for i in range(60):
            if eq[i]<=0:continue
            for j in range(60):
                if eq[j]>0 and k[i,j]!=0:triplets.append((int(eq[i]),int(eq[j]),k[i,j]))
    prefix=(np.array(g['jq'],dtype='<i4').tobytes()+np.array(g['irow'],dtype='<i4').tobytes()+np.array(g['ad'],dtype='<f8').tobytes()+np.array(g['au'],dtype='<f8').tobytes())
    entries=b''.join(struct.pack('<iid',*r) for r in triplets)
    outputs=[]
    for enabled in (0,1):
        payload=struct.pack('<iiii',n,nz,len(triplets),enabled)+prefix+entries
        result=subprocess.run([str(exe)],input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=60)
        (root/f'enabled{enabled}.bin').write_bytes(result.stdout)
        data=np.frombuffer(result.stdout,dtype='<f8');assert len(data)==n+2*nz
        outputs.append(data.copy())
    expected=np.r_[g['ad'],g['au'],g['au']]
    identity=np.array_equal(outputs[0],expected)
    rows=np.array(g['irow'])-1;cols=np.repeat(np.arange(n),np.diff(g['jq']))
    def matvec(data):
        result=data[:n]*direction
        np.add.at(result,rows,data[n:n+nz]*direction[cols])
        np.add.at(result,cols,data[n+nz:]*direction[rows]);return result
    parent_action=action(g,direction); revised=matvec(outputs[1])
    scatter_error=float(np.linalg.norm(revised-parent_action-correction)/np.linalg.norm(revised))
    states={};hashes={}
    for case in ev['cases']:
        p=F/case['run']/'assembly.json';assert sha256(p)==case['assembly_sha256']
        d=json.loads(p.read_text());states[case['h']]=np.array(d['internal']);hashes[str(p.relative_to(F))]=sha256(p)
        del d;gc.collect()
    comparisons=[]
    for h in (1e-6,5e-7):
        fd=(states[h]-states[-h])/(2*h)
        comparisons.append(dict(h=h,corrected_internal_relative=float(np.linalg.norm(revised-fd)/np.linalg.norm(fd)),parent_internal_relative=float(np.linalg.norm(parent_action-fd)/np.linalg.norm(fd))))
    passed=identity and scatter_error<2e-7 and all(r['corrected_internal_relative']<2e-7 for r in comparisons)
    report=dict(checks_passed=passed,disabled_byte_identity=identity,scatter_action_relative=scatter_error,relative_limit=2e-7,comparisons=comparisons,affected_elements=(affected+1).tolist(),entries=len(triplets),run=str(root.relative_to(F)),build_command=cmd,sources={str(p.relative_to(F)):sha256(p) for p in [Path(__file__),c,full,native,evidence,probe/'direction.json',*objects,src/'add_sm_st_as.f',src/'nident.f',exe]},fd_sources=hashes,
       limitations='Isolated compiled native add_sm_st_as scatter prototype on two touched C3D20R elements. Additive exact-minus-native material matrix only, after lower/upper copy. No live nonlingeo/mafillsmas integration, residual/state restoration test, full-pipe sparse solve, independent full-ray FD, follower-load acceptance, pressure improvement, production or physical burst validation.',
       next='Integrate guarded opt-in additive material correction into isolated solver after symmetric-to-asymmetric copy. Require disabled parent identity, unchanged coupon outputs/state restoration, independent full-direction native sparse action/FD and solver ABI checks before pressure successor.')
    with (F/'results/output/volume_material_scatter_20261019.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sources','fd_sources','build_command')},indent=2),flush=True)
    assert passed,'Native scatter/FD prerequisite failed; report preserved.'

if __name__=='__main__':main()

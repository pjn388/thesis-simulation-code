"""Extended precision retained full-frame replay; no native run or gate changes."""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, diags
from audit_plastic_return_map import F, tensors, sha256
from audit_internal_pipe_fields import shape
from verify_heterogeneous_coupon import points

L=np.longdouble

def det(a):
    return sum(a[...,0,j]*(a[...,1,(j+1)%3]*a[...,2,(j+2)%3]-a[...,1,(j+2)%3]*a[...,2,(j+1)%3]) for j in range(3))

def inv(a):
    out=np.empty_like(a)
    for i in range(3):
        for j in range(3):
            out[...,j,i]=a[...,(i+1)%3,(j+1)%3]*a[...,(i+2)%3,(j+2)%3]-a[...,(i+1)%3,(j+2)%3]*a[...,(i+2)%3,(j+1)%3]
    return out/det(a)[...,None,None]

def flow(p,curve):
    k=np.clip(np.searchsorted(curve[:,1],p),1,len(curve)-1)
    a,b=curve[k-1],curve[k]
    value=a[...,0]+(b[...,0]-a[...,0])*(p-a[...,1])/(b[...,1]-a[...,1])
    return np.where(p<=curve[0,1],curve[0,0],np.where(p>=curve[-1,1],curve[-1,0],value))

def stress(c,metric,p,curve):
    mu=L(208000)/L('2.6');bulk=L(208000)/L('1.2')
    j=np.sqrt(det(c));ci=inv(c);j23=j**(L(2)/3)
    tr=np.einsum('...ij,...ij->...',c,metric)/3/j23
    trial=mu*(metric/j23[...,None,None]-tr[...,None,None]*ci)
    sc=trial@c;n=np.sqrt(np.maximum(L(0),np.einsum('...ij,...ji->...',sc,sc)))
    c2=np.sqrt(L(2)/3);f=n-c2*j*flow(p,curve);active=f>L('1e-10')
    lo=np.zeros_like(p);hi=np.where(active,np.maximum(f,0)/(2*mu*tr),0)
    for _ in range(80):
        cop=(lo+hi)/2;r=n-c2*j*flow(p+c2*cop,curve)-2*mu*tr*cop
        lo=np.where(active & (r>0),cop,lo);hi=np.where(active & (r<=0),cop,hi)
    cop=np.where(active,(lo+hi)/2,0)
    factor=np.divide(2*mu*tr*cop,n,out=np.zeros_like(n),where=n>0)
    return bulk/2*(j*j-1)[...,None,None]*ci+(1-factor[...,None,None])*trial

def norm(a):return np.sqrt(np.sum(a*a))

def main():
    assert np.finfo(L).eps<np.finfo(float).eps/100
    root=F/'sym/runs/investigation';bp=root/'volume_material_capture_v5_20261019/assembly.json';tp=root/'late_local_inputs_20261019/assembly.json'
    gate=F/'results/output/volume_material_full_ray_fd_20261019.json';evidence=json.loads(gate.read_text());assert evidence['state']=='failed'
    base=json.loads(bp.read_text());table=json.loads(tp.read_text());curve=np.array(table['ips'][0]['plconloc'][:400],dtype=L).reshape(-1,2)[:,::-1];del table
    xyz=np.array(base['coordinates'],dtype=L).reshape(-1,3);nodes=np.array([base['connectivity'][k:k+20] for k in base['ipkon']])-1
    natural=np.array([np.array([shape(q.astype(complex)+1e-30j*np.eye(3)[j]).imag/1e-30 for j in range(3)]).T for q in points()],dtype=L)
    jac=np.einsum('ena,pnb->epab',xyz[nodes],natural);weight=det(jac);assert weight.min()>0
    inverse=inv(jac);inverse_error=float(abs(jac@inverse-np.eye(3)).max());assert inverse_error<1e-15
    grad=np.einsum('pna,epab->epnb',natural,inverse)
    old=np.array(base['previous_state'],dtype=L).reshape(base['ne'],8,base['nstate']);metric=np.eye(3,dtype=L)-2*tensors(old[...,1:7])
    mapping=np.array(base['mapping']).reshape(-1,4)[:,1:];active=mapping>0
    dp=bp.parent/'full_direction.json';direction=np.array(json.loads(dp.read_text()),dtype=L)
    u=np.array(base['displacement_after'],dtype=L).reshape(-1,4)[:,1:];du=np.zeros_like(xyz);du[active]=direction[mapping[active]-1]
    g=base['global_matrix'];rows=np.array(g['irow'])-1;cols=np.repeat(np.arange(g['neq']),np.diff(g['jq']))
    matrix=coo_matrix((g['au'],(rows,cols)),shape=(g['neq'],g['neq']))+coo_matrix((g['au_upper'],(cols,rows)),shape=(g['neq'],g['neq']))+diags(g['ad'])
    def force(v):
        deform=np.eye(3,dtype=L)+np.einsum('ena,epnb->epab',v[nodes],grad)
        s=stress(deform.swapaxes(-1,-2)@deform,metric,old[...,0],curve)
        local=np.einsum('ep,epna,epba->enb',weight,grad,deform@s)
        nodal=np.zeros_like(xyz);np.add.at(nodal,nodes,local)
        out=np.zeros(base['neq'],dtype=L);np.add.at(out,mapping[active]-1,nodal[active]);return out,s
    frames={};sources=[Path(__file__),bp,tp,gate,dp];checks=[]
    for h in (1e-6,-1e-6,5e-7,-5e-7):
        path=root/f'volume_material_full_ray_fd_20261019/h{h}/assembly.json';sources.append(path)
        assert sha256(path)==next(c['assembly_sha256'] for c in evidence['cases'] if c['h']==h)
        d=json.loads(path.read_text());assert d['previous_state']==base['previous_state']
        v=np.array(d['displacement_after'],dtype=L).reshape(-1,4)[:,1:]
        rounded,s=force(v);ideal,_=force(u+L(str(h))*du)
        native=np.array(d['internal'],dtype=L)
        frames[h]=(native,rounded,ideal,v)
        checks.append(dict(h=h,stress_error_mpa=float(abs(s-tensors(np.array(d['stress'],dtype=L).reshape(base['ne'],8,6))).max()),force_relative=float(norm(rounded-native)/norm(native))))
        print(checks[-1],flush=True)
    comparisons=[]
    for h in (1e-6,5e-7):
        a,b=frames[h],frames[-h];scale=2*L(str(h));native=(a[0]-b[0])/scale;rounded=(a[1]-b[1])/scale;ideal=(a[2]-b[2])/scale
        secant=(a[3]-b[3])/scale;actual=np.zeros(base['neq'],dtype=L);actual[mapping[active]-1]=secant[active]
        # scipy sparse action uses the already independently checked double matrix.
        rounded_action=np.asarray(matrix@np.asarray(actual,dtype=float),dtype=L);ideal_action=np.asarray(matrix@np.asarray(direction,dtype=float),dtype=L)
        comparisons.append(dict(h=h,native_vs_extended_replay=float(norm(native-rounded)/norm(native)),rounded_replay_vs_action=float(norm(rounded-rounded_action)/norm(rounded_action)),ideal_replay_vs_action=float(norm(ideal-ideal_action)/norm(ideal_action)),native_vs_rounded_action=float(norm(native-rounded_action)/norm(native))))
    report=dict(kind='extended_precision_full_frame_force_replay',precision_bits=int(np.finfo(L).nmant),inverse_identity_max=inverse_error,rows=checks,comparisons=comparisons,relative_limit=2e-7,global_fd_accepted=all(c['native_vs_rounded_action']<2e-7 for c in comparisons),sources={str(p.relative_to(F)):sha256(p) for p in sources},limitations='Extended precision starts from retained binary64 JSON values; shape derivatives originate in binary64. Native local intermediate quantities and branch exit not captured. No rollback, pressure improvement or physical acceptance.',next='If native discrepancy remains, capture matched native local deformation, strain, old metric, table, trial and returned stress before attributing global FD failure to a branch. No native amplitude sweep or pressure successor.')
    out=F/'results/output/volume_material_full_ray_extended_20261019.json'
    with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(comparisons,indent=2),flush=True)
if __name__=='__main__':main()

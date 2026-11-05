"""Retained full-frame force replay; isolate evaluation from assembly, no native solve."""
import json
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F, tensors, update, sha256
from audit_internal_pipe_fields import shape
from verify_heterogeneous_coupon import points


def main():
    root=F/'sym/runs/investigation'
    bp=root/'volume_material_capture_v5_20261019/assembly.json'
    tp=root/'late_local_inputs_20261019/assembly.json'
    gate=F/'results/output/volume_material_full_ray_fd_20261019.json'
    evidence=json.loads(gate.read_text()); assert evidence['state']=='failed'
    base=json.loads(bp.read_text()); table=json.loads(tp.read_text())
    curve=np.array(table['ips'][0]['plconloc'][:400]).reshape(-1,2)[:,::-1]; del table
    xyz=np.array(base['coordinates']).reshape(-1,3)
    nodes=np.array([base['connectivity'][k:k+20] for k in base['ipkon']])-1
    natural=np.array([np.array([shape(q.astype(complex)+1e-30j*np.eye(3)[j]).imag/1e-30 for j in range(3)]).T for q in points()])
    jac=np.einsum('ena,pnb->epab',xyz[nodes],natural)
    weight=np.linalg.det(jac); assert weight.min()>0
    grad=np.einsum('pna,epab->epnb',natural,np.linalg.inv(jac))
    old=np.array(base['previous_state']).reshape(base['ne'],8,base['nstate'])
    metric=np.eye(3)-2*tensors(old[...,1:7]); mapping=np.array(base['mapping']).reshape(-1,4)[:,1:]
    def force(deform,stress):
        local=np.einsum('ep,epna,epba->enb',weight,grad,deform@stress)
        nodal=np.zeros_like(xyz); np.add.at(nodal,nodes,local)
        out=np.zeros(base['neq']); active=mapping>0
        np.add.at(out,mapping[active]-1,nodal[active]); return out
    rows=[]; frames={}; sources=[Path(__file__),bp,tp,gate]
    for h in (1e-6,-1e-6,5e-7,-5e-7):
        p=root/f'volume_material_full_ray_fd_20261019/h{h}/assembly.json'
        case=next(c for c in evidence['cases'] if c['h']==h)
        assert sha256(p)==case['assembly_sha256']; sources.append(p)
        d=json.loads(p.read_text()); assert d['previous_state']==base['previous_state']
        u=np.array(d['displacement_after']).reshape(-1,4)[:,1:]
        deform=np.eye(3)+np.einsum('ena,epnb->epab',u[nodes],grad)
        stress=tensors(np.array(d['stress']).reshape(base['ne'],8,6))
        replay,_,peeq,active=update(np.swapaxes(deform,-1,-2)@deform,metric,old[...,0],curve,208000/2.6,208000/1.2)
        native=np.array(d['internal']); assembled=force(deform,stress); independent=force(deform,replay)
        frames[h]=(native,assembled,independent)
        rows.append(dict(h=h,stress_error_mpa=float(abs(replay-stress).max()),peeq_error=float(abs(peeq-np.array(d['current_state']).reshape(base['ne'],8,base['nstate'])[...,0]).max()),native_stress_force_relative=float(np.linalg.norm(assembled-native)/np.linalg.norm(native)),replay_force_relative=float(np.linalg.norm(independent-native)/np.linalg.norm(native)),plastic_points=int(active.sum())))
        print(rows[-1],flush=True)
    comparisons=[]
    for h in (1e-6,5e-7):
        fd=[(a-b)/(2*h) for a,b in zip(frames[h],frames[-h])]
        comparisons.append(dict(h=h,native_vs_native_stress_assembly=float(np.linalg.norm(fd[0]-fd[1])/np.linalg.norm(fd[0])),native_vs_independent_replay=float(np.linalg.norm(fd[0]-fd[2])/np.linalg.norm(fd[0]))))
    report=dict(kind='full_frame_internal_force_replay',acceptance=False,rows=rows,comparisons=comparisons,sources={str(p.relative_to(F)):sha256(p) for p in sources},limitations='Double precision reconstructed geometry and independent return map. Small absolute frame errors can amplify under differentiation. No global FD acceptance, native local branch capture, rollback, convergence benefit or physical acceptance.',next='Use decomposition to select precision or actual native local-input investigation; no pressure successor or amplitude sweep.')
    out=F/'results/output/volume_material_full_ray_force_replay_20261019.json'
    with out.open('x') as f: json.dump(report,f,indent=2); f.write('\n')
    print(json.dumps(comparisons),flush=True)

if __name__=='__main__':main()

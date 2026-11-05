"""Predeclare one resolvable FD pair using retained full-domain curvature checks.

No native solve, changed gate, or acceptance of the existing failed probes. This
tests h=0.1/0.05 mm in the existing unit-norm direction, before a native comparison.
"""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, diags
from audit_full_ray_extended import L, det, inv, stress, norm, flow
from audit_plastic_return_map import F, tensors, sha256
from audit_internal_pipe_fields import shape
from verify_heterogeneous_coupon import points


def main():
    root=F/'sym/runs/investigation/volume_material_capture_v5_20261019'
    bp=root/'assembly.json';dp=root/'full_direction.json'
    tp=F/'sym/runs/investigation/late_local_inputs_20261019/assembly.json'
    gate=F/'results/output/volume_material_full_ray_20261019.json'
    accepted=json.loads(gate.read_text());assert accepted['checks_passed']
    for name,h in accepted['sources'].items():assert sha256(F/name)==h,name
    base=json.loads(bp.read_text());table=json.loads(tp.read_text())
    curve=np.array(table['ips'][0]['plconloc'][:400],dtype=L).reshape(-1,2)[:,::-1]
    xyz=np.array(base['coordinates'],dtype=L).reshape(-1,3)
    nodes=np.array([base['connectivity'][k:k+20] for k in base['ipkon']])-1
    natural=np.array([np.array([shape(q.astype(complex)+1e-30j*np.eye(3)[j]).imag/1e-30
                      for j in range(3)]).T for q in points()],dtype=L)
    jac=np.einsum('ena,pnb->epab',xyz[nodes],natural);weight=det(jac)
    assert weight.min()>0
    grad=np.einsum('pna,epab->epnb',natural,inv(jac))
    old=np.array(base['previous_state'],dtype=L).reshape(base['ne'],8,base['nstate'])
    metric=np.eye(3,dtype=L)-2*tensors(old[...,1:7])
    mapping=np.array(base['mapping']).reshape(-1,4)[:,1:];active=mapping>0
    direction=np.array(json.loads(dp.read_text()),dtype=L)
    u=np.array(base['displacement_after'],dtype=L).reshape(-1,4)[:,1:]
    du=np.zeros_like(u);du[active]=direction[mapping[active]-1]
    g=base['global_matrix'];rows=np.array(g['irow'])-1;cols=np.repeat(np.arange(g['neq']),np.diff(g['jq']))
    matrix=coo_matrix((g['au'],(rows,cols)),shape=(g['neq'],g['neq']))+coo_matrix((g['au_upper'],(cols,rows)),shape=(g['neq'],g['neq']))+diags(g['ad'])
    expected=np.asarray(matrix@np.asarray(direction,dtype=float),dtype=L)
    def response(h):
        v=u+L(str(h))*du
        deform=np.eye(3,dtype=L)+np.einsum('ena,epnb->epab',v[nodes],grad)
        c=deform.swapaxes(-1,-2)@deform
        s=stress(c,metric,old[...,0],curve)
        local=np.einsum('ep,epna,epba->enb',weight,grad,deform@s)
        nodal=np.zeros_like(xyz);np.add.at(nodal,nodes,local)
        out=np.zeros(base['neq'],dtype=L);np.add.at(out,mapping[active]-1,nodal[active])
        j=np.sqrt(det(c));ci=inv(c);mu=L(208000)/L('2.6');bulk=L(208000)/L('1.2')
        z=j**(L(2)/3);tr=np.einsum('...ij,...ij->...',c,metric)/3/z
        trial=mu*(metric/z[...,None,None]-tr[...,None,None]*ci);sc=trial@c
        n=np.sqrt(np.einsum('...ij,...ji->...',sc,sc));a=np.sqrt(L(2)/3)
        yielded=n-a*j*flow(old[...,0],curve)>L('1e-10')
        deviator=s-bulk/2*(j*j-1)[...,None,None]*ci
        factor=np.sum(deviator*trial,axis=(-2,-1))/np.sum(trial*trial,axis=(-2,-1))
        qp=old[...,0]+a*(1-factor)*n/(2*mu*tr)
        segment=np.searchsorted(curve[:,1],qp,side='right')
        return out,yielded,segment,float(det(deform).min())
    _,base_active,base_segment,_=response(0)
    frames={};branch=[]
    for h in (.1,-.1,.05,-.05):
        out,yielded,segment,jmin=response(h);frames[h]=out
        branch.append(dict(h=h,active_switches=int((yielded!=base_active).sum()),
                           plastic_segment_switches=int(((segment!=base_segment)&base_active).sum()),
                           minimum_determinant=jmin))
    comparisons=[];previous=None
    for h in (.1,.05):
        fd=(frames[h]-frames[-h])/(2*L(str(h)))
        comparisons.append(dict(h=h,extended_replay_relative=float(norm(fd-expected)/norm(expected)),
            step_change=None if previous is None else float(norm(fd-previous)/norm(expected))))
        previous=fd
    limit=2e-7
    # Preflight budget reserves at least half the unchanged gate for native noise.
    passed=all(r['extended_replay_relative']<limit/2 and (r['step_change'] is None or r['step_change']<limit/2) for r in comparisons)
    passed=passed and all(r['active_switches']==r['plastic_segment_switches']==0 and r['minimum_determinant']>0 for r in branch)
    internal=np.array(base['internal']);scale=np.linalg.norm(internal)/float(norm(expected))
    sources=[Path(__file__),bp,tp,dp,gate,Path(__file__).with_name('audit_full_ray_extended.py')]
    report=dict(checks_passed=passed,native_accepted=False,relative_limit=limit,preflight_budget=limit/2,
        selected_native_steps=[.1,.05],comparisons=comparisons,branch_checks=branch,
        force_norm=float(np.linalg.norm(internal)),action_norm=float(norm(expected)),
        floating_point_sensitivity={str(h):float(np.finfo(float).eps*scale/h) for h in (1e-6,5e-7,.1,.05)},
        maximum_nodal_perturbation_mm={str(h):float(abs(du).max()*h) for h in (.1,.05)},
        provenance={str(p.relative_to(F)):sha256(p) for p in sources},
        limitations='Independent retained 63-bit-mantissa replay and unchanged local branches, not native verification. eps*norm(force)/(h*norm(action)) is a conditioning estimate, not a rigorous error bound. No claim that this explains all old errors. No pressure/rollback authorization. One fixed pair selected before native execution; no native parameter sweep.',
        next='Only if preflight passes, run four fixed-history native probes at the predeclared pair with original2e-7 gates. Retain failure and stop if they fail; no iterative amplitude fitting. If they pass, residual-evaluation rollback remains a distinct prerequisite to one bounded pressure experiment.')
    output=F/'results/output/full_ray_resolution_20261019.json'
    with output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

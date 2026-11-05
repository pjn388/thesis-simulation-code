"""Independent isotropic endpoint replay; no fitted curve or native solve."""
import json
import numpy as np
from audit_internal_pipe_fields import F, tensors
from pipe_sim.runner import sha256


def update(c, metric, peeq, curve, mu, bulk):
    """incplas.f isotropic/no residual stress/no thermal/no creep branch."""
    j = np.sqrt(np.linalg.det(c))
    ci = np.linalg.inv(c)
    j23 = j**(2/3)
    trace = np.einsum('...ij,...ij->...', c, metric)/3/j23
    trial = mu*(metric/j23[...,None,None]-trace[...,None,None]*ci)
    sc = trial@c
    norm = np.sqrt(np.maximum(0.,np.einsum('...ij,...ji->...',sc,sc)))
    c2 = np.sqrt(2/3)
    flow0 = j*np.interp(peeq,curve[:,1],curve[:,0])
    ftrial = norm-c2*flow0
    active = ftrial > 1e-10
    # Solve the scalar consistency equation independently by bounded bisection.
    low = np.zeros_like(peeq)
    high = np.where(active, np.maximum(ftrial,0)/(2*mu*trace),0.)
    for _ in range(64):
        cop = (low+high)/2
        flow = j*np.interp(peeq+c2*cop,curve[:,1],curve[:,0])
        residual = ftrial-c2*(flow-flow0)-2*mu*trace*cop
        low = np.where(active & (residual>0),cop,low)
        high = np.where(active & (residual<=0),cop,high)
    cop = np.where(active,(low+high)/2,0.)
    correction = np.divide(2*mu*trace*cop,norm,out=np.zeros_like(norm),where=norm>0)
    stress = bulk/2*(j*j-1)[...,None,None]*ci + (1-correction[...,None,None])*trial
    metric_new = metric-(correction*j23/mu)[...,None,None]*trial
    return stress, metric_new, peeq+c2*cop, active


def audit():
    run = F/'sym/runs/investigation/plastic_state_coupon_20261011T095613Z'
    gate = F/'results/output/plastic_state_stress_coupon_20261011T100320Z.json'
    evidence = json.loads(gate.read_text())
    assert evidence['checks_passed']
    assert all(sha256(run/n)==v for n,v in evidence['hashes'].items())
    deck = (run/'job.inp').read_text()
    assert '*INITIAL CONDITIONS' not in deck and 'HARDENING=' not in deck
    assert '*TEMPERATURE' not in deck and '*CREEP' not in deck
    lines = deck.splitlines(); start=lines.index('*PLASTIC')+1; curve=[]
    for line in lines[start:]:
        if line.startswith('*'): break
        curve.append(list(map(float,line.split(','))))
    curve=np.array(curve)
    assert np.all(np.diff(curve[:,1])>0) and np.all(np.diff(curve[:,0])>=0)
    raw=np.memmap(run/'fields.bin',dtype='<f8',mode='r')
    records=json.loads((run/'fields.json').read_text())['records']
    xyz=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    metric=np.broadcast_to(np.eye(3),(1,8,3,3)).copy(); peeq=np.zeros((1,8))
    maxima=dict(stress_mpa=0.,metric=0.,peeq=0.)
    history=[]; plastic_frames=0
    mu=202500/(2*1.3); bulk=202500/(3*.4)
    for rec in records:
        def field(name,width):
            m=rec['fields'][name]
            return raw[m['offset']//8:m['offset']//8+m['count']].reshape((-1,width)) if name=='u' else raw[m['offset']//8:m['offset']//8+m['count']].reshape((1,8,width))
        state=field('state',rec['states']); strain=tensors(field('strain',6))
        c=np.eye(3)+2*strain
        u=field('u',4)[:,1:]; coef=np.linalg.lstsq(np.c_[np.ones(8),xyz],u,rcond=None)[0]
        deformation=np.eye(3)+coef[1:].T; inverse=np.linalg.inv(deformation)
        actual=np.linalg.det(deformation)*(inverse@tensors(field('stress',6))@inverse.T)
        stress,predicted_metric,predicted_peeq,active=update(c,metric,peeq,curve,mu,bulk)
        actual_metric=np.eye(3)-2*tensors(state[...,1:7])
        errors=dict(stress_mpa=float(abs(stress-actual).max()),metric=float(abs(predicted_metric-actual_metric).max()),peeq=float(abs(predicted_peeq-state[...,0]).max()))
        for key,value in errors.items(): maxima[key]=max(maxima[key],value)
        history.append(dict(time=rec['time'],plastic=bool(active.any()),**errors))
        plastic_frames+=int(active.any())
        # Each endpoint independently starts from the retained previous native state.
        metric,peeq=actual_metric.copy(),state[...,0].copy()
    # Fixed identity tolerances; stress permits native consistency stopping error.
    limits=dict(stress_mpa=1e-4,metric=1e-8,peeq=1e-8)
    passed=all(maxima[k]<limits[k] for k in limits)
    assert all(sha256(run/n)==v for n,v in evidence['hashes'].items())
    result=dict(checks_passed=passed,accepted_frames=len(records),plastic_frames=plastic_frames,maximum_errors=maxima,fixed_limits=limits,history=history,run=str(run.relative_to(F)),gate_sha256=sha256(gate),script_sha256=sha256(__file__),source_sha256=sha256(F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'),source_lines='68-104,156-189,264-310,503-580,754-790',limitations='One-increment replay from each previous native state; not accumulated free replay or intermediate trajectory/work convergence. Isotropic monotone hardening, zero backstress/residual stress/thermal/creep only. No pipe/cap/contact/MPC/Abaqus or experimental acceptance.')
    out=F/'results/output/plastic_return_map_coupon_20261011T100722Z.json'
    assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='history'},indent=2))
    assert passed, 'Endpoint replay gate failed; preserve report and investigate without fitting.'


if __name__=='__main__': audit()

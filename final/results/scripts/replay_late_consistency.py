"""Diagnose failed late replay from retained fields, without native execution."""
import json
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,sha256,tensors,update
from audit_internal_pipe_fields import shape
from verify_heterogeneous_coupon import points


def native_loop(norm, trace, j, epini, curve, mu):
    """Source-order isotropic Newton loop, including native bracket fallback."""
    c2=np.sqrt(2/3); stiffness=2*mu*trace
    def flow(cop):
        ep=epini+c2*cop
        i=np.searchsorted(curve[:,1],ep,side='right')-1
        if i<0:return j*curve[0,0],0.
        if i>=len(curve)-1:return j*curve[-1,0],0.
        slope=j*(curve[i+1,0]-curve[i,0])/(curve[i+1,1]-curve[i,1])
        return j*curve[i,0]+slope*(ep-curve[i,1]),slope
    f0,_=flow(0.);ftrial=norm-c2*f0
    if ftrial<=1e-10:return 0.,'elastic',[],0.
    cop=0.;history=[]
    for iteration in range(1,18):
        f,df=flow(cop)
        if abs(cop)<1e-10:f0=f
        residual=ftrial-c2*(f-f0)-stiffness*cop
        dcop=residual/(-2/3*df-stiffness)
        cop-=dcop
        history.append(dict(iteration=iteration,cop=float(cop),dcop=float(dcop)))
        if abs(dcop)<cop*1e-4 or abs(dcop)<1e-10:
            return cop,'newton',history,float(norm-c2*flow(cop)[0]-stiffness*cop)
        if iteration>15 or cop<=0:break
    cop=0.;stage=1
    for count in range(200):
        f,_=flow(cop)
        if abs(cop)<1e-10:f0=f
        fu=ftrial-c2*(f-f0)-stiffness*cop
        if stage==1:
            lo=0.;flo=fu;stage=2;cop=1e-10
        elif stage==2:
            if fu*flo<=0:
                hi=cop;stage=3;cop=(lo+hi)/2
            else:
                cop*=10
                assert cop<=100
        else:
            if fu*flo>=0:lo=cop;flo=fu
            else:hi=cop
            cop=(lo+hi)/2;dcop=(hi-lo)/2
            if abs(dcop)<cop*1e-4 or abs(dcop)<1e-10:
                return cop,'bisection',history,float(norm-c2*flow(cop)[0]-stiffness*cop)
    raise AssertionError('fallback did not terminate')


def main():
    run=F/'sym/runs/investigation/late_intact_capture_20261019'
    base=json.loads((run/'assembly.json').read_text())
    xyz=np.array(base['coordinates']).reshape(-1,3)
    u=np.array(base['displacement_after']).reshape(-1,4)[:,1:]
    prior=np.array(base['previous_state']).reshape(base['ne'],8,13)
    current=np.array(base['current_state']).reshape(base['ne'],8,13)
    native=tensors(np.array(base['stress']).reshape(base['ne'],8,6))
    connect=np.array(base['connectivity'])
    deck=(run/'job.inp').read_text()
    curve=np.array([list(map(float,l.split(','))) for l in deck.split('*PLASTIC\n')[1].split('*')[0].strip().splitlines()])
    mu=208000/2.6;bulk=208000/1.2;rows=[]
    for element in (1189,2669):
        e=element-1;start=base['ipkon'][e];nodes=connect[start:start+20]-1
        for ip,q in enumerate(points()):
            d=np.array([shape(q.astype(complex)+1e-30j*np.eye(3)[j]).imag/1e-30 for j in range(3)]).T
            grad=d@np.linalg.inv(xyz[nodes].T@d)
            deform=np.eye(3)+u[nodes].T@grad;c=deform.T@deform
            metric=np.eye(3)-2*tensors(prior[e,ip,1:7][None])[0]
            stress,_,peeq,_=update(c,metric,np.array(prior[e,ip,0]),curve,mu,bulk)
            j=np.sqrt(np.linalg.det(c));ci=np.linalg.inv(c);z=j**(2/3)
            trace=np.sum(c*metric)/(3*z);trial=mu*(metric/z-trace*ci)
            norm=np.sqrt(np.trace((trial@c)@(trial@c)))
            cop=(current[e,ip,0]-prior[e,ip,0])/np.sqrt(2/3)
            factor=2*mu*trace*cop/norm
            from_native_q=bulk/2*(j*j-1)*ci+(1-factor)*trial
            replayed,exit_kind,iterations,replay_residual=native_loop(norm,trace,j,prior[e,ip,0],curve,mu)
            replay_stress=bulk/2*(j*j-1)*ci+(1-2*mu*trace*replayed/norm)*trial
            consistency=norm-np.sqrt(2/3)*j*np.interp(current[e,ip,0],curve[:,1],curve[:,0])-2*mu*trace*cop
            rows.append(dict(element=element,ip=ip+1, native_exit=exit_kind,newton_history=iterations, replay_cop_error=float(replayed-cop),replay_stress_error_mpa=float(abs(replay_stress-native[e,ip]).max()),replay_consistency_residual_mpa=replay_residual,
                stress_replay_error_mpa=float(abs(stress-native[e,ip]).max()),
                native_peeq_stress_error_mpa=float(abs(from_native_q-native[e,ip]).max()),
                peeq_error=float(peeq-current[e,ip,0]),
                native_scalar_consistency_residual_mpa=float(consistency)))
    report=dict(checks_passed=all(r['replay_stress_error_mpa']<1e-4 for r in rows), stress_gate_mpa=1e-4, rows=rows,source_lines='incplas.f:503-615,754-790',
        interpretation='Source-order scalar loop reproduced from displacement-derived retained geometry. Native exit and stress comparison below; no global causation inference.',
        limitations='Two selected late-state elements only. No gate relaxed and no late tangent/finite-difference acceptance, stability or burst claim.',
        sources={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),run/'assembly.json',F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f',Path(__file__).with_name('audit_plastic_return_map.py'))})
    output=F/'results/output/late_consistency_replay_20261019_v2.json'
    with output.open('x') as out:json.dump(report,out,indent=2);out.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

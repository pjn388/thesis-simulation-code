"""Retained native-input replay. Never changes solver/material acceptance gates."""
import json
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F, sha256, tensors, update
from replay_late_consistency import native_loop
from audit_internal_pipe_fields import shape
from verify_heterogeneous_coupon import points


def main():
    gate=F/'results/output/late_local_inputs_20261019.json'
    capture=json.loads(gate.read_text()); assert capture['checks_passed']
    run=F/capture['run']; base=json.loads((run/'assembly.json').read_text())
    prior=np.array(base['previous_state']).reshape(-1,8,13)
    current=np.array(base['current_state']).reshape(-1,8,13)
    xyz=np.array(base['coordinates']).reshape(-1,3)
    u=np.array(base['displacement_after']).reshape(-1,4)[:,1:]
    native=np.array(base['stress']).reshape(-1,8,6)
    deck=(run/'job.inp').read_text()
    deck_curve=np.array([list(map(float,l.split(','))) for l in deck.split('*PLASTIC\n')[1].split('*')[0].strip().splitlines()])
    rows=[]
    for r in base['ips']:
        e=r['element']-1; ip=r['ip']-1
        a=np.array(r['plconloc']); n=int(a[800]); curve=a[:2*n].reshape(-1,2)[:,::-1]
        assert n==200 and np.isfinite(curve).all() and np.all(np.diff(curve[:,1])>0)
        table_error=float(abs(curve[:,0]-np.interp(curve[:,1],deck_curve[:,1],deck_curve[:,0])).max())
        assert table_error<1e-8
        nodes=np.array(base['connectivity'][base['ipkon'][e]:base['ipkon'][e]+20])-1
        q=points()[ip];d=np.array([shape(q.astype(complex)+1e-30j*np.eye(3)[j]).imag/1e-30 for j in range(3)]).T
        deform=np.eye(3)+u[nodes].T@(d@np.linalg.inv(xyz[nodes].T@d))
        c=np.eye(3)+2*tensors(np.array([r['strain']]))[0]
        metric=np.eye(3)-2*tensors(np.array([r['previous'][1:7]]))[0]
        E,nu=r['elconloc'];mu=E/(2*(1+nu));bulk=E/(3*(1-2*nu))
        assert r['kode']==-51 and r['ielas']==0 and r['dtime']>0
        assert max(abs(x) for x in r['beta']+r['previous'][7:])==0
        s,m,peeq,_=update(c,metric,np.array(r['previous'][0]),curve,mu,bulk)
        wrong,_,_,_=update(c,metric,np.array(r['previous'][0]),deck_curve,mu,bulk)
        j=np.sqrt(np.linalg.det(c));trace=np.sum(c*metric)/(3*j**(2/3));ci=np.linalg.inv(c)
        trial=mu*(metric/j**(2/3)-trace*ci); norm=np.sqrt(np.trace((trial@c)@(trial@c)))
        cop,kind,history,res=native_loop(norm,trace,j,r['previous'][0],curve,mu)
        native_cop=(r['current'][0]-r['previous'][0])/np.sqrt(2/3)
        row=dict(element=e+1,ip=ip+1,table_points=n,table_sample_error_mpa=table_error,
            strain_geometry_error=float(abs(c-deform.T@deform).max()),
            previous_state_error=float(abs(prior[e,ip]-r['previous']).max()),
            returned_state_error=float(abs(current[e,ip]-r['current']).max()),
            returned_stress_error_mpa=float(abs(native[e,ip]-r['stress']).max()),
            native_table_stress_error_mpa=float(abs(s-tensors(np.array([r['stress']]))[0]).max()),
            deck_table_stress_error_mpa=float(abs(wrong-tensors(np.array([r['stress']]))[0]).max()),
            peeq_error=float(abs(peeq-r['current'][0])),
            metric_error=float(abs(m-(np.eye(3)-2*tensors(np.array([r['current'][1:7]]))[0])).max()),
            isotropic_scalar_cop_error=float(abs(cop-native_cop)),scalar_exit_inferred=kind,
            scalar_residual_mpa=res,scalar_history=history)
        rows.append(row)
    limits=dict(strain_geometry_error=1e-10,previous_state_error=1e-12,returned_state_error=1e-12,
        returned_stress_error_mpa=1e-8,native_table_stress_error_mpa=1e-4,peeq_error=1e-8,metric_error=1e-8,isotropic_scalar_cop_error=1e-8)
    passed=len(rows)==16 and all(all(r[k]<v for k,v in limits.items()) for r in rows)
    report=dict(checks_passed=passed,time=base['time'],cycle=base['cycle'],limits=limits,
        maxima={k:max(r[k] for r in rows) for k in limits},rows=rows,
        interpretation='Native captured 200-point hardening samples interpolate the input deck. Re-interpolation of those samples differs from direct deck interpolation between samples and explains the selected late stress discrepancy. Native-input endpoint and isotropic scalar replay agree; tighter local consistency is not justified by the former mismatch.',
        limitations='Two elements/16 IPs at first iteration only. Scalar exit inferred, not observed debug local. Unused/kinematic captured slots include NaNs; no claim of full branch ABI acceptance. No late assembled tangent or independent late finite difference, global causation, production change, burst or physical acceptance.',
        next='Use verified native isotropic hardening table for one retained late assembled-action audit with corrected step duration. Require stress/native matrix gates before any new native solve; no tighter-consistency experiment.',
        sources={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),gate,run/'assembly.json',run/'job.inp',Path(__file__).with_name('replay_late_consistency.py'),Path(__file__).with_name('audit_plastic_return_map.py'))})
    out=F/'results/output/late_native_input_audit_20261019.json'
    with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','sources')},indent=2))
    assert passed

if __name__=='__main__':main()

"""Algebraic metric/reference rates on retained fixed-path native states."""
import json
import numpy as np
from audit_plastic_return_map import F, tensors, sha256
from audit_frozen_metric_work import elastic
from prescribed_metric_coupon import MU, BULK, audit

GATE = F/'results/output/prescribed_metric_coupon_20261011T180008Z.json'
OUT = F/'results/output/prescribed_metric_rates_20261011T183002Z.json'


def reference(m):
    return 1.5*MU*(np.cbrt(np.linalg.det(m))-1)


def run():
    evidence = json.loads(GATE.read_text())
    assert evidence['identity_checks_passed']
    source = F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'
    assert sha256(source) == evidence['source_sha256']
    # Independent centered differences, including off-diagonal tensor direction.
    c = np.diag([1.03,.98,1.01]); m = np.diag([.99,1.02,1.])
    d = np.array([[.3,.1,.2],[.1,-.2,.05],[.2,.05,.4]])
    h = 1e-5; j = np.sqrt(np.linalg.det(c))
    metric_rate = MU/2/j**(2/3)*np.sum(c*d)
    reference_rate = MU/2*np.cbrt(np.linalg.det(m))*np.trace(np.linalg.solve(m,d))
    errors = dict(metric_derivative=abs((elastic(c,m+h*d,MU,BULK)[0]-elastic(c,m-h*d,MU,BULK)[0])/(2*h)-metric_rate),
        reference_derivative=abs((reference(m+h*d)-reference(m-h*d))/(2*h)-reference_rate),
        deformation_derivative=abs((elastic(c+h*d,m,MU,BULK)[0]-elastic(c-h*d,m,MU,BULK)[0])/(2*h)-.5*np.sum(elastic(c,m,MU,BULK)[1]*d)))
    assert max(errors.values()) < 1e-4, errors
    lanes=[]; nodes, weights=np.polynomial.legendre.leggauss(8)
    for retained in evidence['records']:
        directory=F/retained['run']
        assert all(sha256(directory/n)==v for n,v in retained['hashes'].items())
        lines=(directory/'job.inp').read_text().splitlines(); curve=[]
        for line in lines[lines.index('*PLASTIC')+1:]:
            if line.startswith('*'): break
            curve.append(list(map(float,line.split(','))))
        replay=audit(directory,retained['divisions'],np.array(curve))
        raw=np.memmap(directory/'fields.bin',dtype='<f8',mode='r')
        records=json.loads((directory/'fields.json').read_text())['records']
        old_c=np.broadcast_to(np.eye(3),(1,8,3,3)).copy(); old_m=old_c.copy()
        initial=elastic(old_c,old_m,MU,BULK)[0]; frozen=metric_sum=reference_sum=0.
        max_reference_error=max_split_error=0.; history=[]
        for rec in records:
            def field(name,width):
                meta=rec['fields'][name]
                return raw[meta['offset']//8:meta['offset']//8+meta['count']].reshape((1,8,width))
            c=np.eye(3)+2*tensors(field('strain',6))
            m=np.eye(3)-2*tensors(field('state',rec['states'])[...,1:7])
            delta=m-old_m
            assert np.linalg.eigvalsh(m).min()>0
            # Counterfactual split: change C at old M, then M at new C.
            p0=elastic(old_c,old_m,MU,BULK)[0]
            ptrial=elastic(c,old_m,MU,BULK)[0]; p1=elastic(c,m,MU,BULK)[0]
            frozen_increment=float((ptrial-p0).mean())
            metric_increment=float((MU/2/np.sqrt(np.linalg.det(c))**(2/3)*np.einsum('...ij,...ij->...',c,delta)).mean())
            integrated_reference=0.
            for node,weight in zip(nodes,weights):
                interpolated=old_m+(node+1)/2*delta
                integrated_reference+=weight/2*float((MU/2*np.cbrt(np.linalg.det(interpolated))*np.trace(np.linalg.solve(interpolated,delta),axis1=-2,axis2=-1)).mean())
            reference_increment=float((reference(m)-reference(old_m)).mean())
            max_reference_error=max(max_reference_error,abs(integrated_reference-reference_increment))
            max_split_error=max(max_split_error,abs(float((p1-p0).mean())-frozen_increment-metric_increment))
            frozen+=frozen_increment; metric_sum+=metric_increment; reference_sum+=integrated_reference
            storage=float((p1-initial).mean()); native_work=float(field('energy',2)[...,0].mean())
            history.append(dict(time=rec['time'],native_work_n_mm=native_work,
                candidate_potential_change_n_mm=storage,frozen_trial_sum_n_mm=frozen,
                metric_term_sum_n_mm=metric_sum,reference_rate_integral_n_mm=reference_sum,
                work_minus_candidate_potential_n_mm=native_work-storage))
            old_c,old_m=c,m
        telescope_error=abs(frozen+metric_sum-history[-1]['candidate_potential_change_n_mm'])
        assert max_reference_error<1e-8 and max_split_error<1e-8 and telescope_error<1e-8
        assert abs(reference_sum-retained['history'][-1]['reference_change_n_mm'])<1e-8
        assert all(sha256(directory/n)==v for n,v in retained['hashes'].items())
        lanes.append(dict(run=retained['run'],divisions=retained['divisions'],hashes=retained['hashes'],
            replay_checks_passed=replay['replay_checks_passed'],history=history,
            maximum_reference_integral_error_n_mm=max_reference_error,
            maximum_split_error_n_mm=max_split_error,telescoping_error_n_mm=telescope_error))
    result=dict(identity_checks_passed=True,physical_acceptance=False,lanes=lanes,
        derivative_errors=errors,derivative_limit=1e-4,integral_split_limit_n_mm=1e-8,
        gate_sha256=sha256(GATE),source_sha256=sha256(source),script_sha256=sha256(__file__),
        dependencies={n:sha256(F/'results/scripts'/n) for n in ('audit_frozen_metric_work.py','prescribed_metric_coupon.py','audit_plastic_return_map.py')},
        identities='dW/dM:dM=mu/2 J^(-2/3) C:dM; dR=mu/2 det(M)^(1/3) tr(M^-1 dM), R=3mu/2(det(M)^(1/3)-1). W from frozen-metric trial; S=2dW/dC.',
        sources='incplas.f:156-189,754-790; audit_frozen_metric_work.py elastic trial potential',
        limitations='Candidate potential is algebraically stress consistent at fixed M; physical storage and hardening energy are not established. C-then-M split and linear metric interpolation are counterfactual algebra paths, not captured intra-increment plastic evolution. Metric term is not plastic dissipation; reference rate is not a correction. No production, cap/contact/MPC/Abaqus, unloading, experimental or burst acceptance.')
    assert not OUT.exists(); OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(OUT),derivative_errors=errors,endpoints=[r['history'][-1] for r in lanes])))


if __name__=='__main__': run()

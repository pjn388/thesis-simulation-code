"""Source-derived determinant identity on immutable native-verified states."""
import json
import numpy as np
from audit_plastic_return_map import F, sha256

GATE=F/'results/output/elastic_unloading_coupon_20261011T230008Z.json'
OUT=F/'results/output/metric_flow_identity_20261012T003001Z.json'


def run():
    gate=json.loads(GATE.read_text())
    assert gate['frozen_state_work_checks_passed']
    source=F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'
    assert sha256(source)==gate['source_sha256']
    reports=[]
    for r in gate['records']:
        directory=F/r['run']
        assert all(sha256(directory/n)==h for n,h in r['hashes'].items())
        old=np.eye(3); history=[]
        for h in r['history']:
            c=np.diag(1+h['path_fraction']*np.array([.02,-.01,-.01]))**2
            new=np.array(h['metric']); delta=new-old
            direction=np.trace(c@old)/3*np.linalg.inv(c)-old
            beta=float(np.sum(delta*direction)/np.sum(direction**2))
            reconstruction=float(abs(delta-beta*direction).max())
            eigenvalues=np.linalg.eigvalsh(np.diag(np.sqrt(np.diag(c)))@old@np.diag(np.sqrt(np.diag(c))))
            factor=float(eigenvalues.sum()*np.sum(1/eigenvalues)/3-3)
            first=float(np.trace(np.linalg.solve(old,delta)))
            analytic=beta*factor
            exact=float(np.linalg.slogdet(new)[1]-np.linalg.slogdet(old)[1])
            assert eigenvalues.min()>0 and factor>=-1e-12 and beta>=-1e-12
            assert reconstruction<1e-12 and abs(first-analytic)<1e-12
            assert abs(np.sum(c*delta))<1e-12
            history.append(dict(time=h['time'],total_j=float(np.sqrt(np.linalg.det(c))),
                beta=beta,nonnegative_rate_factor=factor,first_order_logdet=first,
                analytic_first_order_logdet=analytic,exact_logdet_increment=exact,
                finite_step_remainder=exact-first,maximum_update_error=reconstruction))
            old=new
        assert abs(sum(x['exact_logdet_increment'] for x in history)-np.linalg.slogdet(old)[1])<1e-12
        reports.append(dict(run=r['run'],divisions=r['divisions'],hashes=r['hashes'],history=history,
            sum_first_order=sum(x['first_order_logdet'] for x in history),
            sum_finite_step=sum(x['finite_step_remainder'] for x in history),
            final_logdet=float(np.linalg.slogdet(old)[1])))
    result=dict(identity_checks_passed=True,physical_acceptance=False,records=reports,
        gate_sha256=sha256(GATE),source_sha256=sha256(source),script_sha256=sha256(__file__),
        derivation='Zero backstress isotropic branch: deltaM=beta[(tr(CM)/3)C^-1-M], beta>=0. Thus tr(M^-1 deltaM)=beta[(tr(CM)/3)tr((CM)^-1)-3]. SPD similarity eigenvalues give sum(lambda)sum(1/lambda)>=9. The first-order determinant rate is nonnegative and generally nonzero; C:deltaM=0 is a different trace constraint. Total J cancels from the direction, but can affect beta through yield consistency.',
        acceptance='Retained hashes/replay gate; source hash; update reconstruction, analytic trace and C:deltaM errors below1e-12; SPD metrics and nonnegative beta/factor; telescoping logdet1e-12.',
        references='incplas.f:156-189,764-790, isotropic zero backstress branch only.',
        limitations='Algebra of implemented update, not proof of physical constitutive error or continuum convergence. No native-state normalization or physical energy/dissipation, cap/contact/MPC/Abaqus or burst acceptance.',
        next='Bounded homogeneous simple shear F=I+.03a e1 tensor e2 has exactly J=1. Load a0 to1, elastic unload1 to.98; nested40/80/160 frames. Distinguish total-volume coupling from nonzero metric trace mechanism without changing production.')
    assert not OUT.exists(); OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ('history','hashes')} for r in reports]))

if __name__=='__main__': run()

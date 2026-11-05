"""Evaluate source plastic-flow logdet rate; no physical volume claim."""
import json
import numpy as np
from audit_plastic_return_map import F, sha256
from prescribed_unloading_coupon import audit, MU

GATE = F/'results/output/prescribed_unloading_coupon_20261011T190005Z.json'
OUT = F/'results/output/metric_flow_20261011T210001Z.json'


def direction(m, c, j):
    a = np.sum(c*m)/3/j**(2/3)
    trial = MU*(m/j**(2/3)-a*np.linalg.inv(c))
    norm = np.sqrt(np.trace((trial@c)@(trial@c)))
    return -2*a*j**(2/3)/norm*trial, a, norm


def run():
    gate = json.loads(GATE.read_text())
    source = F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'
    assert gate['identity_checks_passed'] and sha256(source)==gate['source_sha256']
    records = []
    for retained in gate['records']:
        directory = F/retained['run']
        assert all(sha256(directory/n)==h for n,h in retained['hashes'].items())
        lines = (directory/'job.inp').read_text().splitlines(); curve=[]
        for line in lines[lines.index('*PLASTIC')+1:]:
            if line.startswith('*'): break
            curve.append(list(map(float,line.split(','))))
        replay = audit(directory, retained['divisions'], np.array(curve))
        assert replay['replay_checks_passed']
        old=np.eye(3); peeq=0.; intervals=[]
        for h in replay['history']:
            m=np.array(h['metric']); dp=h['maximum_peeq']-peeq
            if dp>1e-12:
                f=np.diag(1+h['path_fraction']*np.array([.02,-.01,-.01]))
                c=f.T@f; j=float(np.linalg.det(f)); cop=dp/np.sqrt(2/3)
                d,a,norm=direction(old,c,j)
                update_error=float(abs(m-old-cop*d).max())
                rate=float(np.trace(np.linalg.solve(old,d)))
                analytic=-2*a*MU/norm*(3-a*j**(2/3)*np.trace(np.linalg.solve(old,np.linalg.inv(c))))
                endpoint_d,_,_=direction(m,c,j)
                endpoint_rate=float(np.trace(np.linalg.solve(m,endpoint_d)))
                # Independent central difference of logdet along frozen flow direction.
                eps=1e-5
                finite_difference=(np.linalg.slogdet(old+eps*d)[1]-np.linalg.slogdet(old-eps*d)[1])/(2*eps)
                assert update_error<1e-8 and abs(rate-analytic)<1e-10
                assert abs(rate-finite_difference)<1e-8 and abs(np.sum(c*d))<1e-10
                intervals.append(dict(time=h['time'],branch='loading' if h['time']<=.5+1e-10 else 'reverse_plastic',
                    multiplier=cop,source_rate=rate,endpoint_source_rate=endpoint_rate,
                    analytic_error=float(abs(rate-analytic)),finite_difference_error=float(abs(rate-finite_difference)),
                    update_error=update_error,weighted_source_rate=rate*cop,weighted_endpoint_rate=endpoint_rate*cop))
            old=m; peeq=h['maximum_peeq']
        assert all(sha256(directory/n)==h for n,h in retained['hashes'].items())
        records.append(dict(run=retained['run'],divisions=retained['divisions'],hashes=retained['hashes'],intervals=intervals,
            sum_source_rate=sum(x['weighted_source_rate'] for x in intervals),
            sum_endpoint_rate=sum(x['weighted_endpoint_rate'] for x in intervals),
            branch_totals={b:sum(x['weighted_endpoint_rate'] for x in intervals if x['branch']==b) for b in ('loading','reverse_plastic')},
            maximum_endpoint_rate=max(abs(x['endpoint_source_rate']) for x in intervals),
            maximum_finite_difference_error=max(x['finite_difference_error'] for x in intervals)))
    result=dict(identity_checks_passed=True,physical_acceptance=False,records=records,
        source_sha256=sha256(source),gate_sha256=sha256(GATE),script_sha256=sha256(__file__),
        dependencies={n:sha256(F/'results/scripts'/n) for n in ('audit_plastic_return_map.py','prescribed_unloading_coupon.py')},
        source_lines='incplas.f:156-189,760-790; isotropic no-backstress branch',
        derivation='a=(C:M)/(3 J^(2/3)); T=mu(M/J^(2/3)-a C^-1); D=dM/dcop=-2 a J^(2/3) T/||T||C. tr(M^-1 D)=-2 a mu/||T||C [3-a J^(2/3) tr(M^-1 C^-1)]. C:D=0 does not require this rate to vanish.',
        limitations='Frozen source direction and endpoint source evaluation on retained homogeneous states. Endpoint weighting is a diagnostic, not a continuum integration or volume measurement. No determinant normalization, storage/hardening/dissipation, cap/contact/MPC/Abaqus/experimental/burst acceptance.')
    assert not OUT.exists()
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ('intervals','hashes')} for r in records]))


if __name__=='__main__': run()

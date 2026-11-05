"""Source/state audit of isotropic plastic update; no physical energy claim."""
import json
import numpy as np
from audit_plastic_return_map import F, sha256
from prescribed_unloading_coupon import audit, MU, BULK

GATE = F/'results/output/prescribed_unloading_coupon_20261011T190005Z.json'
OUT = F/'results/output/plastic_conventions_20261011T200003Z.json'


def run():
    evidence = json.loads(GATE.read_text())
    assert evidence['identity_checks_passed']
    source = F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'
    assert sha256(source) == evidence['source_sha256']
    reports = []
    for retained in evidence['records']:
        directory = F/retained['run']
        assert all(sha256(directory/n)==h for n,h in retained['hashes'].items())
        lines = (directory/'job.inp').read_text().splitlines()
        assert not any(x in '\n'.join(lines) for x in ('HARDENING=', '*CREEP', '*TEMPERATURE', '*INITIAL CONDITIONS'))
        curve = []
        for line in lines[lines.index('*PLASTIC')+1:]:
            if line.startswith('*'): break
            curve.append(list(map(float,line.split(','))))
        curve = np.array(curve)
        assert np.all(np.diff(curve[:,1])>0) and np.all(np.diff(curve[:,0])>=0)
        replay = audit(directory, retained['divisions'], curve)
        old_m = np.eye(3); old_p = 0.; intervals = []
        for h in replay['history']:
            fraction = h['path_fraction']
            f = np.diag(1+fraction*np.array([.02,-.01,-.01]))
            c = f.T@f; j = np.linalg.det(f); ci = np.linalg.inv(c)
            m = np.array(h['metric']); p = h['maximum_peeq']
            dp = p-old_p; cop = dp/np.sqrt(2/3)
            trace = np.sum(c*old_m)/3/j**(2/3)
            trial = MU*(old_m/j**(2/3)-trace*ci)
            norm = np.sqrt(np.trace((trial@c)@(trial@c)))
            flow = j*np.interp(p,curve[:,1],curve[:,0])
            correction = 0. if norm==0 else 2*MU*trace*cop/norm
            predicted = old_m-correction*j**(2/3)/MU*trial
            error = float(abs(m-predicted).max())
            trace_error = abs(float(np.sum(c*(m-old_m))))
            dev = np.array(h['second_piola_stress_mpa'])-BULK/2*(j*j-1)*ci
            final_norm = np.sqrt(max(0.,np.trace((dev@c)@(dev@c))))
            residual = final_norm-np.sqrt(2/3)*flow
            assert dp >= -1e-12 and error<1e-8 and trace_error<1e-8
            assert residual<1e-4 and (dp<=1e-12 or abs(residual)<1e-4)
            eigen = np.linalg.eigvalsh(m); assert eigen.min()>0
            intervals.append(dict(time=h['time'],branch='loading' if h['time']<=.5+1e-10 else 'unloading',
                plastic=dp>1e-12,peeq_increment=dp,consistency_multiplier=cop,
                metric_update_error=error,endpoint_trace_update_error=trace_error,
                yield_residual_mpa=residual,metric_determinant=float(np.linalg.det(m)),
                minimum_metric_eigenvalue=float(eigen.min()),jacobian=float(j),
                interpolated_cauchy_flow_mpa=float(flow/j)))
            old_m,old_p=m,p
        assert all(sha256(directory/n)==h for n,h in retained['hashes'].items())
        reports.append(dict(run=retained['run'],divisions=retained['divisions'],hashes=retained['hashes'],
            replay_checks_passed=replay['replay_checks_passed'],intervals=intervals,
            loading_plastic_intervals=sum(i['plastic'] and i['branch']=='loading' for i in intervals),
            unloading_plastic_intervals=sum(i['plastic'] and i['branch']=='unloading' for i in intervals),
            maximum_metric_update_error=max(i['metric_update_error'] for i in intervals),
            maximum_endpoint_trace_update_error=max(i['endpoint_trace_update_error'] for i in intervals),
            maximum_plastic_yield_residual_mpa=max(abs(i['yield_residual_mpa']) for i in intervals if i['plastic']),
            maximum_determinant_departure=max(abs(i['metric_determinant']-1) for i in intervals),
            final_determinant=intervals[-1]['metric_determinant']))
    result=dict(identity_checks_passed=True,physical_acceptance=False,records=reports,
        gate_sha256=sha256(GATE),source_sha256=sha256(source),script_sha256=sha256(__file__),
        dependencies={n:sha256(F/'results/scripts'/n) for n in ('audit_plastic_return_map.py','prescribed_unloading_coupon.py')},
        source_lines=dict(state_decode='incplas.f:68-80',trial='incplas.f:156-189',
            jacobian_scaled_flow='incplas.f:264-310',consistency='incplas.f:503-580',
            metric_update='incplas.f:760-790',state_encode='incplas.f:1024-1032'),
        interpretation='State1 is accumulated equivalent plastic strain. M=I-2*state2:7 is the source cpl tensor, used as an inverse plastic metric in the trial contraction. cop=deltaPEEQ/sqrt(2/3). Isotropic table gives piecewise-linear Cauchy flow stress, multiplied by current total J in consistency; outside table source holds endpoint flow constant. Additive M update preserves endpoint C:M contraction, not det(M)=1 exactly. Source inspected branch has no determinant normalization or hardening-energy state.',
        limitations='Agreement checks algebra and replay of retained homogeneous isotropic states; no independent physical hardening energy or dissipation law established. Do not interpret integral(flow dPEEQ) or native-minus-candidate potential as dissipated energy. Determinant departure is quantified numerical sensitivity, not permission to normalize native state or change material. No cap/contact/MPC/Abaqus/experimental/burst acceptance.')
    assert not OUT.exists()
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(OUT),records=[{k:v for k,v in r.items() if k not in ('hashes','intervals')} for r in reports])))


if __name__=='__main__': run()

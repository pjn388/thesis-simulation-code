"""Recover retained shear evidence without rerunning or changing failed gates."""
import json
import numpy as np
from shear_unloading_coupon import ROOT, F, audit, sha256
from audit_elastic_jacobian import elastic_candidate

OUT=F/'results/output/shear_unloading_recovery_20261012T003001Z.json'


def run():
    manifest=json.loads((ROOT/'diagnostic_revision.json').read_text())
    records=[]
    for divisions in (40,80,160):
        directory=ROOT/f'grid_{divisions}'
        before={n:sha256(directory/n) for n in ('job.inp','job.dat','job.sta','solver.log','fields.bin','fields.json')}
        lines=(directory/'job.inp').read_text().splitlines(); curve=[]
        for line in lines[lines.index('*PLASTIC')+1:]:
            if line.startswith('*'): break
            curve.append(list(map(float,line.split(','))))
        report=audit(directory,divisions,np.array(curve))
        start=report['history'][divisions//2-1]
        def potential(h):
            f=np.eye(3); f[0,1]=.03*h['path_fraction']
            return float(elastic_candidate(f.T@f,np.array(h['metric'])))
        errors=[]; elastic_errors=[]
        for h in report['history'][divisions//2:]:
            assert abs(np.array(h['metric'])-np.array(start['metric'])).max()<1e-10
            assert abs(h['maximum_peeq']-start['maximum_peeq'])<1e-12
            work=h['native_work_n_mm']-start['native_work_n_mm']
            errors.append(abs(work-(h['candidate_potential_change_n_mm']-start['candidate_potential_change_n_mm'])))
            elastic_errors.append(abs(work-(potential(h)-potential(start))))
        report.update(loaded_metric_determinant=start['metric_determinant'],
            loaded_peeq=start['maximum_peeq'],maximum_total_j_work_error_n_mm=max(errors),
            maximum_elastic_j_work_error_n_mm=max(elastic_errors))
        assert all(sha256(directory/n)==v for n,v in before.items())
        records.append(report)
    errors=[r['maximum_total_j_work_error_n_mm'] for r in records]
    trend=errors[2]<errors[1]<errors[0]
    result=dict(manifest=manifest,records=records,immutable_replay_checks_passed=True,
        original_monotonic_work_gate_passed=trend,original_diagnostic_passed=trend and errors[2]<1e-6,
        physical_acceptance=False,script_sha256=sha256(__file__),
        dependencies={n:sha256(F/'results/scripts'/n) for n in ('shear_unloading_coupon.py','audit_elastic_jacobian.py')},
        interpretation='Exact totalJ1 prescribed simple shear retains nonunit plastic metric determinant; total volume change is not necessary for this determinant departure. Original decreasing work-error gate failed at approximately1e-11Nmm and remains failed. Replay/frozen-state gates are independent of trend acceptance.',
        limitations='Artificial constrained homogeneous test; no asymptotic or physical energy/hardening/dissipation, cap/contact/MPC/Abaqus or burst acceptance. Original service failure preserved; no tolerance change, normalization or native rerun.')
    assert not OUT.exists();OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict(divisions=r['divisions'],det=r['loaded_metric_determinant'],peeq=r['loaded_peeq'],total_error=r['maximum_total_j_work_error_n_mm'],elastic_error=r['maximum_elastic_j_work_error_n_mm']) for r in records]))

if __name__=='__main__': run()

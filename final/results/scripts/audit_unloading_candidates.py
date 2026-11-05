"""Compare candidate recovery on immutable, native-verified frozen states."""
import json
import numpy as np
from audit_plastic_return_map import F, sha256
from elastic_unloading_coupon import audit
from audit_elastic_jacobian import elastic_candidate, elastic_stress

GATE = F/'results/output/elastic_unloading_coupon_20261011T230008Z.json'
OUT = F/'results/output/unloading_candidates_20261011T233004Z.json'


def run():
    gate = json.loads(GATE.read_text())
    assert gate['frozen_state_work_checks_passed'] and not gate['physical_acceptance']
    records = []
    for retained in gate['records']:
        directory = F/retained['run']
        assert all(sha256(directory/n)==h for n,h in retained['hashes'].items())
        lines = (directory/'job.inp').read_text().splitlines()
        curve = []
        for line in lines[lines.index('*PLASTIC')+1:]:
            if line.startswith('*'): break
            curve.append(list(map(float,line.split(','))))
        replay = audit(directory,retained['divisions'],np.array(curve))
        start = replay['history'][retained['divisions']//2-1]
        m0 = np.array(start['metric'])
        def c_of(h):
            f = np.diag(1+h['path_fraction']*np.array([.02,-.01,-.01]))
            return f.T@f
        w0 = elastic_candidate(c_of(start),m0)
        history = []
        for h in replay['history'][retained['divisions']//2:]:
            m = np.array(h['metric']); c = c_of(h)
            assert np.max(abs(m-m0))<1e-10
            assert abs(h['maximum_peeq']-start['maximum_peeq'])<1e-12
            native = h['native_work_n_mm']-start['native_work_n_mm']
            total = h['candidate_potential_change_n_mm']-start['candidate_potential_change_n_mm']
            candidate = float(elastic_candidate(c,m)-w0)
            history.append(dict(time=h['time'],native_recovered_work_n_mm=native,
                total_j_recovery_n_mm=total,elastic_j_recovery_n_mm=candidate,
                total_j_work_error_n_mm=abs(total-native),
                elastic_j_work_error_n_mm=abs(candidate-native),
                maximum_elastic_j_stress_error_mpa=float(np.max(abs(
                    elastic_stress(c,m)-np.array(h['second_piola_stress_mpa']))))))
        assert all(sha256(directory/n)==h for n,h in retained['hashes'].items())
        assert max(h['total_j_work_error_n_mm'] for h in history)<1e-6
        records.append(dict(run=retained['run'],divisions=retained['divisions'],
            hashes=retained['hashes'],replay_checks_passed=True,history=history,
            maximum_total_j_work_error_n_mm=max(h['total_j_work_error_n_mm'] for h in history),
            maximum_elastic_j_work_error_n_mm=max(h['elastic_j_work_error_n_mm'] for h in history),
            endpoint=history[-1]))
    result = dict(identity_checks_passed=True,physical_acceptance=False,records=records,
        gate_sha256=sha256(GATE),script_sha256=sha256(__file__),
        dependencies={n:sha256(F/'results/scripts'/n) for n in
            ('elastic_unloading_coupon.py','audit_elastic_jacobian.py','audit_plastic_return_map.py')},
        acceptance='Retained native replay limits and immutable hashes; frozen metric1e-10/PEEQ1e-12; total-J recovered work error below1e-6 Nmm. Elastic-J discrepancy is measured, not fitted or required to pass.',
        references='incplas.f:156-189,775-790; node161.html multiplicative isochoric flow; prior elastic_jacobian and bounded unloading evidence.',
        limitations='Candidate comparison on artificial homogeneous frozen-state segment only. No unique physical storage, hardening/dissipation, pipe cap/contact/MPC/Abaqus equivalence or burst acceptance. State-only reference cancels during frozen unloading and cannot repair this discrepancy.',
        next='Quantify detM departures and candidate recovery discrepancy versus loading increment refinement using retained states before any constitutive interpretation or native extension.')
    assert not OUT.exists()
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict(divisions=r['divisions'],endpoint=r['endpoint']) for r in records]))


if __name__=='__main__': run()

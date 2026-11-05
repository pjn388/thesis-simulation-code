"""Separate first-order logdet evolution from finite additive-step terms."""
import json
import numpy as np
from audit_plastic_return_map import F, sha256
from prescribed_unloading_coupon import audit

GATE = F/'results/output/prescribed_unloading_coupon_20261011T190005Z.json'
OUT = F/'results/output/metric_determinant_20261011T203007Z.json'


def run():
    evidence = json.loads(GATE.read_text())
    assert evidence['identity_checks_passed']
    source = F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'
    assert sha256(source) == evidence['source_sha256']
    records = []
    for retained in evidence['records']:
        directory = F/retained['run']
        assert all(sha256(directory/n) == h for n,h in retained['hashes'].items())
        lines = (directory/'job.inp').read_text().splitlines()
        curve = []
        for line in lines[lines.index('*PLASTIC')+1:]:
            if line.startswith('*'): break
            curve.append(list(map(float, line.split(','))))
        replay = audit(directory, retained['divisions'], np.array(curve))
        old = np.eye(3)
        intervals = []
        for state in replay['history']:
            new = np.array(state['metric'])
            delta = new-old
            # Symmetric similarity avoids nonsymmetric eigensolver roundoff.
            eig, vec = np.linalg.eigh(old)
            assert eig.min() > 0
            inverse_root = (vec/np.sqrt(eig))@vec.T
            increments = np.linalg.eigvalsh(inverse_root@delta@inverse_root)
            assert increments.min() > -1
            exact = float(np.log1p(increments).sum())
            first = float(np.trace(np.linalg.solve(old, delta)))
            second = float(-.5*np.sum(increments**2))
            direct = float(np.linalg.slogdet(new)[1]-np.linalg.slogdet(old)[1])
            assert abs(exact-direct) < 1e-12
            intervals.append(dict(time=state['time'],branch='loading' if state['time']<=.5+1e-10 else 'unloading',
                logdet_increment=direct,first_order=first,finite_step=exact-first,
                second_order=second,higher_order=exact-first-second))
            old = new
        total = float(np.linalg.slogdet(old)[1])
        assert abs(sum(i['logdet_increment'] for i in intervals)-total)<1e-12
        assert all(sha256(directory/n) == h for n,h in retained['hashes'].items())
        records.append(dict(run=retained['run'],divisions=retained['divisions'],hashes=retained['hashes'],
            replay_checks_passed=replay['replay_checks_passed'],intervals=intervals,
            final_logdet=total,sum_first_order=sum(i['first_order'] for i in intervals),
            sum_finite_step=sum(i['finite_step'] for i in intervals),
            sum_second_order=sum(i['second_order'] for i in intervals),
            sum_higher_order=sum(i['higher_order'] for i in intervals)))
    result=dict(identity_checks_passed=True,physical_acceptance=False,records=records,
        gate_sha256=sha256(GATE),source_sha256=sha256(source),script_sha256=sha256(__file__),
        dependencies={n:sha256(F/'results/scripts'/n) for n in ('audit_plastic_return_map.py','prescribed_unloading_coupon.py')},
        interpretation='Exact logdet increment decomposed into tr(M_old^-1 deltaM) and finite-step remainder; endpoint C:deltaM=0 does not imply tr(M_old^-1 deltaM)=0.',
        limitations='Retained homogeneous isotropic loading/reverse-plastic paths only. Three grids do not establish a continuum limit. No metric normalization, physical storage/hardening/dissipation, cap, contact/MPC, Abaqus or experimental acceptance.')
    assert not OUT.exists()
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))
    print(json.dumps([{k:v for k,v in r.items() if k not in ('intervals','hashes')} for r in records]))


if __name__ == '__main__': run()

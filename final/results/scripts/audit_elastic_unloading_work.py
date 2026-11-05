"""Check retained elastic unloading, without interpreting plastic energy gaps."""
import json
import numpy as np
from audit_plastic_return_map import F, sha256
from audit_frozen_metric_work import elastic
from prescribed_unloading_coupon import audit, MU, BULK

GATE = F/'results/output/prescribed_unloading_coupon_20261011T190005Z.json'
OUT = F/'results/output/elastic_unloading_work_20261011T193008Z.json'


def deformation(fraction):
    f = np.diag(1+fraction*np.array([.02, -.01, -.01]))
    return f.T@f


def run():
    evidence = json.loads(GATE.read_text())
    assert evidence['identity_checks_passed']
    source = F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'
    assert sha256(source) == evidence['source_sha256']
    nodes, weights = np.polynomial.legendre.leggauss(8)
    reports = []
    for retained in evidence['records']:
        directory = F/retained['run']
        assert all(sha256(directory/n) == h for n,h in retained['hashes'].items())
        lines = (directory/'job.inp').read_text().splitlines(); curve = []
        for line in lines[lines.index('*PLASTIC')+1:]:
            if line.startswith('*'): break
            curve.append(list(map(float,line.split(','))))
        replay = audit(directory, retained['divisions'], np.array(curve))
        intervals = []; plastic_count = 0
        history = replay['history']
        for previous, current in zip(history, history[1:]):
            # Native accepted times can differ from the half-step by roundoff.
            if current['time'] <= .5+1e-10: continue
            dm = float(abs(np.array(current['metric'])-np.array(previous['metric'])).max())
            dp = current['maximum_peeq']-previous['maximum_peeq']
            assert dp >= -1e-12
            if dp > 1e-12:
                plastic_count += 1
                continue
            assert dm < 1e-10
            c0 = deformation(previous['path_fraction']); c1 = deformation(current['path_fraction'])
            metric = np.array(previous['metric'])
            p0,s0 = elastic(c0,metric,MU,BULK); p1,s1 = elastic(c1,metric,MU,BULK)
            # Native work uses endpoint-average S:dC/2. Check independently
            # against captured endpoint stress and exact fixed-M potential.
            native = current['native_work_n_mm']-previous['native_work_n_mm']
            trapezoid = float(.25*np.sum((s0+s1)*(c1-c0)))
            captured = .25*np.sum((np.array(previous['second_piola_stress_mpa'])+
                                  np.array(current['second_piola_stress_mpa']))*(c1-c0))
            exact = float(p1-p0)
            integral = 0.
            for node,weight in zip(nodes,weights):
                c = c0+(node+1)/2*(c1-c0)
                integral += float(weight/4*np.sum(elastic(c,metric,MU,BULK)[1]*(c1-c0)))
            assert abs(native-trapezoid) < 1e-8
            assert abs(captured-trapezoid) < 1e-8
            assert abs(integral-exact) < 1e-8
            intervals.append(dict(time=current['time'],metric_change=dm,peeq_change=dp,
                native_work_increment_n_mm=native,exact_fixed_metric_increment_n_mm=exact,
                native_minus_exact_n_mm=native-exact,
                native_trapezoid_error_n_mm=abs(native-trapezoid),
                integrated_derivative_error_n_mm=abs(integral-exact)))
        expected = next(r for r in evidence['parent_loading_elastic_checks']
                        if r['divisions']==retained['divisions'])['elastic_unloading_intervals']
        assert [r['time'] for r in intervals] == [r['time'] for r in expected]
        assert intervals and plastic_count
        assert len(intervals)+plastic_count == retained['divisions']//2
        assert all(sha256(directory/n) == h for n,h in retained['hashes'].items())
        reports.append(dict(run=retained['run'],divisions=retained['divisions'],
            replay_checks_passed=replay['replay_checks_passed'],hashes=retained['hashes'],
            elastic_intervals=intervals,reverse_plastic_intervals=plastic_count,
            maximum_absolute_quadrature_gap_n_mm=max(abs(i['native_minus_exact_n_mm']) for i in intervals),
            cumulative_elastic_quadrature_gap_n_mm=sum(i['native_minus_exact_n_mm'] for i in intervals)))
    result = dict(identity_checks_passed=True,physical_acceptance=False,records=reports,
        fixed_identity_limit_n_mm=1e-8,gate_sha256=sha256(GATE),source_sha256=sha256(source),
        script_sha256=sha256(__file__),dependencies={n:sha256(F/'results/scripts'/n) for n in
        ('prescribed_unloading_coupon.py','audit_frozen_metric_work.py','audit_plastic_return_map.py')},
        sources='incplas.f fixed-metric elastic trial; resultsmech.f endpoint-average material work; retained accepted native loading/unloading states.',
        limitations='Only intervals with unchanged captured metric and PEEQ are integrated. Linear C interpolation is an algebraic conservative-path check. Native trapezoidal quadrature need not equal exact potential differences at finite increments. Plastic intervals excluded; no physical storage/hardening/dissipation, pipe/cap/contact/MPC/Abaqus, experimental or burst acceptance.')
    assert not OUT.exists()
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(OUT),records=[{k:v for k,v in r.items() if k not in ('hashes','elastic_intervals')} for r in reports])))


if __name__ == '__main__': run()

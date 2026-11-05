"""Candidate state derivative on immutable replay; not physical dissipation."""
import json
import numpy as np
from audit_plastic_return_map import F, sha256
from audit_metric_manual import potential
from audit_metric_flow import direction
from prescribed_unloading_coupon import audit, MU

GATE = F/'results/output/prescribed_unloading_coupon_20261011T190005Z.json'
OUT = F/'results/output/metric_state_power_20261011T220008Z.json'


def run():
    gate = json.loads(GATE.read_text())
    source = F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'
    assert gate['identity_checks_passed'] and sha256(source) == gate['source_sha256']
    records = []
    for retained in gate['records']:
        directory = F/retained['run']
        assert all(sha256(directory/n) == h for n, h in retained['hashes'].items())
        lines = (directory/'job.inp').read_text().splitlines()
        curve = []
        for line in lines[lines.index('*PLASTIC')+1:]:
            if line.startswith('*'): break
            curve.append(list(map(float, line.split(','))))
        replay = audit(directory, retained['divisions'], np.array(curve))
        assert replay['replay_checks_passed']
        old = np.eye(3)
        peeq = 0.
        intervals = []
        for h in replay['history']:
            m = np.array(h['metric'])
            dp = h['maximum_peeq']-peeq
            if dp > 1e-12:
                f = np.diag(1+h['path_fraction']*np.array([.02, -.01, -.01]))
                c = f.T@f
                j = float(np.linalg.det(f))
                cop = dp/np.sqrt(2/3)
                d, _, _ = direction(old, c, j)
                assert np.max(abs(m-old-cop*d)) < 1e-8
                gradient = MU/2/j**(2/3)*c
                analytic = float(np.sum(gradient*d))
                eps = 1e-5
                fd = float((potential(c, old+eps*d)-potential(c, old-eps*d))/(2*eps))
                # Independent direct finite state change at frozen deformation.
                delta = float(potential(c, m)-potential(c, old))
                prediction = float(np.sum(gradient*(m-old)))
                contraction = float(np.sum(c*d))
                assert abs(contraction) < 1e-10  # unchanged prior source-flow gate
                assert abs(fd-analytic) < 1e-4
                assert abs(delta-prediction) < 1e-8 and abs(delta) < 1e-5
                intervals.append(dict(time=h['time'], branch='loading' if h['time'] <= .5+1e-10 else 'reverse_plastic',
                    multiplier=cop, analytic_state_power_mpa=analytic, source_contraction=contraction,
                    finite_difference_state_power_mpa=fd,
                    frozen_deformation_potential_change_mpa=delta,
                    linear_state_change_prediction_mpa=prediction))
            old = m
            peeq = h['maximum_peeq']
        assert all(sha256(directory/n) == h for n, h in retained['hashes'].items())
        records.append(dict(run=retained['run'], divisions=retained['divisions'], hashes=retained['hashes'],
            replay_checks_passed=True, intervals=intervals,
            branch_counts={b:sum(x['branch']==b for x in intervals) for b in ('loading', 'reverse_plastic')},
            maximum_analytic_state_power_mpa=max(abs(x['analytic_state_power_mpa']) for x in intervals),
            maximum_finite_difference_error_mpa=max(abs(x['finite_difference_state_power_mpa']-x['analytic_state_power_mpa']) for x in intervals),
            maximum_frozen_potential_change_mpa=max(abs(x['frozen_deformation_potential_change_mpa']) for x in intervals)))
    result = dict(identity_checks_passed=True, physical_acceptance=False, records=records,
        source_sha256=sha256(source), gate_sha256=sha256(GATE), script_sha256=sha256(__file__),
        dependencies={n:sha256(F/'results/scripts'/n) for n in ('audit_metric_manual.py', 'audit_metric_flow.py', 'audit_plastic_return_map.py', 'prescribed_unloading_coupon.py')},
        derivation='At fixed C, dW/dM=mu C/(2 J^(2/3)). In exact arithmetic source direction obeys C:D=0, hence candidate W has zero state power along D and is linear in M. Actual floating-point powers and frozen-C update differences are reported, not asserted exactly zero; source contraction uses the unchanged prior 1e-10 gate.',
        references='incplas.f:156-189,760-790; local doc/ccx/node161.html:64-120; prior metric_manual and metric_flow audits',
        limitations='This candidate has no state-only reference or hardening energy. Zero candidate state power is not physical zero dissipation: an arbitrary g(M,hardening) changes that power while preserving fixed-M stress. Endpoint source direction and discrete retained states do not establish a continuum thermodynamic model. No production constitutive, pressure, cap, contact/MPC, Abaqus or burst acceptance.',
        next='Inspect source/manual energy and hardening definitions and compare an independently derived physical elastic potential before assigning work-minus-candidate to dissipation. No native rerun required for this source question.')
    assert not OUT.exists()
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ('intervals', 'hashes')} for r in records]))


if __name__ == '__main__': run()

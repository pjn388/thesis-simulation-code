"""Compare two explicitly labelled potentials; no constitutive acceptance."""
import json
import numpy as np
from audit_plastic_return_map import F, sha256
from audit_metric_manual import potential
from prescribed_unloading_coupon import audit, MU, BULK

GATE = F/'results/output/prescribed_unloading_coupon_20261011T190005Z.json'
OUT = F/'results/output/elastic_jacobian_20261011T223003Z.json'


def elastic_candidate(c, m):
    # Fe=F Fp^-1, M=Cp^-1: Je=J sqrt(det M), tr(be)=C:M.
    je = np.sqrt(np.linalg.det(c)*np.linalg.det(m))
    return MU/2*(np.sum(c*m)/je**(2/3)-3)+BULK/4*(je*je-1-2*np.log(je))


def elastic_stress(c, m):
    je = np.sqrt(np.linalg.det(c)*np.linalg.det(m))
    ci = np.linalg.inv(c)
    return MU/je**(2/3)*(m-np.sum(c*m)/3*ci)+BULK/2*(je*je-1)*ci


def run():
    gate = json.loads(GATE.read_text())
    source = F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'
    manual = F/'sym/vendor/CalculiX/ccx_2.23/doc/ccx/node161.html'
    assert gate['identity_checks_passed'] and sha256(source)==gate['source_sha256']
    assert 'plastic flow is isochoric' in manual.read_text()
    records = []
    for retained in gate['records']:
        directory = F/retained['run']
        assert all(sha256(directory/n)==h for n,h in retained['hashes'].items())
        lines = (directory/'job.inp').read_text().splitlines()
        curve = []
        for line in lines[lines.index('*PLASTIC')+1:]:
            if line.startswith('*'): break
            curve.append(list(map(float, line.split(','))))
        replay = audit(directory, retained['divisions'], np.array(curve))
        assert replay['replay_checks_passed']
        history = []
        for h in replay['history']:
            m = np.array(h['metric'])
            f = np.diag(1+h['path_fraction']*np.array([.02, -.01, -.01]))
            c = f.T@f
            candidate_s = elastic_stress(c, m)
            source_s = np.array(h['second_piola_stress_mpa'])
            errors = []
            for i in range(3):
                for k in range(i,3):
                    d = np.zeros((3,3)); d[i,k]=1; d[k,i]=1
                    eps = 1e-6
                    fd = (elastic_candidate(c+eps*d,m)-elastic_candidate(c-eps*d,m))/(2*eps)
                    errors.append(abs(fd-.5*np.sum(candidate_s*d)))
            assert max(errors)<1e-4  # prior independent gradient gate
            # Algebraic equivalence on detM=1; reference states are synthetic,
            # never written to the native metric or used to change the solver.
            unit_m = m/np.linalg.det(m)**(1/3)
            assert abs(elastic_candidate(c,unit_m)-potential(c,unit_m))<1e-8
            history.append(dict(time=h['time'], determinant_metric=float(np.linalg.det(m)),
                total_jacobian=float(np.linalg.det(f)),
                elastic_jacobian=float(np.linalg.det(f)*np.sqrt(np.linalg.det(m))),
                candidate_minus_source_potential_mpa=float(elastic_candidate(c,m)-potential(c,m)),
                maximum_candidate_minus_source_stress_mpa=float(np.max(abs(candidate_s-source_s))),
                gradient_error_mpa=float(max(errors))))
        assert all(sha256(directory/n)==h for n,h in retained['hashes'].items())
        records.append(dict(run=retained['run'],divisions=retained['divisions'],hashes=retained['hashes'],
            replay_checks_passed=True,history=history,
            maximum_stress_discrepancy_mpa=max(x['maximum_candidate_minus_source_stress_mpa'] for x in history),
            maximum_gradient_error_mpa=max(x['gradient_error_mpa'] for x in history),
            maximum_potential_difference_mpa=max(abs(x['candidate_minus_source_potential_mpa']) for x in history)))
    result = dict(identity_checks_passed=True,physical_acceptance=False,records=records,
        source_sha256=sha256(source),manual_sha256=sha256(manual),gate_sha256=sha256(GATE),
        script_sha256=sha256(__file__),dependencies={n:sha256(F/'results/scripts'/n) for n in
            ('audit_plastic_return_map.py','audit_metric_manual.py','prescribed_unloading_coupon.py')},
        derivation='Assuming M=Cp^-1 and Fe=F Fp^-1 gives Je=J sqrt(detM) and tr(be)=C:M. Applying the SAME explicitly chosen neo-Hookean functional form with Je instead of J gives the reported candidate, whose analytic fixed-M gradient is independently checked. It agrees with the prior source-integrable candidate when detM=1. No claim that this functional form or reference is uniquely specified by the manual.',
        references='Local node161.html:64-120 multiplicative/isochoric semantics; incplas.f:156-189,775-790 total-J stress; retained native-verified source replay.',
        limitations='Finite-step detM departures make the two gradients differ. A state-only reference cannot repair a fixed-M stress discrepancy. This comparison neither selects physical energy nor licenses metric normalization or a constitutive patch. Hardening storage/dissipation, continuum convergence, cap/contact/MPC and Abaqus equivalence remain unaccepted.',
        next='Investigate source-consistent recoverable energy by frozen-plastic-state unloading or independent Simo formulation before production load revision. Preserve source states; no fitted reference or normalization.')
    assert not OUT.exists()
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ('history','hashes')} for r in records]))


if __name__=='__main__': run()

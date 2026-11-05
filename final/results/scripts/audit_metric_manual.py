"""Link local manual state semantics to retained replay; no physical acceptance."""
import json
import numpy as np
from audit_plastic_return_map import F, sha256
from prescribed_unloading_coupon import audit, MU, BULK

GATE = F/'results/output/prescribed_unloading_coupon_20261011T190005Z.json'
OUT = F/'results/output/metric_manual_20261011T213005Z.json'


def potential(c, m):
    j = np.sqrt(np.linalg.det(c))
    return MU/2*(np.sum(c*m)/j**(2/3)-3)+BULK/4*(j*j-1-2*np.log(j))


def run():
    gate = json.loads(GATE.read_text())
    manual = F/'sym/vendor/CalculiX/ccx_2.23/doc/ccx/node161.html'
    source = F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'
    text = manual.read_text()
    assert 'inverse plastic right Cauchy-Green tensor' in text
    assert 'plastic flow is isochoric' in text
    assert gate['identity_checks_passed'] and sha256(source)==gate['source_sha256']
    records = []
    for retained in gate['records']:
        directory = F/retained['run']
        assert all(sha256(directory/n)==h for n,h in retained['hashes'].items())
        lines = (directory/'job.inp').read_text().splitlines(); curve=[]
        for line in lines[lines.index('*PLASTIC')+1:]:
            if line.startswith('*'): break
            curve.append(list(map(float,line.split(','))))
        replay = audit(directory,retained['divisions'],np.array(curve))
        assert replay['replay_checks_passed']
        gradient_errors=[]; volumes=[]
        for h in replay['history']:
            m=np.array(h['metric'])
            f=np.diag(1+h['path_fraction']*np.array([.02,-.01,-.01]))
            c=f.T@f
            s=np.array(h['second_piola_stress_mpa'])
            # delta C=2 delta E; symmetric off-diagonal direction has two entries.
            for i in range(3):
                for k in range(i,3):
                    d=np.zeros((3,3)); d[i,k]=1; d[k,i]=1
                    eps=1e-6
                    fd=(potential(c+eps*d,m)-potential(c-eps*d,m))/(2*eps)
                    exact=.5*np.sum(s*d)
                    gradient_errors.append(abs(fd-exact))
            jp=float(np.linalg.det(m)**(-.5))
            # Manual M=Cp^-1; Jp=sqrt(det Cp). This is state implication,
            # not an independently observed physical plastic volume.
            assert abs(jp*jp*np.linalg.det(m)-1)<1e-12
            volumes.append(dict(time=h['time'],implied_plastic_jacobian=jp,
                                determinant_metric=float(np.linalg.det(m))))
        assert max(gradient_errors)<1e-4
        assert all(sha256(directory/n)==h for n,h in retained['hashes'].items())
        records.append(dict(run=retained['run'],divisions=retained['divisions'],hashes=retained['hashes'],
            replay_checks_passed=True,maximum_gradient_error_mpa=float(max(gradient_errors)),
            final_implied_plastic_jacobian=volumes[-1]['implied_plastic_jacobian'],
            maximum_implied_volume_departure=max(abs(v['implied_plastic_jacobian']-1) for v in volumes),
            history=volumes))
    result=dict(identity_checks_passed=True,physical_acceptance=False,records=records,
        manual_sha256=sha256(manual),source_sha256=sha256(source),gate_sha256=sha256(GATE),
        script_sha256=sha256(__file__),dependencies={n:sha256(F/'results/scripts'/n) for n in ('audit_plastic_return_map.py','prescribed_unloading_coupon.py')},
        references=dict(manual='sym/vendor/CalculiX/ccx_2.23/doc/ccx/node161.html:64-115',
                        state='incplas.f:68-80,1024-1032',stress_update='incplas.f:156-189,760-790'),
        interpretation='Manual identifies SDV2:7=(I-Cp^-1)/2, hence source cpl=M=Cp^-1 and implied Jp=det(M)^(-1/2). Manual asserts isochoric plastic flow. Retained source states depart from this identity; do not call this measured physical volume or silently normalize them.',
        candidate_fixed_state_potential='W=mu/2[J^(-2/3) C:M-3]+K/4[J^2-1-2ln J]. At fixed M its C-gradient equals S/2, checked in six symmetric directions against replay/native-verified second Piola stress. Any function of M and hardening state can be added without changing this gradient.',
        limitations='Fixed-state stress integrability does not identify state-only energy reference, hardening energy, physical dissipation or continuum limit. No production constitutive change, cap/contact/MPC/Abaqus/experimental/burst acceptance. Next derive the state derivative and its power along retained plastic flow before interpreting work-minus-potential.')
    assert not OUT.exists()
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ('history','hashes')} for r in records]))


if __name__=='__main__': run()

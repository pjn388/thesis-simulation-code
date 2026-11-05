"""Stress-measure conversion audit; not an assembled FE tangent or solver patch."""
import json
import numpy as np
from verify_geers_reference import update, I
from shear_unloading_coupon import F, sha256


def run():
    assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
    prior_path=F/'results/output/geers_reference_20261012T020003Z.json'
    prior=json.loads(prior_path.read_text())
    assert prior['checks_passed']
    assert sha256(F/'results/scripts/verify_geers_reference.py')==prior['provenance']['script_sha256']
    for name,digest in prior['provenance']['dependencies'].items():
        assert sha256(F/'results/scripts'/name)==digest
    records=[]
    for stretch in (1.,1.015):
        metric=I.copy(); alpha=0.
        for shear in list(np.linspace(0,.03,41)[1:])+list(np.linspace(.03,.0294,11)[1:]):
            f=np.diag([stretch,1.,1.]); f[0,1]=shear
            r=update(f,metric,alpha); inverse=np.linalg.inv(f)
            p=r['tau']@inverse.T
            tangent=np.zeros((9,9)); errors=[]; omitted=[]
            for a in range(3):
                for b in range(3):
                    h=np.zeros((3,3)); h[a,b]=1.
                    dt=update(f,metric,alpha,h)['directional_tau']
                    material=dt@inverse.T
                    dp=material-p@h.T@inverse.T
                    tangent[:,3*a+b]=dp.ravel()
                    for eps in (2e-7,1e-7,5e-8):
                        plus=update(f+eps*h,metric,alpha)
                        minus=update(f-eps*h,metric,alpha)
                        assert plus['active']==minus['active']==r['active']
                        fd=(plus['tau']@np.linalg.inv(f+eps*h).T-minus['tau']@np.linalg.inv(f-eps*h).T)/(2*eps)
                        scale=max(1.,np.linalg.norm(dp))
                        errors.append(float(np.linalg.norm(fd-dp)/scale))
                        omitted.append(float(np.linalg.norm(fd-material)/scale))
            records.append(dict(stretch=stretch,shear=float(shear),active=r['active'],
                fd_relative=max(errors),without_inverse_term_relative=max(omitted),
                major_asymmetry_relative=float(np.linalg.norm(tangent-tangent.T)/np.linalg.norm(tangent))))
            metric,alpha=r['metric'],r['alpha']
    checks=dict(fd_relative=max(r['fd_relative'] for r in records),
        elastic_major_asymmetry=max(r['major_asymmetry_relative'] for r in records if not r['active']),
        plastic_major_asymmetry=max(r['major_asymmetry_relative'] for r in records if r['active']),
        omitted_term_error=max(r['without_inverse_term_relative'] for r in records))
    assert checks['fd_relative']<2e-6
    assert checks['elastic_major_asymmetry']<1e-12
    report=dict(checks=checks,records=records,physical_acceptance=False,
        derivation='P=tau F^-T; dP=dtau F^-T-P dF^T F^-T, at fixed prior metric and alpha. Independent centered differences perturb all nine F components at three step sizes. Elastic major symmetry follows from retained frozen-metric potential; plastic symmetry is measured without imposing acceptance.',
        limitations='Manufactured retained reference only. No FE element integration, spatial tangent, native patch, energy/hardening/PEEQ identification, cap/contact/MPC/Abaqus or physical/burst validation. Plastic asymmetry alone neither proves an error nor supplies a solver stiffness.',
        provenance=dict(script_sha256=sha256(__file__),prior_sha256=sha256(prior_path),
            dependencies={name:sha256(F/'results/scripts'/name) for name in ('verify_geers_reference.py','audit_spherical_metric.py','shear_unloading_coupon.py')}),
        next='Resolve hardening thermodynamics independently; assemble and verify element residual/tangent including geometric and boundary-load terms before any native revision.')
    out=F/'results/output/reference_piola_tangent_20261012T033003Z.json'
    assert not out.exists()
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),checks=checks)))


if __name__=='__main__':
    run()

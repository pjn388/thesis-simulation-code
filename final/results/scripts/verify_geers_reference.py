"""Isolated material-point reference, not a CalculiX patch or physical material fit.

Geers et al. ECCOMAS2000 eqs 8-10, 26-29, 32: linear Kirchhoff
hardening and paper gamma convention. Tangent below differentiates this map
with respect to total F at fixed previous state; it is not a FE stiffness.
"""
import json
from pathlib import Path
import numpy as np
from audit_spherical_metric import reconstruct
from shear_unloading_coupon import F, MU, BULK, sha256

TAG = 'geers_reference_20261012T020003Z'
I = np.eye(3)
Y0, HARD = 250., 1000.  # Manufactured Kirchhoff law, no imported A37 curve.


def dev(a):
    return a - np.trace(a)/3*I


def update(f, metric, alpha, direction=None):
    j = np.linalg.det(f)
    assert j > 0
    k = j**(-2/3)
    b = k*f@metric@f.T
    d = dev(b)
    trial = MU*np.sqrt(1.5*np.sum(d*d))
    active = bool(trial > Y0+HARD*alpha)
    gamma = (trial-Y0-HARD*alpha)/(MU*k+HARD) if active else 0.
    ratio = 1-MU*k*gamma/trial if active else 1.
    dn = ratio*d
    if active:
        # Reconstruct unit-det Bbar, then pull back Be = J^(2/3) Bbar.
        unit_metric, _, _, _ = reconstruct(I, dn)
        inverse = np.linalg.inv(f)
        mn = inverse@(unit_metric/k)@inverse.T
    else:
        mn = metric.copy()  # Freeze the material state on elastic unloading.
    tau = BULK/2*(j*j-1)*I+MU*dn
    result = dict(tau=tau, metric=mn, alpha=alpha+gamma, active=active,
                  gamma=gamma, trial=trial)
    if direction is not None:
        ell = np.trace(np.linalg.solve(f, direction))
        db = k*(direction@metric@f.T+f@metric@direction.T)-2/3*ell*b
        dd = dev(db)
        dr = 0.
        if active:
            dt = 1.5*MU*MU*np.sum(d*dd)/trial
            dk = -2/3*k*ell
            dg = (dt-MU*dk*gamma)/(MU*k+HARD)
            dr = -MU*((dk*gamma+k*dg)/trial-k*gamma*dt/trial**2)
        result['directional_tau'] = BULK*j*j*ell*I+MU*(ratio*dd+dr*d)
    return result


def run():
    assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
    paper = F/'sym/runs/investigation/source_evidence_20261012T013001Z/geers_eccomas2000.pdf'
    retained = F/'results/output/spherical_metric_audit_20261012T013001Z.json'
    prior = json.loads(retained.read_text())
    assert sha256(paper)==prior['provenance']['paper_sha256']
    for r in prior['records']:
        assert all(sha256(F/r['run']/n)==h for n,h in r['hashes'].items())
    rng = np.random.default_rng(731)
    histories=[]; maxima=dict(determinant=0.,yield_mpa=0.,objectivity_mpa=0.,
                             rotated_metric=0.,unloading_metric=0.,tangent_relative=0.)
    for stretch in (1., 1.015):
        metric=I.copy(); alpha=0.; history=[]
        for shear in list(np.linspace(0,.03,81)[1:])+list(np.linspace(.03,.0294,21)[1:]):
            f=np.diag([stretch,1.,1.]); f[0,1]=shear
            r=update(f,metric,alpha)
            maxima['determinant']=max(maxima['determinant'],abs(np.linalg.det(r['metric'])-1))
            assert np.linalg.eigvalsh(r['metric']).min()>0
            vm=np.sqrt(1.5*np.sum(dev(r['tau'])**2))
            error=abs(vm-Y0-HARD*r['alpha']) if r['active'] else max(0.,vm-Y0-HARD*alpha)
            maxima['yield_mpa']=max(maxima['yield_mpa'],error)
            q,_=np.linalg.qr(rng.normal(size=(3,3)))
            if np.linalg.det(q)<0: q[:,0]*=-1
            rotated=update(q@f,metric,alpha)
            maxima['objectivity_mpa']=max(maxima['objectivity_mpa'],float(abs(rotated['tau']-q@r['tau']@q.T).max()))
            maxima['rotated_metric']=max(maxima['rotated_metric'],float(abs(rotated['metric']-r['metric']).max()))
            if not r['active']:
                maxima['unloading_metric']=max(maxima['unloading_metric'],float(abs(r['metric']-metric).max()))
                assert r['alpha']==alpha
            # All nine total-F directions at fixed old state, branch separated.
            for a in range(3):
                for c in range(3):
                    h=np.zeros((3,3)); h[a,c]=1.
                    tangent=update(f,metric,alpha,h)['directional_tau']
                    epsilon=1e-7
                    plus=update(f+epsilon*h,metric,alpha)
                    minus=update(f-epsilon*h,metric,alpha)
                    assert plus['active']==minus['active']==r['active']
                    fd=(plus['tau']-minus['tau'])/(2*epsilon)
                    relative=float(abs(fd-tangent).max()/max(1.,abs(tangent).max()))
                    maxima['tangent_relative']=max(maxima['tangent_relative'],relative)
            history.append(dict(shear=float(shear),alpha=r['alpha'],active=r['active'],
                determinant=float(np.linalg.det(r['metric'])),vm_mpa=float(vm)))
            metric,alpha=r['metric'],r['alpha']
        assert not any(h['active'] for h in history[-20:])
        histories.append(dict(axial_stretch=stretch,history=history))
    limits=dict(determinant=1e-12,yield_mpa=1e-9,objectivity_mpa=1e-8,
                rotated_metric=1e-12,unloading_metric=1e-14,tangent_relative=2e-6)
    passed=all(maxima[n]<v for n,v in limits.items())
    report=dict(checks_passed=passed,physical_acceptance=False,maxima=maxima,limits=limits,
        histories=histories,manufactured_law=dict(initial_kirchhoff_yield=Y0,hardening=HARD),
        mapping='Native isotropic delta_alpha=sqrt(2/3)*cop; matching its deviatoric metric decrement to paper eq26 requires delta_gamma=3*trace(Bbar_trial)/3*J^(2/3)*delta_alpha. Thus paper alpha=gamma is not native PEEQ. Native yield also multiplies Cauchy flow stress by J; this reference uses a fixed Kirchhoff law.',
        limitations='Reference algebra only, no native solve or production patch. No physical equivalence of hardening variables, stored energy/dissipation acceptance, FE assembled tangent, contact/MPC/cap/Abaqus or experimental/burst validation. Original failed work gates preserved.',
        provenance=dict(script_sha256=sha256(__file__),paper_sha256=sha256(paper),
            retained_sha256=sha256(retained),dependencies={n:sha256(F/'results/scripts'/n) for n in
                ('audit_spherical_metric.py','shear_unloading_coupon.py','audit_plastic_return_map.py')}),
        references='Local retained Geers ECCOMAS2000 pp5-9 eq8-10,26-29,32; incplas.f 156-189,503-580,754-790.',
        next='Resolve hardening/PEEQ and energy consistency before native fork; independently verify FE stress-measure tangent and Abaqus convention. Do not relabel gamma as native PEEQ or import production curve unchanged.')
    out=F/'results/output'/f'{TAG}.json'
    assert not out.exists()
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),checks_passed=passed,maxima=maxima)))
    assert passed


if __name__=='__main__':
    run()

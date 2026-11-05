"""Audit the retained manufactured reference; never alter native material states.

Energy is the independently differentiated fixed-metric trial potential already
used in audit_frozen_metric_work. Treat H*gamma^2/2 and Y0*gamma as hypotheses,
not as a sourced identification of gamma with thermodynamic accumulated strain.
"""
import json
import numpy as np
from verify_geers_reference import update, I, Y0, HARD
from shear_unloading_coupon import F, MU, BULK, sha256
from audit_frozen_metric_work import elastic

TAG = 'reference_energy_20261012T023008Z'


def energy(f, metric):
    return float(elastic(f.T@f, metric, MU, BULK)[0])-1.5*MU


def run():
    assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
    prior_path = F/'results/output/geers_reference_20261012T020003Z.json'
    prior = json.loads(prior_path.read_text())
    assert prior['checks_passed']
    assert sha256(F/'results/scripts/verify_geers_reference.py')==prior['provenance']['script_sha256']
    for name, digest in prior['provenance']['dependencies'].items():
        assert sha256(F/'results/scripts'/name)==digest
    paper = F/'sym/runs/investigation/source_evidence_20261012T013001Z/geers_eccomas2000.pdf'
    assert sha256(paper)==prior['provenance']['paper_sha256']
    # Independent frozen-state energy gradient against P = tau F^-T.
    f = np.diag([1.01,.99,1.02]); f[0,1]=.0001
    metric = np.diag([1.0001,1/1.0001,1.])
    r = update(f,metric,0.)
    # Use trial stress from the independent potential even when plastic active.
    _, s = elastic(f.T@f,metric,MU,BULK)
    p = f@s
    errors=[]
    for a in range(3):
        for b in range(3):
            h=np.zeros((3,3)); h[a,b]=1.; eps=1e-6
            fd=(energy(f+eps*h,metric)-energy(f-eps*h,metric))/(2*eps)
            errors.append(abs(fd-p[a,b])/max(1.,abs(p[a,b])))
    assert max(errors)<2e-5
    records=[]
    for stretch in (1.,1.015):
        for n in (40,80,160,320,640):
            metric=I.copy(); alpha=0.; old_f=I.copy(); old_p=np.zeros((3,3))
            work=0.; release=0.; hard=0.; minimum_dissipation=0.; history=[]
            # Ramp stretch and shear together from the same stress-free state,
            # then unload shear at fixed stretch. This differs explicitly from
            # the previous reference's nonidentity starting stretch.
            fractions=list(np.linspace(0,1,n+1)[1:])
            paths=[(v,.03*v) for v in fractions]+[(1.,v) for v in np.linspace(.03,.0294,n//4+1)[1:]]
            for axial_fraction,shear in paths:
                f=np.diag([1+(stretch-1)*axial_fraction,1.,1.]); f[0,1]=shear
                r=update(f,metric,alpha)
                p=r['tau']@np.linalg.inv(f).T
                work+=float(.5*np.sum((p+old_p)*(f-old_f)))
                drop=energy(f,metric)-energy(f,r['metric'])
                dh=.5*HARD*(r['alpha']**2-alpha**2)
                release+=drop; hard+=dh
                minimum_dissipation=min(minimum_dissipation,drop-dh)
                history.append(dict(shear=float(shear),active=r['active'],gamma=r['gamma'],
                    energy_release_mpa=drop,hypothetical_hardening_change_mpa=dh))
                old_f,old_p,metric,alpha=f,p,r['metric'],r['alpha']
            final=energy(f,metric)
            expected=Y0*alpha
            # Work = stored elastic change + sum of metric-induced releases in
            # the infinitesimal-step limit; first audit this independently of
            # any proposed hardening/dissipation interpretation.
            records.append(dict(stretch=stretch,divisions=n,work_mpa=work,
                stored_elastic_mpa=final,metric_release_mpa=release,
                work_balance_error_mpa=work-final-release,
                gamma=alpha,hypothetical_hardening_mpa=hard,
                hypothetical_dissipation_mpa=release-hard,
                hypothetical_yield_work_mpa=expected,
                conjugacy_relative_error=(release-hard-expected)/max(1.,expected),
                minimum_increment_hypothetical_dissipation_mpa=minimum_dissipation,
                unloading_frozen=not any(h['active'] for h in history[-n//4:]),
                history=history))
    checks=dict(frozen_energy_gradient_relative=max(errors),
        finite_outputs=bool(all(np.isfinite(r['work_balance_error_mpa']) for r in records)),
        unloading_frozen=all(r['unloading_frozen'] for r in records),
        refined_work_balance=all(abs(records[i+4]['work_balance_error_mpa'])<abs(records[i]['work_balance_error_mpa']) for i in (0,5)))
    assert checks['finite_outputs'] and checks['unloading_frozen'] and checks['refined_work_balance']
    report=dict(audit_checks=checks,records=records,physical_acceptance=False,
        hypothesis='Test ONLY whether paper gamma can be assigned H*gamma^2/2 stored hardening and Y0*gamma dissipation. Failure does not disprove Geers; it rejects this assumed conjugacy for the retained reference.',
        limitations='Manufactured law and material point only. No native PEEQ equivalence, production revision, FE tangent, cap/contact/MPC/Abaqus or experimental acceptance. Energy density MPa equals N/mm^2; no pipe volume used.',
        provenance=dict(script_sha256=sha256(__file__),reference_report_sha256=sha256(prior_path),paper_sha256=sha256(paper),dependencies={name:sha256(F/'results/scripts'/name) for name in ('verify_geers_reference.py','audit_frozen_metric_work.py','audit_spherical_metric.py','shear_unloading_coupon.py')}),
        next='Derive energy-conjugate accumulated strain and corresponding hardening evolution from the full source before considering a native patch. Preserve native PEEQ and all old work failures.')
    path=F/'results/output'/f'{TAG}.json'
    assert not path.exists()
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(path),checks=checks,finest=[{k:v for k,v in records[i].items() if k!='history'} for i in (4,9)])))


if __name__=='__main__':
    run()

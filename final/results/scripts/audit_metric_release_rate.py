"""Read-only derivative audit of retained reference, not a material revision."""
import json
import numpy as np
from verify_geers_reference import I, MU, dev
from audit_spherical_metric import reconstruct
from audit_reference_energy import energy
from shear_unloading_coupon import F, sha256


def release_rate(f, metric):
    j = np.linalg.det(f)
    k = j**(-2/3)
    b = k*f@metric@f.T
    d = dev(b)
    equivalent = MU*np.sqrt(1.5*np.sum(d*d))
    normal = MU*d/equivalent
    inverse = np.linalg.inv(b)
    # dB/dgamma=-k*N+s*I. det(B)=1 implies
    # s=k*tr(B^-1 N)/tr(B^-1); W=MU/2 tr(B)+Wvol.
    rate = -1.5*MU*k*np.trace(inverse@normal)/np.trace(inverse)
    return float(rate), float(equivalent), d, normal, k


def run():
    assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
    prior_path=F/'results/output/reference_energy_20261012T023008Z.json'
    prior=json.loads(prior_path.read_text())
    assert sha256(F/'results/scripts/audit_reference_energy.py')==prior['provenance']['script_sha256']
    for name, digest in prior['provenance']['dependencies'].items():
        assert sha256(F/'results/scripts'/name)==digest
    # Finite differences change only plastic correction at fixed total F.
    # This is independent of the total-F stress derivative in the prior audit.
    samples=[]
    for stretch in (1.,1.015):
        for shear in (.005,.015,.03):
            f=np.diag([stretch,1.,1.]); f[0,1]=shear
            rate,eq,d,normal,k=release_rate(f,I)
            eps=1e-6
            inv=np.linalg.inv(f)
            def corrected(g):
                b,_,_,_=reconstruct(I,d-k*g*normal)
                return inv@(b/k)@inv.T
            fd=-(energy(f,corrected(eps))-energy(f,corrected(-eps)))/(2*eps)
            relative=abs(fd-rate)/max(1.,abs(rate))
            samples.append(dict(stretch=stretch,shear=shear,rate_mpa=rate,
                finite_difference_mpa=fd,relative_error=relative,rate_over_equivalent=rate/eq))
    # Use immutable reference trajectories: reconstruct each previous state
    # from the retained map, compare endpoint-rate integration with exact drop.
    from verify_geers_reference import update
    records=[]
    for rec in prior['records']:
        n=rec['divisions']; stretch=rec['stretch']; metric=I.copy(); alpha=0.
        ratesum=0.; exactsum=0.; minimum=float('inf'); ratios=[]
        fractions=list(np.linspace(0,1,n+1)[1:])
        paths=[(v,.03*v) for v in fractions]+[(1.,v) for v in np.linspace(.03,.0294,n//4+1)[1:]]
        for axial,shear in paths:
            f=np.diag([1+(stretch-1)*axial,1.,1.]); f[0,1]=shear
            r=update(f,metric,alpha)
            if r['active']:
                q,eq,_,_,_=release_rate(f,r['metric'])
                ratesum+=q*r['gamma']; minimum=min(minimum,q)
                ratios.append(q/eq)
            exactsum+=energy(f,metric)-energy(f,r['metric'])
            metric,alpha=r['metric'],r['alpha']
        assert abs(exactsum-rec['metric_release_mpa'])<1e-9
        records.append(dict(divisions=n,stretch=stretch,exact_release_mpa=exactsum,
            endpoint_rate_integral_mpa=ratesum,error_mpa=ratesum-exactsum,
            minimum_active_rate_mpa=minimum,rate_over_equivalent_range=[min(ratios),max(ratios)]))
    checks=dict(derivative_relative=max(s['relative_error'] for s in samples),
        positive_active_release=all(r['minimum_active_rate_mpa']>0 for r in records),
        refinement_decreases=all(abs(records[i+1]['error_mpa'])<abs(records[i]['error_mpa']) for i in (0,1,2,3,5,6,7,8)))
    assert checks['derivative_relative']<2e-5
    assert checks['positive_active_release'] and checks['refinement_decreases']
    report=dict(checks=checks,samples=samples,records=records,physical_acceptance=False,
        derivation='At fixed F, determinant-constrained Bbar correction obeys dBbar/dgamma=-J^(-2/3)N+sI, s=J^(-2/3)tr(Bbar^-1 N)/tr(Bbar^-1). Minus elastic energy derivative q=-3MU/2 J^(-2/3)tr(Bbar^-1 N)/tr(Bbar^-1). This is elastic release per reference gamma, not a hardening-storage law.',
        references='Geers ECCOMAS2000 pp5-8 eq8,10,13,15,16,26,27,32; https://congress2.cimne.com/eccomas/proceedings/eccomas2000/pdf/90.pdf; independent inference from determinant constraint and retained elastic potential.',
        limitations='Manufactured reference only. Positive elastic release does not establish dissipation after unknown hardening storage. No evolution revision, native PEEQ equivalence, production curve, FE tangent, cap/contact/MPC/Abaqus or physical/burst acceptance. Original energy failures preserved.',
        provenance=dict(script_sha256=sha256(__file__),prior_sha256=sha256(prior_path),
            dependencies={name:sha256(F/'results/scripts'/name) for name in ('verify_geers_reference.py','audit_reference_energy.py','audit_spherical_metric.py','audit_frozen_metric_work.py')}),
        next='Establish hardening free energy and accumulated strain convention from thermodynamic source; this derivative explains why gamma alone is not assumed work conjugate. Verify FE tangent before native fork.')
    out=F/'results/output/metric_release_rate_20261012T030006Z.json'
    assert not out.exists()
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),checks=checks,finest=[records[4],records[9]])))


if __name__=='__main__':
    run()

"""Manufactured total-Lagrangian hex audit, not native C3D20R validation."""
import json
import itertools
import numpy as np
from verify_geers_reference import update, I
from shear_unloading_coupon import F, sha256


def run():
    assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
    prior=F/'results/output/reference_piola_tangent_20261012T033003Z.json'
    assert json.loads(prior.read_text())['checks']['fd_relative']<2e-6
    signs=np.array(list(itertools.product((-1.,1.),repeat=3)))
    x=signs@np.array([[1.1,.1,0],[0,.9,.05],[.02,0,1.2]])
    points=[]
    for q in itertools.product((-1/np.sqrt(3),1/np.sqrt(3)),repeat=3):
        q=np.array(q)
        dn=np.array([[s[j]*np.prod([1+s[k]*q[k] for k in range(3) if k!=j])/8
                      for j in range(3)] for s in signs])
        jac=x.T@dn
        assert np.linalg.det(jac)>0
        points.append((dn@np.linalg.inv(jac),np.linalg.det(jac)))

    def assemble(u, stiffness=False):
        residual=np.zeros((8,3)); tangent=np.zeros((24,24)); branches=[]
        for grad,weight in points:
            f=I+u.T@grad
            r=update(f,I,0.); inv=np.linalg.inv(f)
            p=r['tau']@inv.T
            residual+=weight*grad@p.T
            branches.append(r['active'])
            if stiffness:
                for node in range(8):
                    for component in range(3):
                        h=np.zeros((3,3)); h[component]=grad[node]
                        dt=update(f,I,0.,h)['directional_tau']
                        dp=dt@inv.T-p@h.T@inv.T
                        tangent[:,3*node+component]=(weight*grad@dp.T).ravel()+tangent[:,3*node+component]
        return residual,tangent,branches

    records=[]
    for shear in (.0001,.015,.03):
        f=I.copy(); f[0,1]=shear
        u=x@(f-I).T
        # Nonaffine perturbation exercises differing integration-point states.
        u[:,0]+=shear*.04*signs[:,1]*signs[:,2]
        residual,k,branches=assemble(u,True)
        errors=[]
        for col in range(24):
            h=np.zeros((8,3)); h.ravel()[col]=1
            for eps in (2e-7,1e-7,5e-8):
                rp,_,bp=assemble(u+eps*h); rm,_,bm=assemble(u-eps*h)
                assert bp==bm==branches
                fd=((rp-rm)/(2*eps)).ravel()
                errors.append(float(np.linalg.norm(fd-k[:,col])/max(1.,np.linalg.norm(k[:,col]))))
        translation=np.tile([.3,-.2,.1],(8,1)).ravel()
        force=np.linalg.norm(residual.sum(axis=0))/max(1.,np.linalg.norm(residual))
        moment=np.linalg.norm(np.cross(x+u,residual).sum(axis=0))/max(1.,np.linalg.norm(residual))
        null=np.linalg.norm(k@translation)/max(1.,np.linalg.norm(k)*np.linalg.norm(translation))
        records.append(dict(shear=shear,active_points=sum(branches),fd_relative=max(errors),
            translation_null_relative=float(null),net_force_relative=float(force),
            net_moment_relative=float(moment),major_asymmetry=float(np.linalg.norm(k-k.T)/np.linalg.norm(k))))
    assert max(r['fd_relative'] for r in records)<2e-6
    assert max(max(r['translation_null_relative'],r['net_force_relative'],r['net_moment_relative']) for r in records)<1e-12
    assert records[0]['active_points']==0 and records[-1]['active_points']==8
    assert records[0]['major_asymmetry']<1e-12
    out=F/'results/output/reference_element_tangent_20261012T040000Z.json'
    assert not out.exists()
    report=dict(checks_passed=True,physical_acceptance=False,records=records,
        method='Distorted trilinear eight-node hex, eight Gauss points, fixed old metric I and alpha0. Internal residual integral P gradN and total-F directional tangent. All24 nodal directions, three perturbation sizes, nonaffine elastic/plastic states; translation nullspace and current-configuration force/moment balance.',
        limitations='Manufactured reference internal residual only; not native C3D20R integration, history transfer, boundary/follower pressure tangent, contact/MPC/cap, hardening thermodynamics, Abaqus or physical/burst validation. Plastic major asymmetry measured without acceptance requirement.',
        provenance=dict(script_sha256=sha256(__file__),prior_sha256=sha256(prior),dependencies={n:sha256(F/'results/scripts'/n) for n in ('verify_geers_reference.py','shear_unloading_coupon.py','audit_spherical_metric.py')}),
        next='Resolve hardening thermodynamics and separately verify follower boundary residual/tangent before a native revision.')
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    run()

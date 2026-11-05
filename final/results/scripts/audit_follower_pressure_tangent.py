"""Manufactured closed-surface pressure audit; no native solver acceptance."""
import itertools
import json
from pathlib import Path
import hashlib
import numpy as np

FINAL = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    assert json.loads((FINAL/'results/automation/paused.json').read_text())['state'] == 'resumed'
    # Outward oriented four-node faces of a cube; include both end faces.
    x = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype=float)
    faces = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    signs = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    pressure = 2.3  # Outward traction, opposite to positive external DLOAD.

    def assemble(y):
        force = np.zeros((8,3)); tangent = np.zeros((24,24)); volume = 0.
        face_forces = []
        for ids in faces:
            ids = np.array(ids); points = y[ids]; ff = np.zeros((4,3))
            for r,s in itertools.product((-1/np.sqrt(3),1/np.sqrt(3)), repeat=2):
                n = (1+signs[:,0]*r)*(1+signs[:,1]*s)/4
                dr = signs[:,0]*(1+signs[:,1]*s)/4
                ds = signs[:,1]*(1+signs[:,0]*r)/4
                a = dr@points; b = ds@points; area = np.cross(a,b)
                ff += pressure*n[:,None]*area
                volume += float((n@points)@area)/3
                for local,node in enumerate(ids):
                    for component in range(3):
                        h = np.eye(3)[component]
                        da = np.cross(dr[local]*h,b)+np.cross(a,ds[local]*h)
                        for j,target in enumerate(ids):
                            tangent[3*target:3*target+3,3*node+component] += pressure*n[j]*da
            force[ids] += ff
            face_forces.append(ff.sum(axis=0).tolist())
        return force,tangent,volume,face_forces

    records = []
    for stretch in (1.,1.2,1.7):
        y = x@np.array([[stretch,.2,.05],[0,.9,.1],[.1,0,1.1]]).T
        y[:,0] += .08*x[:,1]*x[:,2]  # Warped bilinear faces.
        y += [.4,-.3,.2]
        force,k,v,ff = assemble(y)
        assert v > 0
        fd_errors = []; work_errors = []
        for col in range(24):
            h = np.zeros((8,3)); h.ravel()[col] = 1
            for eps in (2e-6,1e-6,5e-7):
                fp,_,vp,_ = assemble(y+eps*h)
                fm,_,vm,_ = assemble(y-eps*h)
                fd = ((fp-fm)/(2*eps)).ravel()
                fd_errors.append(float(np.linalg.norm(fd-k[:,col])/max(1.,np.linalg.norm(k[:,col]))))
                work_errors.append(float(abs(pressure*(vp-vm)/(2*eps)-force.ravel()[col])/max(1.,abs(force.ravel()[col]))))
        scale = max(1.,np.linalg.norm(force))
        records.append(dict(stretch=stretch,volume=v,face_resultants=ff,
            fd_relative=max(fd_errors),volume_work_relative=max(work_errors),
            net_force_relative=float(np.linalg.norm(force.sum(axis=0))/scale),
            net_moment_relative=float(np.linalg.norm(np.cross(y,force).sum(axis=0))/scale),
            closed_tangent_asymmetry=float(np.linalg.norm(k-k.T)/np.linalg.norm(k))))
    assert max(r['fd_relative'] for r in records) < 2e-6
    assert max(r['volume_work_relative'] for r in records) < 2e-6
    assert max(max(r['net_force_relative'],r['net_moment_relative'],r['closed_tangent_asymmetry']) for r in records) < 1e-12
    out = FINAL/'results/output/follower_pressure_tangent_20261012T043007Z.json'
    assert not out.exists()
    source = FINAL/'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f'
    report = dict(checks_passed=True,physical_acceptance=False,records=records,
        method='Bilinear four-node faces, 2x2 Gauss quadrature, outward constant pressure: f=integral p N (x_r cross x_s); df=integral p N (dx_r cross x_s+x_r cross dx_s). Six closed faces including both ends; 24 nodal directions and three finite-difference sizes. Independent divergence volume verifies f dot dx=p dV.',
        sources=['CalculiX2.23 e_c3d.f:1597-1625 inward DLOAD sign and pressure stiffness source comment; 1761-1810 native distributed-load stiffness implementation. Native formulation is not compared by this audit.'],
        provenance={'script_sha256':digest(__file__),'native_source_sha256':digest(source)},
        limitations='Manufactured algebra only, not native C3D20R/shape8q, captured assembly, pipe finite-cap mapping, open-surface symmetrization, contact/MPC, material thermodynamics, Abaqus or physical/burst acceptance. No production changes.',
        next='Compare native pressure stiffness/load conventions with independent assembled derivatives; resolve hardening energy before any native or production revision.')
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    run()

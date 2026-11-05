"""Independent quadratic open-face pressure derivative; not native acceptance."""
import hashlib
import json
from pathlib import Path
import numpy as np

FINAL = Path(__file__).resolve().parents[2]

def shape(r, s):
    signs = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    a, b = signs.T
    n = (1+a*r)*(1+b*s)*(a*r+b*s-1)/4
    dr = a*(1+b*s)*(2*a*r+b*s)/4
    ds = b*(1+a*r)*(a*r+2*b*s)/4
    return (np.r_[n, (1-r*r)*(1-s)/2, (1+r)*(1-s*s)/2,
                  (1-r*r)*(1+s)/2, (1-r)*(1-s*s)/2],
            np.r_[dr, -r*(1-s), (1-s*s)/2, -r*(1+s), -(1-s*s)/2],
            np.r_[ds, -(1-r*r)/2, -(1+r)*s, (1-r*r)/2, -(1-r)*s])

def assemble(y, order):
    f = np.zeros((8,3)); k = np.zeros((24,24))
    q, w = np.polynomial.legendre.leggauss(order)
    for ir,r in enumerate(q):
        for js,s in enumerate(q):
            n,dr,ds = shape(r,s)
            assert abs(n.sum()-1) < 1e-14
            assert max(abs(dr.sum()),abs(ds.sum())) < 1e-14
            a,b = dr@y,ds@y
            weight = w[ir]*w[js]*2.3
            f += weight*n[:,None]*np.cross(a,b)
            for node in range(8):
                for c,h in enumerate(np.eye(3)):
                    da = np.cross(dr[node]*h,b)+np.cross(a,ds[node]*h)
                    k[:,3*node+c] += (weight*n[:,None]*da).ravel()
    return f,k

def run():
    assert json.loads((FINAL/'results/automation/paused.json').read_text())['state']=='resumed'
    uv = np.array([[-1,-1],[1,-1],[1,1],[-1,1],[0,-1],[1,0],[0,1],[-1,0]],float)
    records=[]
    for curvature in (0.,.08,.2):
        r,s=uv.T
        y=np.column_stack((r+.07*r*s,s+.04*r*r,curvature*(r*r+.6*s*s)+.03*r*s))
        y=y@np.array([[1.2,.1,.03],[0,.9,.08],[.1,0,1.1]]).T
        f,k=assemble(y,3); f5,k5=assemble(y,5)
        fd=[]; sym=[]
        for col in range(24):
            h=np.zeros((8,3)); h.ravel()[col]=1
            for eps in (2e-6,1e-6,5e-7):
                fp,_=assemble(y+eps*h,3); fm,_=assemble(y-eps*h,3)
                d=((fp-fm)/(2*eps)).ravel()
                scale=max(1.,np.linalg.norm(d))
                fd.append(float(np.linalg.norm(d-k[:,col])/scale))
                sym.append(float(np.linalg.norm(d-((k+k.T)/2)[:,col])/scale))
        translations=[]
        for c in range(3):
            h=np.zeros((8,3)); h[:,c]=1
            translations.append(float(np.linalg.norm(k@h.ravel())))
        records.append(dict(curvature=curvature,fd_relative=max(fd),
            symmetrized_fd_relative=max(sym),asymmetry=float(np.linalg.norm(k-k.T)/np.linalg.norm(k)),
            quadrature_force_relative=float(np.linalg.norm(f-f5)/max(1.,np.linalg.norm(f5))),
            quadrature_tangent_relative=float(np.linalg.norm(k-k5)/np.linalg.norm(k5)),
            translation_error=max(translations),resultant=f.sum(axis=0).tolist()))
    assert max(x['fd_relative'] for x in records)<2e-6
    assert max(max(x['quadrature_force_relative'],x['quadrature_tangent_relative'],x['translation_error']) for x in records)<1e-12
    assert min(x['symmetrized_fd_relative'] for x in records)>.01
    sources=[FINAL/'sym/vendor/CalculiX/ccx_2.23/src'/name for name in ('shape8q.f','e_c3d.f')]
    report=dict(checks_passed=True,native_acceptance=False,physical_acceptance=False,records=records,
        method='Eight-node serendipity open face; local polynomial shape derivatives; 3x3 versus 5x5 quadrature; exact area-vector derivative against 24 directions and 3 central differences at 3 curvatures. Outward pressure2.3; native positive external DLOAD uses opposite sign.',
        sources=['shape8q.f local shape functions/derivatives; e_c3d.f:1597-1625 pressure sign;1761-1810 symmetrized stiffness term. No port of native global derivative or xkl mapping is asserted.'],
        provenance={str(p.relative_to(FINAL)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*sources]},
        interpretation='An isolated open face has boundary terms and generally nonsymmetric exact load derivative. Symmetrization changes that derivative. This does not prove native solver error: native mapping, assembled closed surfaces, constraints and solver matrix conventions are not compared.',
        next='Compare assembled native pressure stiffness at fixed geometry and its residual sign, including global shape derivative/xkl mapping, closed-face cancellation and constrained subspace. Hardening-energy identification remains a separate blocker. Production untouched.')
    out=FINAL/'results/output/quadratic_pressure_derivative_20261012T050003Z.json'
    assert not out.exists()
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(records))

if __name__=='__main__':
    run()

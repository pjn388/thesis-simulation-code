"""Source-path audit and native-order reference; no native execution claim."""
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_quadratic_pressure_derivative import FINAL, assemble


def run():
    assert json.loads((FINAL/'results/automation/paused.json').read_text())['state'] == 'resumed'
    src = FINAL/'sym/vendor/CalculiX/ccx_2.23/src'
    names = ('nonlingeo.c', 'mafillsmmain.c', 'mafillsm.f', 'e_c3d.f', 'shape8q.f')
    sources = {name: (src/name).read_text() for name in names}
    assert 'mass[2]={0,0},stiffness=1, buckling=0, rhsi=1' in sources['nonlingeo.c']
    assert 'ncmat_,mass,&stiffness,&buckling,&rhsi' in sources['nonlingeo.c']
    assert 'ncmat_,mass(1),stiffness,buckling' in sources['mafillsm.f']
    assert 'if(rhsi.eq.1) then' in sources['e_c3d.f']
    assert 'elseif((mass.eq.1).and.' in sources['e_c3d.f']
    deck = FINAL/'sym/runs/campaign/orientation_gouge_00_adaptive/job.inp'
    assert '*STATIC' in deck.read_text().upper()
    assert '*DYNAMIC' not in deck.read_text().upper()
    uv = np.array([[-1,-1],[1,-1],[1,1],[-1,1],[0,-1],[1,0],[0,1],[-1,0]], float)
    records = []
    for curvature in (0., .08, .2):
        r,s = uv.T
        y = np.column_stack((r+.07*r*s, s+.04*r*r, curvature*(r*r+.6*s*s)+.03*r*s))
        y = y@np.array([[1.2,.1,.03],[0,.9,.08],[.1,0,1.1]]).T
        f,k = assemble(y,2)
        f3,k3 = assemble(y,3)
        errors = []
        for col in range(24):
            h = np.zeros((8,3)); h.ravel()[col] = 1
            for eps in (2e-6,1e-6,5e-7):
                fp,_ = assemble(y+eps*h,2); fm,_ = assemble(y-eps*h,2)
                d = ((fp-fm)/(2*eps)).ravel()
                errors.append(float(np.linalg.norm(d-k[:,col])/max(1.,np.linalg.norm(d))))
        records.append(dict(curvature=curvature, native_order_reference_fd_relative=max(errors),
            force_2x2_vs_3x3_relative=float(np.linalg.norm(f-f3)/np.linalg.norm(f3)),
            tangent_2x2_vs_3x3_relative=float(np.linalg.norm(k-k3)/np.linalg.norm(k3))))
    assert max(x['native_order_reference_fd_relative'] for x in records) < 2e-6
    report = dict(reference_checks_passed=True, native_execution=False, physical_acceptance=False,
        records=records, source_trace=[
            'nonlingeo.c:118 static defaults mass[2]={0,0}, rhsi=1;811-812 sets mass only within nmethod==4 dynamics.',
            'nonlingeo.c:2492-2502 forwards flags to mafillsmmain; mafillsm.f:286-293 forwards mass(1) and rhsi to e_c3d.',
            'e_c3d.f:1478 force branch if rhsi==1;1625 stiffness branch is ELSEIF mass==1 and perturbation. At documented static defaults the cited symmetrized term at1761-1810 is not entered.',
            'e_c3d.f:287-294 C3D20R mint2d=4;1460-1465 selects gauss2d2. Reference here uses2x2; previous audit used3x3.',
            'shape8q.f:30-33 iflag3 uses global shape derivatives; reference here still uses local area-vector derivatives, not a native global-derivative/xkl port.'],
        inference='The previously cited symmetrized branch cannot be presumed to be the active static pressure tangent. Source defaults indicate it is skipped on this call path. This is a source-level inference, not runtime instrumentation or proof of a solver defect; alternative assembly paths and effective runtime flags require capture.',
        quadrature_interpretation='Finite differences verify the independent2x2 discrete load derivative. Differences from3x3 are measured quadrature sensitivity for manufactured warped faces, not pipe-model accuracy or acceptance.',
        next='Instrument actual static assembly flags and pressure tangent at fixed geometry, search alternative load-tangent assembly, then compare constrained residual derivative. Do not revise native stiffness based on inactive code. Hardening and cap/contact/MPC/Abaqus equivalence remain independent prerequisites.',
        provenance={str(p.relative_to(FINAL)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [Path(__file__),Path(__file__).with_name('audit_quadratic_pressure_derivative.py'),deck,*[src/n for n in names]]})
    out = FINAL/'results/output/static_pressure_path_20261012T053008Z.json'
    assert not out.exists()
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(records))


if __name__ == '__main__':
    run()

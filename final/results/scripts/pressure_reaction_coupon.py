"""Isolated native verification of RF subtraction on a loaded restraint."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from pipe_sim.model import HEX_LOCAL
from pipe_sim.results import dat_blocks
from pipe_sim.runner import sha256
from audit_nodal_pressure_loads import nodal_loads
from audit_pressure_failures import converged_times, matches_sta

FINAL = Path(__file__).resolve().parents[2]


def run(name):
    root = FINAL / 'sym/runs/investigation' / name
    root.mkdir(parents=True, exist_ok=False)
    exe = FINAL / 'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
    build = json.loads((exe.parent / 'build.json').read_text())
    assert build['verified'] and sha256(exe) == build['solver_sha256']
    points = (HEX_LOCAL + 1) * np.array([2., 3., 1.])
    top = [i+1 for i,p in enumerate(points) if p[2] == 2]
    bottom = [i+1 for i,p in enumerate(points) if p[2] == 0]
    # P2 top outward ordering, including its four edge nodes.
    face = [5,6,7,8,13,14,15,16]
    records = []
    for fixed in [False, True]:
        directory = root / ('loaded_fixed' if fixed else 'loaded_free')
        directory.mkdir()
        lines = ['*HEADING', 'Pressure reaction semantic coupon: mm N MPa', '*NODE']
        lines += [f'{i},{p[0]},{p[1]},{p[2]}' for i,p in enumerate(points,1)]
        lines += ['*ELEMENT,TYPE=C3D20R,ELSET=CUBE', '1,'+','.join(map(str,range(1,16)))+',',
                  ','.join(map(str,range(16,21))),
                  '*NSET,NSET=ALL', ','.join(map(str,range(1,17))), ','.join(map(str,range(17,21))),
                  '*MATERIAL,NAME=ELASTIC', '*ELASTIC', '210000,0.3',
                  '*SOLID SECTION,ELSET=CUBE,MATERIAL=ELASTIC', '*BOUNDARY']
        lines += [f'{n},1,3,0' for n in bottom]
        # Suppress single reduced-element lateral modes: uniaxial strain fixture.
        lines += [f'{n},1,2,0' for n in range(1,21) if n not in bottom]
        if fixed:
            lines += [f'{n},3,3,0' for n in top]
        lines += ['*STEP,NLGEOM,INC=100', '*STATIC,SOLVER=PARDISO',
                  '1,1,1e-8,1', '*DLOAD', '1,P2,2', '*NODE PRINT,NSET=ALL',
                  'U,RF', '*END STEP']
        deck = directory/'job.inp'
        deck.write_text('\n'.join(lines)+'\n')
        before = sha256(deck)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MKL_THREADING_LAYER='GNU')
        with (directory/'solver.log').open('w') as log:
            result = subprocess.run([str(exe),'-i','job'],cwd=directory,env=env,
                                    stdout=log,stderr=subprocess.STDOUT,timeout=60)
        log = (directory/'solver.log').read_text()
        assert result.returncode == 0 and 'Job finished' in log and '*ERROR' not in log
        assert sha256(deck) == before and matches_sta(converged_times(directory/'job.sta'),1)
        blocks = {k:rows for k,g,t,rows in dat_blocks(directory/'job.dat') if abs(t-1)<1e-7}
        u = {int(r[0]):np.array(r[1:]) for r in blocks['U']}
        rf = {int(r[0]):np.array(r[1:]) for r in blocks['RF']}
        assert set(u) == set(rf) == set(range(1,21))
        applied = {n:np.zeros(3) for n in u}
        for n,v in zip(face,nodal_loads([points[n-1]+u[n] for n in face],-2)):
            applied[n] += v
        constraints = {(n,d) for n in bottom for d in range(3)}
        constraints |= {(n,d) for n in range(1,21) for d in [0,1]}
        if fixed:
            constraints |= {(n,2) for n in top}
        reaction = sum((np.eye(3)[d]*(rf[n][d]-applied[n][d]) for n,d in constraints),np.zeros(3))
        total = sum(applied.values(),np.zeros(3))
        free_error = max(abs(rf[n][d]-applied[n][d]) for n in u for d in range(3) if (n,d) not in constraints)
        top_rf = sum(rf[n][2] for n in top)
        top_applied = sum(applied[n][2] for n in top)
        assert abs(top_applied+48) < .01
        assert free_error < 1e-4 and np.max(abs(total+reaction)) < 1e-4
        assert abs(top_rf-(0 if fixed else top_applied)) < 1e-4
        if fixed:
            assert max(abs(u[n][2]) for n in top) < 1e-12
        records.append(dict(run=str(directory.relative_to(FINAL)),loaded_face_z_constrained=fixed,
            accepted_time=1.,applied_resultant_n=total.tolist(),loaded_face_rf_z_n=top_rf,
            loaded_face_applied_z_n=top_applied,recovered_support_reaction_n=reaction.tolist(),
            equilibrium_residual_n=(total+reaction).tolist(),free_dof_max_error_n=free_error,
            input_sha256=before,native_sha256={n:sha256(directory/n) for n in ['job.dat','job.sta','solver.log']},
            solver_sha256=sha256(exe),returncode=result.returncode))
    report = dict(records=records,script_sha256=sha256(__file__),
        sources=['sym/vendor/CalculiX/ccx_2.23/doc/ccx/node212.html:62-64,99-138',
                 'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f:1360-1486,1597-1615'],
        limitations='Elastic one-element planar pressure coupon with all lateral DOFs fixed (uniaxial strain); verifies loaded constrained RF subtraction, '
                     'not finite-expansion pipe reactions, mesh convergence, Abaqus equivalence or burst.')
    (root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    output = FINAL/'results/output/pressure_restraint_candidates'/f'{name}.json'
    assert not output.exists()
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    run(parser.parse_args().name)

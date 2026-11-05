"""Bounded native central residual derivative diagnostic; not equilibrium/burst."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from verify_shared_node_capture import FINAL, EXE, coupon
from pipe_sim.runner import sha256


def matrix(g):
    a = np.diag(g['ad'])
    for j in range(57):
        for k in range(g['jq'][j]-1,g['jq'][j+1]-1):
            i = g['irow'][k]-1
            a[i,j] = a[j,i] = g['au'][k]
    return a


def main(tag):
    acceptance = FINAL/'results/output/equation_identity_recovery_v2_20261017T0234.json'
    reference = json.loads(acceptance.read_text())
    for p,h in reference['provenance'].items():
        assert sha256(FINAL/p) == h, p
    for case in reference['cases']:
        for p,h in case['hashes'].items():
            assert sha256(FINAL/case['run']/p) == h
    root = FINAL/'sym/runs/investigation'/('coherent_residual_'+tag)
    root.mkdir(exist_ok=False)
    capture = Path(__file__).with_name('capture_coherent_residual_gdb.py')
    source = FINAL/'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f'
    direction = np.random.default_rng(1741).normal(size=57)
    direction /= np.linalg.norm(direction)
    cases, comparisons = [], []
    for side in ('minus','plus'):
        states = {}
        for pressure in (0,2):
            for amplitude in (0.,1e-4,-1e-4,5e-5,-5e-5):
                assert json.loads((FINAL/'results/automation/paused.json').read_text())['state'] == 'resumed'
                folder = root/f'{side}_p{pressure}_u{amplitude}'
                folder.mkdir()
                deck,nodes,elements,_,pins = coupon(side,False,.04)
                if pressure == 0:
                    for e in elements:
                        deck = deck.replace(f'{e},P{2 if side == "plus" else 1},2\n',f'{e},P{2 if side == "plus" else 1},0\n')
                (folder/'job.inp').write_text(deck)
                (folder/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
                env = dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CCX_NPROC_RESULTS='1',CCX_NPROC_STIFFNESS='1',CCX_NPROC_EQUATION_SOLVER='1',THESIS_MATRIX_SOURCE=str(source),THESIS_MATRIX_OUTPUT=str(folder/'matrix.json'),THESIS_DIRECTION=json.dumps(direction.tolist()),THESIS_AMPLITUDE=str(amplitude))
                with (folder/'solver.log').open('w') as stream:
                    result = subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(EXE),'-i','job'],cwd=folder,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
                assert result.returncode == 0, str(folder)
                data = json.loads((folder/'matrix.json').read_text())
                r,g = data['residual'],data['global_matrix']
                expected = np.array(data['imposed']['displacement'])
                assert np.array_equal(r['displacement'],expected)
                assert r['mapping'] == data['imposed']['mapping']
                assert np.max(abs(np.array(r['residual'])-(np.array(r['fext'])-r['internal']))) < 1e-10
                assert np.max(abs(np.array(g['rhs'])-r['residual'])) < 1e-10
                for record in data['records']:
                    local = expected.reshape(len(nodes),4)[np.array(record['connectivity'])-1,1:]
                    assert np.array_equal(record['displacements'],local)
                    assert np.isfinite(record['initial_stress']).all()
                    if amplitude:
                        assert np.linalg.norm(record['initial_stress']) > 0
                if amplitude:
                    assert np.linalg.norm(r['internal']) > 0
                states[pressure,amplitude] = data
                cases.append(dict(run=str(folder.relative_to(FINAL)),side=side,pressure=pressure,amplitude=amplitude,internal_norm=float(np.linalg.norm(r['internal'])),hashes={p.name:sha256(p) for p in (folder/'job.inp',folder/'matrix.json',folder/'solver.log')}))
        for pressure in (0,2):
            k = matrix(states[pressure,0.]['global_matrix'])
            previous = None
            for h in (1e-4,5e-5):
                plus,minus = (states[pressure,s*h]['residual'] for s in (1,-1))
                derivative = {key:(np.array(plus[key])-minus[key])/(2*h) for key in ('internal','fext','residual')}
                expected = -k@direction
                comparisons.append(dict(side=side,pressure=pressure,h=h,internal_matrix_relative_error=float(np.linalg.norm(derivative['internal']-k@direction)/np.linalg.norm(k@direction)),complete_residual_matrix_relative_error=float(np.linalg.norm(derivative['residual']-expected)/np.linalg.norm(expected)),external_derivative_norm=float(np.linalg.norm(derivative['fext'])),residual_derivative_norm=float(np.linalg.norm(derivative['residual'])),step_change_relative=None if previous is None else float(np.linalg.norm(derivative['residual']-previous)/np.linalg.norm(previous))))
                previous = derivative['residual']
    report = dict(cases=cases,comparisons=comparisons,direction=direction.tolist(),provenance={str(p.relative_to(FINAL)):sha256(p) for p in (Path(__file__),capture,capture.with_name('capture_equation_identity_gdb.py'),source,source.with_name('results.c'),source.with_name('nonlingeo.c'),EXE,acceptance)},limitation='Two constrained elastic C3D20R coupons, one random admissible direction, zero-centered states with two central step sizes, pressure0/2 on both faces. Displacement injected before first iout=-1 native results; updated displacements/stress/internal force verified at assembly and residual. Intentionally terminated before factorization. Numerical differences measured, not proof of native defect or production cause. No later finite/plastic/contact/MPC/cap/Abaqus or physical validation.')
    output = FINAL/'results/output'/('coherent_residual_'+tag+'.json')
    assert not output.exists()
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(comparisons))

if __name__ == '__main__':
    import sys
    main(sys.argv[1])

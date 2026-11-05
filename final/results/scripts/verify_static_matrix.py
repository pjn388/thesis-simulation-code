"""Compare first native matrices at verified common zero-state geometry."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from verify_shared_node_capture import FINAL, EXE, coupon
from pipe_sim.runner import sha256


def run(tag):
    capture = Path(__file__).with_name('capture_static_matrix_gdb.py')
    source = FINAL/'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f'
    assert sha256(EXE) == json.loads((EXE.parent/'build.json').read_text())['solver_sha256']
    root = FINAL/'sym/runs/investigation'/('static_matrix_'+tag)
    root.mkdir(exist_ok=False)
    cases, comparisons = [], []
    for side in ('minus', 'plus'):
        states = {}
        for pressure in (0, 2):
            assert json.loads((FINAL/'results/automation/paused.json').read_text())['state'] == 'resumed'
            folder = root/f'{side}_p{pressure}'
            folder.mkdir()
            deck, nodes, elements, _, _ = coupon(side, False, .04)
            if pressure == 0:
                for e in elements:
                    deck = deck.replace(f'{e},P{2 if side == "plus" else 1},2\n',
                                        f'{e},P{2 if side == "plus" else 1},0\n')
            (folder/'job.inp').write_text(deck)
            (folder/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MKL_THREADING_LAYER='GNU',
                CCX_NPROC_RESULTS='1', CCX_NPROC_STIFFNESS='1', CCX_NPROC_EQUATION_SOLVER='1',
                THESIS_MATRIX_SOURCE=str(source), THESIS_MATRIX_OUTPUT=str(folder/'matrix.json'))
            with (folder/'solver.log').open('w') as stream:
                result = subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(EXE),'-i','job'],
                    cwd=folder,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
            assert result.returncode == 0 and 'exited normally' in (folder/'solver.log').read_text()
            data = json.loads((folder/'matrix.json').read_text())['records']
            assert [r['nelem'] for r in data] == [1,2]
            for r in data:
                assert r['connectivity'] == elements[r['nelem']]
                assert np.array_equal(r['coordinates'], [nodes[n] for n in r['connectivity']])
                assert np.max(np.abs(r['displacements'])) == 0
                assert np.max(np.abs(r['initial_stress'])) == 0
                assert r['flags'] == dict(nmethod=1,mass=0,stiffness=1,rhsi=1,buckling=0,nload=2,iinc=1)
                assert r['time'] == .0625
                assert r['xload'][0] == pressure*.0625
                assert np.isfinite(r['s']).all() and np.linalg.norm(r['s']) > 0
            states[pressure] = data
            cases.append(dict(run=str(folder.relative_to(FINAL)),
                hashes={p.name:sha256(p) for p in (folder/'job.inp',folder/'job.sta',folder/'matrix.json',folder/'solver.log')}))
        for zero, loaded in zip(states[0], states[2]):
            assert zero['coordinates'] == loaded['coordinates']
            a, b = np.array(zero['s']), np.array(loaded['s'])
            comparisons.append(dict(side=side,element=zero['nelem'],
                matrix_max_difference=float(np.max(abs(a-b))),
                matrix_relative_difference=float(np.linalg.norm(a-b)/np.linalg.norm(a)),
                rhs_max_difference=float(np.max(abs(np.array(zero['ff'])-loaded['ff'])))))
    report = dict(cases=cases, comparisons=comparisons,
        limitation='First element assembly at zero displacement and stress only, two elastic C3D20R coupons. Element matrix before constraints; no global residual derivative, geometric stress stiffness, contact, MPC, pipe, cap or Abaqus acceptance. ABI checked against connectivity, coordinates, flags and time. Equality does not establish a solver defect.',
        provenance={str(p.relative_to(FINAL)):sha256(p) for p in (Path(__file__),capture,source,EXE,Path(__file__).with_name('verify_shared_node_capture.py'))})
    output = FINAL/'results/output'/('static_matrix_'+tag+'.json')
    assert not output.exists()
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(comparisons))


if __name__ == '__main__':
    import sys
    run(sys.argv[1])

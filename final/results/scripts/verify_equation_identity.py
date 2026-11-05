"""Compare actual first PARDISO matrices at verified common zero-state geometry."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from verify_shared_node_capture import FINAL, EXE, coupon
from pipe_sim.runner import sha256


def run(tag):
    capture = Path(__file__).with_name('capture_equation_identity_gdb.py')
    source = FINAL/'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f'
    assert sha256(EXE) == json.loads((EXE.parent/'build.json').read_text())['solver_sha256']
    root = FINAL/'sym/runs/investigation'/('equation_identity_'+tag)
    root.mkdir(exist_ok=False)
    cases, comparisons = [], []
    for side in ('minus', 'plus'):
        states = {}
        for pressure in (0, 2):
            assert json.loads((FINAL/'results/automation/paused.json').read_text())['state'] == 'resumed'
            folder = root/f'{side}_p{pressure}'
            folder.mkdir()
            deck, nodes, elements, _, pins = coupon(side, False, .04)
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
            assert result.returncode == 0
            global_data = json.loads((folder/'matrix.json').read_text())['global_matrix']
            assert global_data['neq'] == 57
            assert np.isfinite(global_data['ad']).all() and np.isfinite(global_data['au']).all()
            assert global_data['jq'][0] == 1 and global_data['jq'][-1] == global_data['nzs']+1
            assert all(1 <= i <= 57 for i in global_data['irow'])
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
            residual = json.loads((folder/'matrix.json').read_text())['residual']
            assert residual['nmethod'] == 1 and residual['neq'] == [57,57]
            assert residual['nk'] == len(nodes) and residual['mi'][1] == 3
            mapping = np.array(residual['mapping']).reshape(len(nodes),4)
            assert np.max(abs(np.array(residual['displacement']))) == 0
            active = mapping[:,1:]
            equations = active[active>0]
            assert sorted(equations.tolist()) == list(range(1,58))
            assert all(np.all(active[n-1] <= 0) for n in pins)
            assert all(np.all(active[n-1] > 0) for n in nodes if n not in pins)
            external = np.zeros(57)
            assembled = np.zeros((57,57))
            for r in data:
                eq = active[np.array(r['connectivity'])-1].ravel()
                local = np.array(r['s']).reshape((60,60),order='F')
                # e_c3d stores the upper triangle; mirror only that triangle.
                local = np.triu(local)+np.triu(local,1).T
                ff = np.array(r['ff'])
                for i, e in enumerate(eq):
                    if e > 0:
                        external[e-1] += ff[i]
                        for j, f in enumerate(eq):
                            if f > 0:
                                assembled[e-1,f-1] += local[i,j]
            actual = np.diag(global_data['ad'])
            for col in range(57):
                for k in range(global_data['jq'][col]-1,global_data['jq'][col+1]-1):
                    row = global_data['irow'][k]-1
                    actual[row,col] = global_data['au'][k]
                    actual[col,row] = global_data['au'][k]
            matrix_error = float(np.max(abs(actual-assembled)))
            assert matrix_error < 1e-9, matrix_error
            assert np.max(abs(external-residual['fext'])) < 1e-10
            assert np.max(abs(np.array(residual['internal']))) == 0
            assert np.max(abs(np.array(residual['residual'])-(external-np.array(residual['internal'])))) < 1e-10
            assert np.max(abs(np.array(global_data['rhs'])-residual['residual'])) < 1e-10
            comparisons.append(dict(side=side,pressure=pressure,kind='equation_residual_assembly_identity',
                matrix_max_error=matrix_error,external_max_error=float(np.max(abs(external-residual['fext']))),
                active_equations=57, internal_force_max=0.0))
            states[pressure] = data
            states['global'+str(pressure)] = global_data
            cases.append(dict(run=str(folder.relative_to(FINAL)),
                hashes={p.name:sha256(p) for p in (folder/'job.inp',folder/'matrix.json',folder/'solver.log')}))
        a, b = states['global0'], states['global2']
        assert all(a[k] == b[k] for k in ('neq','nzs','irow','jq','symmetryflag','inputformat'))
        comparisons.append(dict(side=side,kind='actual_pardiso_entry',neq=a['neq'],nzs=a['nzs'],
            symmetryflag=a['symmetryflag'], inputformat=a['inputformat'],
            diagonal_max_difference=float(np.max(abs(np.array(a['ad'])-b['ad']))),
            off_diagonal_max_difference=float(np.max(abs(np.array(a['au'])-b['au']))),
            rhs_max_difference=float(np.max(abs(np.array(a['rhs'])-b['rhs'])))))
        for zero, loaded in zip(states[0], states[2]):
            assert zero['coordinates'] == loaded['coordinates']
            a, b = np.array(zero['s']), np.array(loaded['s'])
            comparisons.append(dict(side=side,element=zero['nelem'],
                matrix_max_difference=float(np.max(abs(a-b))),
                matrix_relative_difference=float(np.linalg.norm(a-b)/np.linalg.norm(a)),
                rhs_max_difference=float(np.max(abs(np.array(zero['ff'])-loaded['ff'])))))
    report = dict(cases=cases, comparisons=comparisons,
        limitation='Native equation mapping, static residual sign and element-to-global assembly identity at zero displacement/stress on two constrained elastic C3D20R coupons. Probes intentionally terminated before factorization/equilibrium; no converged response or campaign completion. Element connectivity/geometry/flags and sparse dimensions/index ranges checked. Internal force/residual derivative, later states, contact/MPC/pipe/cap/Abaqus and production convergence causation remain unverified. Equality does not establish a solver defect.',
        provenance={str(p.relative_to(FINAL)):sha256(p) for p in (Path(__file__),capture,source,EXE,Path(__file__).with_name('verify_shared_node_capture.py'))})
    output = FINAL/'results/output'/('equation_identity_'+tag+'.json')
    assert not output.exists()
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(comparisons))


if __name__ == '__main__':
    import sys
    run(sys.argv[1])

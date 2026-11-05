"""Native external-load derivatives at common zero-stress coupon geometry.

Assembly probes are deliberately terminated after two element returns. They
are not converged solves and are never included in campaign completion.
"""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from verify_shared_node_capture import FINAL, EXE, coupon
from audit_nodal_pressure_loads import nodal_loads
from pipe_sim.runner import sha256


def run(tag):
    capture = Path(__file__).with_name('capture_pressure_derivative_gdb.py')
    source = FINAL/'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f'
    parent = FINAL/'results/output/static_matrix_v2_20261012T063002Z.json'
    assert sha256(EXE) == json.loads((EXE.parent/'build.json').read_text())['solver_sha256']
    root = FINAL/'sym/runs/investigation'/('pressure_derivative_'+tag)
    root.mkdir(exist_ok=False)
    cases, comparisons = [], []
    for side in ('minus', 'plus'):
        deck, nodes, elements, face, pins = coupon(side, False, .04)
        free = np.array([3*(n-1)+d for n in nodes if n not in pins for d in range(3)])
        for direction in range(3):
            vector = np.zeros((len(nodes), 3))
            # Deterministic smooth fields, zero on all constrained nodes.
            for n, (x, y, z) in nodes.items():
                if n not in pins:
                    vector[n-1, direction] = (1+x/8)*(1+y/6)
            vector /= np.linalg.norm(vector)
            derivatives = []
            for h in (1e-4, 5e-5):
                values = []
                for sign in (-1, 1):
                    assert json.loads((FINAL/'results/automation/paused.json').read_text())['state'] == 'resumed'
                    folder = root/f'{side}_d{direction}_h{h}_{sign}'
                    folder.mkdir()
                    imposed = sign*h*vector
                    (folder/'job.inp').write_text(deck)
                    (folder/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
                    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MKL_THREADING_LAYER='GNU',
                        CCX_NPROC_RESULTS='1', CCX_NPROC_STIFFNESS='1', CCX_NPROC_EQUATION_SOLVER='1',
                        THESIS_MATRIX_SOURCE=str(source), THESIS_MATRIX_OUTPUT=str(folder/'matrix.json'),
                        THESIS_IMPOSED_DISPLACEMENT=json.dumps({n:imposed[n-1].tolist() for n in nodes}))
                    with (folder/'solver.log').open('w') as stream:
                        result = subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(EXE),'-i','job'],
                            cwd=folder, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=30)
                    assert result.returncode == 0, folder
                    records = json.loads((folder/'matrix.json').read_text())['records']
                    assert [r['nelem'] for r in records] == [1, 2]
                    assembled, expected = np.zeros_like(vector), np.zeros_like(vector)
                    for r in records:
                        conn = elements[r['nelem']]
                        assert r['connectivity'] == conn
                        assert np.array_equal(r['coordinates'], [nodes[n] for n in conn])
                        assert np.array_equal(r['displacements'], imposed[np.array(conn)-1])
                        assert np.max(np.abs(r['initial_stress'])) == 0
                        assert r['time'] == .0625 and r['xload'][0] == .125
                        assert r['flags'] == dict(nmethod=1,mass=0,stiffness=1,rhsi=1,buckling=0,nload=2,iinc=1)
                        xyz = np.array(r['coordinates'])+np.array(r['displacements'])
                        local = np.zeros((20, 3))
                        local[face] = nodal_loads(xyz[face], -.125)
                        native = np.array(r['ff']).reshape(20, 3)
                        assert np.max(abs(local-native)) < 1e-9
                        for n, a, b in zip(conn, native, local):
                            assembled[n-1] += a
                            expected[n-1] += b
                    values.append((assembled, expected))
                    cases.append(dict(run=str(folder.relative_to(FINAL)),state='assembly_probe_terminated',
                        maximum_load_error_n=float(np.max(abs(assembled-expected))),
                        hashes={p.name:sha256(p) for p in (folder/'job.inp', folder/'matrix.json', folder/'solver.log')}))
                native = (values[1][0]-values[0][0])/(2*h)
                independent = (values[1][1]-values[0][1])/(2*h)
                assert np.max(abs(native-independent)) < 1e-7
                derivatives.append(native.ravel()[free])
            difference = float(np.max(abs(derivatives[0]-derivatives[1])))
            assert difference < 1e-7
            comparisons.append(dict(side=side,direction=direction,free_dofs=len(free),
                derivative_max_n_per_mm=float(np.max(abs(derivatives[1]))),
                derivative_norm_n_per_mm=float(np.linalg.norm(derivatives[1])),
                half_step_difference_n_per_mm=difference))
    report = dict(cases=cases,comparisons=comparisons,
        limitation='External pressure-load directional derivatives only at imposed zero-stress initial geometry. Shared nodes aggregated and homogeneous constraints projected. Native probes terminated before equilibrium; no accepted response/PEEQ/burst. Not the full constrained residual derivative or captured global solver matrix; internal force derivative and alternative assembly paths remain unverified. No production solver/load or Abaqus revision.',
        provenance={str(p.relative_to(FINAL)):sha256(p) for p in (Path(__file__),capture,source,EXE,parent,Path(__file__).with_name('audit_nodal_pressure_loads.py'))})
    output = FINAL/'results/output'/('pressure_derivative_'+tag+'.json')
    assert not output.exists()
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(comparisons))


if __name__ == '__main__':
    import sys
    run(sys.argv[1])

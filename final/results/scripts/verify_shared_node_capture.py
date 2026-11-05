"""Bounded two-element load aggregation diagnostic; independent service only."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from pipe_sim.results import dat_blocks
from pipe_sim.runner import sha256
from audit_nodal_pressure_loads import nodal_loads
from audit_pressure_failures import converged_times, matches_sta

FINAL = Path(__file__).resolve().parents[2]
EXE = FINAL/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
CAPTURE = FINAL/'results/scripts/capture_native_element_loads_gdb.py'
LOCAL = [(0,0,0),(4,0,0),(4,6,0),(0,6,0),(0,0,2),(4,0,2),(4,6,2),(0,6,2),
         (2,0,0),(4,3,0),(2,6,0),(0,3,0),(2,0,2),(4,3,2),(2,6,2),(0,3,2),
         (0,0,1),(4,0,1),(4,6,1),(0,6,1)]


def coupon(side, fixed, curvature):
    nodes, lookup, elements = {}, {}, {}
    for e, shift in [(1,0),(2,4)]:
        conn = []
        for x,y,z in LOCAL:
            key = (x+shift,y,z)
            if key not in lookup:
                n = len(lookup)+1
                lookup[key] = n
                nodes[n] = (x+shift,y,z+curvature*(x+shift)*(8-x-shift))
            conn.append(lookup[key])
        elements[e] = conn
    assert len(nodes) == 32 and len(set(elements[1]) & set(elements[2])) == 8
    face = [4,5,6,7,12,13,14,15] if side == 'plus' else [0,3,2,1,11,10,9,8]
    opposite = [0,1,2,3,8,9,10,11] if side == 'plus' else [4,5,6,7,12,13,14,15]
    pins = set(nodes) if fixed else {conn[i] for conn in elements.values() for i in opposite}
    lines = ['*HEADING','Shared-node pressure diagnostic: mm N MPa','*NODE']
    lines += [f'{n},'+','.join(map(str,p)) for n,p in nodes.items()]
    lines += ['*ELEMENT,TYPE=C3D20R,ELSET=SOLID']
    for e,conn in elements.items():
        lines += [f'{e},'+','.join(map(str,conn[:15]))+',', ','.join(map(str,conn[15:]))]
    lines += ['*NSET,NSET=ALL']
    ids = list(nodes)
    lines += [','.join(map(str,ids[i:i+16])) for i in range(0,len(ids),16)]
    lines += ['*MATERIAL,NAME=ELASTIC','*ELASTIC','210,0.3',
              '*SOLID SECTION,ELSET=SOLID,MATERIAL=ELASTIC','*BOUNDARY']
    lines += [f'{n},1,3,0' for n in sorted(pins)]
    lines += ['*STEP,NLGEOM,INC=100','*STATIC,SOLVER=PARDISO','0.0625,1,1e-8,0.0625','*DLOAD']
    lines += [f'{e},P{2 if side == "plus" else 1},2' for e in elements]
    lines += ['*NODE PRINT,NSET=ALL','U,RF','*END STEP']
    return '\n'.join(lines)+'\n', nodes, elements, face, pins


def main(tag):
    root = FINAL/'sym/runs/investigation'/tag
    assert json.loads((FINAL/'results/automation/paused.json').read_text())['state'] == 'resumed'
    assert sha256(EXE) == json.loads((EXE.parent/'build.json').read_text())['solver_sha256']
    root.mkdir(exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MKL_THREADING_LAYER='GNU',
               CCX_NPROC_RESULTS='1', CCX_NPROC_STIFFNESS='1', CCX_NPROC_EQUATION_SOLVER='1')
    cases = []
    for side in ['minus','plus']:
        for curvature in [0,0.04]:
            for fixed in [True,False]:
                assert json.loads((FINAL/'results/automation/paused.json').read_text())['state'] == 'resumed'
                run = root/f'{side}_{curvature}_{"fixed" if fixed else "free"}'
                run.mkdir()
                deck,nodes,elements,face,pins = coupon(side,fixed,curvature)
                (run/'job.inp').write_text(deck)
                (run/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(CAPTURE)+'\nquit\n')
                env['THESIS_CAPTURE_OUTPUT'] = str(run/'capture.json')
                with (run/'solver.log').open('w') as stream:
                    result = subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(EXE),'-i','job'],
                                            cwd=run,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=90)
                assert result.returncode == 0 and 'exited normally' in (run/'solver.log').read_text()
                records = json.loads((run/'capture.json').read_text())['records']
                errors = []
                batches = []
                for i in range(0,len(records),2):
                    pair = records[i:i+2]
                    assert [r['nelem'] for r in pair] == [1,2]
                    assert len({(r['routine'],r['istep'],r['iinc'],r['time'],r['dtime']) for r in pair}) == 1
                    assembled = {n:np.zeros(3) for n in nodes}
                    expected = {n:np.zeros(3) for n in nodes}
                    geometry = {}
                    for r in pair:
                        assert r['connectivity'] == elements[r['nelem']]
                        assert r['routine'] == 'e_c3d_'  # RHS remains an explicit unverified gate.
                        assert abs(r['xload'][0]-2*r['time']) < 1e-10
                        native = np.array(r['ff']).reshape(20,3)
                        current = np.array(r['coordinates'])+np.array(r['displacements_at_assembly'])
                        local_expected = np.zeros((20,3))
                        local_expected[face] = nodal_loads(current[face],-r['xload'][0])
                        errors.append(float(np.max(abs(native-local_expected))))
                        for n,p,v,w in zip(r['connectivity'],current,native,local_expected):
                            if n in geometry:
                                assert np.max(abs(geometry[n]-p)) < 1e-12
                            geometry[n] = p
                            assembled[n] += v
                            expected[n] += w
                    errors.append(max(float(np.max(abs(assembled[n]-expected[n]))) for n in nodes))
                    batches.append((pair[0],assembled))
                assert max(errors) < 1e-9
                accepted = converged_times(run/'job.sta')
                frames = {}
                for kind,group,time,rows in dat_blocks(run/'job.dat'):
                    assert matches_sta(accepted,time)
                    frames.setdefault(time,{})[kind] = rows
                assert abs(max(frames)-1) < 1e-8
                history = []
                for time,blocks in sorted(frames.items()):
                    candidates = [(r,a) for r,a in batches if abs(r['time']-time)<1e-7]
                    assert candidates
                    r,assembled = candidates[-1]
                    rf = {int(row[0]):np.array(row[1:]) for row in blocks['RF']}
                    u = {int(row[0]):np.array(row[1:]) for row in blocks['U']}
                    assert set(rf) == set(u) == set(nodes)
                    assert max(float(np.max(abs(u[n]))) for n in pins) == 0
                    free_error = max((float(np.max(abs(rf[n]-assembled[n]))) for n in nodes if n not in pins),default=0)
                    scale = max(float(np.max(abs(v))) for v in assembled.values())
                    assert free_error < 1e-4 and free_error/scale < 2e-5, (run.name,time,free_error)
                    history.append(dict(time=time,free_rf_error_n=free_error,assembly_pairs=len(candidates),
                                        resultant_n=sum(assembled.values(),np.zeros(3)).tolist()))
                cases.append(dict(run=str(run.relative_to(FINAL)),fixed=fixed,curvature=curvature,side=side,
                                  captures=len(records),assembly_pairs=len(batches),accepted_frames=len(frames),
                                  maximum_load_error_n=max(errors),history=history,
                                  hashes={n:sha256(run/n) for n in ['job.inp','job.sta','job.dat','capture.json','solver.log']}))
    report = dict(cases=cases,solver_sha256=sha256(EXE),capture_sha256=sha256(CAPTURE),script_sha256=sha256(__file__),
                  sources=['sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f:19-38,131,333','sym/vendor/CalculiX/ccx_2.23/src/mafillsm.f:588-612'],
                  limitations='Two C3D20R elements, serial assembly, no MPC/contact/RHS. Pair identity verified by element order, shared geometry and metadata; not a general pipe assembly-cycle identifier. No support RF subtraction, production load revision, cap tangent/work, Abaqus equivalence or physical validation.')
    (root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    output = FINAL/'results/output'/f'{tag}.json'
    assert not output.exists()
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),max_load_error_n=max(c['maximum_load_error_n'] for c in cases),
                          max_free_rf_error_n=max(h['free_rf_error_n'] for c in cases for h in c['history']))))

if __name__ == '__main__':
    import sys
    main(sys.argv[1])

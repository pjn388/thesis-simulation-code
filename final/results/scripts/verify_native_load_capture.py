"""Fresh bounded coupon verification of diagnostic pre-elimination load capture.

Run only as an independent systemd unit. No production-process attachment.
"""
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
TAG = '20261011T000006Z'
ROOT = FINAL/'sym/runs/investigation'/f'native_capture_batch_v4_{TAG}'
SOURCE = FINAL/'sym/runs/investigation/curved_dt16_20261010T230002Z'
EXE = FINAL/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
CAPTURE = FINAL/'results/scripts/capture_native_element_loads_gdb.py'


def main():
    pause = FINAL/'results/automation/paused.json'
    assert json.loads(pause.read_text())['state'] == 'resumed'
    assert sha256(EXE) == json.loads((EXE.parent/'build.json').read_text())['solver_sha256']
    ROOT.mkdir(exist_ok=False)
    records = []
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MKL_THREADING_LAYER='GNU',
               CCX_NPROC_RESULTS='1', CCX_NPROC_STIFFNESS='1', CCX_NPROC_EQUATION_SOLVER='1')
    for source in sorted(SOURCE.iterdir()):
        if not source.is_dir():
            continue
        for fixed in (False, True):
            assert json.loads(pause.read_text())['state'] == 'resumed'
            run = ROOT/(source.name+('_fixed' if fixed else '_free'))
            run.mkdir()
            deck = (source/'job.inp').read_text()
            if fixed:
                begin = deck.index('*BOUNDARY')
                end = deck.index('*STEP', begin)
                deck = deck[:begin]+'*BOUNDARY\n'+''.join(f'{n},1,3,0\n' for n in range(1,21))+deck[end:]
            (run/'job.inp').write_text(deck)
            (run/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(CAPTURE)+'\nquit\n')
            env['THESIS_CAPTURE_OUTPUT'] = str(run/'capture.json')
            with (run/'solver.log').open('w') as log:
                result = subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(EXE),'-i','job'],
                                        cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
            assert result.returncode == 0 and 'exited normally' in (run/'solver.log').read_text()
            captures = json.loads((run/'capture.json').read_text())['records']
            assert captures
            side = source.name.split('_')[0]
            face = [5,6,7,8,13,14,15,16] if side == 'plus' else [1,4,3,2,12,11,10,9]
            errors = []
            rhs_count = 0
            for record in captures:
                assert record['connectivity'] == list(range(1,21)) and record['nelem'] == 1
                points = np.array(record['coordinates'])+np.array(record['displacements_at_assembly'])
                pressure = record['xload'][0]
                assert abs(pressure-2*record['time']) < 1e-10
                expected = np.zeros((20,3))
                expected[np.array(face)-1] = nodal_loads(points[np.array(face)-1],-pressure)
                native = np.array(record['ff']).reshape(20,3)
                error = float(np.max(np.abs(native-expected)))
                assert error < 1e-9, (source.name,record['routine'],error)
                errors.append(error)
                rhs_count += record['routine'] == 'e_c3d_rhs_'
            accepted = converged_times(run/'job.sta')
            frames = {}
            for kind, group, time, rows in dat_blocks(run/'job.dat'):
                assert matches_sta(accepted,time)
                frames.setdefault(time,{})[kind] = rows
            assert abs(max(frames)-1) < 1e-8
            last = captures[-1]
            records.append(dict(run=str(run.relative_to(FINAL)),source=str(source.relative_to(FINAL)),
                                fixed=fixed,captures=len(captures),rhs_captures=rhs_count,
                                accepted_frames=len(frames),max_all_component_error_n=max(errors),
                                final_resultant_n=np.array(last['ff']).reshape(20,3).sum(axis=0).tolist(),
                                max_assembly_displacement_mm=float(np.max(np.abs([r['displacements_at_assembly'] for r in captures]))),
                                hashes={n:sha256(run/n) for n in ['job.inp','capture.json','job.dat','job.sta','solver.log']},
                                source_deck_sha256=sha256(source/'job.inp')))
    report = dict(records=records,solver_sha256=sha256(EXE),capture_script_sha256=sha256(CAPTURE),
                  verification_script_sha256=sha256(__file__),
                  sources=['sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f:19-38','sym/vendor/CalculiX/ccx_2.23/src/e_c3d_rhs.f:19-31','sym/vendor/CalculiX/ccx_2.23/src/mafillsm.f:588-612','sym/vendor/CalculiX/ccx_2.23/src/rhs.f:233-272'],
                  limitations='Single-element x86-64 SysV diagnostic only, one stiffness/RHS thread. Captures every assembly, not a pressure-frame mapping. No multi-element summation, MPC transfer, support RF subtraction, coupled cap tangent/work, Abaqus equivalence, production load revision or physical validation.')
    (ROOT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (FINAL/'results/output'/f'native_capture_verification_{TAG}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases=len(records),max_error_n=max(r['max_all_component_error_n'] for r in records))))


if __name__ == '__main__':
    main()

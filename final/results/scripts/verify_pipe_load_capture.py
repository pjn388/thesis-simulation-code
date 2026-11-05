"""Fresh serial elastic-pipe assembly diagnostic; preserve production/native parents."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from pipe_sim.model import Model
from pipe_sim.results import dat_blocks
from pipe_sim.runner import sha256
from audit_end_surface_traction import face_nodes
from audit_nodal_pressure_loads import nodal_loads
from audit_pressure_failures import converged_times, matches_sta

FINAL = Path(__file__).resolve().parents[2]
EXE = FINAL/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
CAPTURE = FINAL/'results/scripts/capture_native_element_loads_gdb.py'
SOURCE = FINAL/'sym/runs/pressure_restraints_allip_20261008T160000Z'


def main(tag, plastic=False, strict_field=False, max_increment=None, pressure_mpa=20, field_residual=1e-8, field_correction=1e-6):
    if Path(tag).name != tag:
        raise ValueError('Use a new directory name')
    if (strict_field or max_increment is not None) and not plastic:
        raise ValueError('Sensitivity options require the bounded20MPa diagnostic')
    if max_increment is not None and not 0 < max_increment <= .05:
        raise ValueError('Diagnostic maximum increment must be in (0, .05]')
    if not 0 < pressure_mpa <= 24 or (pressure_mpa != 20 and not (plastic and strict_field)):
        raise ValueError('Pressure extensions require passed FIELD controls and at most24MPa')
    if field_residual not in (1e-8, 1e-7, 1e-6) or (field_residual != 1e-8 and not strict_field):
        raise ValueError('Residual sensitivity requires strict FIELD mode and a supported discrete tolerance')
    if field_correction not in (1e-6, 1e-5, 1e-4) or (field_correction != 1e-6 and not strict_field):
        raise ValueError('Correction sensitivity requires strict FIELD mode and a bounded discrete tolerance')
    root = FINAL/'sym/runs/investigation'/tag
    root.mkdir(exist_ok=False)
    assert sha256(EXE) == json.loads((EXE.parent/'build.json').read_text())['solver_sha256']
    cases = []
    names = ['intact_support_bounded_20261010T170002Z'] if plastic else ['bottom','tangential']
    for name in names:
        assert json.loads((FINAL/'results/automation/paused.json').read_text())['state'] == 'resumed'
        source = (FINAL/'sym/runs/investigation'/name) if plastic else SOURCE/name
        run = root/name
        run.mkdir()
        for f in ['job.inp','mesh.inp','model.json','case.json']:
            shutil.copyfile(source/f,run/f)
        deck = (run/'job.inp').read_text()
        if plastic:
            original = deck.split('*DLOAD\n')[1].split('*')[0]
            scaled = ''.join(','.join(line.split(',')[:2])+f', {float(line.split(",")[2])*pressure_mpa/35:.12g}\n' for line in original.strip().splitlines())
            deck = deck.replace(original,scaled).replace('*END STEP','*NODE PRINT, NSET=PIPE_NODES, FREQUENCY=1\nU, RF\n*END STEP')
            if strict_field:
                # CalculiX node251.html convergence controls; diagnostic only.
                deck = deck.replace('*STATIC, SOLVER=PARDISO', f'*CONTROLS, PARAMETERS=FIELD\n{field_residual:.12g},{field_correction:.12g},,,1e-8,1e-5,1e-6,1e-12\n*STATIC, SOLVER=PARDISO')
            if max_increment is not None:
                assert '0.025, 1., 1e-06, 0.05' in deck
                deck = deck.replace('0.025, 1., 1e-06, 0.05', f'0.025, 1., 1e-06, {max_increment:.12g}')
            (run/'job.inp').write_text(deck)
            case = json.loads((run/'case.json').read_text())
            assert case['analysis']['max_pressure'] == 35
            case['analysis']['max_pressure'] = pressure_mpa
            (run/'case.json').write_text(json.dumps(case,indent=2)+'\n')
            (run/'diagnostic_revision.json').write_text(json.dumps(dict(parent=str(source.relative_to(FINAL)),comparison_parent='sym/runs/investigation/plastic_pipe_capture_20261011T003000Z/'+name,dload_scale=pressure_mpa/35,pressure_mpa=pressure_mpa,added_output='PIPE_NODES U/RF',strict_field=strict_field,field_residual=field_residual,field_correction=field_correction,max_increment=max_increment,serial_capture=True,native_timeout_seconds=900,production_revision=False),indent=2)+'\n')
        assert not any(k in deck.upper() for k in ['*MPC','*EQUATION','*CONTACT','*CLOAD'])
        model = Model.load(run/'model.json')
        pressure_faces = {}
        for line in deck.split('*DLOAD\n')[1].split('*')[0].strip().splitlines():
            e,f,p = [x.strip() for x in line.split(',')]
            pressure_faces.setdefault(int(e),[]).append((int(f[1:]),float(p)))
        pins = set()
        for line in deck.split('*BOUNDARY\n')[1].split('*')[0].strip().splitlines():
            n,first,last,value = [x.strip() for x in line.split(',')]
            assert float(value) == 0
            ids = [int(n)] if n.isdigit() else model.nsets[n]
            pins.update((i,d-1) for i in ids for d in range(int(first),int(last)+1))
        (run/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(CAPTURE)+'\nquit\n')
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MKL_THREADING_LAYER='GNU',
                   CCX_NPROC_RESULTS='1', CCX_NPROC_STIFFNESS='1', CCX_NPROC_EQUATION_SOLVER='1',
                   THESIS_CAPTURE_OUTPUT=str(run/'capture.json'),THESIS_CAPTURE_JSONL=str(run/'capture.jsonl'))
        with (run/'solver.log').open('w') as stream:
            result = subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(EXE),'-i','job'],
                                    cwd=run,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=900 if plastic else 600)
        assert result.returncode == 0 and 'exited normally' in (run/'solver.log').read_text()
        accepted = converged_times(run/'job.sta')
        frames = {}
        for kind,group,time,rows in dat_blocks(run/'job.dat'):
            if kind not in ['U','RF'] or group != ('PIPE_NODES' if plastic else 'BENCH'):
                continue
            assert matches_sta(accepted,time)
            frames.setdefault(time,{})[kind] = rows
        assert abs(max(frames)-1) < 1e-8
        last_batches = {}
        element_ids = sorted(model.elements)
        assembled = {}; geometry = {}; errors = []; cycles = 0; seen = []; metadata = None
        with (run/'capture.jsonl').open() as stream:
            for line in stream:
                r = json.loads(line)
                key = (r['routine'],r['istep'],r['iinc'],r['time'],r['dtime'])
                if not seen:
                    metadata = key
                    assembled = {n:np.zeros(3) for n in model.nodes}
                    geometry = {}
                assert key == metadata and r['routine'] == 'e_c3d_'
                e = r['nelem']; seen.append(e)
                conn = r['connectivity']
                assert conn == model.elements[e][1]
                current = np.array(r['coordinates'])+np.array(r['displacements_at_assembly'])
                native = np.array(r['ff']).reshape(20,3)
                expected = np.zeros((20,3))
                for f,p in pressure_faces.get(e,[]):
                    indices = [conn.index(n) for n in face_nodes(model,e,f)]
                    expected[indices] += nodal_loads(current[indices],-p*r['time'])
                error = float(np.max(abs(native-expected)))
                assert error < 1e-8, (name,e,r['time'],error)
                errors.append(error)
                for n,p,v in zip(conn,current,native):
                    if n in geometry:
                        assert np.max(abs(geometry[n]-p)) < 1e-11
                    geometry[n] = p
                    assembled[n] += v
                if len(seen) == len(element_ids):
                    assert seen == element_ids
                    cycles += 1
                    last_batches[r['time']] = assembled
                    seen = []
        assert not seen
        history = []
        for time,blocks in sorted(frames.items()):
            matching = [t for t in last_batches if abs(t-time)<1e-7]
            assert len(matching) == 1
            a = last_batches[matching[0]]
            rf = {int(row[0]):np.array(row[1:]) for row in blocks['RF']}
            u = {int(row[0]):np.array(row[1:]) for row in blocks['U']}
            assert set(rf) == set(u) == set(model.nodes)
            assert max(abs(u[n][d]) for n,d in pins) == 0
            free_error = max(abs(rf[n][d]-a[n][d]) for n in model.nodes for d in range(3) if (n,d) not in pins)
            scale = max(float(np.max(abs(v))) for v in a.values())
            within_tolerance = free_error < 1e-4 and free_error/scale < 2e-5
            if not plastic:
                assert within_tolerance, (name,time,free_error,scale)
            reaction = np.zeros(3)
            for n,d in pins:
                reaction[d] += rf[n][d]-a[n][d]
            net = sum(a.values(),np.zeros(3))
            history.append(dict(time=time,max_free_rf_error_n=float(free_error),within_retained_coupon_tolerance=bool(within_tolerance),net_applied_n=net.tolist(),
                                reconstructed_support_reaction_n=reaction.tolist(),equilibrium_residual_n=(net+reaction).tolist()))
        parent_comparison = []
        if plastic:
            import csv
            parent = list(csv.DictReader((FINAL/'sym/runs/campaign/orientation_intact_00_adaptive/results/history.csv').open()))
            parent = [r for r in parent if r['stage']=='pressurise']
            crown = model.nsets['CROWN'][0]
            for time,blocks in sorted(frames.items()):
                u = {int(row[0]):np.array(row[1:]) for row in blocks['U']}
                pressure = pressure_mpa*time
                covered = float(parent[0]['pressure_mpa'])<=pressure<=float(parent[-1]['pressure_mpa'])
                parent_u = np.interp(pressure,[float(r['pressure_mpa']) for r in parent],[float(r['crown_inward_displacement_mm']) for r in parent]) if covered else None
                parent_comparison.append(dict(pressure_mpa=pressure,crown_inward_mm=-float(u[crown][1]),parent_interpolated_crown_mm=parent_u,crown_difference_mm=None if parent_u is None else -float(u[crown][1])-parent_u))
        cases.append(dict(parent_comparison=parent_comparison,run=str(run.relative_to(FINAL)),source=str(source.relative_to(FINAL)),
                          elements=len(element_ids),assembly_cycles=cycles,accepted_frames=len(frames),
                          maximum_local_load_error_n=max(errors),history=history,
                          input_identical=all(sha256(run/f)==sha256(source/f) for f in ['job.inp','mesh.inp','model.json','case.json']),
                          hashes={f:sha256(run/f) for f in ['job.inp','mesh.inp','job.dat','job.sta','capture.jsonl','solver.log']}))
        (root/f'{name}_audit.json').write_text(json.dumps(cases[-1],indent=2)+'\n')
    report = dict(cases=cases,solver_sha256=sha256(EXE),capture_script_sha256=sha256(CAPTURE),script_sha256=sha256(__file__),
                  strict_field=strict_field,field_residual=field_residual,field_correction=field_correction,max_increment=max_increment,pressure_mpa=pressure_mpa,
                  limitations=('Bounded intact pressure diagnostic (at most24MPa) with retained plastic material; classify actual PEEQ extent separately. Original cap convention preserved, not validated. Printed RF rounding audit required; raw absolute comparisons are not the full verdict. ' if plastic else 'Unchanged elastic 1MPa diagnostic decks. ')+'Serial assembly; no contact, MPC, RHS, general assembly identifier, cap tangent/work, Abaqus equivalence or burst validation. No general support-transfer acceptance inferred.')
    (root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    (FINAL/'results/output'/f'{tag}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),max_load_error_n=max(c['maximum_local_load_error_n'] for c in cases),
                          max_free_rf_error_n=max(h['max_free_rf_error_n'] for c in cases for h in c['history']))))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tag')
    parser.add_argument('--plastic20', action='store_true')
    parser.add_argument('--strict-field', action='store_true')
    parser.add_argument('--max-increment', type=float)
    parser.add_argument('--field-correction', type=float, default=1e-6)
    parser.add_argument('--field-residual', type=float, default=1e-8)
    parser.add_argument('--pressure-mpa', type=float, default=20)
    args = parser.parse_args()
    main(args.tag,plastic=args.plastic20,strict_field=args.strict_field,max_increment=args.max_increment,pressure_mpa=args.pressure_mpa,field_residual=args.field_residual,field_correction=args.field_correction)

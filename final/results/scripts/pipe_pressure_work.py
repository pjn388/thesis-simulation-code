"""Isolated elastic pipe work/energy benchmark; campaign decks are immutable."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from pipe_sim.model import Model
from pipe_sim.results import dat_blocks
from pipe_sim.runner import run, sha256
from audit_nodal_pressure_loads import nodal_loads
from audit_end_surface_traction import face_nodes
from audit_pressure_failures import converged_times, matches_sta

FINAL = Path(__file__).resolve().parents[2]


def benchmark(name):
    if Path(name).name != name or name in {'.', '..'}:
        raise ValueError('Use a new directory name')
    root = FINAL/'sym/runs/investigation'/name
    root.mkdir(exist_ok=False)
    records = []
    for mesh, parent_name in [('coarse', 'pressure_restraints_20261008T143009Z'),
                              ('refined', 'pressure_restraints_refined_20261008T150008Z')]:
        parent = FINAL/'sym/runs'/parent_name/'tangential'
        manifest = json.loads((parent/'manifest.json').read_text())
        assert all(sha256(parent/f) == h for f, h in manifest['files'].items())
        assert json.loads((parent/'status.json').read_text())['state'] == 'completed'
        model = Model.load(parent/'model.json')
        case = json.loads((parent/'case.json').read_text())
        assert not case['material']['plastic']
        ro = case['pipe']['od']/2
        ri = ro-case['pipe']['thickness']
        p = case['analysis']['max_pressure']
        ratio = ri**2/(ro**2-ri**2)
        a = p*ratio
        b = a*ro**2
        nu = case['material']['poisson']
        theory = np.pi*case['pipe']['length']/case['material']['young'] * (
            (3-6*nu)*a*a*(ro*ro-ri*ri)/2 + (1+nu)*b*b*(1/(ri*ri)-1/(ro*ro)))
        original = (parent/'job.inp').read_text()
        lines = original.upper().split('*DLOAD\n')[1].split('*')[0].strip().splitlines()
        actual = {(int(e), int(f.strip()[1:])): float(v) for e, f, v in
                  (line.split(',') for line in lines)}
        expected = {(e, f): p*(1 if s == 'INNER' else -ratio)
                    for s in ['INNER', 'END_MINUS', 'END_PLUS'] for e, f in model.surfaces[s]}
        assert actual.keys() == expected.keys()
        assert all(np.isclose(actual[k], v, rtol=1e-11) for k, v in expected.items())
        faces = [(s, face_nodes(model, e, f), actual[e, f])
                 for s in ['INNER', 'END_MINUS', 'END_PLUS'] for e, f in model.surfaces[s]]
        for dt in [.25, .125]:
            target = root/f'{mesh}_dt_{dt}'
            target.mkdir()
            for f in ['case.json', 'mesh.inp', 'model.json']:
                shutil.copy2(parent/f, target/f)
            settings = json.loads((target/'case.json').read_text())
            settings['analysis'].update(threads=1, timeout_seconds=120)
            (target/'case.json').write_text(json.dumps(settings, indent=2)+'\n')
            deck = original.replace('0.025, 1., 1e-06, 0.1', f'{dt}, 1., 1e-06, {dt}')
            assert deck != original
            deck = deck.replace('*END STEP', '*EL PRINT, ELSET=PIPE, FREQUENCY=1\nELSE\n*END STEP')
            (target/'job.inp').write_text(deck)
            inputs = {f: sha256(target/f) for f in ['job.inp', 'case.json', 'mesh.inp', 'model.json']}
            child_manifest = dict(manifest, files=inputs,
                                  parent=str(parent.relative_to(FINAL)),
                                  parent_manifest_sha256=sha256(parent/'manifest.json'),
                                  benchmark_script_sha256=sha256(__file__),
                                  controlled_change='First-step ELSE and fixed increment; inherited tangential elastic pipe')
            (target/'manifest.json').write_text(json.dumps(child_manifest, indent=2)+'\n')
            (target/'status.json').write_text(json.dumps(dict(state='generated'))+'\n')
            status = run(target, memory_limit_mb=1500)
            assert status['state'] == 'completed', status
            frames = {}
            accepted = converged_times(target/'job.sta')
            for kind, group, time, rows in dat_blocks(target/'job.dat'):
                assert matches_sta(accepted, time)
                if group == 'BENCH' and kind == 'U':
                    frames.setdefault(time, {})['U'] = {int(r[0]): np.array(r[1:]) for r in rows}
                if group == 'PIPE' and kind is None:
                    frames.setdefault(time, {})['energy'] = sum(r[1] for r in rows)
                    assert len(rows) == len(model.elements)
            previous_u = {n: np.zeros(3) for n in model.nodes}
            previous_load = {s: {n: np.zeros(3) for n in model.nodes}
                             for s in ['INNER', 'END_MINUS', 'END_PLUS']}
            work = {s: 0. for s in previous_load}
            history = []
            for time, frame in sorted(frames.items()):
                u = frame['U']
                assert set(u) == set(model.nodes)
                loads = {s: {n: np.zeros(3) for n in model.nodes} for s in work}
                for s, nodes, pressure in faces:
                    values = nodal_loads([np.array(model.nodes[n])+u[n] for n in nodes], -pressure*time)
                    for n, value in zip(nodes, values):
                        loads[s][n] += value
                for s in work:
                    work[s] += sum(float(.5*(loads[s][n]+previous_load[s][n]) @ (u[n]-previous_u[n]))
                                   for n in model.nodes)
                energy = frame['energy']
                error = sum(work.values())/energy-1
                assert abs(error) < 5e-4, (mesh, dt, time, error)
                history.append(dict(time=time, external_work_by_surface_n_mm=dict(work),
                                    internal_energy_n_mm=energy, relative_work_energy_error=error))
                previous_u, previous_load = u, loads
            assert abs(max(frames)-1) < 1e-8
            analytic_error = sum(work.values())/theory-1
            assert abs(analytic_error) < .02, analytic_error
            assert all(sha256(target/f) == h for f, h in inputs.items())
            records.append(dict(run=str(target.relative_to(FINAL)), mesh=mesh, increment=dt,
                                elements=len(model.elements), history=history,
                                analytical_energy_n_mm=theory, relative_analytical_error=analytic_error,
                                input_sha256=inputs, native_sha256={f: sha256(target/f) for f in
                                ['job.dat', 'job.sta', 'job.frd', 'solver.log']},
                                parent=str(parent.relative_to(FINAL)), parent_manifest_sha256=sha256(parent/'manifest.json'),
                                solver_sha256=status['solver_sha256']))
    report = dict(records=records, script_sha256=sha256(__file__),
                  sources=['sym/vendor/CalculiX/ccx_2.23/doc/ccx/node280.html:179-186',
                           'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f:1360-1486,1597-1615'],
                  limitations='Low-pressure elastic tangential-restraint pipe, two surface meshes and two increments; wall layers unchanged. End loads remain reference-area-ratio follower tractions, not verified finite-expansion caps. No plastic continuation, burst, experimental or Abaqus equivalence validation.')
    (root/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    out = FINAL/'results/output/pressure_restraint_candidates'/f'{name}.json'
    assert not out.exists()
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    benchmark(parser.parse_args().name)

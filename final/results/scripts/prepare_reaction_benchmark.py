"""Freeze an output-only intact benchmark; preparation never launches a solver."""
import argparse
import json
import shutil
from pathlib import Path

from pipe_sim.model import Model
from pipe_sim.runner import sha256, write_json

FINAL = Path(__file__).resolve().parents[2]
PARENT = FINAL / 'sym/runs/campaign/orientation_intact_00_adaptive'
OUTPUT = '*NODE PRINT, NSET=SUPPORT_AUDIT, FREQUENCY=1\nU, RF\n'


def additions(model):
    constraints = [(n, dof) for name, dof in
                   [('BOTTOM', 2), ('LATERAL_ANCHORS', 1), ('AXIAL_ANCHOR', 3)]
                   for n in model.nsets[name]]
    nodes = sorted({n for n, _ in constraints})
    mesh = '\n*NSET, NSET=SUPPORT_AUDIT\n' + '\n'.join(
        ', '.join(map(str, nodes[i:i+16])) for i in range(0, len(nodes), 16)) + '\n'
    return constraints, mesh


def verify(target):
    parent = json.loads((PARENT / 'manifest.json').read_text())
    child = json.loads((target / 'manifest.json').read_text())
    for directory, manifest in [(PARENT, parent), (target, child)]:
        assert all(sha256(directory / n) == h for n, h in manifest['files'].items())
    assert child['parent_manifest_sha256'] == sha256(PARENT / 'manifest.json')
    constraints, extra = additions(Model.load(PARENT / 'model.json'))
    original = (PARENT / 'job.inp').read_text()
    boundary = '*BOUNDARY\nBOTTOM, 2, 2, 0.\nLATERAL_ANCHORS, 1, 1, 0.\nAXIAL_ANCHOR, 3, 3, 0.'
    assert original.count('*BOUNDARY') == 1 and original.count(boundary) == 1
    assert original.count('*END STEP') == 1
    assert len(constraints) == 85 and len(set(constraints)) == 85
    assert 'SUPPORT_AUDIT' not in original
    assert (target / 'job.inp').read_text() == original.replace('*END STEP', OUTPUT + '*END STEP')
    assert (target / 'mesh.inp').read_text() == (PARENT / 'mesh.inp').read_text() + extra
    for name in ['case.json', 'model.json']:
        assert sha256(target / name) == sha256(PARENT / name)
    return dict(run=str(target.relative_to(FINAL)), parent=str(PARENT.relative_to(FINAL)),
                output_only_verified=True, constrained_components=len(constraints),
                constrained_node_dofs=[list(c) for c in constraints],
                output_node_count=len({n for n, _ in constraints}),
                added_mesh_text=extra, added_deck_text=OUTPUT,
                input_sha256=child['files'], parent_input_sha256=parent['files'],
                parent_manifest_sha256=child['parent_manifest_sha256'],
                parent_solver_sha256=json.loads((PARENT / 'status.json').read_text())['solver_sha256'],
                limitations='Prepared, not solved. Same failed physical load path; additional reaction evidence only. '
                'RF balance must use constrained components only and accepted frames. '
                'Geometric pressure integration is not captured assembled load evidence. '
                'No burst, experimental validation, mesh convergence or Abaqus load equivalence established.')


def prepare(target):
    parent = json.loads((PARENT / 'manifest.json').read_text())
    assert all(sha256(PARENT / n) == h for n, h in parent['files'].items())
    target.mkdir(exist_ok=False)
    for name in ['case.json', 'model.json', 'mesh.inp', 'job.inp']:
        shutil.copy2(PARENT / name, target / name)
    _, extra = additions(Model.load(PARENT / 'model.json'))
    with (target / 'mesh.inp').open('a') as stream:
        stream.write(extra)
    (target / 'job.inp').write_text((PARENT / 'job.inp').read_text().replace('*END STEP', OUTPUT + '*END STEP'))
    manifest = dict(parent)
    manifest.update(parent=str(PARENT.relative_to(FINAL)),
                    parent_manifest_sha256=sha256(PARENT / 'manifest.json'),
                    preparation_script_sha256=sha256(Path(__file__)),
                    controlled_change='Only add union support NSET and per-increment U/RF print. Case and model byte-identical.',
                    files={n: sha256(target / n) for n in parent['files']})
    write_json(target / 'manifest.json', manifest)
    evidence = verify(target)
    write_json(target / 'status.json', {'state': 'generated', 'launch_deferred': True,
                                      'reason': 'Production sequential queue occupies large-job slot.'})
    out = FINAL / 'results/output/pressure_restraint_candidates' / (target.name + '_preparation.json')
    write_json(out, evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'verify'])
    parser.add_argument('run_id')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {'.', '..'}:
        parser.error('run_id must be one new directory name')
    target = FINAL / 'sym/runs' / args.run_id
    evidence = prepare(target) if args.action == 'prepare' else verify(target)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()

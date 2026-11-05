"""Kinematic pressure-restraint candidates; no solver or physical model changes."""
import json
from pathlib import Path

import numpy as np
from pipe_sim.model import Model
from audit_pressure_failures import digest

FINAL = Path(__file__).resolve().parents[2]


def audit(run):
    model = Model.load(run / 'model.json')
    manifest = json.loads((run / 'manifest.json').read_text())
    assert all(digest(run / name) == sha for name, sha in manifest['files'].items())
    case = json.loads((run / 'case.json').read_text())
    length = case['pipe']['length']
    radius = case['pipe']['od'] / 2
    pipe_nodes = sorted({n for e in model.elsets['PIPE'] for n in model.elements[e][1]})

    def nearest(target):
        return min(pipe_nodes, key=lambda n: np.linalg.norm(np.array(model.nodes[n])-target))

    # Tangential constraints at cardinal nodes, plus axial displacement at z=0.
    targets = [(0, -radius, -length/2, 0), (0, -radius, length/2, 0),
               (0, radius, -length/2, 0), (radius, 0, -length/2, 1),
               (radius, 0, length/2, 1), (0, -radius, 0, 2)]
    candidate = [(nearest(np.array(t[:3])), t[3]) for t in targets]
    for (node, dof), target in zip(candidate, targets):
        assert np.linalg.norm(np.array(model.nodes[node])-target[:3]) < 1e-6
    current = ([(n, 1) for n in model.nsets['BOTTOM']]
               + [(n, 0) for n in model.nsets['LATERAL_ANCHORS']]
               + [(n, 2) for n in model.nsets['AXIAL_ANCHOR']])

    def describe(constraints):
        rows, blocked = [], []
        for node, dof in constraints:
            x, y, z = np.array(model.nodes[node]) / length
            rotation = [[0, z, -y], [-z, 0, x], [y, -x, 0]][dof]
            rows.append(list(np.eye(3)[dof])+rotation)
            # Two independent axisymmetric expansion fields, normalized so
            # outer radius / end displacement is 1 mm. Not a Lamé solve.
            px, py, pz = model.nodes[node]
            blocked.append([[px/radius, py/radius, 0][dof],
                            [0, 0, 2*pz/length][dof]])
        matrix = np.array(rows)
        singular = np.linalg.svd(matrix, compute_uv=False)
        rank = int(np.linalg.matrix_rank(matrix))
        return dict(constraint_rows=len(rows), rigid_body_rank=rank,
                    unrestrained_rigid_modes=6-rank, singular_values=singular.tolist(),
                    maximum_blocked_unit_radial_expansion_mm=float(np.max(np.abs(np.array(blocked)[:, 0]))),
                    maximum_blocked_unit_axial_expansion_mm=float(np.max(np.abs(np.array(blocked)[:, 1]))))

    proposed = describe(candidate)
    assert proposed['rigid_body_rank'] == 6
    assert proposed['maximum_blocked_unit_radial_expansion_mm'] < 1e-10
    assert proposed['maximum_blocked_unit_axial_expansion_mm'] < 1e-10
    return dict(run=str(run.relative_to(FINAL)),
                evidence_sha256={name: digest(run/name) for name in ['case.json', 'model.json', 'job.inp', 'manifest.json']},
                retained_bottom_restraints=describe(current),
                bottom_removed_without_replacement=describe(current[len(model.nsets['BOTTOM']):]),
                candidate_tangential_restraints=proposed,
                candidate_node_dof_coordinates=[dict(node=n, dof=d+1, coordinates_mm=model.nodes[n]) for n,d in candidate],
                limitations='Infinitesimal rigid-body rank and undeformed axisymmetric expansion compatibility only. '
                'No nonlinear stiffness, reaction, Lamé agreement, local pin stress, contact transfer, or solver validation. '
                'Candidate is not applied to any deck. Replacing dent fixture requires stage-specific release and '
                'verification in both CalculiX and Abaqus. These cardinal pins may constrain non-axisymmetric deformation.')


def main():
    out = FINAL / 'results/output/pressure_restraint_candidates'
    out.mkdir(parents=True, exist_ok=True)
    results = [audit(FINAL / 'sym/runs/campaign' / name) for name in
               ['orientation_intact_00_adaptive', 'full_shuai2018_dent_adaptive']]
    (out/'audit.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps([{k: r[k] for k in ['run','retained_bottom_restraints','bottom_removed_without_replacement','candidate_tangential_restraints']} for r in results]))


if __name__ == '__main__':
    main()

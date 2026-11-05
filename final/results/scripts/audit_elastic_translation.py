"""Separate constant translation from elastic displacement error in frozen benchmarks."""
import argparse
import json
from pathlib import Path

import numpy as np
from pipe_sim.config import Case
from pipe_sim.model import Model
from pipe_sim.runner import sha256, write_json
from audit_intact_end_fields import complete_fields
from audit_pressure_failures import converged_times, matches_sta

FINAL = Path(__file__).resolve().parents[2]


def decompose(xyz, observed, expected, mask):
    """Least-squares constant translation only; never fit deformation or rotation."""
    assert xyz.shape == observed.shape == expected.shape
    assert mask.any()
    translation = np.mean((observed - expected)[mask], axis=0)
    radius = np.linalg.norm(xyz[:, :2], axis=1)
    direction = xyz[:, :2] / radius[:, None]
    theory = np.sum(expected[:, :2] * direction, axis=1)
    assert np.all(theory > 0)
    errors = {}
    for name, disp in [('raw', observed), ('translation_removed', observed-translation)]:
        radial = np.sum(disp[:, :2]*direction, axis=1)
        relative = abs(radial[mask]-theory[mask])/theory[mask]
        errors[name] = dict(mean=float(relative.mean()), maximum=float(relative.max()),
                           vector_rms_mm=float(np.sqrt(np.mean(np.sum((disp[mask]-expected[mask])**2, axis=1)))))
    return dict(translation_mm=translation.tolist(), nodes=int(mask.sum()), radial_errors=errors)


def audit(root):
    records = []
    for name in ['bottom', 'tangential']:
        run = root/name
        status = json.loads((run/'status.json').read_text())
        assert status['state'] == 'completed'
        manifest = json.loads((run/'manifest.json').read_text())
        assert all(sha256(run/n) == h for n, h in manifest['files'].items())
        native = {n: sha256(run/n) for n in ['job.frd', 'job.sta', 'job.dat', 'solver.log']}
        assert matches_sta(converged_times(run/'job.sta'), 1)
        case = Case.load(run/'case.json')
        assert not case.material.plastic and case.analysis.max_pressure == 1
        model = Model.load(run/'model.json')
        blocks = []
        with (run/'job.frd').open() as stream:
            for time, kind, labels, rows in complete_fields(stream):
                if kind == 'DISP' and abs(time-1) < 1e-7:
                    # FRD includes a derived ALL descriptor without a fourth value.
                    assert labels == ['D1', 'D2', 'D3', 'ALL']
                    blocks.append(rows)
        assert len(blocks) == 1
        ids = sorted({n for e in model.elsets['PIPE'] for n in model.elements[e][1]})
        xyz = np.array([model.nodes[n] for n in ids])
        observed = np.array([blocks[0][n] for n in ids])
        ro = case.pipe.od/2; ri = ro-case.pipe.thickness
        a = ri**2/(ro**2-ri**2); b = a*ro**2
        radius = np.linalg.norm(xyz[:, :2], axis=1)
        nu = case.material.poisson; young = case.material.young
        radial = ((1-2*nu)*a*radius+(1+nu)*b/radius)/young
        expected = np.column_stack([radial[:, None]*xyz[:, :2]/radius[:, None], (1-2*nu)*a*xyz[:, 2]/young])
        mask = abs(xyz[:, 2]) < case.pipe.length/4
        result = decompose(xyz, observed, expected, mask)
        assert native == {n: sha256(run/n) for n in native}
        records.append(dict(variant=name, run=str(run.relative_to(FINAL)), time=1,
                            pressure_mpa=1, **result, native_sha256=native,
                            input_hashes=manifest['files'], solver_sha256=status['solver_sha256']))
    return dict(records=records, reporter_sha256=sha256(Path(__file__)),
                limitations='Unweighted central-half nodal least-squares translation diagnostic. Raw errors retained; no acceptance gate changed. Translation cannot establish fixture validity, nonlinear end equilibrium, pin effects, mesh convergence, Abaqus equivalence or experimental validation. Candidate remains unadopted.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args = parser.parse_args()
    assert Path(args.run_id).name == args.run_id and args.run_id not in {'.', '..'}
    result = audit(FINAL/'sym/runs'/args.run_id)
    write_json(FINAL/'results/output/pressure_restraint_candidates'/(args.run_id+'_translation.json'), result)
    print(json.dumps(result, indent=2))

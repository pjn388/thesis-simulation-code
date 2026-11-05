"""Projected quadratic end contours of the retained intact attempt; no model edit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pipe_sim.model import Model, HEX_FACES, HEX_LOCAL
from audit_intact_end_fields import complete_fields
from audit_pressure_failures import digest, converged_times, matches_sta

FINAL = Path(__file__).resolve().parents[2]


def edge_area(points):
    """Signed Green area integral along a quadratic edge (start, mid, end)."""
    result = 0.
    for t, weight in zip(*np.polynomial.legendre.leggauss(3)):
        shape = np.array([t*(t-1)/2, 1-t*t, t*(t+1)/2])
        derivative = np.array([t-.5, -2*t, t+.5])
        x, y = shape @ np.asarray(points)[:, :2]
        dx, dy = derivative @ np.asarray(points)[:, :2]
        result += weight*(x*dy-y*dx)/2
    return float(result)


def contours(model, surface, ri, ro):
    result = {'inner': [], 'outer': []}
    for element, face in model.surfaces[surface]:
        kind, nodes = model.elements[element]
        assert kind == 'C3D20R'
        corners = HEX_FACES[face]
        for i, j in zip(corners, corners[1:]+corners[:1]):
            radii = [np.linalg.norm(model.nodes[nodes[k]][:2]) for k in (i, j)]
            for label, radius in [('inner', ri), ('outer', ro)]:
                if not np.allclose(radii, radius, atol=1e-7, rtol=0):
                    continue
                mid = np.flatnonzero(np.all(HEX_LOCAL == (HEX_LOCAL[i]+HEX_LOCAL[j])/2, axis=1))
                assert len(mid) == 1
                edge = [nodes[i], nodes[int(mid[0])], nodes[j]]
                a, b = np.asarray(model.nodes[edge[0]]), np.asarray(model.nodes[edge[2]])
                if a[0]*b[1]-a[1]*b[0] < 0:
                    edge.reverse()
                result[label].append(edge)
    for edges in result.values():
        assert edges and len({tuple(sorted(e)) for e in edges}) == len(edges)
        assert sorted(e[0] for e in edges) == sorted(e[2] for e in edges), 'Open contour'
    return result


def areas(edges, coordinates):
    return {label: sum(edge_area([coordinates[n] for n in e]) for e in rows)
            for label, rows in edges.items()}


def main():
    run = FINAL/'sym/runs/campaign/orientation_intact_00_adaptive'
    status = json.loads((run/'status.json').read_text())
    assert status['state'] == 'failed'
    manifest = json.loads((run/'manifest.json').read_text())
    assert all(digest(run/n) == h for n, h in manifest['files'].items())
    names = ['model.json', 'case.json', 'job.inp', 'job.frd', 'job.sta', 'results/history.csv']
    hashes = {n: digest(run/n) for n in names}
    case = json.loads((run/'case.json').read_text())
    model = Model.load(run/'model.json')
    latest = pd.read_csv(run/'results/history.csv').iloc[-1]
    assert matches_sta(converged_times(run/'job.sta'), latest.time)
    displacement = None
    with (run/'job.frd').open() as stream:
        for time, kind, labels, rows in complete_fields(stream):
            if kind == 'DISP' and abs(time-latest.time) <= 1e-6:
                displacement = rows
    assert displacement is not None
    ro = case['pipe']['od']/2
    ri = ro-case['pipe']['thickness']
    ratio = ri**2/(ro**2-ri**2)
    records = []
    for surface in ['END_MINUS', 'END_PLUS']:
        edges = contours(model, surface, ri, ro)
        nodes = {n for rows in edges.values() for e in rows for n in e}
        assert nodes.issubset(displacement)
        current = {n: np.asarray(model.nodes[n])+displacement[n] for n in nodes}
        reference, deformed = areas(edges, model.nodes), areas(edges, current)
        wall = deformed['outer']-deformed['inner']
        ideal_force = latest.pressure_mpa*deformed['inner']
        deck_force = latest.pressure_mpa*ratio*wall
        records.append(dict(surface=surface, quadratic_edges={k: len(v) for k,v in edges.items()},
                            reference_projected_area_mm2=reference, current_projected_area_mm2=deformed,
                            reference_inner_circle_relative_error=reference['inner']/(np.pi*ri**2)-1,
                            ideal_closed_cap_axial_resultant_n=float(ideal_force),
                            reference_ratio_traction_axial_resultant_n=float(deck_force),
                            relative_resultant_mismatch=float(deck_force/ideal_force-1)))
    assert hashes == {n: digest(run/n) for n in names}, 'Native evidence changed'
    report = dict(run=str(run.relative_to(FINAL)), accepted_time=float(latest.time),
                  pressure_mpa=float(latest.pressure_mpa), records=records, native_sha256=hashes,
                  solver_sha256=status['solver_sha256'], script_sha256=digest(Path(__file__)),
                  sources=['sym/src/pipe_sim/decks.py:92-96', 'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f:1360-1486',
                           'data/evidence/zhao2021/text.txt:313-317'],
                  limitations='Geometric diagnostic of an ideal closed cap, not a force/reaction balance or physical fixture model. '
                  'Uses projected quadratic end contours and accepted rounded FRD displacement. '
                  'End surfaces may warp; current pressure resultant projected axially uses Green area. '
                  'No proof of failure cause, burst, or validation. Abaqus finite-load semantics still require verification; no adapter changed.')
    out = FINAL/'results/output/pressure_restraint_candidates/intact_end_cap_resultant.json'
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

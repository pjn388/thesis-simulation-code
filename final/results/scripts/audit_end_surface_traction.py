"""Independent 3D pressure-vector integral; no solver inputs are changed."""
import json
from pathlib import Path

import numpy as np
from pipe_sim.model import Model, HEX_FACES, HEX_LOCAL
from audit_end_cap_resultant import FINAL, areas, contours
from audit_intact_end_fields import complete_fields
from audit_pressure_failures import digest, converged_times, matches_sta


def shape8(r, s):
    return np.array([
        -(1-r)*(1-s)*(1+r+s)/4, -(1+r)*(1-s)*(1-r+s)/4,
        -(1+r)*(1+s)*(1-r-s)/4, -(1-r)*(1+s)*(1+r-s)/4,
        (1-r*r)*(1-s)/2, (1+r)*(1-s*s)/2,
        (1-r*r)*(1+s)/2, (1-r)*(1-s*s)/2])


def face_vector(points):
    """Integral of oriented normal dA, using exact polynomial quadrature."""
    points = np.asarray(points)
    total = np.zeros(3)
    q, weights = np.polynomial.legendre.leggauss(3)
    for r, wr in zip(q, weights):
        for s, ws in zip(q, weights):
            dr = shape8(r+1e-30j, s).imag/1e-30 @ points
            ds = shape8(r, s+1e-30j).imag/1e-30 @ points
            total += wr*ws*np.cross(dr, ds)
    return total


def face_nodes(model, element, face):
    kind, nodes = model.elements[element]
    assert kind == 'C3D20R'
    corners = HEX_FACES[face]
    mids = []
    for i, j in zip(corners, corners[1:]+corners[:1]):
        mid = np.flatnonzero(np.all(HEX_LOCAL == (HEX_LOCAL[i]+HEX_LOCAL[j])/2, axis=1))
        assert len(mid) == 1
        mids.append(int(mid[0]))
    return [nodes[i] for i in corners+mids]


def main():
    prior_path = FINAL/'results/output/pressure_restraint_candidates/intact_end_cap_resultant.json'
    prior = json.loads(prior_path.read_text())
    run = FINAL/prior['run']
    hashes = prior['native_sha256']
    assert all(digest(run/n) == h for n, h in hashes.items())
    status = json.loads((run/'status.json').read_text())
    assert status['state'] == 'failed'
    time = prior['accepted_time']
    assert matches_sta(converged_times(run/'job.sta'), time)
    displacement = None
    with (run/'job.frd').open() as stream:
        for t, kind, labels, rows in complete_fields(stream):
            if kind == 'DISP' and abs(t-time) <= 1e-6:
                displacement = rows
    assert displacement is not None
    model = Model.load(run/'model.json')
    case = json.loads((run/'case.json').read_text())
    ro = case['pipe']['od']/2
    ri = ro-case['pipe']['thickness']
    ratio = ri**2/(ro**2-ri**2)
    records = []
    for previous in prior['records']:
        surface = previous['surface']
        faces = [face_nodes(model, e, f) for e, f in model.surfaces[surface]]
        nodes = {n for face in faces for n in face}
        assert nodes.issubset(displacement)
        current = {n: np.asarray(model.nodes[n])+displacement[n] for n in nodes}
        vector = sum((face_vector([current[n] for n in face]) for face in faces), np.zeros(3))
        sign = -1 if surface == 'END_MINUS' else 1
        assert sign*vector[2] > 0, 'Unexpected end normal orientation'
        projected = areas(contours(model, surface, ri, ro), current)
        wall = projected['outer']-projected['inner']
        relative = sign*vector[2]/wall-1
        assert abs(relative) < 1e-10, 'Surface and contour integrals disagree'
        # Tensile end-cap traction acts outward; CCX deck uses negative pressure.
        traction_force = prior['pressure_mpa']*ratio*vector
        prior_force = previous['reference_ratio_traction_axial_resultant_n']
        assert abs(sign*traction_force[2]/prior_force-1) < 1e-10
        records.append(dict(surface=surface, quadratic_faces=len(faces),
                            outward_area_vector_mm2=vector.tolist(),
                            outward_traction_resultant_n=traction_force.tolist(),
                            surface_vs_contour_relative_error=float(relative),
                            transverse_over_axial_resultant=float(np.linalg.norm(vector[:2])/abs(vector[2])),
                            ideal_cap_mismatch=previous['relative_resultant_mismatch']))
    assert all(digest(run/n) == h for n, h in hashes.items())
    report = dict(run=prior['run'], accepted_time=time, pressure_mpa=prior['pressure_mpa'],
                  records=records, native_sha256=hashes, solver_sha256=status['solver_sha256'],
                  script_sha256=digest(Path(__file__)), parent_report_sha256=digest(prior_path),
                  sources=prior['sources'],
                  limitations='Independent 3D integration of prescribed end traction at accepted rounded FRD geometry. '
                  'Not an assembled solver nodal-load/reaction balance; constrained reactions and inner-wall loads '
                  'are not integrated. Does not prove numerical failure cause, physical cap behavior, or burst. '
                  'Abaqus finite-load semantics remain unverified; no production adapter changes.')
    out = FINAL/'results/output/pressure_restraint_candidates/intact_end_surface_traction.json'
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

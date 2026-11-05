"""Independent virtual-work prerequisite for finite-cap investigation; no native solve."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pipe_sim.model import Model
from pipe_sim.runner import sha256
from audit_end_cap_resultant import contours, areas
from audit_end_surface_traction import face_nodes, shape8
from audit_nodal_pressure_loads import nodal_loads

FINAL = Path(__file__).resolve().parents[2]


def audit():
    # Predeclared manufactured motion: additive radial expansion of 5% of bore
    # radius, axial extension of 2%; linear path, no equilibrium/material claim.
    # Predeclared numerical tolerances: 1e-8 relative work/resultant at 3x3
    # surface quadrature; 2x2 errors are reported, not silently discarded.
    records = []
    for mesh, name in [('coarse', 'pressure_restraints_20261008T143009Z'),
                       ('refined', 'pressure_restraints_refined_20261008T150008Z')]:
        parent = FINAL/'sym/runs'/name/'tangential'
        manifest = json.loads((parent/'manifest.json').read_text())
        assert all(sha256(parent/f) == h for f, h in manifest['files'].items())
        model = Model.load(parent/'model.json')
        case = json.loads((parent/'case.json').read_text())
        ro = case['pipe']['od']/2
        ri = ro-case['pipe']['thickness']
        length = case['pipe']['length']
        reference_ratio = ri**2/(ro**2-ri**2)
        x = {n: np.asarray(v) for n, v in model.nodes.items()}
        velocity = {n: np.array([.05*ri*v[0]/np.linalg.norm(v[:2]),
                                  .05*ri*v[1]/np.linalg.norm(v[:2]), .02*v[2]])
                    for n, v in x.items()}
        edges = {s: contours(model, s, ri, ro) for s in ['END_MINUS', 'END_PLUS']}
        faces = {s: [face_nodes(model, e, f) for e, f in model.surfaces[s]]
                 for s in ['INNER', 'END_MINUS', 'END_PLUS']}
        def geometry(t):
            coords = {n: v+t*velocity[n] for n, v in x.items()}
            area = areas(edges['END_PLUS'], coords)
            return coords, area
        # Adaptive bore is not necessarily an extrusion of the end contour.
        # Divergence integral on side + virtual planar bore caps, independent
        # scalar 5x5 integration without consistent nodal-load assembly.
        side_integral = 0.
        qv, wv = np.polynomial.legendre.leggauss(5)
        for nodes in faces['INNER']:
            points = np.array([x[n] for n in nodes])
            for r, wr in zip(qv, wv):
                for s, ws in zip(qv, wv):
                    pos = shape8(r, s) @ points
                    dr = shape8(r+1e-30j, s).imag/1e-30 @ points
                    ds = shape8(r, s+1e-30j).imag/1e-30 @ points
                    side_integral -= wr*ws*float(pos @ np.cross(dr, ds))
        a0 = geometry(0.)[1]['inner']
        volume0 = (side_integral+a0*length)/3
        assert volume0 > 0
        # All bore nodes have radius ri: declared inner motion is affine.
        # V(t)=V0*(1+.05*t)^2*(1+.02*t), p(t)=t MPa.
        expected_work = volume0*((.10+.02)/2 +
                                2*(.0025+.10*.02)/3 + 3*.0025*.02/4)
        endpoint_checks = []
        for order in [2, 3]:
            work = dict(side=0., candidate_cap=0., reference_cap=0.)
            q, weights = np.polynomial.legendre.leggauss(8)
            for sample, weight in zip((q+1)/2, weights/2):
                coords, area = geometry(float(sample))
                ratio = area['inner']/(area['outer']-area['inner'])
                for surface, groups in faces.items():
                    for nodes in groups:
                        points = [coords[n] for n in nodes]
                        v = np.array([velocity[n] for n in nodes])
                        if surface == 'INNER':
                            work['side'] += weight*float(np.sum(nodal_loads(points, -sample, order)*v))
                        else:
                            for key, factor in [('candidate_cap', ratio), ('reference_cap', reference_ratio)]:
                                work[key] += weight*float(np.sum(nodal_loads(points, sample*factor, order)*v))
            coords, area = geometry(1.)
            ratio = area['inner']/(area['outer']-area['inner'])
            resultants = {}
            for surface in ['END_MINUS', 'END_PLUS']:
                force = sum((nodal_loads([coords[n] for n in nodes], ratio, order).sum(axis=0)
                             for nodes in faces[surface]), np.zeros(3))
                sign = -1 if surface == 'END_MINUS' else 1
                error = sign*force[2]/area['inner']-1
                resultants[surface] = dict(force_n=force.tolist(), relative_axial_error=float(error))
                if order == 3:
                    assert abs(error) < 1e-8, error
            candidate_error = (work['side']+work['candidate_cap'])/expected_work-1
            if order == 3:
                assert abs(candidate_error) < 1e-8, candidate_error
            endpoint_checks.append(dict(surface_quadrature_order=order, work_n_mm=work,
                pressure_volume_work_n_mm=expected_work,
                candidate_relative_work_error=candidate_error,
                reference_relative_work_error=(work['side']+work['reference_cap'])/expected_work-1,
                endpoint_resultants=resultants,
                endpoint_reference_cap_relative_resultant_error=reference_ratio/ratio-1))
        assert all(sha256(parent/f) == h for f, h in manifest['files'].items())
        records.append(dict(mesh=mesh, parent=str(parent.relative_to(FINAL)),
            parent_manifest_sha256=sha256(parent/'manifest.json'), frozen_input_sha256=manifest['files'],
            elements=len(model.elements), radial_travel_mm=.05*ri, axial_strain=.02,
            reference_area_ratio=reference_ratio, initial_bore_volume_mm3=volume0,
            end_area_times_length_mm3=a0*length,
            extrusion_volume_relative_error=a0*length/volume0-1, checks=endpoint_checks))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), records=records,
        script_sha256=sha256(__file__),
        convention='Prescribed axisymmetric quadratic-surface tube; p=t MPa, '
            'r(t)=r0+0.05*ri*t, z(t)=z0*(1+0.02*t). Virtual caps translate axially '
            'with planar ends. Initial bore volume is an independent divergence integral, '
            'not assumed end area times length. W=integral p(t)*dV(t).',
        sources=['sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f:1360-1486,1597-1615',
            'sym/src/pipe_sim/decks.py:92-96', 'data/evidence/zhao2021/text.txt:313-317'],
        limitations='Manufactured geometric virtual-work diagnostic only. Candidate current-area '
            'ratio follows the declared ideal-cap convention, not a recovered experimental fixture. '
            'No native equilibrium, ELSE, plasticity, warped-end cap work, coupling implementation, '
            'physical validation, failure causation or Abaqus equivalence established. '
            'Production loads remain immutable. Next prerequisite is bounded native verification.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), 'Use new evidence path'
    report = audit()
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps([dict(mesh=r['mesh'], checks=r['checks']) for r in report['records']]))

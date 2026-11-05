"""Directional verification of ideal-cap load coupling, without native/model edits."""
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


def area_action(edges, x, v):
    result = {}
    for label, rows in edges.items():
        total = 0.
        for nodes in rows:
            points = np.array([x[n][:2] for n in nodes])
            motion = np.array([v[n][:2] for n in nodes])
            for t, w in zip(*np.polynomial.legendre.leggauss(3)):
                n = np.array([t*(t-1)/2, 1-t*t, t*(t+1)/2])
                dn = np.array([t-.5, -2*t, t+.5])
                p, dp, u, du = n@points, dn@points, n@motion, dn@motion
                total += w*(u[0]*dp[1]+p[0]*du[1]-u[1]*dp[0]-p[1]*du[0])/2
        result[label] = total
    return result


def face_action(points, motion, order):
    result = np.zeros((8, 3))
    q, weights = np.polynomial.legendre.leggauss(order)
    for r, wr in zip(q, weights):
        for s, ws in zip(q, weights):
            n = shape8(r, s)
            nr = shape8(r+1e-30j, s).imag/1e-30
            ns = shape8(r, s+1e-30j).imag/1e-30
            derivative = np.cross(nr@motion, ns@points)+np.cross(nr@points, ns@motion)
            result += wr*ws*n[:, None]*derivative
    return result


def audit():
    records = []
    for mesh, folder in [('coarse', 'pressure_restraints_20261008T143009Z'),
                         ('refined', 'pressure_restraints_refined_20261008T150008Z')]:
        parent = FINAL/'sym/runs'/folder/'tangential'
        manifest = json.loads((parent/'manifest.json').read_text())
        assert all(sha256(parent/f)==h for f,h in manifest['files'].items())
        model = Model.load(parent/'model.json')
        case = json.loads((parent/'case.json').read_text())
        ro = case['pipe']['od']/2; ri = ro-case['pipe']['thickness']
        for surface in ['END_MINUS', 'END_PLUS']:
            edges = contours(model, surface, ri, ro)
            faces = [face_nodes(model,e,f) for e,f in model.surfaces[surface]]
            ids = sorted({n for face in faces for n in face})
            index = {n:i for i,n in enumerate(ids)}
            for expansion in [0., .05*ri]:
                x = {n:np.array(model.nodes[n])+np.r_[expansion*np.array(model.nodes[n][:2])/np.linalg.norm(model.nodes[n][:2]),0.] for n in ids}
                for direction in ['inner_radial', 'outer_radial', 'local_radial', 'axial_warp']:
                    v = {n:np.zeros(3) for n in ids}
                    for n in ids:
                        original = np.array(model.nodes[n]); radius = np.linalg.norm(original[:2])
                        selected = ((direction=='inner_radial' and abs(radius-ri)<1e-7) or
                                    (direction=='outer_radial' and abs(radius-ro)<1e-7) or
                                    (direction=='local_radial' and n==edges['inner'][0][1]))
                        if selected: v[n][:2] = original[:2]/radius
                        if direction=='axial_warp': v[n][2] = original[0]/ro
                    a = areas(edges,x); da = area_action(edges,x,v)
                    wall = a['outer']-a['inner']; ratio = a['inner']/wall
                    # p=1 MPa held fixed. Derivative includes nonlocal ratio term.
                    dq = (a['outer']*da['inner']-a['inner']*da['outer'])/wall**2
                    for order in [2,3]:
                        def force(coords):
                            b = areas(edges,coords); q = b['inner']/(b['outer']-b['inner'])
                            f = np.zeros((len(ids),3))
                            for face in faces:
                                local = nodal_loads([coords[n] for n in face],q,order)
                                for n,value in zip(face,local): f[index[n]] += value
                            return f
                        tangent = np.zeros((len(ids),3)); uncoupled = tangent.copy()
                        for face in faces:
                            points = np.array([x[n] for n in face]); motion = np.array([v[n] for n in face])
                            geometric = ratio*face_action(points,motion,order)
                            coupled = geometric+dq*nodal_loads(points,1.,order)
                            for n,c,g in zip(face,coupled,geometric):
                                tangent[index[n]] += c; uncoupled[index[n]] += g
                        checks = []
                        for h in [1e-3,1e-4]:
                            plus = {n:x[n]+h*v[n] for n in ids}; minus = {n:x[n]-h*v[n] for n in ids}
                            fd = (force(plus)-force(minus))/(2*h)
                            scale = max(1.,float(np.max(abs(tangent))))
                            error = float(np.max(abs(fd-tangent)))/scale
                            assert error<1e-7, (mesh,surface,direction,order,h,error)
                            checks.append(dict(step_mm=h,relative_error=error,omitted_ratio_term_relative_error=float(np.max(abs(fd-uncoupled)))/scale))
                        if direction in ('inner_radial','outer_radial'):
                            assert checks[-1]['omitted_ratio_term_relative_error']>1e-3
                        records.append(dict(mesh=mesh,surface=surface,expansion_mm=expansion,direction=direction,quadrature_order=order,ratio=ratio,ratio_action_per_mm=dq,checks=checks,parent=str(parent.relative_to(FINAL)),parent_manifest_sha256=sha256(parent/'manifest.json')))
        assert all(sha256(parent/f)==h for f,h in manifest['files'].items())
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,records=records,script_sha256=sha256(__file__),relative_tolerance=1e-7,dependency_sha256={n:sha256(FINAL/'results/scripts'/n) for n in ['audit_end_cap_resultant.py','audit_end_surface_traction.py','audit_nodal_pressure_loads.py']},
        sources=['sym/src/pipe_sim/decks.py:92-96','sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f:1597-1655','results/scripts/manufactured_cap_work.py'],
        limitations='Independent analytic tangent actions versus centered finite differences of geometric candidate forces, p=1 MPa. Both ends, two meshes, two quadrature orders, four directions and reference/expanded states. Not a full Jacobian, native coupled tangent implementation, equilibrium, general warped-end ideal-cap work, contact/MPC transfer, fixture evidence or Abaqus equivalence. Production loads unchanged; geometric follower term alone misses area-ratio coupling.')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('output',type=Path); args=parser.parse_args()
    assert not args.output.exists()
    result = audit(); args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],records=len(result['records']),maximum_relative_error=max(c['relative_error'] for r in result['records'] for c in r['checks']))))

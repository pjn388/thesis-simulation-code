"""Read-only force balance from completed all-node elastic benchmark outputs."""
import json
from pathlib import Path

import numpy as np
from pipe_sim.model import Model
from pipe_sim.results import dat_blocks
from pipe_sim.runner import sha256
from audit_end_surface_traction import face_nodes, shape8, FINAL
from audit_pressure_restraints import audit


def integrate(points, order):
    points = np.asarray(points)
    q, weights = np.polynomial.legendre.leggauss(order)
    result = np.zeros(3)
    for r, wr in zip(q, weights):
        for s, ws in zip(q, weights):
            dr = shape8(r+1e-30j, s).imag/1e-30 @ points
            ds = shape8(r, s+1e-30j).imag/1e-30 @ points
            result += wr*ws*np.cross(dr, ds)
    return result


def constrained_sum(rows, constraints):
    rf = {int(row[0]): row[1:] for row in rows}
    result = np.zeros(3)
    for node, dof in set(constraints):
        result[dof-1] += rf[node][dof-1]
    return result


def main():
    root = FINAL/'sym/runs/pressure_restraints_allip_20261008T160000Z'
    pins = audit(root/'common_adaptive')['candidate_node_dof_coordinates']
    records = []
    for name in ['bottom', 'tangential']:
        run = root/name
        status = json.loads((run/'status.json').read_text())
        assert status['state'] == 'completed'
        manifest = json.loads((run/'manifest.json').read_text())
        assert all(sha256(run/n) == h for n,h in manifest['files'].items())
        hashes = {n:sha256(run/n) for n in ['job.dat','job.sta','job.frd','solver.log']}
        model = Model.load(run/'model.json')
        case = json.loads((run/'case.json').read_text())
        assert '*STEP, NLGEOM' in (run/'job.inp').read_text().upper()
        assert not case['material']['plastic']
        blocks = {kind:rows for kind,group,time,rows in dat_blocks(run/'job.dat')
                  if group == 'BENCH' and abs(time-1) < 1e-7}
        assert set(model.nodes) == {int(row[0]) for row in blocks['RF']}
        displacement = {int(row[0]):np.asarray(row[1:]) for row in blocks['U']}
        assert set(model.nodes) == set(displacement)
        current = {n:np.asarray(p)+displacement[n] for n,p in model.nodes.items()}
        constraints = ([(n,d) for group,d in [('BOTTOM',2),('LATERAL_ANCHORS',1),('AXIAL_ANCHOR',3)]
                        for n in model.nsets[group]] if name == 'bottom'
                       else [(p['node'],p['dof']) for p in pins])
        # CCX node212: RF includes applied nodal loads at constrained DOFs.
        # This filtered RF sum is not independently recovered support reaction.
        filtered_rf = constrained_sum(blocks['RF'], constraints)
        ro = case['pipe']['od']/2
        ri = ro-case['pipe']['thickness']
        pressure = case['analysis']['max_pressure']
        ratio = ri**2/(ro**2-ri**2)
        orders = []
        for order in [2,3]:
            forces = {}
            for surface in ['INNER','END_MINUS','END_PLUS']:
                vector = sum((integrate([current[n] for n in face_nodes(model,e,f)],order)
                              for e,f in model.surfaces[surface]), np.zeros(3))
                forces[surface] = pressure*(-1 if surface == 'INNER' else ratio)*vector
            net = sum(forces.values(),np.zeros(3))
            scale = np.mean([np.linalg.norm(forces[s]) for s in ['END_MINUS','END_PLUS']])
            orders.append(dict(quadrature_order=order, prescribed_vectors_n={s:v.tolist() for s,v in forces.items()},
                               net_prescribed_n=net.tolist(), net_prescribed_plus_constrained_rf_n=(net+filtered_rf).tolist(),
                               combined_vector_over_mean_end_force=float(np.linalg.norm(net+filtered_rf)/scale)))
        assert all(sha256(run/n) == h for n,h in hashes.items())
        records.append(dict(variant=name,run=str(run.relative_to(FINAL)),constrained_dofs=len(set(constraints)),
                            constrained_rf_n=filtered_rf.tolist(), quadratures=orders,
                            native_sha256=hashes,input_sha256=manifest['files'],solver_sha256=status['solver_sha256']))
    report = dict(records=records,script_sha256=sha256(Path(__file__)),
                  rf_semantics_source='sym/vendor/CalculiX/ccx_2.23/doc/ccx/node212.html:62-64,135-138',
                  limitations='RF is applied nodal load plus support reaction, including at constrained DOFs. '
                  'Filtering constrained components alone does not remove DLOAD contributions. '
                  'Reported combined vector is descriptive, not verified reaction equilibrium. '
                  'Geometric 2x2/3x3 integrals are not independently captured solver assembled loads. '
                  'Accepted low-pressure NLGEOM DAT coordinates only; rounded output; no large-expansion balance, '
                  'physical cap validation, failure cause, burst or production/Abaqus revision.')
    # Keep the earlier report to expose, rather than erase, its interpretation error.
    out = FINAL/'results/output/pressure_restraint_candidates/elastic_force_balance_rf_semantics.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__ == '__main__':
    main()

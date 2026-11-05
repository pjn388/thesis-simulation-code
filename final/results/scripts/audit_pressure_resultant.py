"""Prescribed whole-pipe pressure resultant and recorded reaction coverage."""
import json
from pathlib import Path
import numpy as np
from pipe_sim.model import Model
from audit_end_surface_traction import face_nodes, face_vector, FINAL
from audit_intact_end_fields import complete_fields
from audit_pressure_failures import digest, converged_times, matches_sta


def surface_vector(model, name, positions):
    return sum((face_vector([positions[n] for n in face_nodes(model, e, f)])
                for e, f in model.surfaces[name]), np.zeros(3))


def reaction_coverage(model):
    constraints = {(n, dof) for name, dof in
                   [('BOTTOM', 2), ('LATERAL_ANCHORS', 1), ('AXIAL_ANCHOR', 3)]
                   for n in model.nsets[name]}
    history = set(model.nsets['HISTORY'])
    recorded = sorted((n, dof) for n, dof in constraints if n in history)
    missing = sorted(constraints-set(recorded))
    return dict(constrained_dofs=len(constraints), recorded_constrained_dofs=recorded,
                missing_constrained_dofs=missing, complete=not missing)


def main():
    parent_path = FINAL/'results/output/pressure_restraint_candidates/intact_end_surface_traction.json'
    parent = json.loads(parent_path.read_text())
    run = FINAL/parent['run']
    hashes = parent['native_sha256']
    assert all(digest(run/n) == h for n,h in hashes.items())
    assert json.loads((run/'status.json').read_text())['state'] == 'failed'
    time = parent['accepted_time']
    assert matches_sta(converged_times(run/'job.sta'), time)
    displacement = None
    with (run/'job.frd').open() as stream:
        for t, kind, _, rows in complete_fields(stream):
            if kind == 'DISP' and abs(t-time) <= 1e-6:
                displacement = rows
    assert displacement is not None
    model = Model.load(run/'model.json')
    current = {n:np.asarray(p)+displacement[n] for n,p in model.nodes.items()}
    reference = {n:np.asarray(p) for n,p in model.nodes.items()}
    names = ['INNER','OUTER','END_MINUS','END_PLUS']
    vectors = {name:surface_vector(model,name,current) for name in names}
    reference_vectors = {name:surface_vector(model,name,reference) for name in names}
    scale = sum(np.linalg.norm(v) for v in vectors.values())
    closure = sum(vectors.values(), np.zeros(3))
    assert np.linalg.norm(closure)/scale < 1e-10, 'Closed boundary normals do not cancel'
    for old in parent['records']:
        assert np.allclose(vectors[old['surface']],old['outward_area_vector_mm2'],rtol=1e-10,atol=1e-8)
    case = json.loads((run/'case.json').read_text())
    ro = case['pipe']['od']/2
    ri = ro-case['pipe']['thickness']
    ratio = ri**2/(ro**2-ri**2)
    pressure = parent['pressure_mpa']
    # INNER normals point out of material into bore; positive pressure is compressive.
    forces = {'INNER':-pressure*vectors['INNER'],
              'END_MINUS':pressure*ratio*vectors['END_MINUS'],
              'END_PLUS':pressure*ratio*vectors['END_PLUS']}
    resultant = sum(forces.values(),np.zeros(3))
    normalization = np.mean([np.linalg.norm(forces[n]) for n in ['END_MINUS','END_PLUS']])
    coverage = reaction_coverage(model)
    assert not coverage['complete'], 'Reassess reaction availability before drawing conclusions'
    assert '*NODE FILE, FREQUENCY=1\nU\n' in (run/'job.inp').read_text()
    assert '*NODE PRINT, NSET=HISTORY, FREQUENCY=1\nU, RF\n' in (run/'job.inp').read_text()
    assert all(digest(run/n) == h for n,h in hashes.items())
    report = dict(run=parent['run'],accepted_time=time,pressure_mpa=pressure,
                  quadratic_faces={n:len(model.surfaces[n]) for n in names},
                  reference_outward_area_vectors_mm2={n:v.tolist() for n,v in reference_vectors.items()},
                  current_outward_area_vectors_mm2={n:v.tolist() for n,v in vectors.items()},
                  closed_boundary_vector_mm2=closure.tolist(),closed_boundary_relative_error=float(np.linalg.norm(closure)/scale),
                  prescribed_force_vectors_n={n:v.tolist() for n,v in forces.items()},
                  prescribed_net_force_n=resultant.tolist(),
                  balancing_reaction_required_n=(-resultant).tolist(),
                  net_force_over_mean_end_force=float(np.linalg.norm(resultant)/normalization),
                  reaction_output_coverage=coverage,native_sha256=hashes,
                  solver_sha256=parent['solver_sha256'],parent_report_sha256=digest(parent_path),
                  script_sha256=digest(Path(__file__)),sources=parent['sources'],
                  limitations='Geometric prescribed pressure integral on rounded accepted FRD coordinates; '
                  'not solver assembled loads or measured support reactions. HISTORY omits constrained DOFs; '
                  'retained output cannot establish complete reaction balance. Net load can be balanced by supports '
                  'and does not establish disequilibrium, failure cause, physical cap behavior, burst or mesh convergence. '
                  'No production or Abaqus adapter revision.')
    output = FINAL/'results/output/pressure_restraint_candidates/intact_pressure_resultant.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['prescribed_force_vectors_n','prescribed_net_force_n','net_force_over_mean_end_force','closed_boundary_relative_error','reaction_output_coverage']}))


if __name__ == '__main__':
    main()

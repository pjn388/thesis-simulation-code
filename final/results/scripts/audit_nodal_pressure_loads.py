"""Reconstruct consistent pressure nodal loads; preserve all native evidence."""
import json
from pathlib import Path
import numpy as np
from pipe_sim.model import Model
from pipe_sim.results import dat_blocks
from pipe_sim.runner import sha256
from audit_end_surface_traction import face_nodes, shape8, FINAL
from audit_pressure_restraints import audit
from audit_elastic_force_balance import constrained_sum
from audit_pressure_failures import converged_times, matches_sta


def nodal_loads(points, traction, order=2):
    """Eight-node outward normal traction; CCX C3D20R uses 2x2 quadrature."""
    points = np.asarray(points)
    result = np.zeros((8, 3))
    q, weights = np.polynomial.legendre.leggauss(order)
    for r, wr in zip(q, weights):
        for s, ws in zip(q, weights):
            dr = shape8(r+1e-30j, s).imag/1e-30 @ points
            ds = shape8(r, s+1e-30j).imag/1e-30 @ points
            result += traction*wr*ws*shape8(r,s)[:,None]*np.cross(dr,ds)
    return result


def main():
    root = FINAL/'sym/runs/pressure_restraints_allip_20261008T160000Z'
    pins = audit(root/'common_adaptive')['candidate_node_dof_coordinates']
    records = []
    for name in ['bottom','tangential']:
        run = root/name
        status = json.loads((run/'status.json').read_text())
        assert status['state'] == 'completed'
        assert matches_sta(converged_times(run/'job.sta'),1.0)
        manifest = json.loads((run/'manifest.json').read_text())
        assert all(sha256(run/n) == h for n,h in manifest['files'].items())
        hashes = {n:sha256(run/n) for n in ['job.dat','job.frd','job.sta','solver.log']}
        deck = (run/'job.inp').read_text().upper()
        assert '*STEP, NLGEOM' in deck
        assert not any(k in deck for k in ['*CLOAD','*EQUATION','*MPC','*CONTACT'])
        model = Model.load(run/'model.json')
        case = json.loads((run/'case.json').read_text())
        assert not case['material']['plastic']
        blocks = {k:rows for k,g,t,rows in dat_blocks(run/'job.dat') if g == 'BENCH' and abs(t-1)<1e-7}
        u = {int(r[0]):np.asarray(r[1:]) for r in blocks['U']}
        rf = {int(r[0]):np.asarray(r[1:]) for r in blocks['RF']}
        assert set(u) == set(rf) == set(model.nodes)
        current = {n:np.asarray(p)+u[n] for n,p in model.nodes.items()}
        constraints = set([(n,d) for g,d in [('BOTTOM',2),('LATERAL_ANCHORS',1),('AXIAL_ANCHOR',3)]
                           for n in model.nsets[g]] if name=='bottom' else [(p['node'],p['dof']) for p in pins])
        ro = case['pipe']['od']/2
        ri = ro-case['pipe']['thickness']
        pressure = case['analysis']['max_pressure']
        ratio = ri**2/(ro**2-ri**2)
        # Verify the frozen deck actually contains exactly these constant pressures.
        expected = {(e,f):pressure*(1 if s=='INNER' else -ratio)
                    for s in ['INNER','END_MINUS','END_PLUS'] for e,f in model.surfaces[s]}
        lines = deck.split('*DLOAD\n')[1].split('*')[0].strip().splitlines()
        actual = {}
        for line in lines:
            e,f,p = [x.strip() for x in line.split(',')]
            key = (int(e), int(f[1:]))
            assert key not in actual
            actual[key] = float(p)
        assert actual.keys() == expected.keys()
        assert all(np.isclose(actual[k],p,rtol=1e-11) for k,p in expected.items())
        quadratures=[]
        for order in [2,3]:
            loads = {n:np.zeros(3) for n in model.nodes}
            for (e,f),p in actual.items():
                nodes=face_nodes(model,e,f)
                values=nodal_loads([current[n] for n in nodes],-p,order)
                for n,v in zip(nodes,values):
                    loads[n] += v
            free = [(n,d) for n in model.nodes for d in range(1,4) if (n,d) not in constraints]
            errors=np.array([rf[n][d-1]-loads[n][d-1] for n,d in free])
            scale=max(np.linalg.norm(v) for v in loads.values())
            applied_support=constrained_sum([[n,*v] for n,v in loads.items()],constraints)
            filtered_rf=constrained_sum(blocks['RF'],constraints)
            reactions=filtered_rf-applied_support
            net=sum(loads.values(),np.zeros(3))
            quadratures.append(dict(order=order,free_dofs=len(free),free_rf_minus_load_max_n=float(np.max(abs(errors))),
                free_rf_minus_load_rms_n=float(np.sqrt(np.mean(errors**2))),maximum_nodal_load_norm_n=float(scale),
                free_error_over_maximum_nodal_load=float(np.max(abs(errors))/scale),
                constrained_rf_n=filtered_rf.tolist(),applied_at_constrained_dofs_n=applied_support.tolist(),
                reconstructed_support_reaction_n=reactions.tolist(),net_applied_n=net.tolist(),
                reconstructed_equilibrium_residual_n=(net+reactions).tolist()))
        assert all(sha256(run/n)==h for n,h in hashes.items())
        records.append(dict(variant=name,run=str(run.relative_to(FINAL)),accepted_time=1.0,quadratures=quadratures,
                            native_sha256=hashes,input_sha256=manifest['files'],solver_sha256=status['solver_sha256']))
    report=dict(records=records,script_sha256=sha256(Path(__file__)),
        sources=['sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f:1360-1486,1597-1615',
                 'sym/vendor/CalculiX/ccx_2.23/doc/ccx/node212.html:62-64,99-138'],
        limitations='Reconstruction at rounded final DAT geometry, not captured solver assembly. '
                    '2x2 follows C3D20R DLOAD quadrature; 3x3 is sensitivity only. Free-DOF errors and '
                    'equilibrium residuals must be inspected before accepting reaction recovery. '
                    'Low-pressure elastic evidence only, not finite-expansion failure causation, cap validation '
                    'or burst. No production/Abaqus changes; previous reports retained.')
    out=FINAL/'results/output/pressure_restraint_candidates/elastic_nodal_pressure_loads.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    main()

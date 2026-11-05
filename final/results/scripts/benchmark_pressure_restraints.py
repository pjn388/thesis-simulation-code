"""Isolated matched elastic benchmark; never modifies a campaign input."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from pipe_sim.config import Case
from pipe_sim.model import Model
from pipe_sim.results import dat_blocks, extract
from pipe_sim.runner import generate, run, sha256, write_json
from pipe_sim.verification import elastic_benchmark
from audit_pressure_restraints import audit

FINAL = Path(__file__).resolve().parents[2]
SOURCE = FINAL / 'sym/runs/pressure_restraints_20261008T143009Z/common_adaptive'
ROOT = FINAL / 'sym/runs/pressure_restraints_20261008T143009Z'
OUT = FINAL / 'results/output/pressure_restraint_candidates/elastic_benchmark.json'


def prepare(mesh_scale=1.0, all_pipe_stress=False):
    case = Case.load(FINAL/'sym/cases/elastic_intact.json')
    case.analysis.linear_solver = 'pardiso'
    case.mesh.strategy = 'adaptive_quads'
    case.mesh.fine_size *= mesh_scale
    case.mesh.far_size *= mesh_scale
    generate(case, SOURCE)
    candidate = audit(SOURCE)['candidate_node_dof_coordinates']
    model = Model.load(SOURCE / 'model.json')
    original = (SOURCE / 'job.inp').read_text()
    old = 'BOTTOM, 2, 2, 0.\nLATERAL_ANCHORS, 1, 1, 0.\nAXIAL_ANCHOR, 3, 3, 0.'
    assert original.count(old) == 1
    for name in ['bottom', 'tangential']:
        target = ROOT / name
        target.mkdir(parents=True, exist_ok=False)
        case = Case.load(SOURCE / 'case.json')
        case.name = 'elastic_restraints_' + name
        case.analysis.linear_solver = 'pardiso'
        case.analysis.threads = 1
        case.analysis.timeout_seconds = 120
        case.save(target / 'case.json')
        shutil.copy2(SOURCE / 'model.json', target / 'model.json')
        mesh = (SOURCE / 'mesh.inp').read_text() + '\n*NSET, NSET=BENCH\n'
        nodes = sorted(model.nodes)
        mesh += '\n'.join(', '.join(map(str, nodes[i:i+16])) for i in range(0,len(nodes),16)) + '\n'
        (target / 'mesh.inp').write_text(mesh)
        deck = original.replace('*STATIC\n', '*STATIC,SOLVER=PARDISO\n')
        if name == 'tangential':
            replacement = '\n'.join(f"{r['node']}, {r['dof']}, {r['dof']}, 0." for r in candidate)
            deck = deck.replace(old, replacement)
        deck = deck.replace('*END STEP', '*NODE PRINT, NSET=BENCH, FREQUENCY=1\nU, RF\n*END STEP')
        if all_pipe_stress:
            old_print = '*EL PRINT, ELSET=DEFECT, FREQUENCY=1\nS, E, COORD'
            assert deck.count(old_print) == 1
            deck = deck.replace(old_print, '*EL PRINT, ELSET=PIPE, FREQUENCY=1\nS, E, COORD')
        (target / 'job.inp').write_text(deck)
        manifest = json.loads((SOURCE / 'manifest.json').read_text())
        manifest.update(case=case.name, parent=str(SOURCE.relative_to(FINAL)),
                        parent_manifest_sha256=sha256(SOURCE/'manifest.json'),
                        benchmark_script_sha256=sha256(Path(__file__)),
                        mesh_scale=mesh_scale,
                        all_pipe_stress=all_pipe_stress,
                        controlled_change='bottom versus six tangential/axial pins; all other deck lines matched',
                        files={n:sha256(target/n) for n in ['case.json','model.json','mesh.inp','job.inp']})
        write_json(target/'manifest.json', manifest)
        write_json(target/'status.json', {'state':'generated'})
    print(ROOT)


def report():
    records = []
    candidate = audit(SOURCE)['candidate_node_dof_coordinates']
    for name in ['bottom', 'tangential']:
        target = ROOT/name
        manifest = json.loads((target/'manifest.json').read_text())
        assert all(sha256(target/n)==h for n,h in manifest['files'].items())
        status = json.loads((target/'status.json').read_text())
        if status['state'] != 'completed':
            raise ValueError(f'{target}: report requires completed immutable native outputs')
        extract(target)
        import pandas as pd
        fields = pd.read_csv(target/'results/integration_points.csv')
        assert abs(fields.time.max()-1) < 1e-7, 'Final stress frame missing'
        assert abs(fields[fields.time == fields.time.max()].pressure_mpa.iloc[0]-1) < 1e-7
        stress = elastic_benchmark(target)
        case = Case.load(target/'case.json')
        model = Model.load(target/'model.json')
        blocks = {(q,g):np.array(rows) for q,g,t,rows in dat_blocks(target/'job.dat')
                  if g=='BENCH' and abs(t-1)<1e-7 and rows}
        displacement, reaction = blocks['U','BENCH'], blocks['RF','BENCH']
        ids = displacement[:,0].astype(int)
        xyz = np.array([model.nodes[n] for n in ids])
        radius = np.linalg.norm(xyz[:,:2],axis=1)
        p=case.analysis.max_pressure; ro=case.pipe.od/2; ri=ro-case.pipe.thickness
        A=p*ri**2/(ro**2-ri**2); B=A*ro**2
        theory=((1-2*case.material.poisson)*A*radius+(1+case.material.poisson)*B/radius)/case.material.young
        radial=np.sum(displacement[:,1:3]*xyz[:,:2],axis=1)/radius
        remote=abs(xyz[:,2])<case.pipe.length/4
        axial_theory=(1-2*case.material.poisson)*A*xyz[:,2]/case.material.young
        constraints = ([(n,2) for n in model.nsets['BOTTOM']] + [(n,1) for n in model.nsets['LATERAL_ANCHORS']] + [(n,3) for n in model.nsets['AXIAL_ANCHOR']] if name=='bottom' else [(r['node'],r['dof']) for r in candidate])
        rf = {int(row[0]):row[1:] for row in reaction}
        pin_values = [float(rf[n][d-1]) for n,d in constraints]
        records.append(dict(variant=name, run=str(target.relative_to(FINAL)),
                            stress=stress, radial_remote_mean_relative_error=float(np.mean(abs(radial[remote]-theory[remote])/theory[remote])),
                            axial_max_error_over_end_displacement=float(np.max(abs(displacement[:,3]-axial_theory))/(abs(A*(1-2*case.material.poisson)*case.pipe.length/2/case.material.young))),
                            maximum_constrained_dof_reaction_n=max(map(abs,pin_values)),
                            constrained_dof_reactions=[dict(node=n,dof=d,reaction_n=v) for (n,d),v in zip(constraints,pin_values)],
                            net_reaction_n=np.sum(reaction[:,1:],axis=0).tolist(),
                            input_hashes=manifest['files'], solver=json.loads((target/'status.json').read_text()),
                            native_sha256={n:sha256(target/n) for n in ['job.dat','job.sta','job.frd','solver.log']}))
    checks = dict(axial_stress_2percent=records[1]['stress']['axial_mean_relative_error'] < .02,
                  hoop_stress_2percent=records[1]['stress']['hoop_mean_pointwise_relative_error'] < .02,
                  radial_displacement_2percent=records[1]['radial_remote_mean_relative_error'] < .02)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    write_json(OUT,dict(records=records, checks=checks, passed=all(checks.values()), reporter_sha256=sha256(Path(__file__)), limitations='Low-pressure elastic matched adaptive mesh only. Stress metrics cover DEFECT integration points, not the full pipe. RF at free loaded nodes includes applied loads; pin metrics use only constrained DOFs. Not experimental validation, mesh convergence, finite-expansion equilibrium or stage-specific release verification. Candidate not adopted in CalculiX/Abaqus campaign.'))
    print(json.dumps(dict(records=records,checks=checks),indent=2))
    if not all(checks.values()):raise SystemExit('Benchmark gate failed; recorded evidence; candidate remains unadopted')


def main():
    global ROOT, SOURCE, OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','solve','report'])
    parser.add_argument('--run-id', help='New immutable benchmark directory name under sym/runs')
    parser.add_argument('--mesh-scale', type=float, default=1.0,
                        help='Scale fine/far surface sizes only; wall layers unchanged')
    parser.add_argument('--all-pipe-stress', action='store_true',
                        help='Print stress/strain/coordinates at all PIPE integration points')
    args=parser.parse_args()
    if not 0 < args.mesh_scale <= 1:
        parser.error('mesh scale must be in (0, 1]')
    if args.run_id:
        if Path(args.run_id).name != args.run_id or args.run_id in {'.', '..'}:
            parser.error('run-id must be a single directory name')
        ROOT = FINAL/'sym/runs'/args.run_id
        SOURCE = ROOT/'common_adaptive'
        OUT = FINAL/'results/output/pressure_restraint_candidates'/f'{args.run_id}.json'
    elif args.mesh_scale != 1:
        parser.error('mesh refinement requires a new run-id')
    if args.all_pipe_stress and not args.run_id:
        parser.error('all-pipe output requires a new run-id')
    if args.action=='prepare':prepare(args.mesh_scale, args.all_pipe_stress)
    elif args.action=='solve':
        for name in ['bottom','tangential']:
            status=run(ROOT/name,memory_limit_mb=1500)
            print(name,json.dumps(status),flush=True)
            if status['state']!='completed':raise RuntimeError('Benchmark failed; preserve native evidence')
    else:report()


if __name__=='__main__':main()

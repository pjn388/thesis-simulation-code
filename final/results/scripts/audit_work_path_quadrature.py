"""Isolate temporal quadrature on piecewise linear rounded DAT displacement paths."""
import json
from pathlib import Path
import numpy as np
from pipe_sim.model import Model
from pipe_sim.results import dat_blocks
from pipe_sim.runner import sha256
from audit_pressure_failures import converged_times, matches_sta
from audit_end_surface_traction import face_nodes, shape8
F = Path(__file__).resolve().parents[2]

def audit(tag):
    report_path = F/'results/output'/f'{tag}_terminal.json'
    report = json.loads(report_path.read_text())
    run = F/report['run']
    assert report['completed']
    hashes = {**report['input_hashes'], **report['output_hashes']}
    assert all(sha256(run/n) == h for n,h in hashes.items())
    model = Model.load(run/'model.json')
    deck = (run/'job.inp').read_text().split('*DLOAD\n')[1].split('*',1)[0]
    magnitudes = {(int(e),int(f.strip()[1:])):float(p) for e,f,p in (l.split(',') for l in deck.strip().splitlines())}
    spatial, weights = np.polynomial.legendre.leggauss(2)
    kernels = [(shape8(r,s), shape8(r+1e-30j,s).imag/1e-30,
                shape8(r,s+1e-30j).imag/1e-30, wr*ws)
               for r,wr in zip(spatial,weights) for s,ws in zip(spatial,weights)]
    groups = {}
    for surface in ('INNER','END_MINUS','END_PLUS'):
        faces = [(face_nodes(model,e,f),magnitudes[e,f]) for e,f in model.surfaces[surface]]
        groups[surface] = (np.array([n for n,p in faces]), np.array([-p for n,p in faces]))
    previous = {n:np.zeros(3) for n in model.nodes}; previous_t=0.
    totals = {method:{s:0. for s in groups} for method in ('trapezoid','gauss2','gauss3')}
    accepted = converged_times(run/'job.sta'); count=0
    for k,g,t,rows in dat_blocks(run/'job.dat'):
        if k!='U' or g!='PIPE_NODES' or not matches_sta(accepted,t): continue
        current = {int(r[0]):np.array(r[1:]) for r in rows}
        assert set(current)==set(model.nodes) and t>previous_t
        for surface,(nodes,pressure) in groups.items():
            start = np.array([[model.nodes[int(n)]+previous[int(n)] for n in face] for face in nodes])
            delta = np.array([[current[int(n)]-previous[int(n)] for n in face] for face in nodes])
            def virtual_work(a):
                points=start+a*delta; value=0.
                for N,dr,ds,w in kernels:
                    normal=np.cross(np.einsum('i,fij->fj',dr,points),np.einsum('i,fij->fj',ds,points))
                    motion=np.einsum('i,fij->fj',N,delta)
                    value += float(np.sum(pressure*(previous_t+a*(t-previous_t))*w*np.einsum('fj,fj->f',normal,motion)))
                return value
            totals['trapezoid'][surface]+=.5*(virtual_work(0)+virtual_work(1))
            for order in (2,3):
                q,w=np.polynomial.legendre.leggauss(order)
                totals[f'gauss{order}'][surface]+=sum(.5*wi*virtual_work(.5*(qi+1)) for qi,wi in zip(q,w))
        previous=current;previous_t=t;count+=1
    endpoint=report['work_audit']['history'][-1]
    assert previous_t==1. and abs(sum(totals['trapezoid'].values())-endpoint['external_work_n_mm'])<1e-4
    assert abs(sum(totals['gauss2'].values())-sum(totals['gauss3'].values()))<1e-4
    assert all(sha256(run/n)==h for n,h in hashes.items())
    return dict(run=report['run'],accepted_frames=count,work_by_surface_n_mm=totals,
                external_minus_internal_n_mm={m:sum(v.values())-endpoint['internal_work_n_mm'] for m,v in totals.items()},
                gauss3_minus_trapezoid_n_mm=sum(totals['gauss3'].values())-sum(totals['trapezoid'].values()),
                input_output_hashes=hashes,terminal_sha256=sha256(report_path))

if __name__=='__main__':
    report=dict(cases=[audit(t) for t in ('plastic_tangential_work_p24_20261011T070440Z','plastic_tangential_work_quarterdt_p24_20261011T072144Z')],
                script_sha256=sha256(__file__),sources=['results/scripts/audit_nodal_pressure_loads.py','results/scripts/review_tangential_p24_work_quarterdt.py:74-102'],
                limitations='Gauss2 integrates cubic pressure-times-area exactly along assumed linear displacement/time segments; Gauss3 checks implementation. Spatial2x2 retained. Rounded DAT path is not native assembly trajectory; displacement/time rounding, true nonlinear path, assembly residual and internal integration remain unbounded. No force/work acceptance, cap revision, Abaqus equivalence or physical validation.')
    out=F/'results/output/work_path_quadrature_20261011T080003Z.json'
    assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in c.items() if k not in ('input_output_hashes',)} for c in report['cases']],indent=2))

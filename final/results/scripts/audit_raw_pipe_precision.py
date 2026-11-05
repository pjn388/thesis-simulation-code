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

def quadrature(tag, raw_records=None, report_override=None):
    report_path = F/'results/output'/f'{tag}_terminal.json'
    report = report_override or json.loads(report_path.read_text())
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
        if raw_records is not None:
            record = raw_records[count]
            t = record['time']
            current = {n:np.array(record['u'][n-1]) for n in model.nodes}
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
    assert abs(previous_t-1.)<1e-12
    if raw_records is None and report_override is None:
        assert abs(sum(totals['trapezoid'].values())-endpoint['external_work_n_mm'])<1e-4
    assert abs(sum(totals['gauss2'].values())-sum(totals['gauss3'].values()))<1e-4
    assert all(sha256(run/n)==h for n,h in hashes.items())
    return dict(run=report['run'],accepted_frames=count,work_by_surface_n_mm=totals,
                external_minus_internal_n_mm={m:sum(v.values())-endpoint['internal_work_n_mm'] for m,v in totals.items()},
                gauss3_minus_trapezoid_n_mm=sum(totals['gauss3'].values())-sum(totals['trapezoid'].values()),
                input_output_hashes=hashes,terminal_sha256=sha256(report_path))


def audit(lane):
    from review_raw_pipe_precision import PARENTS, paths
    tag,run=paths(lane)
    terminal_path=F/'results/output'/f'{tag}_terminal.json'
    terminal=json.loads(terminal_path.read_text())
    assert terminal['completed_native']
    parent_path=F/'results/output'/f'{PARENTS[lane]}_terminal.json'
    parent=json.loads(parent_path.read_text()); parent_run=F/parent['run']
    assert sha256(parent_path)==terminal['parent_report_sha256']
    assert terminal['input_hashes']==parent['input_hashes']
    assert sha256(F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso')==terminal['solver_sha256']==parent['solver_sha256']
    for report,directory in ((terminal,run),(parent,parent_run)):
        assert all(sha256(directory/n)==h for n,h in {**report['input_hashes'],**report['output_hashes']}.items())
    identical_dat=(run/'job.dat').read_bytes()==(parent_run/'job.dat').read_bytes()
    records=json.loads((run/'raw.json').read_text())['records']
    accepted=converged_times(run/'job.sta'); frames={}
    for kind,group,t,rows in dat_blocks(run/'job.dat'):
        assert matches_sta(accepted,t)
        frames.setdefault(t,{})[kind]=rows
    assert len(records)==len(frames)
    max_u=max_energy=max_time=0.
    def allowance(value):
        return 0. if value==0 else .5*10**(int(np.floor(np.log10(abs(value))))-6)
    model=Model.load(run/'model.json')
    for record,(t,fields) in zip(records,sorted(frames.items())):
        assert matches_sta(accepted,record['time'])
        assert abs(record['time']-t)<=allowance(t)+1e-14
        max_time=max(max_time,abs(record['time']-t))
        assert record['elements']==len(model.elements)
        for row in fields['U']:
            n=int(row[0])
            for actual,value in zip(record['u'][n-1],row[1:]):
                assert abs(actual-value)<=allowance(value)+1e-13
                max_u=max(max_u,abs(actual-value))
        assert all(record['u'][n-1][d]==0 for n,d in parent['pins'])
        if record is not records[-1]: assert not record['energy']
    endpoint=records[-1]; assert abs(endpoint['time']-1)<1e-12
    energy=dict(endpoint['energy']); printed={int(r[0]):r[1] for r in frames[sorted(frames)[-1]][None]}
    assert set(energy)==set(printed)==set(model.elements)
    # Each energy is a difference between two native cumulative doubles.
    # Sum telescopes; bound individual subtraction rounding conservatively.
    cumulative=0.; subtraction_bound=0.
    for e,value in endpoint['energy']:
        assert abs(value-printed[e])<=allowance(printed[e])+1e-7
        max_energy=max(max_energy,abs(value-printed[e]))
        cumulative+=value
        subtraction_bound+=np.finfo(float).eps*(2*abs(cumulative)+abs(value))
    raw_energy=float(__import__('math').fsum(energy.values()))
    printed_energy=float(__import__('math').fsum(printed.values()))
    override=dict(terminal,completed=True,work_audit=parent['work_audit'])
    rounded=quadrature(PARENTS[lane],report_override=override); raw=quadrature(PARENTS[lane],records,override)
    parent_quad=quadrature(PARENTS[lane])
    parent_frames={}
    for kind,group,t,rows in dat_blocks(parent_run/'job.dat'):
        if kind=='U' and group=='PIPE_NODES': parent_frames.setdefault(t,{})['crown']=-next(r[2] for r in rows if int(r[0])==model.nsets['CROWN'][0])
        if kind=='PEEQ': parent_frames.setdefault(t,{})['peeq']=max(r[-1] for r in rows)
    crown_diff=peeq_diff=0.
    for t,fields in frames.items():
        crown=-next(r[2] for r in fields['U'] if int(r[0])==model.nsets['CROWN'][0])
        crown_diff=max(crown_diff,abs(crown-parent_frames[t]['crown']))
        peeq_diff=max(peeq_diff,abs(max(r[-1] for r in fields['PEEQ'])-parent_frames[t]['peeq']))
    raw_work={m:sum(v.values()) for m,v in raw['work_by_surface_n_mm'].items()}
    rounded_work={m:sum(v.values()) for m,v in rounded['work_by_surface_n_mm'].items()}
    return dict(run=terminal['run'],parent=parent['run'],accepted_frames=len(records),identical_parent_dat=identical_dat,maximum_parent_crown_difference_mm=crown_diff,maximum_parent_peeq_difference=peeq_diff,
        captured_rounded_minus_parent_gauss2_n_mm=sum(rounded["work_by_surface_n_mm"]["gauss2"].values())-sum(parent_quad["work_by_surface_n_mm"]["gauss2"].values()),
        maximum_u_rounding_mm=max_u,maximum_time_rounding=max_time,maximum_element_energy_rounding_n_mm=max_energy,
        raw_endpoint_energy_n_mm=raw_energy,printed_endpoint_energy_n_mm=printed_energy,
        raw_minus_printed_energy_n_mm=raw_energy-printed_energy,conservative_subtraction_bound_n_mm=float(subtraction_bound),
        raw_work_by_surface_n_mm=raw['work_by_surface_n_mm'],rounded_work_by_surface_n_mm=rounded['work_by_surface_n_mm'],
        raw_external_minus_internal_n_mm={m:w-raw_energy for m,w in raw_work.items()},
        rounded_external_minus_internal_n_mm={m:w-printed_energy for m,w in rounded_work.items()},
        raw_minus_rounded_external_n_mm={m:w-rounded_work[m] for m,w in raw_work.items()},
        terminal_sha256=sha256(terminal_path),parent_terminal_sha256=sha256(parent_path),input_output_hashes={**terminal['input_hashes'],**terminal['output_hashes']},
        verdict='Accepted-frame raw output matches literal rounding intervals, constrained U zero and parent response differences quantified; serial rerun DAT may differ. Work gaps are quantified, not accepted as convergence.',
        limitations='Linear segments between native corrected outputs are assumed; actual nonlinear trajectory and native internal stress/strain temporal integration remain unresolved. Floating bound addresses capture subtraction only. No cap revision, contact/MPC transfer, Abaqus equivalence, burst or experimental validation.')

if __name__=='__main__':
    import sys
    lane=sys.argv[1]
    report=audit(lane);report['script_sha256']=sha256(__file__)
    out=F/'results/output'/f'raw_precision_audit_{lane}_20261011T085724Z.json'
    assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='input_output_hashes'},indent=2))

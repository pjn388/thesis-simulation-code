"""Refine pressure sampling on frozen C3D20R validation meshes, in two lanes."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import campaign
from comparison_utils import loop_branches, error_metrics
from plot_investigation import reference
from pipe_sim.config import Case
from pipe_sim.model import Model
from pipe_sim.decks import write_deck
from pipe_sim.runner import sha256, write_json
from terminal_progress import solver_directories

FINAL=campaign.FINAL
TAG='pressure005_20261020'
OUT=FINAL/'results/output'/TAG
LANES={'a':('p1','p2'), 'b':('p3','p4')}


def identifiers(specimen):
    parent=f'full_zhao2021_{specimen}_adaptive'
    return parent,parent+'_'+TAG


def prepare():
    plan=json.loads(campaign.PLAN.read_text())
    prepared=[]
    for specimen in ('p1','p2','p3','p4'):
        parent_id,name=identifiers(specimen)
        assert not any(j['id']==name for j in plan['jobs']),name
        parent=next(j for j in plan['jobs'] if j['id']==parent_id)
        source=FINAL/parent['case_file'];old=FINAL/parent['run']
        assert sha256(source)==parent['case_sha256']
        manifest=json.loads((old/'manifest.json').read_text())
        for filename,expected in manifest['files'].items():
            assert sha256(old/filename)==expected,filename
        case=Case.load(source);before=case.to_dict()
        assert before==Case.load(old/'case.json').to_dict()
        assert case.mesh.order==2 and case.analysis.pressure_max_increment is None
        assert case.analysis.max_pressure==35 and case.analysis.max_increment==.05
        case.name=name;case.analysis.pressure_max_increment=.005
        case.notes.append('Pressure resolution only: retain frozen C3D20R mesh and original material, fixture, denting/withdrawal steps, loads and failure strain. Pressure-stage initial and maximum increments capped at0.005, equivalent to0.175MPa at35MPa target. Minimum increment and convergence tolerances unchanged. First crossing remains provisional; no new fracture calibration.')
        case.validate()
        for k in ('pipe','gouge','dent','material','mesh','probe_positions'):assert case.to_dict()[k]==before[k]
        assert {k:v for k,v in case.to_dict()['analysis'].items() if k!='pressure_max_increment'}=={k:v for k,v in before['analysis'].items() if k!='pressure_max_increment'}
        cp=campaign.CASES/(name+'.json');assert not cp.exists();case.save(cp)
        directory=campaign.RUNS/name;directory.mkdir()
        case.save(directory/'case.json')
        write_json(directory/'status.json',{'state':'building'})
        for filename in ('model.json','mesh.vtu','geometry.png'):
            shutil.copy2(old/filename,directory/filename)
        write_deck(case,Model.load(old/'model.json'),directory)
        assert sha256(directory/'mesh.inp')==sha256(old/'mesh.inp')
        prefix,pressure=(old/'job.inp').read_text().split('** Stage: pressurise')
        prefix=prefix.replace(parent_id,name,1)
        original=f'{case.analysis.initial_increment:.12g}, 1., {case.analysis.min_increment:.12g}, {case.analysis.max_increment:.12g}\n'
        assert pressure.count(original)==1
        expected=prefix+'** Stage: pressurise'+pressure.replace(original,'0.005, 1., 1e-06, 0.005\n',1)
        assert (directory/'job.inp').read_text()==expected,'Unexpected deck change'
        manifest.update(case=name,created_utc=datetime.now(timezone.utc).isoformat(),
            parent_run=parent['run'],parent_manifest_sha256=sha256(old/'manifest.json'),
            preparation_script_sha256=sha256(Path(__file__)),
            mesh_provenance='Exact frozen parent model/mesh, no remeshing',
            files={n:sha256(directory/n) for n in ('case.json','model.json','mesh.inp','job.inp')},
            source_hashes={str(p.relative_to(FINAL/'sym')):sha256(p) for p in (FINAL/'sym/src/pipe_sim').glob('*.py')})
        write_json(directory/'manifest.json',manifest)
        write_json(directory/'status.json',{'state':'generated'})
        child=copy.deepcopy(parent)
        child.update(id=name,parent_attempt=parent_id,case_file=str(cp.relative_to(FINAL)),case_sha256=sha256(cp),
            run=str(directory.relative_to(FINAL)),changes={'analysis':{'pressure_max_increment':.005}},
            description='Frozen C3D20R pressure-step refinement; unchanged denting and physics')
        child.pop('superseded_by',None)
        child['entry'].update(id=name,case_file=child['case_file'],case_sha256=child['case_sha256'],assumptions=case.notes)
        plan['jobs'].append(child);prepared.append(name)
    write_json(campaign.PLAN,plan)
    print('Prepared exact-parent-mesh cases:',prepared,flush=True)


def report():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _report()


def _report():
    jobs={j['id']:j for j in json.loads(campaign.PLAN.read_text())['jobs']}
    tests=pd.read_csv(FINAL/'data/tables/zhao2021/table04_burst_tests.csv')
    fe=pd.read_csv(FINAL/'data/figures/zhao2021/fig14_failure_pressure.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    burst,bax=plt.subplots(figsize=(9,5),layout='constrained')
    rows,metrics,sources=[],[],{}
    for index,specimen in enumerate(('p1','p2','p3','p4')):
        parent,child=identifiers(specimen);ax=axes.flat[index]
        refs=reference(jobs[parent]['entry'])
        for role,ref in refs.items():
            ax.plot(ref.x,ref.y,'o-' if role=='experiment' else '--',ms=2,label='Physical test' if role=='experiment' else 'Paper FE')
        measured=float(tests.loc[tests.specimen==specimen.upper(),'burst_pressure_MPa'].iloc[0])
        point=fe[abs(fe.x-measured)<=fe.x_digitization_uncertainty];assert len(point)==1
        point=point.iloc[0]
        bax.plot(index-.27,measured,'ko',label='Physical burst' if index==0 else None)
        bax.errorbar(index-.09,point.y,yerr=point.y_digitization_uncertainty,fmt='^',color='gray',label='Paper FE' if index==0 else None)
        for refined,name,color,offset in [(False,parent,'#0072b2',.09),(True,child,'#d55e00',.27)]:
            j=jobs[name];directory=FINAL/j['run'];case=Case.load(FINAL/j['case_file'])
            status=json.loads((directory/'status.json').read_text())
            row=dict(case_id=name,specimen=specimen.upper(),refined=refined,solver_state=status['state'],experiment_mpa=measured,
                paper_fe_mpa=float(point.y),criterion_pressure_mpa=None,lower_pressure_mpa=None,error_percent=None,validated=False)
            hp=directory/'results/history.csv'
            if status['state'] in ('running','building','generated') or not hp.exists():
                row['assessment']='pending';rows.append(row);continue
            from compare import check_run
            check_run(j,j['entry'])
            h=pd.read_csv(hp);s=json.loads((directory/'results/summary.json').read_text())
            loop=h[h.stage.isin(['indent','withdraw'])].copy()
            loop['x']=loop.indenter_travel_mm-case.dent.clearance;loop['y']=loop.indenter_force_n/1000
            ax.plot(loop.x,loop.y,color=color,label='Refined pressure steps' if refined else 'Parent C3D20R')
            row.update(assessment=s['status'],last_pressure_mpa=float(h.pressure_mpa.max()),max_peeq=float(h.max_peeq.max()),
                contact_checks_passed=s.get('contact_checks_passed'),withdrawal_completed=s.get('withdrawal_completed'),
                peak_error_percent=100*(loop.y.max()/refs['experiment'].y.max()-1) if len(loop) else None)
            usable=s.get('status')=='criterion_crossed' and s.get('contact_checks_passed') is True and s.get('withdrawal_completed') is True
            if usable:
                lower,upper=s['pressure_bracket_mpa']
                row.update(lower_pressure_mpa=lower,criterion_pressure_mpa=upper,bracket_width_mpa=upper-lower,
                    error_percent=100*(upper/measured-1),lower_error_percent=100*(lower/measured-1))
                bax.errorbar(index+offset,upper,yerr=[[upper-lower],[0]],fmt='s',color=color,capsize=3)
            for role,ref in refs.items():
                for branch,part in loop_branches(ref).items():
                    sim=loop[loop.stage==branch]
                    if branch=='withdraw' and len(sim):sim=pd.concat([loop[loop.stage=='indent'].tail(1),sim])
                    m,_=error_metrics(part,sim);m.update(case_id=name,reference=role,branch=branch);metrics.append(m)
            rows.append(row)
            for p in (hp,directory/'results/summary.json',directory/'manifest.json',directory/'status.json',directory/'job.sta'):
                sources[str(p.relative_to(FINAL))]=sha256(p)
        ax.set(title=specimen.upper(),xlabel='Tool penetration (mm)',ylabel='Force (kN)');ax.grid(alpha=.2);ax.legend(fontsize=8)
        refpath=FINAL/jobs[parent]['entry']['reference_file'];sources[str(refpath.relative_to(FINAL))]=sha256(refpath)
    for label,color in [('Parent criterion bracket','#0072b2'),('Refined criterion bracket','#d55e00')]:bax.plot([],[],'s',color=color,label=label)
    bax.set(xticks=range(4),xticklabels=['P1','P2','P3','P4'],ylabel='Pressure (MPa)',title='Physical tests, published FE and provisional criterion brackets')
    bax.grid(alpha=.2);bax.legend(fontsize=8)
    burst.text(.02,.01,'Missing markers mean no usable crossing. Brackets are load-step intervals, not confidence intervals.',fontsize=8)
    for f,name in ((fig,'denting_comparison'),(burst,'burst_comparison')):
        f.savefig(OUT/(name+'.png'),dpi=160);f.savefig(OUT/(name+'.pdf'));plt.close(f)
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'branch_metrics.csv',index=False)
    for p in (Path(__file__),FINAL/'data/tables/zhao2021/table04_burst_tests.csv',FINAL/'data/figures/zhao2021/fig14_failure_pressure.csv'):
        sources[str(p.relative_to(FINAL))]=sha256(p)
    write_json(OUT/'provenance.json',dict(sources=sources,limitations='Terminal extracted results only. Physical error is descriptive; no new accuracy cutoff or calibrated failure law. Frozen mesh and pre-pressure deck identity verified at preparation. Refined histories still require bracket/mesh/material/fixture assessment. No solver stop is labelled burst.'))
    print('Refreshed physical comparisons:',OUT,flush=True)


def queue(lane):
    progress=FINAL/'results/output/campaign'/('queue_bracket_'+lane+'.json')
    names=[identifiers(s)[1] for s in LANES[lane]]
    for index,name in enumerate(names):
        while True:
            if json.loads((FINAL/'results/automation/paused.json').read_text()).get('state')!='resumed':
                write_json(progress,dict(state='paused',current=name,index=index,total=len(names)));return
            native=solver_directories(set(),run_root=FINAL/'sym')
            if native is not None and len(native)<2 and campaign.available_memory_mb()>=13048 and shutil.disk_usage(FINAL).free>=40*1024**3:break
            write_json(progress,dict(state='waiting_resources',current=name,index=index,total=len(names)))
            time.sleep(10)
        write_json(progress,dict(state='running',current=name,index=index,total=len(names),cases=names))
        campaign.execute(name)
        report()
    write_json(progress,dict(state='finished',total=len(names),cases=names))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','report','queue']);p.add_argument('lane',nargs='?',choices=['a','b']);args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='report':report()
    else:
        if args.lane is None:p.error('queue requires lane a or b')
        queue(args.lane)

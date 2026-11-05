"""Frozen validation and gouge-orientation campaign, preserving native attempts."""
import argparse,copy,json,math,os,shutil,sys
import time
import subprocess
import fcntl
from pathlib import Path
import numpy as np
import pandas as pd
from pipe_sim.config import Case
from pipe_sim.geometry import symmetric_grid
from pipe_sim.runner import generate,run,write_json,sha256
from pipe_sim.results import extract
from prepare_cases import FINAL,RESULTS
PLAN=RESULTS/'config/full_campaign.json'
CASES=FINAL/'sym/cases/campaign'
RUNS=FINAL/'sym/runs/campaign'
OUT=RESULTS/'output/campaign'


def usable_predictions(frame):
    """Accept pressure crossings only after valid contact and required withdrawal."""
    return frame[(frame.failure_status=='criterion_crossed') &
                 (frame.contact_checks_passed==True) &
                 ((frame.dent==False) | (frame.withdrawal_completed==True)) &
                 frame.criterion_pressure_mpa.notna()]


def estimate(c):
    g,m,p=c.gouge,c.mesh,c.pipe;a=math.radians(g.angle_deg)
    contact=c.dent.indenter_diameter/2+m.refinement_padding if c.dent.enabled else 2*m.fine_size
    sx=max(contact,(g.length*math.sin(a)+g.width*math.cos(a))/2+m.refinement_padding if g.enabled else 0)
    zx=max(contact,(g.length*math.cos(a)+g.width*math.sin(a))/2+m.refinement_padding if g.enabled else 0)
    return (len(symmetric_grid(math.pi*p.od/2,sx,m.fine_size,m.far_size))-1)*(len(symmetric_grid(p.length/2,zx,m.fine_size,m.far_size))-1)*m.wall_layers


def prepare():
    if PLAN.exists():raise FileExistsError('Campaign already frozen')
    entries=json.loads((RESULTS/'config/literature_cases.json').read_text());jobs=[]
    CASES.mkdir(exist_ok=True);RUNS.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    def add(name,c,kind,entry=None,angles=None):
        c.name=name;c.analysis.linear_solver='pardiso';c.analysis.threads=3;c.analysis.timeout_seconds=14400
        c.notes+=['Frozen campaign: no force fitting or orientation-dependent material/criterion adjustment.',
                  'Pressure target is an applied load, not a predicted burst pressure. Missing criterion crossings remain missing.']
        c.validate();path=CASES/(name+'.json');c.save(path)
        e=copy.deepcopy(entry)
        if e:e.update(id=name,case_file=str(path.relative_to(FINAL)),case_sha256=sha256(path),assumptions=c.notes)
        jobs.append(dict(id=name,kind=kind,case_file=str(path.relative_to(FINAL)),case_sha256=sha256(path),
                         run=str((RUNS/name).relative_to(FINAL)),entry=e,estimated_elements=estimate(c),angles=angles))
    selected=[e for e in entries if e['id'].endswith('_figure_curve') or e['id'].startswith('zhao2021_') or e['id']=='shuai2018_dent']
    for e in selected:
        c=Case.load(FINAL/e['case_file'])
        c.analysis.max_pressure=35. if e['paper']=='zhao2021' else (40. if e['paper'].startswith('allouti') else 10.7)
        c.analysis.max_increment=.05
        c.notes+=['Campaign supersedes the earlier 5 MPa workflow check; failure criterion remains study-specific/unvalidated as documented.']
        add('full_'+e['id'],c,'validation',e)
    parent=next(e for e in entries if e['id']=='zhao2021_p1');base=Case.load(FINAL/parent['case_file'])
    base.analysis.max_pressure=35.;base.analysis.max_increment=.05
    for kind,angles in [('intact',[0]),('dent',[0]),('gouge',[0,15,30,45,60,75,90]),('combined',[15,30,45,60,75,90])]:
        for a in angles:
            c=Case.from_dict(base.to_dict());c.dent.enabled=kind in ('dent','combined');c.gouge.enabled=kind in ('gouge','combined');c.gouge.angle_deg=a
            c.notes+=['Zhao P1 geometry/material; gouge width 20 mm and cosine profile remain provisional.', 'Orientation is measured from the pipe axis.']
            add(f'orientation_{kind}_{a:02d}',c,'orientation',angles=[a])
    reference=next(e for e in entries if e['id']=='allouti2012_3_percent_od_figure_curve')
    for name,changes in [('a37_refined',{'mesh':{'fine_size':3.,'wall_layers':6,'indenter_size':2.}}),('a37_finite_support',{'analysis':{'support':'bottom_patch','support_length':88.9}})]:
        c=Case.load(FINAL/reference['case_file']);c.analysis.max_pressure=5.
        for group,params in changes.items():
            for key,value in params.items():setattr(getattr(c,group),key,value)
        c.notes+=['Controlled sensitivity, not a fitted material or a recovered fixture dimension.']
        add(name,c,'iteration',reference)
    plan={'angles_deg':[0,15,30,45,60,75,90],'orientation_combined_0':'full_zhao2021_p1',
          'baseline':'Zhao P1; provisional 20 mm cosine gouge width','jobs':jobs,
          'solver_build':json.loads((FINAL/'sym/vendor/ccx_pardiso/build.json').read_text())}
    write_json(PLAN,plan);print(pd.DataFrame(jobs)[['id','kind','estimated_elements']].to_string(index=False))


def execute(name):
    """Exclude duplicate builds/solves even when independent queues overlap."""
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('Invalid campaign case ID')
    locks=OUT/'.execution_locks';locks.mkdir(parents=True,exist_ok=True)
    with (locks/(name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _execute(name)


def _execute(name):
    plan=json.loads(PLAN.read_text());j=next(j for j in plan['jobs'] if j['id']==name);path=FINAL/j['case_file'];directory=FINAL/j['run']
    if j.get('superseded_by'):
        print('Superseded by',j['superseded_by'],flush=True);return
    if sha256(path)!=j['case_sha256']:raise ValueError('Frozen input changed')
    if shutil.disk_usage(FINAL).free<25*1024**3:raise RuntimeError('Less than 25 GiB free: retain outputs and free space before continuing')
    if directory.exists() and (directory/'status.json').exists():
        state=json.loads((directory/'status.json').read_text())['state']
        if state!='generated':
            print('Preserving existing attempt:',name,state,flush=True);return
    else:generate(Case.load(path),directory)
    print('Solving',name,flush=True);status=run(directory,memory_limit_mb=11000);print(json.dumps(status),flush=True)
    if (directory/'job.dat').exists() and (directory/'job.dat').stat().st_size:
        try:extract(directory)
        except ValueError as exc:write_json(directory/'extraction_error.json',{'error':str(exc)})


def report():
    OUT.mkdir(exist_ok=True)
    with (OUT/'.report.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _report()


def _report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import compare
    OUT.mkdir(exist_ok=True);compare.OUTPUT=OUT
    plan=json.loads(PLAN.read_text());rows=[];metrics=[]
    for j in plan['jobs']:
        d=FINAL/j['run'];c=Case.load(FINAL/j['case_file'])
        state=json.loads((d/'status.json').read_text()) if (d/'status.json').exists() else {'state':'not_run'}
        s=json.loads((d/'results/summary.json').read_text()) if (d/'results/summary.json').exists() else {}
        row=dict(case_id=j['id'],kind=j['kind'],angle_deg=c.gouge.angle_deg,dent=c.dent.enabled,gouge=c.gouge.enabled,
                 active=not bool(j.get('superseded_by')),superseded_by=j.get('superseded_by'),
                 solver_state=state['state'],failure_status=s.get('status'),last_pressure_mpa=s.get('last_converged_pressure_mpa'),
                 observed_peeq=s.get('maximum_observed_peeq'),criterion_pressure_mpa=s.get('burst_pressure_mpa'),
                 pressure_bracket_lower_mpa=s.get('pressure_bracket_mpa',[None,None])[0],
                 contact_checks_passed=s.get('contact_checks_passed'),withdrawal_completed=s.get('withdrawal_completed'),
                 elapsed_seconds=state.get('elapsed_seconds'),run=j['run'],validated=False)
        if (d/'results/history.csv').exists():
            h=pd.read_csv(d/'results/history.csv');row.update(peak_force_kn=float(h.indenter_force_n.max()/1000),
                complete_loading=bool(np.isclose(h.time,1.).any()) if c.dent.enabled else None,
                last_pressure_mpa=float(h.pressure_mpa.max()))
        if j['entry'] and not j.get('superseded_by'):
            e=j['entry'];record={'run':j['run']} if (d/'results/history.csv').exists() else None
            result,errors=(compare.shuai_overlay(e,record) if e.get('comparison_kind')=='pressure_strain' else compare.load_overlay(e,record))
            metrics+=errors
        rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(OUT/'status.csv',index=False);pd.DataFrame(metrics).to_csv(OUT/'validation_errors.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for kind,style in [('gouge','o-'),('combined','s-')]:
        subset=df[(df.kind=='orientation')&df.active&df.gouge&(df.dent==(kind=='combined'))].copy()
        if kind=='combined':subset=pd.concat([subset,df[df.case_id==plan['orientation_combined_0']]])
        valid=usable_predictions(subset)
        ax.errorbar(valid.angle_deg,valid.criterion_pressure_mpa,
                    yerr=[valid.criterion_pressure_mpa-valid.pressure_bracket_lower_mpa,np.zeros(len(valid))],
                    fmt=style,capsize=3,label=kind+' (provisional criterion)')
    ax.set(xlabel='Gouge angle from pipe axis (degrees)',ylabel='First PEEQ 0.8 crossing (MPa)',title='Orientation study — provisional, not experimentally validated',xticks=plan['angles_deg'])
    ax.grid(alpha=.2);ax.legend()
    orientation=df[(df.kind=='orientation')&df.active]
    orientation=pd.concat([orientation,df[df.case_id==plan['orientation_combined_0']]])
    if usable_predictions(orientation).empty:ax.text(.5,.5,'No usable criterion-crossing predictions yet.\nPressure targets and solver stops are not burst results.',ha='center',transform=ax.transAxes)
    fig.savefig(OUT/'orientation_burst.png',dpi=180);fig.savefig(OUT/'orientation_burst.pdf');plt.close(fig)
    # Independent physical burst tests, published FE, and our matching criterion.
    tests=pd.read_csv(FINAL/'data/tables/zhao2021/table04_burst_tests.csv')
    paper_fe=pd.read_csv(FINAL/'data/figures/zhao2021/fig14_failure_pressure.csv')
    burst=[];fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for index,test in tests.iterrows():
        matches=paper_fe[abs(paper_fe.x-test.burst_pressure_MPa)<=paper_fe.x_digitization_uncertainty]
        if len(matches)!=1:raise ValueError('Published FE point/specimen association ambiguous')
        point=matches.iloc[0];key='full_zhao2021_'+test.specimen.lower()+'_adaptive'
        ours=df[df.case_id==key].iloc[0]
        usable=ours.failure_status=='criterion_crossed' and ours.contact_checks_passed==True and ours.withdrawal_completed==True
        prediction=float(ours.criterion_pressure_mpa) if usable else None
        burst.append(dict(specimen=test.specimen,experiment_mpa=float(test.burst_pressure_MPa),paper_fe_mpa=float(point.y),
                          paper_fe_uncertainty_mpa=float(point.y_digitization_uncertainty),our_criterion_mpa=prediction,
                          our_lower_bracket_mpa=ours.pressure_bracket_lower_mpa if usable else None,
                          error_percent=100*(prediction/test.burst_pressure_MPa-1) if usable else None,
                          solver_state=ours.solver_state,status='provisional_prediction' if usable else 'no_usable_prediction',
                          paper_fe_association='Unique Figure14 point within digitization uncertainty of Table4 test pressure'))
        ax.plot(index-.15,test.burst_pressure_MPa,'ko',label='Physical experiment' if index==0 else None)
        ax.errorbar(index,float(point.y),yerr=float(point.y_digitization_uncertainty),fmt='^',color='#888888',label='Paper FE (digitized)' if index==0 else None)
        if usable:ax.errorbar(index+.15,prediction,yerr=[[prediction-ours.pressure_bracket_lower_mpa],[0]],fmt='s',color='#0072b2')
    ax.plot([],[],'s',color='#0072b2',label='Our provisional criterion crossing')
    if not any(r['our_criterion_mpa'] is not None for r in burst):ax.text(.03,.05,'Our results: no usable criterion crossings yet',transform=ax.transAxes)
    ax.set(xticks=range(4),xticklabels=tests.specimen.tolist(),xlabel='Zhao specimen',ylabel='Pressure (MPa)',title='Physical burst tests, published FE and our simulation')
    ax.grid(alpha=.2);ax.legend();fig.savefig(OUT/'zhao_burst_comparison.png',dpi=180);fig.savefig(OUT/'zhao_burst_comparison.pdf');plt.close(fig)
    pd.DataFrame(burst).to_csv(OUT/'burst_validation.csv',index=False)
    active=df[df.active];usable=usable_predictions(active)
    lines=['# Campaign summary','',f'{sum(active.solver_state=="completed")} of {len(active)} active jobs completed their requested load history.',
           'See [live progress](PROGRESS.md) for running stages and converged applied pressures.',
           'Results are provisional: experimental validation, gouge-width/profile confirmation and mesh convergence remain outstanding.','',
           '## Gouge orientation','']
    for kind in ['gouge','combined']:
        selected=usable[(usable.kind=='orientation')&usable.gouge&(usable.dent==(kind=='combined'))]
        if kind=='combined':selected=pd.concat([selected,usable[usable.case_id==plan['orientation_combined_0']]])
        selected=selected.sort_values('angle_deg')
        lines.append(f'{kind.capitalize()}: {len(selected)} of 7 angles have a usable numerical PEEQ 0.8 crossing.')
        if len(selected):
            lines+=['','| Angle from axis | First criterion crossing (MPa) |','|---:|---:|']
            lines += [f'| {r.angle_deg:g}° | {r.criterion_pressure_mpa:.3f} |' for _,r in selected.iterrows()]
        if len(selected)<7:lines+=['','The incomplete angle set cannot establish the overall orientation trend.']
        lines+=['']
    lines+=['Pressure targets, memory/time stops and nonconvergence are not burst pressures.',
            'See status.csv, validation_errors.csv and the per-case three-way response figures for evidence.']
    (OUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print(df[['case_id','solver_state','criterion_pressure_mpa']].to_string(index=False))


def available_memory_mb():
    """Linux reclaimable memory; total free RAM alone excludes usable cache."""
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])/1024
    raise RuntimeError('MemAvailable unavailable; cannot admit another large job')


def request_capacity_review():
    """Compatibility hook; publication workflows do not launch local services."""
    return


def queue(names, queue_id=None):
    """One job per queue lane; reuse terminal attempts and report progress."""
    if queue_id is not None and (not queue_id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in queue_id)):
        raise ValueError('queue_id must contain only lowercase letters, digits, underscores or hyphens')
    progress=OUT/(f'queue_{queue_id}.json' if queue_id else 'queue.json');OUT.mkdir(exist_ok=True)
    for index,name in enumerate(names):
        plan=json.loads(PLAN.read_text());job=next(j for j in plan['jobs'] if j['id']==name)
        status=FINAL/job['run']/'status.json'
        write_json(progress,dict(state='running',current=name,index=index,total=len(names),pid=os.getpid()))
        while status.exists() and json.loads(status.read_text())['state'] in ('running','building'):
            print('Waiting for existing attempt',name,flush=True);time.sleep(30)
        while (not status.exists() or json.loads(status.read_text())['state']=='generated') and available_memory_mb()<11000+2048:
            write_json(progress,dict(state='waiting_memory',current=name,index=index,total=len(names),pid=os.getpid(),required_available_mb=13048))
            print('Waiting for memory before',name,flush=True);time.sleep(30)
        write_json(progress,dict(state='running',current=name,index=index,total=len(names),pid=os.getpid()))
        try:execute(name)
        except Exception as exc:
            write_json(OUT/(name+'_queue_error.json'),{'error':str(exc)})
            print(name,str(exc),flush=True)
        subprocess.run([sys.executable,__file__,'report'],check=True)
    write_json(progress,dict(state='finished',total=len(names)))
    request_capacity_review()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','report','queue']);p.add_argument('names',nargs='*');p.add_argument('--queue-id');a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='report':report()
    elif a.action=='queue':queue(a.names,a.queue_id)
    else:
        for name in a.names:execute(name)

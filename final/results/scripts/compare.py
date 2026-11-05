"""Rebuild paper/CalculiX overlays and descriptive errors from registered runs."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipe_sim.config import Case
from pipe_sim.model import Model
from prepare_cases import FINAL, RESULTS, DATA, digest
from comparison_utils import canonical_curve, loop_branches, error_metrics, pressure_gauge_curve

OUTPUT=RESULTS/'output'


def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def check_run(record,entry):
    directory=FINAL/record['run']
    wanted=Case.load(FINAL/entry['case_file']).to_dict()
    actual=Case.load(directory/'case.json').to_dict()
    if wanted!=actual:raise ValueError('Registered run does not match current literature case')
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,sha in manifest['files'].items():
        if digest(directory/name)!=sha:raise ValueError('Run input hash mismatch: '+name)
    for name,sha in entry['sources'].items():
        if digest(FINAL/name)!=sha:raise ValueError('Reference source changed: '+name)
    if digest(FINAL/entry['material']['source'])!=entry['material']['source_sha256']:
        raise ValueError('Material source changed')
    return directory


def load_overlay(entry, record):
    out=OUTPUT/entry['id'];out.mkdir(parents=True,exist_ok=True)
    ref=pd.read_csv(FINAL/entry['reference_file']).fillna({'panel':''})
    ref=ref[(ref.panel==entry['panel'])&ref.series.isin([entry['experimental_series'],entry['numerical_series']])]
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    reference=[];experimental=None
    for series,part in ref.groupby('series',sort=False):
        part=canonical_curve(part,'mm','kN');reference.append(part)
        exp=series==entry['experimental_series']
        if exp:experimental=part
        ax.plot(part.x,part.y,'o-' if exp else '--',color='#222222' if exp else '#959595',
                ms=3,lw=1.4,label='Paper experiment (digitized)' if exp else 'Paper FE (digitized)')
    pd.concat(reference).to_csv(out/'paper_curves.csv',index=False)
    status={'case_id':entry['id'],'comparison_status':'awaiting_simulation','run':None,
            'reference_file':entry['reference_file'],'reference_sha256':digest(FINAL/entry['reference_file']),
            'validation_status':'preliminary_unvalidated','assumptions':entry['assumptions'],
            'material':entry['material'],'metrics_interpretation':'Descriptive digitized-curve errors, not confidence intervals or a validation pass.'}
    metrics=[]
    if record:
        try:
            directory=check_run(record,entry);status['run']=record['run']
            solver=json.loads((directory/'status.json').read_text());status['solver_state']=solver['state']
            history_path=directory/'results/history.csv'
            if history_path.exists():
                history=pd.read_csv(history_path)
                sim=history[history.stage.isin(['indent','withdraw'])].copy()
                case=Case.load(directory/'case.json')
                sim['x']=sim.indenter_travel_mm-case.dent.clearance;sim['y']=sim.indenter_force_n/1000
                sim['x_unit']='mm';sim['y_unit']='kN'
                sim.to_csv(out/'simulation_curve.csv',index=False)
                ax.plot(sim.x,sim.y,color='#0072b2',lw=2.2,label=f'Our CalculiX ({solver["state"]})')
                status['comparison_status']='overlay_available' if solver['state']=='completed' else 'partial_simulation_overlay'
                status['maximum_observed_peeq']=float(history.max_peeq.max())
                status['material_supported_range_exceeded']=bool(history.max_peeq.max()>entry['material']['source_supported_plastic_strain_max'])
                summary_path=directory/'results/summary.json'
                quality=json.loads(summary_path.read_text()).get('response_quality_issues',[]) if summary_path.exists() else []
                status['response_quality_issues']=quality
                if quality:
                    status['comparison_status']='contact_issue_overlay'
                    ax.text(.03,.85,'Contact quality check failed; diagnostic response only',transform=ax.transAxes,
                            va='top',fontsize=9,color='#b22222')
                if experimental is not None:
                    for branch,points in loop_branches(experimental).items():
                        subset=sim[sim.stage==branch]
                        if branch=='withdraw' and len(subset):
                            subset=pd.concat([sim[sim.stage=='indent'].tail(1),subset],ignore_index=True)
                        result,errors=error_metrics(points,subset)
                        result.update(case_id=entry['id'],branch=branch,error_unit='kN',solver_state=solver['state'],contact_quality_issue=bool(quality))
                        metrics.append(result);errors.to_csv(out/f'{branch}_point_errors.csv',index=False)
                status['run_manifest_sha256']=digest(directory/'manifest.json')
                status['history_sha256']=digest(history_path)
                if status['material_supported_range_exceeded']:
                    ax.text(.03,.94,'Material strain exceeds the recovered curve extent',transform=ax.transAxes,
                            va='top',fontsize=9,color='#a35000')
            else:status['comparison_status']='no_extracted_simulation'
        except (ValueError,FileNotFoundError) as exc:
            status['comparison_status']='rejected_mismatch';status['reason']=str(exc)
    if status['comparison_status'] not in ('overlay_available','partial_simulation_overlay','contact_issue_overlay'):
        ax.text(.04,.94,'Our simulation: '+status['comparison_status'].replace('_',' '),transform=ax.transAxes,va='top',color='#a33')
    ax.set(xlabel='Nominal tool penetration below original OD (mm)',ylabel='Indenting force (kN)',
           title=f'{entry["paper"]} · {entry["specimen"]}\nLoad–displacement comparison — preliminary, unvalidated')
    ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=9)
    fig.text(.02,.005,'Curve order preserved. Contact/support/material assumptions are listed in comparison.json.',fontsize=8)
    fig.savefig(out/'load_displacement.png',dpi=180);fig.savefig(out/'load_displacement.pdf');plt.close(fig)
    save_json(out/'comparison.json',status)
    return status,metrics


def material_plots(entries):
    from textwrap import fill
    out=OUTPUT/'materials';out.mkdir(exist_ok=True)
    for paper in sorted({e['paper'] for e in entries}):
        entry=next(e for e in entries if e['paper']==paper);case=Case.load(FINAL/entry['case_file'])
        source=pd.read_csv(FINAL/entry['material']['source']).sort_values('point_order')
        plastic=np.array(case.material.plastic);total=plastic[:,1]+plastic[:,0]/case.material.young
        fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
        ax.plot(source.x,source.y,'k.',ms=3,label='Published curve (digitized)')
        ax.plot(np.r_[0,total],np.r_[0,plastic[:,0]],color='#0072b2',label='Solver material reconstructed as total strain')
        ax.axvline(source.x.max(),color='#d55e00',ls=':',label='End of recovered curve')
        ax.set(xlabel='Total strain (1)',ylabel='Stress (MPa)',xlim=(0,max(.22,source.x.max()*1.1)),
               title=f'{paper}: material input audit\n'+fill(entry['material']['processing'],75))
        ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.savefig(out/f'{paper}.png',dpi=160);fig.savefig(out/f'{paper}.pdf');plt.close(fig)
        pd.DataFrame({'true_stress_mpa':plastic[:,0],'plastic_strain':plastic[:,1],'reconstructed_total_strain':total}).to_csv(out/f'{paper}_solver_material.csv',index=False)


def burst_plots(entries,records):
    out=OUTPUT/'burst';out.mkdir(exist_ok=True)
    rows=[];latest={r['case_id']:r for r in records}
    selection_path=RESULTS/'config/burst_case_selection.json'
    selections=json.loads(selection_path.read_text()) if selection_path.exists() else {}
    specs=[('allouti2012','table04_burst_tests.csv','vessel'),('allouti2014','table03_burst_and_fracture.csv','vessel'),
           ('zhao2021','table04_burst_tests.csv','specimen')]
    for paper,filename,key in specs:
        source=DATA/'tables'/paper/filename;frame=pd.read_csv(source)
        pressure_column=next(c for c in frame if c.lower()=='burst_pressure_mpa')
        fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
        ax.plot(frame[key].astype(str),frame[pressure_column],'ko',label='Published burst tests (table)')
        predicted=0
        for _,r in frame.iterrows():
            row={'paper':paper,'specimen':str(r[key]),'paper_pressure_mpa':float(r[pressure_column]),
                         'our_burst_pressure_mpa':None,'status':'no_calibrated_matched_burst_prediction',
                         'source':str(source.relative_to(FINAL)),'source_sha256':digest(source)}
            # Zhao uses the same PEEQ criterion; other papers do not. No silent
            # transfer of the 0.8 threshold to Allouti/Oyane/notch criteria.
            matches=[e for e in entries if e['paper']==paper and str(e.get('vessel'))==str(r[key])]
            selected=selections.get(paper,{}).get(str(r[key]))
            if selected:
                matches=[e for e in matches if e['id']==selected]
                if not matches:raise ValueError('Burst case selection is not a matching registered specimen: '+selected)
            else:
                matches=[e for e in matches if e['id'] in latest and
                         (FINAL/latest[e['id']]['run']/'results/summary.json').exists()]
                if len(matches)>1:row['status']='ambiguous_run_selection'
            if paper=='zhao2021' and len(matches)==1 and matches[0]['id'] in latest:
                entry=matches[0];directory=check_run(latest[entry['id']],entry)
                summary=directory/'results/summary.json'
                if summary.exists():
                    s=json.loads(summary.read_text());row['status']=s['status']
                    if s['status']=='criterion_crossed':
                        case=Case.load(directory/'case.json')
                        if case.analysis.failure_peeq!=.8:raise ValueError('Zhao comparison expects the documented 0.8 criterion')
                        p=float(s['burst_pressure_mpa']);lower=float(s['pressure_bracket_mpa'][0])
                        row.update(our_burst_pressure_mpa=p,pressure_lower_mpa=lower,
                            signed_error_percent=100*(p/r[pressure_column]-1),status='provisional_criterion_prediction',
                            run=str(directory.relative_to(FINAL)),case_id=entry['id'],
                            assumptions=json.dumps(entry['assumptions']),
                            material_supported_range_exceeded=s.get('maximum_observed_peeq',0)>entry['material']['source_supported_plastic_strain_max'])
                        ax.errorbar(str(r[key]),p,yerr=[[p-lower],[0]],fmt='s',color='#0072b2',
                            label='Our provisional criterion crossing' if predicted==0 else None);predicted+=1
            rows.append(row)
        ax.text(.03,.05,('No matched criterion predictions available.' if not predicted else 'Criterion predictions retain unresolved geometry/material assumptions.')+'\nA pressure target or solver stop is not plotted as burst.',
                transform=ax.transAxes,fontsize=9,color='#a33')
        maximum=max([float(frame[pressure_column].max())]+[r['our_burst_pressure_mpa'] for r in rows if r['paper']==paper and r['our_burst_pressure_mpa'] is not None])
        ax.set(xlabel='Specimen',ylabel='Burst pressure (MPa)',title=paper+' — burst validation pending',ylim=(0,maximum*1.2))
        ax.grid(alpha=.2);ax.legend();fig.savefig(out/f'{paper}.png',dpi=170);fig.savefig(out/f'{paper}.pdf');plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'burst_comparison.csv',index=False)


def shuai_overlay(entry,record):
    out=OUTPUT/entry['id'];out.mkdir(parents=True,exist_ok=True)
    ref=pd.read_csv(FINAL/entry['reference_file'])
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    status={'case_id':entry['id'],'comparison_status':'awaiting_simulation','run':None,
            'validation_status':'preliminary_unvalidated','assumptions':entry['assumptions']}
    model=None;probes=None;metrics=[];mappings=[]
    if record:
        try:
            directory=check_run(record,entry);status['run']=record['run']
            status['solver_state']=json.loads((directory/'status.json').read_text())['state']
            if (directory/'results/probes.csv').exists():
                model=Model.load(directory/'model.json');probes=pd.read_csv(directory/'results/probes.csv')
        except (ValueError,FileNotFoundError) as exc:status.update(comparison_status='rejected_mismatch',reason=str(exc))
    for ax,gauge in zip(axes.flat,entry['gauges']):
        for direction,component,color in [('circumferential','hoop_strain','#0072b2'),('axial','axial_strain','#d55e00')]:
            experiment=None
            for role in ['Exp','FE']:
                part=ref[(ref.panel==gauge['panel'])&(ref.series==direction+'_'+role)]
                part=canonical_curve(part,'MPa','microstrain')
                if role=='Exp':experiment=part
                ax.plot(part.x,part.y,'o' if role=='Exp' else '--',ms=3,lw=1,color=color,alpha=.65,
                        label=f'Paper {direction} {role}')
            if model is not None:
                radius=Case.load(directory/'case.json').pipe.od/2;s=gauge['arc_mm'];z=gauge['axial_mm']
                target=np.array([radius*np.sin(s/radius),radius*np.cos(s/radius),z])
                ids=model.nsets['GAUGES'];node=min(ids,key=lambda n:np.linalg.norm(np.array(model.nodes[n])-target))
                try:
                    sim=pressure_gauge_curve(probes,node,component)
                    sim['x']=sim.pressure_mpa;sim['y']=sim.gauge_engineering_strain*1e6
                    ax.plot(sim.x,sim.y,color=color,lw=2,label='Our '+direction)
                    sim.to_csv(out/f'gauge_{gauge["gauge"]}_{direction}.csv',index=False)
                    result,errors=error_metrics(experiment,sim);result.update(case_id=entry['id'],branch=f'gauge_{gauge["gauge"]}_{direction}',error_unit='microstrain',solver_state=status['solver_state']);metrics.append(result)
                    errors.to_csv(out/f'gauge_{gauge["gauge"]}_{direction}_errors.csv',index=False)
                    mappings.append(dict(**gauge,direction=direction,node=node,
                        node_offset_mm=float(np.linalg.norm(np.array(model.nodes[node])-target)),
                        integration_point_offset_mm=float(sim.initial_offset_mm.iloc[0])))
                    status['comparison_status']='nearest_ip_strain_overlay'
                except ValueError as exc:status['strain_unavailable_reason']=str(exc)
        ax.set(title=f'Gauge {gauge["gauge"]} (panel {gauge["panel"]})',xlabel='Pressure (MPa)',ylabel='Engineering strain since withdrawal (µε)')
        ax.grid(alpha=.2)
    axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(handles,labels,loc='center',fontsize=9)
    axes.flat[-1].text(.03,.1,'Point 18 excluded: coordinates unresolved.\nOur curves appear after a matching run.\nNearest-IP values are not surface gauge averages.',fontsize=9)
    fig.suptitle('Shuai 2018 — pressure/strain comparison; preliminary, unvalidated')
    fig.savefig(out/'pressure_strain.png',dpi=170);fig.savefig(out/'pressure_strain.pdf');plt.close(fig)
    save_json(out/'gauge_mappings.json',mappings);save_json(out/'comparison.json',status)
    return status,metrics


def strain_overlays():
    """Optional explicit gauge mappings; never guess a gauge's coordinates or zero."""
    config=RESULTS/'config/strain_comparisons.json'
    specs=json.loads(config.read_text()) if config.exists() else []
    for spec in specs:
        if not spec.get('enabled',False):continue
        if not spec.get('mapping_confirmed',False):raise ValueError('Confirm gauge mapping before enabling '+spec['id'])
        out=OUTPUT/'strain'/spec['id'];out.mkdir(parents=True,exist_ok=True)
        directory=FINAL/spec['run'];probe=pd.read_csv(directory/'results/probes.csv')
        sim=pressure_gauge_curve(probe,spec['requested_node'],spec['component'],spec.get('baseline_time',2.))
        reference=pd.read_csv(FINAL/spec['reference_file']).fillna({'panel':''})
        ref=reference[(reference.panel==spec['panel'])&(reference.series==spec['series'])]
        ref=canonical_curve(ref,spec['x_unit'],spec['y_unit'])
        from comparison_utils import conversion
        pressure=sim.pressure_mpa.to_numpy()*conversion('MPa',spec['pressure_unit'])
        strain=sim.gauge_engineering_strain.to_numpy()*conversion('1',spec['strain_unit'])
        x,y=(pressure,strain) if spec['pressure_axis']=='x' else (strain,pressure)
        fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
        ax.plot(ref.x,ref.y,'ko-',ms=3,label='Paper '+spec['series']);ax.plot(x,y,color='#0072b2',label='CalculiX nearest-IP gauge approximation')
        ax.set(xlabel=spec['x_label'],ylabel=spec['y_label'],title=spec['id']+' — gauge mapping requires verification')
        ax.legend();ax.grid(alpha=.2);fig.savefig(out/'comparison.png',dpi=170);fig.savefig(out/'comparison.pdf');plt.close(fig)
        sim.to_csv(out/'simulation_gauge.csv',index=False);ref.to_csv(out/'paper_gauge.csv',index=False)
        save_json(out/'mapping.json',spec)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    entries=json.loads((RESULTS/'config/literature_cases.json').read_text())
    path=RESULTS/'config/runs.json';records=json.loads(path.read_text()) if path.exists() else []
    # Latest registered attempt is explicit; earlier attempts remain in runs.json.
    latest={r['case_id']:r for r in records}
    statuses=[];metrics=[]
    for entry in entries:
        function=shuai_overlay if entry.get('comparison_kind')=='pressure_strain' else load_overlay
        status,rows=function(entry,latest.get(entry['id']));statuses.append(status);metrics+=rows
    pd.DataFrame(metrics,columns=['case_id','branch','solver_state','error_unit','reference_points','overlap_points',
         'coverage_fraction','rmse','mae','max_absolute_error','normalized_rmse_by_reference_range','overlap_x_min','overlap_x_max']).to_csv(OUTPUT/'curve_metrics.csv',index=False)
    save_json(OUTPUT/'comparison_status.json',statuses)
    material_plots(entries);burst_plots(entries,records);strain_overlays()
    pd.DataFrame([{k:s.get(k) for k in ['case_id','comparison_status','solver_state','run','material_supported_range_exceeded']} for s in statuses]).to_csv(OUTPUT/'comparison_status.csv',index=False)
    print(f'Wrote {len(entries)} comparison figures and status records to {OUTPUT}')


if __name__=='__main__':main()

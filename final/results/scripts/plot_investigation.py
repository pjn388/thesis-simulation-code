"""Three-way response plots and control-study errors; no fitting of FE responses."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipe_sim.config import Case
from prepare_cases import FINAL,RESULTS,digest
from comparison_utils import canonical_curve,loop_branches,error_metrics
from compare import check_run
from investigate import PLAN

OUT=RESULTS/'output/investigation'


def reference(entry):
    frame=pd.read_csv(FINAL/entry['reference_file']).fillna({'panel':''})
    result={}
    for role,series in [('experiment',entry['experimental_series']),('paper_fe',entry['numerical_series'])]:
        result[role]=canonical_curve(frame[(frame.panel==entry['panel'])&(frame.series==series)],'mm','kN')
    return result


def plot_material(plan):
    source=pd.read_csv(FINAL/'data/figures/allouti2012/fig09_material_curve.csv').sort_values('point_order')
    baseline=Case.load(FINAL/'sym/cases/literature/allouti2012_3_percent_od.json')
    candidate=Case.load(FINAL/'sym/cases/investigation/a37_curve_only.json')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,limit in zip(axes,[.04,.21]):
        ax.plot(source.x,source.y,'k.',ms=2.5,label='Paper Figure 9 (digitized)')
        for case,color,label in [(baseline,'#d55e00','Previous table-yield splice'),(candidate,'#0072b2','Curve-preserving input')]:
            p=np.array(case.material.plastic)
            ax.plot(np.r_[0,p[:,1]+p[:,0]/case.material.young],np.r_[0,p[:,0]],color=color,lw=2,label=label)
        ax.set(xlim=(0,limit),ylim=(0,470),xlabel='Assumed true total strain (1)',ylabel='Stress (MPa)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Material discrepancy: tabulated yield and plotted curve cannot be silently combined')
    fig.savefig(OUT/'material_discrepancy.png',dpi=180);fig.savefig(OUT/'material_discrepancy.pdf');plt.close(fig)
    plastic_strains=np.array([0,.002,.005,.01,.02,.05,.1,.19])
    row={'plastic_strain':plastic_strains}
    for name,case in [('previous',baseline),('curve_preserving',candidate)]:
        p=np.array(case.material.plastic);row[name+'_stress_mpa']=np.interp(plastic_strains,p[:,1],p[:,0])
    pd.DataFrame(row).to_csv(OUT/'material_comparison.csv',index=False)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=json.loads(PLAN.read_text());registry=json.loads((RESULTS/'config/literature_cases.json').read_text())
    baseline=next(e for e in registry if e['id']=='allouti2012_3_percent_od')
    records=json.loads((RESULTS/'config/runs.json').read_text())
    base_run=next(r for r in records if r['case_id']==baseline['id'])
    jobs=[{'id':'previous_baseline','entry':baseline,'run':base_run['run'],
           'description':'Previous baseline: 355 MPa yield splice, coarse mesh'}]+plan['jobs']
    rows=int(np.ceil(len(jobs)/3))
    metrics=[];summary=[];provenance=[];fig,axes=plt.subplots(rows,3,figsize=(16,5*rows),layout='constrained',squeeze=False)
    for ax in list(axes.flat)[len(jobs):]:ax.set_visible(False)
    for job,ax in zip(jobs,axes.flat):
        entry=job['entry'];refs=reference(entry)
        directory=FINAL/job['run'];target=OUT/job['id'];target.mkdir(exist_ok=True)
        status={'state':'not_run'}
        if (directory/'status.json').exists():status=json.loads((directory/'status.json').read_text())
        sim=None;case=Case.load(FINAL/entry['case_file'])
        record={'case_id':job['id'],'description':job['description'],'solver_state':status['state'],
            'complete_loading':False,'complete_withdrawal':False,'peak_force_kn':None,'peak_error_vs_experiment_percent':None}
        if (directory/'results/history.csv').exists():
            check_run({'run':job['run']},entry)
            h=pd.read_csv(directory/'results/history.csv')
            sim=h[h.stage.isin(['indent','withdraw'])].copy()
            sim['x']=sim.indenter_travel_mm-case.dent.clearance;sim['y']=sim.indenter_force_n/1000
            record['minimum_force_kn']=float(sim.y.min())
            record['unphysical_tensile_force']=bool(sim.y.min() < -max(.001,sim.y.max()*.001))
            record['complete_loading']=bool(np.isclose(h.time,1.).any())
            record['complete_withdrawal']=bool(np.isclose(h.time,2.).any())
            if record['complete_loading']:
                peak=float(sim[sim.stage=='indent'].y.max());record['peak_force_kn']=peak
                record['peak_error_vs_experiment_percent']=100*(peak/refs['experiment'].y.max()-1)
                record['peak_error_vs_paper_fe_percent']=100*(peak/refs['paper_fe'].y.max()-1)
            if record['complete_withdrawal'] and not record['unphysical_tensile_force']:
                record['residual_crown_displacement_mm']=float(h[np.isclose(h.time,2.)].crown_inward_displacement_mm.iloc[-1])
            record['max_observed_peeq']=float(h.max_peeq.max())
            record['material_supported_range_exceeded']=bool(h.max_peeq.max()>entry['material']['source_supported_plastic_strain_max'])
            record['elapsed_seconds']=status.get('elapsed_seconds')
            if (directory/'results/summary.json').exists():
                assessment=json.loads((directory/'results/summary.json').read_text())
                record['excessive_penetration']=assessment.get('max_nodal_contact_penetration_mm',0.)>case.pipe.thickness*.1
                record['max_contact_penetration_mm']=assessment.get('max_nodal_contact_penetration_mm')
            for role,ref in refs.items():
                for branch,part in loop_branches(ref).items():
                    curve=sim[sim.stage==branch]
                    if branch=='withdraw' and len(curve):curve=pd.concat([sim[sim.stage=='indent'].tail(1),curve])
                    values,errors=error_metrics(part,curve)
                    values.update(case_id=job['id'],reference=role,branch=branch,error_unit='kN')
                    values['contact_quality_issue']=record['unphysical_tensile_force'] or record.get('excessive_penetration',False)
                    metrics.append(values);errors.to_csv(target/f'{role}_{branch}_errors.csv',index=False)
            sim.to_csv(target/'our_response.csv',index=False)
            provenance.append({'case_id':job['id'],'run':job['run'],
                'case_sha256':digest(directory/'case.json'),'history_sha256':digest(directory/'results/history.csv'),
                'manifest_sha256':digest(directory/'manifest.json'),'reference_sha256':digest(FINAL/entry['reference_file'])})
        for role,frame in refs.items():frame.to_csv(target/(role+'.csv'),index=False)
        def draw(a):
            ref=refs['experiment'];a.plot(ref.x,ref.y,'o-',color='#202020',lw=1.2,ms=3,label='Paper experiment')
            ref=refs['paper_fe'];a.plot(ref.x,ref.y,'--',color='#7c7c7c',lw=1.8,label='Paper simulation')
            if sim is not None:a.plot(sim.x,sim.y,color='#0072b2',lw=2,label='Our CalculiX')
            else:a.text(.04,.88,status['state'].replace('_',' '),transform=a.transAxes,color='#a33')
            a.grid(alpha=.2);a.set(xlabel='Tool penetration (mm)',ylabel='Force (kN)')
            title=job['description'].replace(';',';\n')
            if len(title)>95:title=title[:95]+'…'
            a.set_title(title,fontsize=10)
            if record['peak_force_kn'] is not None:
                a.text(.03,.96,f'Peak: {record["peak_force_kn"]:.2f} kN ({record["peak_error_vs_experiment_percent"]:+.1f}% vs test)',
                       transform=a.transAxes,va='top',fontsize=9)
            a.legend(loc='lower right',fontsize=8)
            if record.get('unphysical_tensile_force'):
                a.text(.03,.87,'Unphysical tensile tool force: unloading unreliable',transform=a.transAxes,
                       va='top',fontsize=8,color='#b22222')
            if record.get('excessive_penetration'):
                a.text(.03,.81,'INVALID contact: penetration exceeds 10% of wall thickness',transform=a.transAxes,
                       va='top',fontsize=8,color='#b22222')
        draw(ax)
        single,one=plt.subplots(figsize=(9,6),layout='constrained');draw(one)
        single.suptitle(entry['paper']+' · '+entry['specimen']+' — three-way response comparison\nUnvalidated investigation; '+status['state'],fontsize=13)
        single.savefig(target/'three_way_response.png',dpi=180);single.savefig(target/'three_way_response.pdf');plt.close(single)
        summary.append(record)
    fig.suptitle('Controlled mismatch investigation — same solver workflow\nBlack: experiment · dashed grey: published simulation · blue: our CalculiX',fontsize=15)
    fig.savefig(OUT/'controlled_comparisons.png',dpi=180);fig.savefig(OUT/'controlled_comparisons.pdf');plt.close(fig)
    pd.DataFrame(metrics).to_csv(OUT/'response_errors.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    plot_material(plan)
    # Compact reading view; the full grid retains all failed/diagnostic controls.
    selected=['a37_contact_fixed','a37_8pct_contact_fixed']
    overview,panels=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    for name,axis in zip(selected,panels):
        job=next(j for j in jobs if j['id']==name);entry=job['entry'];refs=reference(entry)
        for role,color,style,label in [('experiment','#202020','o-','Paper experiment'),('paper_fe','#888888','--','Paper simulation')]:
            ref=refs[role];axis.plot(ref.x,ref.y,style,color=color,lw=1.6,ms=3,label=label)
        path=OUT/name/'our_response.csv';record=next(r for r in summary if r['case_id']==name)
        if path.exists():
            sim=pd.read_csv(path);axis.plot(sim.x,sim.y,color='#0072b2',lw=2,label='Our CalculiX (contact patch)')
        axis.set(xlabel='Tool penetration (mm)',ylabel='Force (kN)',
                 title='Allouti 2012 · '+entry['specimen'].replace('_',' '))
        if record['peak_force_kn'] is not None:
            axis.text(.04,.95,f'Peak error vs experiment: {record["peak_error_vs_experiment_percent"]:+.1f}%',transform=axis.transAxes,va='top')
        if record['solver_state']!='completed':
            axis.text(.04,.87,'Partial run: '+record['solver_state'],transform=axis.transAxes,color='#b22222')
        if record.get('unphysical_tensile_force') or record.get('excessive_penetration'):
            axis.text(.04,.8,'Contact check failed',transform=axis.transAxes,color='#b22222')
        axis.grid(alpha=.2)
    handles,labels=panels[0].get_legend_handles_labels()
    overview.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=10)
    overview.suptitle('Same model and material treatment, two prescribed dent depths\nThree-way response comparison — preliminary; fixture and mesh checks outstanding',fontsize=13)
    overview.savefig(OUT/'three_way_overview.png',dpi=180);overview.savefig(OUT/'three_way_overview.pdf');plt.close(overview)
    print(pd.DataFrame(summary)[['case_id','solver_state','peak_force_kn','peak_error_vs_experiment_percent']].to_string(index=False))


if __name__=='__main__':main()

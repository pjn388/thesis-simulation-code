"""Compare five completed denting checks with experiments and published FE."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare import check_run
from comparison_utils import error_metrics, loop_branches
from pipe_sim.config import Case
from plot_investigation import reference
from prepare_cases import FINAL, RESULTS, digest


def main():
    output=RESULTS/'output/campaign/dent_validation';output.mkdir(parents=True,exist_ok=True)
    investigation=json.loads((RESULTS/'config/investigation.json').read_text())['jobs']
    jobs=[next(j for j in investigation if j['id']==name)
          for name in ['a37_contact_fixed','a37_8pct_contact_fixed']]
    entries=json.loads((RESULTS/'config/literature_cases.json').read_text())
    records=json.loads((RESULTS/'config/runs.json').read_text())
    for depth in [10,13,16]:
        key=f'allouti2012_{depth}_percent_od_figure_curve'
        entry=next(e for e in entries if e['id']==key)
        record=next(r for r in reversed(records) if r['case_id']==key)
        jobs.append(dict(id=key,entry=entry,run=record['run']))
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    metrics=[];provenance=[]
    for job,ax in zip(jobs,axes.flat):
        entry=job['entry'];refs=reference(entry);directory=check_run({'run':job['run']},entry)
        case=Case.load(directory/'case.json');h=pd.read_csv(directory/'results/history.csv')
        summary=json.loads((directory/'results/summary.json').read_text())
        if not (summary['withdrawal_completed'] and summary['contact_checks_passed']):
            raise ValueError('This comparison requires a complete, physically admissible denting history')
        sim=h[h.stage.isin(['indent','withdraw'])].copy()
        sim['x']=sim.indenter_travel_mm-case.dent.clearance;sim['y']=sim.indenter_force_n/1000
        peak=float(sim.y.max());error=100*(peak/refs['experiment'].y.max()-1)
        row=dict(case_id=job['id'],specimen=entry['specimen'],peak_force_kn=peak,
                 experimental_peak_kn=float(refs['experiment'].y.max()),peak_error_percent=error,
                 residual_crown_displacement_mm=summary['residual_crown_displacement_mm'])
        for branch in ['indent','withdraw']:
            curve=sim[sim.stage==branch]
            if branch=='withdraw':curve=pd.concat([sim[sim.stage=='indent'].tail(1),curve])
            value,_=error_metrics(loop_branches(refs['experiment'])[branch],curve)
            row[branch+'_rmse_kn']=value['rmse']
            row[branch+'_coverage_fraction']=value['coverage_fraction']
        metrics.append(row)
        provenance.append(dict(case_id=job['id'],run=job['run'],history_sha256=digest(directory/'results/history.csv'),
                               reference_file=entry['reference_file'],reference_sha256=digest(FINAL/entry['reference_file'])))
        for name,style,label in [('experiment','o-','Physical experiment'),('paper_fe','--','Paper FE')]:
            frame=refs[name];ax.plot(frame.x,frame.y,style,color='black' if name=='experiment' else '0.6',ms=3,label=label)
        ax.plot(sim.x,sim.y,color='#0072b2',label='Our CalculiX')
        ax.set(title=f"{entry['specimen'].replace('_',' ')} — peak error {error:+.1f}%",
               xlabel='Tool indentation (mm)',ylabel='Force (kN)');ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0,.9,'Same material and contact model across depths.\n\n'
        'Good peak force agreement does not imply\naccurate unloading or burst prediction.\n\n'
        'Material strain convention, support and mesh\nconvergence remain unresolved.\n\n'
        'These checks include a 5 MPa workflow ramp;\nthat ramp is not a physical burst test.',va='top',fontsize=11)
    fig.suptitle('Allouti 2012: denting validation across five depths — provisional')
    fig.savefig(output/'three_way_dent_validation.png',dpi=180);fig.savefig(output/'three_way_dent_validation.pdf');plt.close(fig)
    pd.DataFrame(metrics).to_csv(output/'metrics.csv',index=False)
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__=='__main__':main()

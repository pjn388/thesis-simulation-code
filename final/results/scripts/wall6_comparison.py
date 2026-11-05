"""Terminal C3D8 wall or independent surface comparison; no solve or acceptance fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from formulation_trials import FINAL, TAG, sha256, reference, loop_branches, error_metrics

OUT = FINAL/'results/output/c3d8_20261019/wall6'

def report(variant="wall6"):
    assert variant in ("wall6", "fine4")
    out = OUT if variant == "wall6" else OUT.parent / "fine4"
    parameter = "wall_layers" if variant == "wall6" else "fine_size"
    mesh_values = (4, 6) if variant == "wall6" else (5, 4)
    jobs = {j['id']: j for j in json.loads((FINAL/'results/config/full_campaign.json').read_text())['jobs']}
    rows, branches, common, sources = [], [], [], {}
    out.mkdir(exist_ok=True)
    for specimen in ('p1', 'p4'):
        base = f'full_zhao2021_{specimen}_adaptive_{TAG}'
        refs = reference(jobs[base]['entry'])
        for rel in (jobs[base]['entry']['reference_file'], 'results/output/c3d8_20261019/provenance.json', 'results/output/c3d8_20261019/p4/provenance.json'):
            sources[rel] = sha256(FINAL/rel)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        for role, ref in refs.items():
            axes[0].plot(ref.x, ref.y, 'o-' if role == 'experiment' else '--', ms=2, label=role)
        for layers, name in ((mesh_values[0], base), (mesh_values[1], base+"_"+variant)):
            job=jobs[name]; run=FINAL/job['run']
            status=json.loads((run/'status.json').read_text())
            assert status['state'] in ('failed', 'completed'), 'Terminal results required'
            manifest=json.loads((run/'manifest.json').read_text())
            for filename, expected in manifest['files'].items():
                assert sha256(run/filename)==expected
            case=json.loads((FINAL/job['case_file']).read_text())
            parent=json.loads((FINAL/jobs[base]['case_file']).read_text())
            for key in ('pipe','gouge','dent','material','analysis','probe_positions'):
                assert case[key]==parent[key], key
            assert {k:v for k,v in case['mesh'].items() if k!=parameter}=={k:v for k,v in parent['mesh'].items() if k!=parameter}
            h=pd.read_csv(run/'results/history.csv'); summary=json.loads((run/'results/summary.json').read_text())
            loop=h[h.stage.isin(['indent','withdraw'])].copy()
            loop['x']=loop.indenter_travel_mm-case['dent']['clearance'];loop['y']=loop.indenter_force_n/1000
            axes[0].plot(loop.x,loop.y,label=f"{parameter}={layers}")
            p=h[h.stage=='pressurise'].sort_values('pressure_mpa')
            axes[1].plot(p.pressure_mpa,p.max_peeq,label=f"{parameter}={layers}")
            for role,ref in refs.items():
                for branch,part in loop_branches(ref).items():
                    curve=loop[loop.stage==branch]
                    if branch=='withdraw':curve=pd.concat([loop[loop.stage=='indent'].tail(1),curve])
                    values,_=error_metrics(part,curve)
                    branches.append(dict(specimen=specimen,**{parameter:layers},reference=role,branch=branch,**values))
            supported=job['entry']['material']['source_supported_plastic_strain_max']
            rows.append(dict(specimen=specimen,**{parameter:layers},case_id=name,state=status['state'],last_pressure_mpa=float(h.pressure_mpa.max()),max_peeq=float(h.max_peeq.max()),material_supported_peeq=supported,within_material_range=bool(h.max_peeq.max()<=supported),peak_force_kn=float(loop.y.max()),peak_error_percent=float(100*(loop.y.max()/refs['experiment'].y.max()-1)),**{k:summary.get(k) for k in ('status','withdrawal_completed','contact_checks_passed','max_nodal_contact_penetration_mm','residual_crown_displacement_mm','pressure_bracket_mpa','burst_pressure_mpa')}))
            for pressure in (5.,10.,15.,16.53,20.,22.79,25.,26.):
                if len(p) and p.pressure_mpa.min()<=pressure<=p.pressure_mpa.max():
                    common.append(dict(specimen=specimen,**{parameter:layers},pressure_mpa=pressure,max_peeq=float(np.interp(pressure,p.pressure_mpa,p.max_peeq))))
            for path in (run/'results/history.csv',run/'results/summary.json',run/'status.json',run/'manifest.json',FINAL/job['case_file']):
                sources[str(path.relative_to(FINAL))]=sha256(path)
        axes[0].set(title=specimen.upper()+' denting',xlabel='Tool penetration (mm)',ylabel='Force (kN)')
        axes[1].axhline(.8,color='black',linestyle=':',label='Unchanged criterion')
        axes[1].set(title=specimen.upper()+' pressure response',xlabel='Pressure (MPa)',ylabel='Maximum PEEQ')
        for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.savefig(out/f'{specimen}.pdf');fig.savefig(out/f'{specimen}.png',dpi=140);plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(branches).to_csv(out/'branch_metrics.csv',index=False)
    pd.DataFrame(common).to_csv(out/'common_pressure.csv',index=False)
    sources[str(Path(__file__).relative_to(FINAL))]=sha256(Path(__file__))
    (out/'provenance.json').write_text(json.dumps(dict(sources=sources,limitations='Terminal numerical convergence stops are not burst. Branch overlap only; no alignment or fitting. Crown is not measured residual shape. Common-pressure PEEQ uses linear interpolation of accepted frames. No mesh convergence or physical validation established.'),indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('wall6','fine4'), default='wall6')
    report(parser.parse_args().variant)

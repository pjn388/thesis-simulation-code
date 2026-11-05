import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import pandas as pd
from .config import Case
from .runner import ROOT, generate, run, solver_path, write_json, collect_external
from .results import extract, stage_at
from .verification import elastic_benchmark, compare_reference
from .comparison import comparison_group, interaction_table


def default_run(case):
    return ROOT/'runs'/f'{datetime.now():%Y%m%d_%H%M%S_%f}_{case.name}_calculix'


def matrix_cases(base):
    """Intact, dent-only, and three matched gouge/combined pairs."""
    for kind,angles in [('intact',[0]),('dent',[0]),('gouge',[0,45,90]),('combined',[0,45,90])]:
        for angle in angles:
            c=Case.from_dict(base.to_dict())
            c.name=f'{base.name}_{kind}_{angle:02d}'
            c.dent.enabled=kind in ('dent','combined')
            c.gouge.enabled=kind in ('gouge','combined')
            c.gouge.angle_deg=angle
            yield c


def compare(directories, output):
    records=[]
    for directory in map(Path,directories):
        c=Case.load(directory/'case.json')
        summary=json.loads((directory/'results/summary.json').read_text())
        records.append({'case':c.name,'angle_deg':c.gouge.angle_deg,'dent':c.dent.enabled,
                        'gouge':c.gouge.enabled,'status':summary['status'],
                        'comparison_group':comparison_group(c),
                        'burst_pressure_mpa':summary.get('burst_pressure_mpa'),
                        'pressure_lower_mpa':summary.get('pressure_bracket_mpa',[None,None])[0],
                        'validation_status':'unvalidated','run':str(directory.resolve())})
    df=pd.DataFrame(records);Path(output).parent.mkdir(parents=True,exist_ok=True);df.to_csv(output,index=False)
    interaction_table(df).to_csv(Path(output).with_name(Path(output).stem+'_interactions.csv'),index=False)
    return df


def main(argv=None):
    parser=argparse.ArgumentParser(description='Parametric pipe dent/gouge simulations (mm, N, MPa).')
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ['generate','run']:
        p=sub.add_parser(command);p.add_argument('case',type=Path)
        p.add_argument('--out',type=Path)
    p=sub.add_parser('solve');p.add_argument('directory',type=Path)
    p=sub.add_parser('extract');p.add_argument('directory',type=Path)
    p=sub.add_parser('collect');p.add_argument('directory',type=Path)
    p=sub.add_parser('status');p.add_argument('directory',type=Path)
    p=sub.add_parser('matrix');p.add_argument('case',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    p=sub.add_parser('convergence');p.add_argument('case',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--factors',nargs='+',type=float,default=[1.,.75,.5]);p.add_argument('--run',action='store_true')
    p=sub.add_parser('compare');p.add_argument('directories',nargs='+');p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('benchmark');p.add_argument('directory',type=Path)
    p=sub.add_parser('validate');p.add_argument('directory',type=Path);p.add_argument('--specimen',required=True)
    p.add_argument('--reference',type=Path,required=True)
    sub.add_parser('doctor')
    args=parser.parse_args(argv)
    try:
        if args.command=='doctor':
            import gmsh
            print(json.dumps({'python':sys.executable,'in_venv':sys.prefix!=sys.base_prefix,
                              'gmsh':gmsh.__version__,'calculix':solver_path()},indent=2));return 0
        if args.command=='extract':print(json.dumps(extract(args.directory),indent=2));return 0
        if args.command=='collect':print(json.dumps(collect_external(args.directory),indent=2));return 0
        if args.command=='status':
            status=json.loads((args.directory/'status.json').read_text())
            case=Case.load(args.directory/'case.json');sta=args.directory/'job.sta'
            manifest=json.loads((args.directory/'manifest.json').read_text())
            if manifest['solver']=='calculix' and sta.exists():
                for line in reversed(sta.read_text(errors='replace').splitlines()):
                    fields=line.split()
                    if len(fields)==7 and fields[0].isdigit() and fields[2].isdigit():
                        t=float(fields[4]);stage,p=stage_at(case,t)
                        status['last_converged']={'stage':stage,'total_time':t,'pressure_mpa':p,'increment':int(fields[1])};break
            print(json.dumps(status,indent=2));return 0
        if args.command=='benchmark':print(json.dumps(elastic_benchmark(args.directory),indent=2));return 0
        if args.command=='validate':print(json.dumps(compare_reference(args.directory,args.specimen,args.reference),indent=2));return 0
        if args.command=='compare':print(compare(args.directories,args.out).to_string(index=False));return 0
        if args.command=='solve':
            status=run(args.directory);print(json.dumps(status,indent=2))
            manifest=json.loads((args.directory/'manifest.json').read_text())
            if manifest['solver']=='calculix' and (args.directory/'job.dat').exists() and (args.directory/'job.dat').stat().st_size:
                print(json.dumps(extract(args.directory),indent=2))
            return 0 if status['state']=='completed' else 2
        case=Case.load(args.case)
        if args.command in ('generate','run'):
            out=generate(case,args.out or default_run(case))
            print(f'Generated: {out}',flush=True)
            if args.command=='run':
                status=run(out);print(json.dumps(status,indent=2),flush=True)
                if (out/'job.dat').exists() and (out/'job.dat').stat().st_size:
                    print(json.dumps(extract(out),indent=2))
                return 0 if status['state']=='completed' else 2
        elif args.command in ('matrix','convergence'):
            if args.out.exists() and any(args.out.iterdir()):raise FileExistsError('Study directory must be new or empty')
            args.out.mkdir(parents=True,exist_ok=True)
            configs=args.out/'cases';configs.mkdir()
            if args.command=='matrix':cases=list(matrix_cases(case))
            else:
                cases=[]
                for factor in args.factors:
                    if factor<=0:raise ValueError('Mesh factors must be positive')
                    c=Case.from_dict(case.to_dict());c.name+=f'_mesh_{factor:g}'.replace('.','p')
                    c.mesh.fine_size*=factor;c.mesh.far_size*=factor;c.mesh.indenter_size*=factor
                    c.mesh.wall_layers=max(2,round(case.mesh.wall_layers/factor));cases.append(c.validate())
            results=[];failed=False
            for c in cases:
                c.save(configs/(c.name+'.json'))
                out=generate(c,args.out/c.name,'calculix');print(out,flush=True)
                if args.run:
                    status=run(out)
                    if (out/'job.dat').exists() and (out/'job.dat').stat().st_size:
                        extract(out);results.append(out)
                    if status['state']!='completed':
                        failed=True;print(f'{c.name}: {status["state"]}',file=sys.stderr)
            if results:compare(results,args.out/'comparison.csv')
            if failed:return 2
        return 0
    except (ValueError,TypeError,FileNotFoundError,FileExistsError,RuntimeError) as e:
        print(f'Error: {e}',file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())

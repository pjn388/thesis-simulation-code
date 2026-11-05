"""Run selected literature cases through the existing solver, then rebuild plots."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from pipe_sim.config import Case
from pipe_sim.runner import generate, run, write_json, sha256
from pipe_sim.results import extract
from prepare_cases import FINAL, RESULTS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_ids',nargs='+',help='IDs in config/literature_cases.json, or all')
    parser.add_argument('--generate-only',action='store_true')
    parser.add_argument('--solver',choices=['calculix','abaqus'],default='calculix')
    args=parser.parse_args()
    entries=json.loads((RESULTS/'config/literature_cases.json').read_text())
    selected=entries if args.case_ids==['all'] else [e for e in entries if e['id'] in args.case_ids]
    if args.case_ids!=['all'] and {e['id'] for e in selected}!=set(args.case_ids):
        parser.error('Unknown case ID; inspect config/literature_cases.json')
    register=RESULTS/'config/runs.json'; records=json.loads(register.read_text()) if register.exists() else []
    failed=False
    for entry in selected:
        path=FINAL/entry['case_file']
        if sha256(path)!=entry['case_sha256']:raise ValueError('Case changed; update its registry/source assumptions before running: '+str(path))
        case=Case.load(path)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        directory=generate(case,FINAL/'sym/runs/literature'/f'{stamp}_{case.name}',args.solver)
        record={'case_id':entry['id'],'run':str(directory.relative_to(FINAL)),
                'case_sha256':entry['case_sha256'],'registry_entry':entry,'created_utc':stamp}
        records.append(record);write_json(register,records)
        print(f'Generated {directory}',flush=True)
        if args.generate_only:continue
        status=run(directory);print(json.dumps(status),flush=True)
        if args.solver=='calculix' and (directory/'job.dat').exists() and (directory/'job.dat').stat().st_size:
            try: print(json.dumps(extract(directory)),flush=True)
            except ValueError as exc: print(f'No extractable frames: {exc}',file=sys.stderr)
        failed |= status['state']!='completed'
    subprocess.run([sys.executable,str(RESULTS/'scripts/compare.py')],check=True)
    return 2 if failed else 0


if __name__=='__main__':raise SystemExit(main())

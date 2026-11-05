"""Read-only terminal comparison and bounded next diagnostic selection."""
import csv, hashlib, json, re
from pathlib import Path
import numpy as np
F=Path(__file__).resolve().parents[2]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parent=F/'sym/runs/campaign/orientation_intact_00_adaptive'
    run=F/'sym/runs/investigation/static_linesearch_strict_20261019/intact'
    paths=[parent,run]; sources={}
    histories=[]; costs=[]
    for p in paths:
        for n in ('job.sta','job.cvg','solver.log','results/history.csv','status.json','job.inp','mesh.inp','case.json','model.json'):
            sources[str((p/n).relative_to(F))]=digest(p/n)
        histories.append(list(csv.DictReader((p/'results/history.csv').open())))
        sta=[line.split() for line in (p/'job.sta').read_text().splitlines() if len(line.split())==7 and line.split()[0].isdigit()]
        costs.append(dict(run=str(p.relative_to(F)),accepted_increments=sum('U' not in r[2] for r in sta),failed_attempts=sum('U' in r[2] for r in sta),iterations=sum(int(r[3]) for r in sta),elapsed_seconds=json.loads((p/'status.json').read_text())['elapsed_seconds']))
    assert all(digest(parent/n)==digest(run/n) for n in ('job.inp','mesh.inp','case.json','model.json'))
    a,b=histories; assert len(a)==len(b) and [r['time'] for r in a]==[r['time'] for r in b]
    shared={k:float(max(abs(float(x[k])-float(y[k])) for x,y in zip(a,b))) for k in ('pressure_mpa','max_peeq','crown_inward_displacement_mm')}
    factors=list(map(float,re.findall(r'line search factor=([\d.eE+-]+)',(run/'solver.log').read_text())))
    cvg=[list(map(float,l.split())) for l in (run/'job.cvg').read_text().splitlines() if len(l.split())==9 and l.split()[0].isdigit()]
    last=[r for r in cvg if r[:3]==cvg[-1][:3]]
    confirmation=F/'results/output/pipe_assembled_action_confirmation_20261019.json'
    report=dict(physical_validation=False,improved=False,common_frames=len(a),shared_load_max_absolute_differences=shared,costs=costs,last_pressure_mpa=float(b[-1]['pressure_mpa']),last_peeq=float(b[-1]['max_peeq']),line_search=dict(evaluations=len(factors),minimum=min(factors),maximum=max(factors)),late_attempt=dict(iterations=len(last),correction_percent=[r[6] for r in last],force_residual_percent=[r[5] for r in last],median_last_eight_correction_ratio=float(np.median([last[i][6]/last[i-1][6] for i in range(len(last)-7,len(last))]))),confirmation_passed=json.loads(confirmation.read_text())['checks_passed'],next='Capture the original intact pipe first matrix/residual immediately beyond its last accepted time .7504883 using unchanged production binary/deck. Diagnose displacement correction versus force residual and compare with early-state approximation; no new FD sweep or production method promotion.',limitations='Same endpoint and increment schedule; line search executed but did not unlock pressure. Residual contraction is iteration behavior, not proof of physical instability. Last observed strain alone cannot exclude a constitutive problem. No burst, boundary validity or Abaqus acceptance.',sources=sources)
    report['sources'][str(confirmation.relative_to(F))]=digest(confirmation)
    out=F/'results/output/static_linesearch_outcome_20261019.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

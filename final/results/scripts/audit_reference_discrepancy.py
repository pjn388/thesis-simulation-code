"""Compare paper FE and CalculiX on the same digitized experimental points.

Descriptive branch interpolation only: no fit, uncertainty bound or acceptance gate.
Run from repository root, passing a new JSON output path under final/results/reviews.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from comparison_utils import loop_branches
from plot_investigation import reference

FINAL = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    plan_path = FINAL/'results/config/investigation.json'
    plan = json.loads(plan_path.read_text())
    sources = {str(plan_path.relative_to(FINAL)): digest(plan_path)}
    records = []
    for parent in ('a37_contact_fixed', 'a37_8pct_contact_fixed'):
        job = next(j for j in plan['jobs'] if j['id'] == parent)
        refs = {k: loop_branches(v) for k, v in reference(job['entry']).items()}
        paper = FINAL/job['entry']['reference_file']
        sources[str(paper.relative_to(FINAL))] = digest(paper)
        for case_id in (parent, parent+'_wall3_20261018'):
            current = next(j for j in plan['jobs'] if j['id'] == case_id)
            run = FINAL/current['run']
            for name in ('case.json', 'status.json', 'results/history.csv'):
                sources[str((run/name).relative_to(FINAL))] = digest(run/name)
            assert json.loads((run/'status.json').read_text())['state'] == 'completed'
            case = json.loads((run/'case.json').read_text())
            history = pd.read_csv(run/'results/history.csv')
            history['x'] = history.indenter_travel_mm-case['dent']['clearance']
            history['y'] = history.indenter_force_n/1000
            for branch in ('indent', 'withdraw'):
                model = history[history.stage == branch]
                if branch == 'withdraw':
                    model = pd.concat([history[history.stage == 'indent'].tail(1), model])
                curves = {'paper_fe': refs['paper_fe'][branch], 'calculix': model}
                curves = {k: v.groupby('x', as_index=False).y.mean().sort_values('x')
                          for k, v in curves.items()}
                low = max(float(v.x.min()) for v in curves.values())
                high = min(float(v.x.max()) for v in curves.values())
                exp = refs['experiment'][branch]
                exp = exp[(exp.x >= low) & (exp.x <= high)]
                assert len(exp) >= 2
                values = {}
                for name, curve in curves.items():
                    prediction = np.interp(exp.x, curve.x, curve.y)
                    errors = prediction-exp.y.to_numpy()
                    values[name] = dict(rmse_kn=float(np.sqrt(np.mean(errors**2))),
                                        predictions_kn=prediction.tolist(), errors_kn=errors.tolist())
                records.append(dict(case_id=case_id, branch=branch,
                    experimental_branch_points=len(refs['experiment'][branch]),
                    shared_points=len(exp), penetration_mm=exp.x.tolist(),
                    experimental_force_kn=exp.y.tolist(), comparisons=values))
    return dict(utc=datetime.now(timezone.utc).isoformat(), sources=sources,
        script_sha256=digest(Path(__file__)), records=records,
        limitations='Digitized branch points, linear interpolation within exact shared overlap only; no alignment, extrapolation, fitting, statistical uncertainty or physical acceptance. RMSEs cannot be subtracted as independent error components.')


if __name__ == '__main__':
    target = Path(sys.argv[1]).resolve()
    assert target.is_relative_to(FINAL/'results/reviews') and not target.exists()
    result = audit()
    target.write_text(json.dumps(result, indent=2)+'\n')
    for row in result['records']:
        print(row['case_id'], row['branch'], row['shared_points'],
              {k: v['rmse_kn'] for k, v in row['comparisons'].items()})

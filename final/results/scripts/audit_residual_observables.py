"""Distinguish sampled tool release from unloaded crown in terminal A37 runs."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from plot_investigation import reference

FINAL = Path(__file__).resolve().parents[2]


def inspect():
    plan = json.loads((FINAL/'results/config/investigation.json').read_text())
    records = []
    sources = {}
    for parent in ('a37_contact_fixed', 'a37_8pct_contact_fixed'):
        for name in (parent, parent+'_wall3_20261018'):
            job = next(j for j in plan['jobs'] if j['id'] == name)
            run = FINAL/job['run']
            case = json.loads((run/'case.json').read_text())
            status = json.loads((run/'status.json').read_text())
            assert status['state'] == 'completed'
            manifest = json.loads((run/'manifest.json').read_text())
            for f, expected in manifest['files'].items():
                assert hashlib.sha256((run/f).read_bytes()).hexdigest() == expected
            h = pd.read_csv(run/'results/history.csv')
            w = h[h.stage == 'withdraw'].reset_index(drop=True)
            assert abs(w.iloc[-1].time-2.) < 1e-8
            # Reporting tolerance only; bracket retained samples, never infer exact separation.
            threshold = 1.0  # N
            loaded = w[w.indenter_force_n > threshold]
            assert len(loaded)
            last = int(loaded.index[-1])
            assert last+1 < len(w)
            a, b = w.iloc[last], w.iloc[last+1]
            assert (w.iloc[last+1:].indenter_force_n.abs() <= threshold).all()
            clearance = case['dent']['clearance']
            refs = reference(job['entry'])
            endpoints = {role: dict(tool_mm=float(ref.iloc[-1].x), force_kn=float(ref.iloc[-1].y))
                         for role, ref in refs.items()}
            record = dict(case_id=name, reporting_force_tolerance_n=threshold,
                last_loaded=dict(time=float(a.time), tool_mm=float(a.indenter_travel_mm-clearance), force_n=float(a.indenter_force_n)),
                first_near_zero=dict(time=float(b.time), tool_mm=float(b.indenter_travel_mm-clearance), force_n=float(b.indenter_force_n), crown_mm=float(b.crown_inward_displacement_mm)),
                fully_withdrawn=dict(time=float(w.iloc[-1].time), tool_mm=float(w.iloc[-1].indenter_travel_mm-clearance), crown_mm=float(w.iloc[-1].crown_inward_displacement_mm)),
                reference_endpoints=endpoints)
            records.append(record)
            for f in ('case.json', 'manifest.json', 'status.json', 'job.sta', 'solver.log', 'results/history.csv', 'results/summary.json'):
                path = run/f
                sources[str(path.relative_to(FINAL))] = hashlib.sha256(path.read_bytes()).hexdigest()
            path = FINAL/job['entry']['reference_file']
            sources[str(path.relative_to(FINAL))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(utc=datetime.now(timezone.utc).isoformat(), records=records, sources=sources,
        limitations='Retained extracted frames, not native DAT re-extraction. One-newton tolerance defines sampled near-zero force only, not a contact acceptance gate. Release time/travel is bracketed; crown node displacement is not tool travel or a measured residual shape. No residual error, burst or full-loop validation inferred.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = inspect()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for record in result['records']:
        print(json.dumps(record))

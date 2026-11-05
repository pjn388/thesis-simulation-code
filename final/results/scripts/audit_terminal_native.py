"""Read terminal A37 native files independently; never overwrite extraction outputs."""
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

FINAL = Path(__file__).resolve().parents[2]
NUMBER = re.compile(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect(name):
    run = FINAL/'sym/runs/investigation'/name
    model = json.loads((run/'model.json').read_text())
    summary = json.loads((run/'results/summary.json').read_text())
    assert json.loads((run/'status.json').read_text())['state'] == 'completed'
    ref, crown = model['nsets']['INDENTER_REF'][0], model['nsets']['CROWN'][0]
    expected_count = 8*len(model['elsets']['PIPE'])
    frames = {}
    kind = None
    with (run/'job.dat').open() as stream:
        for line in stream:
            if 'for set ' in line and 'and time' in line:
                time = float(line.split('and time')[1].replace('D', 'E'))
                kind = next((k for p, k in [(' displacements', 'U'), (' forces', 'RF'),
                    (' equivalent plastic strain', 'PEEQ')] if line.startswith(p)), None)
                frame = frames.setdefault(time, dict(count=0, maximum=0., U={}, RF={}))
            elif kind and line.strip() and line.split()[0].isdigit():
                values = line.split()
                if kind == 'PEEQ':
                    frame['count'] += 1
                    frame['maximum'] = max(frame['maximum'], float(values[2]))
                elif int(values[0]) in (ref, crown):
                    frame[kind][int(values[0])] = float(values[2])
    history = list(csv.DictReader((run/'results/history.csv').open()))
    assert len(frames) == len(history)
    errors = dict(max_peeq=0., indenter_force_n=0., indenter_travel_mm=0., crown_inward_displacement_mm=0.)
    for row in history:
        f = frames[float(row['time'])]
        assert f['count'] == expected_count
        native = dict(max_peeq=f['maximum'], indenter_force_n=-f['RF'][ref],
            indenter_travel_mm=-f['U'][ref], crown_inward_displacement_mm=-f['U'][crown])
        for key, value in native.items():
            errors[key] = max(errors[key], abs(value-float(row[key])))
    assert max(errors.values()) <= 1e-10, errors
    active = False
    blocks = 0
    penetration = 0.
    minimum_pressure = 0.
    with (run/'job.frd').open() as stream:
        for line in stream:
            if line.startswith(' -4'):
                active = line.split()[1] == 'CONTACT'
            elif active and line.startswith(' -1'):
                values = [float(v.replace('D', 'E')) for v in NUMBER.findall(line[3:])]
                assert len(values) >= 7
                penetration = max(penetration, -values[1])
                minimum_pressure = min(minimum_pressure, values[4])
            elif line.startswith(' -3') and active:
                blocks += 1
                active = False
    assert blocks > 0 and not active
    assert abs(penetration-summary['max_nodal_contact_penetration_mm']) < 1e-10
    assert abs(minimum_pressure-summary['min_nodal_contact_pressure_mpa']) < 1e-10
    assert max(frames) == 3. and 2. in frames
    return dict(case=name, frames=len(frames), peeq_rows_per_frame=expected_count,
        max_absolute_history_errors=errors, contact_blocks=blocks,
        max_penetration_mm=penetration, min_contact_pressure_mpa=minimum_pressure,
        max_peeq=max(f['maximum'] for f in frames.values()),
        residual_crown_mm=-frames[2.]['U'][crown],
        sources={str((run/f).relative_to(FINAL)): digest(run/f) for f in
            ('job.dat', 'job.frd', 'job.sta', 'model.json', 'case.json', 'status.json',
             'results/history.csv', 'results/summary.json')})


if __name__ == '__main__':
    records = [inspect(p+'_wall3_20261018') for p in ('a37_contact_fixed', 'a37_8pct_contact_fixed')]
    result = dict(utc=datetime.now(timezone.utc).isoformat(), records=records,
        scope='Independent DAT nodal U/RF and all PIPE PEEQ maxima/counts, FRD contact extrema only. No stress/strain tensor audit, source material identification, contact patch verification, or physical burst validation.')
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for r in records:
        print({k:v for k,v in r.items() if k != 'sources'})

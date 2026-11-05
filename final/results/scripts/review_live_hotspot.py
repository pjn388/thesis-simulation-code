"""Bounded fixed-point pressure diagnostic; preserves all native output."""
import csv
import argparse
import json
import re

from pipe_sim.results import HEADER, stress_invariants
from review_live_withdrawal import FINAL, RUN, OUT, bounded_lines


def diagnostic_frames(result, criterion):
    """Select the sampled crossing when present and latest accepted pressure."""
    pressure = [r for r in result['history'] if r['stage'] == 'pressurise']
    if not pressure:
        raise ValueError('Snapshot has no accepted pressure frames')
    bracket = result.get('pressure_comparison', {}).get('criterion_first_sampled_bracket')
    if bracket:
        crossing = bracket['upper']
    else:
        crossing = next((r for r in pressure if r['max_peeq'] >= criterion), None)
    return pressure, crossing, ([crossing] if crossing else []) + [pressure[-1]]


def selected_blocks(path, targets, expected, evidence):
    """Accept only header-terminated, full declared-set blocks; omit the EOF block."""
    kind = None
    time = None
    count = 0
    selected = {}
    blocks = {}
    for line in bounded_lines(path, evidence):
        match = HEADER.search(line)
        if match:
            if kind and count == expected[kind]:
                blocks[(time, kind)] = selected
            time = float(match[2].replace('D', 'E'))
            title = line.strip().lower()
            kind = ('PEEQ' if title.startswith('equivalent plastic strain') else
                    'S' if title.startswith('stresses') else None)
            if match[1].upper() != ('PIPE' if kind == 'PEEQ' else 'DEFECT'):
                kind = None
            count = 0
            selected = {}
        elif kind and re.match(r'^\s*\d+\s', line):
            fields = line.split()
            if len(fields) != (3 if kind == 'PEEQ' else 8):
                continue
            point = (int(fields[0]), int(fields[1]))
            count += 1
            if point in targets:
                selected[point] = [float(v.replace('D', 'E')) for v in fields[2:]]
    return blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_id', nargs='?')
    args = parser.parse_args()
    run, out = RUN, OUT
    if args.case_id:
        from pathlib import Path
        if Path(args.case_id).name != args.case_id:
            parser.error('case_id must be a campaign directory name')
        run = FINAL / 'sym/runs/campaign' / args.case_id
        out = FINAL / 'results/output/live_cases' / args.case_id
    result = json.loads((out/'snapshot.json').read_text())
    assert (FINAL / result['run']).resolve() == run.resolve(), 'Snapshot run mismatch'
    from audit_pressure_failures import digest
    manifest = json.loads((run/'manifest.json').read_text())
    assert all(digest(run/name) == sha for name, sha in manifest['files'].items()), 'Frozen inputs changed'
    history = result['history']
    from pipe_sim.config import Case
    pressure, crossing, selected_frames = diagnostic_frames(
        result, Case.load(run/'case.json').analysis.failure_peeq)
    targets = {(r['max_peeq_element'], r['max_peeq_ip']) for r in selected_frames}
    model = json.loads((run/'model.json').read_text())
    expected = {kind: len(model['elsets'][name])*8 for kind, name in [('PEEQ', 'PIPE'), ('S', 'DEFECT')]}
    assert all(p[0] in model['elsets']['DEFECT'] for p in targets), 'Hotspot outside available stress-output set'
    evidence = {}
    blocks = selected_blocks(run/'job.dat', targets, expected, evidence)
    rows = []
    for frame in history:
        time = frame['time']
        stresses, peeq = blocks.get((time, 'S'), {}), blocks.get((time, 'PEEQ'), {})
        for point in sorted(targets):
            if point not in stresses or point not in peeq:
                continue
            vm, tri, principal = stress_invariants(stresses[point])
            rows.append(dict(time=time, stage=frame['stage'], pressure_mpa=frame['pressure_mpa'],
                             element=point[0], ip=point[1], peeq=peeq[point][0],
                             mises_mpa=vm, triaxiality=tri, maximum_principal_mpa=principal,
                             global_max_peeq=frame['max_peeq'],
                             global_max_element=frame['max_peeq_element'],
                             global_max_ip=frame['max_peeq_ip']))
    for frame in selected_frames:
        point = (frame['max_peeq_element'], frame['max_peeq_ip'])
        row = next(r for r in rows if r['time'] == frame['time'] and (r['element'], r['ip']) == point)
        assert abs(row['peeq']-frame['max_peeq']) < 1e-12
        assert point[0] in model['elsets']['PIPE']
    with (out/'fixed_hotspot.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    diagnostic = dict(run=result['run'], snapshot_created_utc=result['created_utc'],
                      criterion_crossing_observed=crossing is not None,
                      snapshot_prefix=result['prefix_evidence']['job.dat'],
                      fixed_point_prefix=evidence, targets=sorted(targets),
                      pressure_global_max_points=sorted({(r['max_peeq_element'],r['max_peeq_ip']) for r in pressure}),
                      crossing_rows=[] if crossing is None else [r for r in rows if r['time'] == crossing['time']],
                      latest_rows=[r for r in rows if r['time'] == pressure[-1]['time']],
                      rows=len(rows), limitations='Separate bounded DAT read matched to complete accepted snapshot frames. Header-terminated full DEFECT stress and PIPE PEEQ blocks only. Maxima ties choose first encountered point. No damage integration, mesh convergence, failure causation or physical burst established.')
    (out/'fixed_hotspot.json').write_text(json.dumps(diagnostic, indent=2)+'\n')
    print(json.dumps(diagnostic, indent=2))


if __name__ == '__main__':
    main()

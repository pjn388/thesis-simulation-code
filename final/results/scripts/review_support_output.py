"""Bounded support U/RF evidence; RF is not independently recovered reaction."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pipe_sim.model import Model
from pipe_sim.results import HEADER
from prepare_reaction_benchmark import additions, PARENT, verify
from review_live_withdrawal import bounded_lines, FINAL
from audit_pressure_failures import converged_times, matches_sta, digest


def complete_node_blocks(lines, expected, allow_eof=False):
    """Require exact unique coverage; accept EOF only for terminal runs."""
    key = None
    rows = {}
    duplicate = False
    for line in lines:
        match = HEADER.search(line)
        if match:
            if key and not duplicate and set(rows) == expected[key[1]]:
                yield key, rows
            title = line.strip().lower()
            kind = 'U' if title.startswith('displacements') else 'RF' if title.startswith('forces') else None
            group = match[1].upper()
            key = (float(match[2].replace('D', 'E')), group, kind) if kind and group in expected else None
            rows, duplicate = {}, False
        elif key:
            fields = line.split()
            if len(fields) == 4 and fields[0].isdigit():
                node = int(fields[0])
                duplicate |= node in rows
                rows[node] = [float(v.replace('D', 'E')) for v in fields[1:]]
    if allow_eof and key and not duplicate and set(rows) == expected[key[1]]:
        yield key, rows


def audit(run):
    run = run.resolve()
    manifest = json.loads((run/'manifest.json').read_text())
    assert all(digest(run/n) == h for n, h in manifest['files'].items())
    parent_manifest = json.loads((PARENT/'manifest.json').read_text())
    assert all(digest(PARENT/n) == h for n, h in parent_manifest['files'].items())
    assert digest(run/'model.json') == digest(PARENT/'model.json')
    source = FINAL/'sym/runs/intact_support_output_20261008T193004Z'
    verify(source)
    assert digest(run/'job.inp') == digest(source/'job.inp')
    assert digest(run/'mesh.inp') == digest(source/'mesh.inp')
    child_case = json.loads((run/'case.json').read_text())
    parent_case = json.loads((PARENT/'case.json').read_text())
    for key in ('threads', 'timeout_seconds'):
        child_case['analysis'][key] = parent_case['analysis'][key]
    assert child_case == parent_case
    model = Model.load(run/'model.json')
    constraints, _ = additions(model)
    expected = {'SUPPORT_AUDIT': {n for n, _ in constraints}, 'HISTORY': set(model.nsets['HISTORY'])}
    accepted = converged_times(run/'job.sta')
    evidence = {'job.sta': {'sha256': digest(run/'job.sta')}}
    status = json.loads((run/'status.json').read_text())
    terminal = status['state'] in ('failed', 'completed', 'interrupted')
    blocks = dict(complete_node_blocks(bounded_lines(run/'job.dat', evidence), expected,
                                      allow_eof=terminal))
    parent_rows = list(csv.DictReader((PARENT/'results/history.csv').open()))
    parent_accepted = converged_times(PARENT/'job.sta')
    assert all(matches_sta(parent_accepted, float(r['time'])) for r in parent_rows)
    parent_pressure = np.array([float(r['pressure_mpa']) for r in parent_rows])
    parent_crown = np.array([float(r['crown_inward_displacement_mm']) for r in parent_rows])
    pressure_max = parent_case['analysis']['max_pressure']
    records = []
    for time in sorted({k[0] for k in blocks}):
        keys = [(time, group, kind) for group in expected for kind in ('U', 'RF')]
        if not all(k in blocks for k in keys) or not matches_sta(accepted, time):
            continue
        u, rf = blocks[(time, 'SUPPORT_AUDIT', 'U')], blocks[(time, 'SUPPORT_AUDIT', 'RF')]
        constrained_u = [u[n][d-1] for n, d in constraints]
        filtered = [sum(rf[n][d-1] for n, d in constraints if d == axis) for axis in (1, 2, 3)]
        crown = -blocks[(time, 'HISTORY', 'U')][model.nsets['CROWN'][0]][1]
        pressure = time*pressure_max
        comparison = None
        if parent_pressure.min() <= pressure <= parent_pressure.max():
            reference = float(np.interp(pressure, parent_pressure, parent_crown))
            comparison = {'parent_interpolated_crown_mm': reference, 'difference_mm': crown-reference}
        records.append(dict(time=time, pressure_mpa=pressure, constrained_components=len(constraints),
                            constrained_max_abs_u_mm=max(abs(v) for v in constrained_u),
                            constrained_rf_sum_n=filtered, constrained_max_abs_rf_n=max(abs(rf[n][d-1]) for n, d in constraints),
                            crown_inward_mm=crown, common_pressure_comparison=comparison))
    assert records, 'No complete accepted support/history U/RF frames yet'
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), run=str(run.relative_to(FINAL)),
                parent=str(PARENT.relative_to(FINAL)), solver_status=json.loads((run/'status.json').read_text()),
                records=records, terminal_eof_allowed=terminal, prefix_evidence=evidence, input_sha256=manifest['files'],
                parent_history_sha256=digest(PARENT/'results/history.csv'), script_sha256=digest(Path(__file__)),
                sources=['sym/vendor/CalculiX/ccx_2.23/doc/ccx/node212.html:62-64,135-138'],
                limitations='Bounded accepted full-set U/RF evidence only. Common-pressure crown interpolation is descriptive, not a validation tolerance. RF includes external load and reaction; no assembled constrained-load subtraction or reaction equilibrium established. No cap tangent/work, Abaqus equivalence, mesh convergence, physical burst or production revision.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve previous evidence'
    report = audit(args.run)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'frames': len(report['records']), 'latest': report['records'][-1], 'limitations': report['limitations']}))

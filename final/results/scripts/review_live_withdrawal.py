"""Bounded, read-only native snapshot; never extract into an active run."""
import csv
import hashlib
import json
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pipe_sim.results import HEADER, NUMBER, stage_at
from pipe_sim.config import Case
from audit_pressure_failures import converged_times, digest, matches_sta

FINAL = Path(__file__).resolve().parents[2]
RUN = FINAL / 'sym/runs/campaign/full_zhao2021_p1_adaptive'
OUT = FINAL / 'results/output/live_withdrawal'


def bounded_lines(path, evidence):
    """Hash and read only the initial byte prefix; discard an incomplete last line."""
    limit = path.stat().st_size
    h = hashlib.sha256()
    consumed = 0
    with path.open('rb') as stream:
        while consumed < limit:
            raw = stream.readline(limit - consumed)
            if not raw:
                break
            consumed += len(raw)
            h.update(raw)
            if raw.endswith(b'\n'):
                yield raw.decode(errors='replace')
    evidence[path.name] = {'prefix_bytes': consumed, 'prefix_sha256': h.hexdigest(),
                           'size_after_read': path.stat().st_size}


def snapshot(run):
    case = Case.load(run / 'case.json')
    model = json.loads((run / 'model.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    checks = {name: digest(run / name) == sha for name, sha in manifest['files'].items()}
    assert all(checks.values()), 'Frozen inputs changed'
    sta = converged_times(run / 'job.sta')
    evidence = {'job.sta': {'sha256': digest(run / 'job.sta')}}
    ref = model['nsets']['INDENTER_REF'][0] if case.dent.enabled else None
    crown = model['nsets']['CROWN'][0]
    expected = 8 * len(model['elsets']['PIPE'])
    frames = {}
    kind = None
    frame = None
    count = 0
    max_peeq = 0.
    max_point = None

    def end_block():
        if frame is not None and kind == 'PEEQ':
            frame['peeq_count'] = count
            frame['max_peeq'] = max_peeq
            frame['max_peeq_element'], frame['max_peeq_ip'] = max_point or (None, None)

    for line in bounded_lines(run / 'job.dat', evidence):
        match = HEADER.search(line)
        if match:
            end_block()
            t = float(match[2].replace('D', 'E'))
            frame = frames.setdefault(t, {'time': t})
            if not case.dent.enabled:
                frame.update(tool_travel_mm=0., tool_force_n=0.)
            title = line.strip().lower()
            kind = next((k for prefix, k in [('equivalent plastic strain', 'PEEQ'),
                        ('displacements', 'U'), ('forces', 'RF')] if title.startswith(prefix)), None)
            count = 0
            max_peeq = 0.
            max_point = None
        elif frame is not None and kind and re.match(r'^\s*\d+\s', line):
            vals = [float(v.replace('D', 'E')) for v in re.findall(NUMBER, line)]
            if kind == 'PEEQ' and len(vals) == 3:
                count += 1
                if max_point is None or vals[2] > max_peeq:
                    max_peeq = vals[2]
                    max_point = (int(vals[0]), int(vals[1]))
            elif kind in ('U', 'RF') and len(vals) == 4:
                if int(vals[0]) == ref:
                    frame['tool_travel_mm' if kind == 'U' else 'tool_force_n'] = -vals[2]
                if kind == 'U' and int(vals[0]) == crown:
                    frame['crown_inward_mm'] = -vals[2]
    # The last block is not finalized at EOF: partial native writes are possible.
    rows = []
    rejected = []
    for t, f in sorted(frames.items()):
        required = {'tool_travel_mm', 'tool_force_n', 'crown_inward_mm', 'max_peeq'}
        if f.get('peeq_count') != expected or not required.issubset(f) or not matches_sta(sta, t):
            rejected.append(t)
            continue
        f['stage'], f['pressure_mpa'] = stage_at(case, t)
        rows.append(f)
    assert rows, 'No complete STA-consistent frames'
    contacts = []
    active = False
    t = 0.
    for line in bounded_lines(run / 'job.frd', evidence):
        if line.lstrip().startswith('100CL'):
            t = float(line.split()[2])
        elif line.startswith(' -4'):
            active = line.split()[1] == 'CONTACT'
            penetration = pressure = minimum = 0.
        elif active and line.startswith(' -1'):
            vals = re.findall(NUMBER, line[3:])
            if len(vals) >= 7:
                penetration = max(penetration, -float(vals[1]))
                pressure = max(pressure, float(vals[4]))
                minimum = min(minimum, float(vals[4]))
        elif line.startswith(' -3'):
            if active and any(abs(r['time']-t) <= 1e-6 for r in rows):
                contacts.append({'time': t, 'penetration_mm': penetration,
                                 'max_pressure_mpa': pressure, 'min_pressure_mpa': minimum})
            active = False
    supported = json.loads(case.material.source)['source_supported_plastic_strain_max']
    last = rows[-1]
    completed = None if not case.dent.enabled else any(r['stage'] == 'withdraw' and abs(r['time']-2.) < 1e-7 for r in rows)
    peak = max(r['tool_force_n'] for r in rows)
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), run=str(run.relative_to(FINAL)),
                frozen_inputs_verified=checks, solver_sha256=json.loads((run/'status.json').read_text())['solver_sha256'],
                solver_state=json.loads((run/'status.json').read_text())['state'],
                prefix_evidence=evidence, history=rows, contacts=contacts, rejected_frame_times=rejected,
                material_supported_peeq=supported, maximum_peeq=max(r['max_peeq'] for r in rows),
                peak_tool_force_n=peak, tensile_force_detected=min(r['tool_force_n'] for r in rows) < -max(1., peak*.001),
                withdrawal_completed=completed, residual_crown_mm=None if not case.dent.enabled else next((r['crown_inward_mm'] for r in rows if abs(r['time']-2.) < 1e-7), None),
                latest_frame=last, limitations='Independent bounded DAT/FRD prefixes, not an atomic snapshot. Only complete PEEQ blocks with 8 points per pipe element and STA-consistent times are accepted; incomplete final blocks are excluded. Contact frames require complete FRD terminators and matching accepted DAT times. No burst, residual before complete withdrawal, mesh convergence or experimental validation inferred.')


def compare_snapshot(result, run=RUN, out=OUT):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_investigation import reference
    from comparison_utils import error_metrics, loop_branches
    plan = json.loads((FINAL/'results/config/full_campaign.json').read_text())
    job = next(j for j in plan['jobs'] if j['id'] == run.name)
    refs = reference(job['entry'])
    case = Case.load(run/'case.json')
    history = pd.DataFrame(result['history'])
    indent = history[history.stage == 'indent'].copy()
    indent['x'] = indent.tool_travel_mm - case.dent.clearance
    indent['y'] = indent.tool_force_n / 1000.
    metrics = {}
    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
    for role, ref in refs.items():
        for branch, target in loop_branches(ref).items():
            part = history[history.stage == branch].copy()
            if branch == 'withdraw':
                part = pd.concat([history[history.stage == 'indent'].tail(1), part])
            part['x'] = part.tool_travel_mm - case.dent.clearance
            part['y'] = part.tool_force_n / 1000.
            metrics[role+'_'+branch], _ = error_metrics(target, part)
        ax.plot(ref.x, ref.y, 'o-' if role == 'experiment' else '--', label=role)
    for stage, part in history.groupby('stage', sort=False):
        ax.plot(part.tool_travel_mm-case.dent.clearance, part.tool_force_n/1000., label='our '+stage)
    ax.set(xlabel='Tool indentation (mm)', ylabel='Compression-positive force (kN)',
           title=textwrap.fill((run.name.replace('_', ' ') + ': completed withdrawal; pressure study incomplete'
                  if result['withdrawal_completed'] else
                  run.name.replace('_', ' ') + ': incomplete withdrawal; residual unavailable'), 65))
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(out/'three_way_snapshot.png', dpi=180)
    plt.close(fig)
    result['branch_comparison'] = metrics
    result['reference_sha256'] = digest(FINAL/job['entry']['reference_file'])
    return result


def compare_pressure(result):
    """Compare accepted histories only; endpoint secants are not burst tests."""
    import pandas as pd
    import matplotlib.pyplot as plt
    control = FINAL/'sym/runs/campaign/orientation_gouge_00_adaptive'
    audit = json.loads((FINAL/'results/output/pressure_failure_audit/audit.json').read_text())
    record = next(r for r in audit['records'] if r['case_id'] == control.name)
    for name, sha in record['evidence_sha256'].items():
        assert digest(control/name) == sha, 'Control evidence changed; rebuild failure audit'
    history = pd.DataFrame(result['history'])
    pressure = history[history.stage == 'pressurise'].copy()
    other = pd.read_csv(control/'results/history.csv')
    other = other[other.stage == 'pressurise']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for rows, peeq, crown, label in [
            (pressure, 'max_peeq', 'crown_inward_mm', f"Combined P1 ({result['solver_state']})"),
            (other, 'max_peeq', 'crown_inward_displacement_mm', 'Gouge-only 0° (failed)')]:
        axes[0].plot(rows[peeq], rows.pressure_mpa, '.-', label=label)
        axes[1].plot(-rows[crown], rows.pressure_mpa, '.-', label=label)
    axes[0].axvline(result['material_supported_peeq'], color='grey', linestyle=':', label='Material extent')
    axes[0].axvline(.8, color='black', linestyle='--', label='Provisional PEEQ criterion')
    axes[0].set_xlabel('Maximum observed PEEQ (hotspot may change)')
    axes[1].set_xlabel('Crown outward displacement from initial surface (mm)')
    for ax in axes:
        ax.set_ylabel('Applied pressure (MPa)')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('Accepted pressure histories; endpoints are not burst predictions')
    fig.savefig(OUT/'pressure_snapshot.png', dpi=180)
    plt.close(fig)
    summary = dict(accepted_pressure_frames=len(pressure),
                   control_run=record['run'], control_evidence_sha256=record['evidence_sha256'],
                   control_last_pressure_mpa=record['last_pressure_mpa'],
                   control_maximum_peeq=record['max_observed_peeq'],
                   interpretation='Different dent histories; not a physical burst or orientation ranking. Global maximum PEEQ can switch integration points. Secants do not establish a limit point.')
    if len(pressure):
        summary['latest_accepted'] = pressure.iloc[-1].to_dict()
        summary['material_extent_exceeded'] = bool((pressure.max_peeq > result['material_supported_peeq']).any())
        criterion = Case.load(RUN/'case.json').analysis.failure_peeq
        summary['criterion_peeq'] = criterion
        for label, threshold in [('criterion', criterion), ('material_extent', result['material_supported_peeq'])]:
            above = pressure[pressure.max_peeq >= threshold]
            bracket = None
            if len(above):
                upper = above.iloc[0]
                below = pressure[pressure.time < upper.time]
                bracket = dict(lower=None if below.empty else below.iloc[-1].to_dict(),
                               upper=upper.to_dict())
            summary[label+'_first_sampled_bracket'] = bracket
    if len(pressure) > 1:
        dp = float(pressure.pressure_mpa.diff().iloc[-1])
        de = float(pressure.max_peeq.diff().iloc[-1])
        du = float(pressure.crown_inward_mm.diff().iloc[-1])
        summary['last_secant_mpa_per_peeq'] = None if de == 0 else dp/de
        summary['last_secant_mpa_per_outward_crown_mm'] = None if du == 0 else dp/-du
    result['pressure_comparison'] = summary
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = compare_pressure(compare_snapshot(snapshot(RUN)))
    (OUT/'snapshot.json').write_text(json.dumps(result, indent=2)+'\n')
    with (OUT/'history.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['history'][0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(result['history'])
    print(json.dumps({k: result[k] for k in ['latest_frame','maximum_peeq','material_supported_peeq',
                     'peak_tool_force_n','tensile_force_detected','withdrawal_completed','residual_crown_mm']}))
    print('Complete contact frames:', len(result['contacts']))


if __name__ == '__main__':
    main()

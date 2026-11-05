"""Read-only ASCII campaign dashboard. No solver commands or file writes."""
import argparse
from collections import Counter
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
import textwrap
from progress_inventory import extra_jobs, tail_text, deck_duration, management_snapshot

FINAL = Path(__file__).resolve().parents[2]
PLAN = FINAL / 'results/config/full_campaign.json'
EARLIER = FINAL / 'results/output/campaign/dent_validation/provenance.json'


def read_json(path):
    return json.loads(path.read_text())


def last_increment(path):
    """Use only complete, successful native STA records, never failed attempts."""
    if not path.exists():
        return None
    for line in reversed(path.read_text(errors='replace').splitlines(keepends=True)):
        if not line.endswith('\n'):
            continue
        fields = line.split()
        if len(fields) != 7 or not fields[0].isdigit() or not fields[2].isdigit():
            continue
        try:
            value = float(fields[4].replace('D', 'E'))
            if math.isfinite(value) and value >= 0:
                return value
        except ValueError:
            continue
    return None


def progress_bar(fraction, width=20):
    if fraction is None:
        return '[' + '?' * width + ']'
    scaled = min(1., max(0., fraction)) * width
    full = int(scaled)
    partial = int(scaled > full + 1e-9 and full < width)
    return '[' + '#' * full + '>' * partial + '.' * (width-full-partial) + ']'


def solver_directories(targets, run_root=None):
    """One /proc scan per refresh, to flag stale running records on Linux."""
    if not Path('/proc/self').exists():
        return None
    found = {}
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            cwd = (process / 'cwd').resolve(strict=True)
            if cwd not in targets and (run_root is None or not cwd.is_relative_to(run_root)):
                continue
            command = (process / 'cmdline').read_bytes().split(b'\0')
            executable = Path(os.fsdecode(command[0])).name.lower()
            if executable.startswith(('ccx', 'abaqus', 'standard', 'explicit')):
                fields = (process/'stat').read_text().rsplit(')', 1)[1].split()
                if fields[0] == 'Z':
                    continue
                paused = fields[0] in ('T', 't')
                for line in (process/'cgroup').read_text().splitlines():
                    if line.startswith('0::'):
                        events = Path('/sys/fs/cgroup') / line[3:].lstrip('/') / 'cgroup.events'
                        if events.exists():
                            paused |= 'frozen 1' in events.read_text()
                item = found.setdefault(cwd, {'pids': [], 'paused': True})
                item['pids'].append(int(process.name)); item['paused'] &= paused
        except (OSError, IndexError):
            continue  # A process may exit while /proc is being read.
    return found


def task_label(job, case, earlier=False):
    if earlier:
        depth = {'a37_contact_fixed': '3', 'a37_8pct_contact_fixed': '8'}.get(job['id'])
        if depth is None:
            depth = job['id'].removeprefix('allouti2012_').split('_')[0]
        return f'A37 {depth}% dent (5 MPa check)'
    entry = job.get('entry') or {}
    if job.get('kind') == 'validation':
        paper = entry.get('paper', '')
        specimen = entry.get('specimen', '')
        if paper.startswith('allouti'):
            return f'Allouti {paper[-4:]} {specimen.split("_")[0]}%'
        if paper == 'zhao2021':
            return f'Zhao {specimen}'
        if paper == 'shuai2018':
            return 'Shuai dent'
    if job.get('kind') == 'orientation':
        dent = case.get('dent', {}).get('enabled', True)
        gouge = case.get('gouge', {}).get('enabled', False)
        if not gouge:
            return 'Dent-only baseline' if dent else 'Intact baseline'
        prefix = 'Dent + gouge' if dent else 'Gouge only'
        return f'{prefix} {case["gouge"]["angle_deg"]:g} deg'
    return job['id'].replace('_', ' ').removeprefix('full ')


def task_row(job, root, live=None, earlier=False):
    directory = (root / job['run']).resolve()
    row = dict(id=job['id'], label=job['id'], earlier=earlier,
               kind=job.get('kind', 'earlier'), state='QUEUED', fraction=0.,
               stage='-', stage_percent=0., pressure=None, error=None,
               run=str(directory), campaign=job.get('campaign', not earlier),
               superseded=bool(job.get('superseded_by')))
    try:
        case_path = directory / 'case.json' if earlier or not job.get('case_file') else root / job['case_file']
        case = read_json(case_path) if case_path.exists() else {}
        row['label'] = job.get('display_path') or task_label(job, case, earlier)
        if (directory / 'status.json').exists():
            status = read_json(directory / 'status.json')
        elif not row['campaign']:
            log = tail_text(directory/'solver.log')
            state = ('running' if live is not None and directory in live else
                     'failed' if '*ERROR' in log else 'solved' if 'Job finished' in log else
                     'incomplete' if (directory/'job.sta').exists() else
                     'generated' if (directory/'job.inp').exists() else 'not_run')
            status = {'state': state}
        else:
            status = {'state': 'not_run'}
        state = status['state']
        row['state'] = {'not_run': 'QUEUED', 'generated': 'QUEUED', 'building': 'BUILDING',
                        'running': 'RUNNING', 'completed': 'DONE', 'failed': 'FAILED',
                        'interrupted': 'INTERRUPTED'}.get(state, state.upper())
        if state == 'running' and live is not None and directory not in live:
            row['state'] = 'NO SOLVER?'
        if state == 'running' and isinstance(live, dict) and live.get(directory, {}).get('paused'):
            row['state'] = 'PAUSED'
        if not row['campaign'] and state in ('not_run', 'generated'):
            row['state'] = 'PREPARED' if state == 'generated' else 'NOT RUN'
        if job.get('superseded_by'):
            row['state'] = 'SUPERSEDED'
        total = last_increment(directory / 'job.sta')
        continuation = directory / 'continuation.json'
        if continuation.exists():
            from pipe_sim.continuation import progress_at
            if total is not None:
                row.update(progress_at(json.loads(continuation.read_text()), total))
            return row
        if not case:
            duration = deck_duration(directory/'job.inp')
            row['fraction'] = min(1., total/duration) if total is not None and duration else None
            if total is not None:
                row.update(stage='native', stage_percent=100*row['fraction'] if row['fraction'] is not None else None,
                           native_time=total)
            return row
        dent = case.get('dent', {}).get('enabled', True)
        stages = ['indent', 'withdraw', 'pressure'] if dent else ['pressure']
        if total is not None:
            index = min(len(stages)-1, max(0, math.ceil(total-1e-8)-1))
            row.update(fraction=min(1., total/len(stages)), stage=stages[index],
                       stage_percent=min(100., max(0., total-index)*100),
                       pressure=case['analysis']['max_pressure'] * min(1., max(0., total-(len(stages)-1))))
        elif state == 'completed':
            row['error'] = 'Completion recorded but native progress is unavailable'
    except (OSError, ValueError, KeyError, TypeError) as exc:
        row.update(state='READ ERROR', error=str(exc))
    return row


def snapshot(plan_path=PLAN, root=FINAL, earlier_path=EARLIER, include_earlier=True, all_attempts=False, include_other=True):
    plan = read_json(plan_path)
    jobs = [j for j in plan['jobs'] if all_attempts or not j.get('superseded_by')]
    earlier = []
    if include_earlier and earlier_path.exists():
        earlier = [dict(j, id=j['case_id']) for j in read_json(earlier_path)]
    known = {(root / j['run']).resolve() for j in jobs + earlier}
    extras, errors = extra_jobs(root, plan['jobs']) if include_other else ([], [])
    extras = [j for j in extras if (root/j['run']).resolve() not in known]
    targets = known | {(root/j['run']).resolve() for j in extras}
    live = solver_directories(targets, root/'sym/runs' if include_other else None)
    if live is not None and include_other:
        for directory in set(live)-targets:
            extras.append(dict(id=directory.name, run=str(directory), kind='additional', campaign=False,
                               display_path=str(directory.relative_to(root))))
    rows = [task_row(j, root, live, earlier=True) for j in earlier]
    rows += [task_row(j, root, live) for j in jobs]
    rows += [task_row(j, root, live) for j in extras]
    plan['_inventory_errors'] = errors
    return plan, rows


def overall_progress(rows):
    """Estimate current campaign completion, excluding historical/failed work."""
    required = {r['id']: r for r in rows
                if r.get('campaign', not r['earlier']) and not r.get('superseded')}
    completed = underway = attention = 0
    credit = 0.
    for row in required.values():
        fraction = row.get('fraction')
        known = fraction is not None and math.isfinite(fraction) and not row.get('error')
        if row['state'] == 'DONE' and known and fraction >= 1. - 1e-6:
            completed += 1
            credit += 1.
        elif row['state'] in ('RUNNING', 'PAUSED'):
            underway += 1
            if known:
                credit += min(.999, max(0., fraction))
        elif row['state'] not in ('QUEUED', 'BUILDING'):
            attention += 1
    total = len(required)
    return dict(total=total, completed=completed, underway=underway, attention=attention,
                fraction=credit/total if total else None)


def overall_lines(rows, width):
    progress = overall_progress(rows)
    fraction = progress['fraction']
    # Never round incomplete work up to a displayed 100%.
    percent = ('n/a' if fraction is None else
               f'{100. if progress["completed"] == progress["total"] else min(99.9, fraction*100):.1f}%')
    return [f'THESIS SIMULATIONS {progress_bar(fraction, 24 if width >= 90 else 12)} {percent} estimated',
            f'Complete: {progress["completed"]}/{progress["total"]} | In progress/paused: {progress["underway"]} | Needs attention: {progress["attention"]}',
            'Current plan; equal cases + live loading. Failed attempts add no completion.']


def render(plan, rows, width=110, management=None):
    bar_width = 20 if width >= 105 else 12
    label_width = 29 if width >= 105 else 23
    groups = [
        ('EARLIER DENTING CHECKS (separate from full-pressure campaign)', lambda r: r['earlier']),
        ('FULL-PRESSURE VALIDATIONS', lambda r: not r['earlier'] and r['kind'] == 'validation'),
        ('SENSITIVITY CHECKS', lambda r: not r['earlier'] and r['kind'] == 'iteration'),
        ('BASELINES AND GOUGE ORIENTATION', lambda r: not r['earlier'] and r['kind'] == 'orientation'),
        ('ADDITIONAL RUNS / DIAGNOSTICS / HISTORICAL ATTEMPTS', lambda r: not r['earlier'] and r['kind'] == 'additional'),
        ('OTHER TASKS', lambda r: not r['earlier'] and r['kind'] not in ('validation', 'iteration', 'orientation', 'additional')),
    ]
    active = [r for r in rows if r.get('campaign', not r['earlier']) and not r.get('superseded')]
    lines = overall_lines(rows, width) + [f'Coverage: {len(rows)} simulation records; {len(active)} active campaign entries.',
             'New runs, diagnostics and service tasks are rediscovered at each refresh.', '']
    if management is not None:
        lines.append('MANAGEMENT AND SCHEDULES (status only; no estimated completion percentage)')
        for unit in management['units']:
            lines.extend(textwrap.wrap(f'{unit["Id"]}: {unit["display_state"]} ({unit.get("SubState", "?")})', width=max(20,width-1)))
            details = []
            if unit.get('Result') not in (None, '', 'success'):
                details.append('result='+unit['Result'])
            if unit.get('NextElapseUSecRealtime'):
                details.append('next='+unit['NextElapseUSecRealtime'])
            if details:
                lines.extend(textwrap.wrap('  '+', '.join(details), width=max(20,width-1), subsequent_indent='  '))
        review = management.get('review')
        if review:
            lines += [f'Latest review {review.get("review_id", "?")}: {review["display_state"]}']
            if review.get('last_error'):
                lines.extend(textwrap.wrap('  Latest review error: '+review['last_error'],width=max(20,width-1)))
        counts = management.get('review_counts', {})
        if counts:
            lines.append('Finished review attempts: '+', '.join(f'{k}={v}' for k,v in sorted(counts.items())))
        lines.extend(management.get('errors', [])); lines.append('')
    lines.extend(plan.get('_inventory_errors', []))
    for title, select in groups:
        selected = [r for r in rows if select(r)]
        if not selected:
            continue
        lines += [title]
        for row in selected:
            label = row['label']
            if row['kind'] == 'additional':
                lines.extend(textwrap.wrap(label, width=max(20,width-1)))
                label = '  native run'
            if len(label) > label_width:
                label = label[:label_width-1] + '~'
            pressure = f'{row["pressure"]:.2f} MPa' if row['stage'] == 'pressure' else ''
            detail = (f'native t={row["native_time"]:g}' if row['stage']=='native' else
                      f'{row["stage"]} {row["stage_percent"]:.0f}%' if row['stage'] != '-' else '')
            if width < 105 and row['stage'] == 'pressure':
                detail, pressure = f'P:{row["pressure"]:.2f}MPa', ''
            percent = f'{100*row["fraction"]:5.1f}%' if row['fraction'] is not None else '   n/a'
            lines.append(f'{label:<{label_width}} {progress_bar(row["fraction"], bar_width)} '
                         f'{percent} {row["state"]:<12} {detail} {pressure}'.rstrip())
            if row['error']:
                lines.append(f'  {row["id"]}: {row["error"]}')
        lines.append('')
    shared = plan.get('orientation_combined_0')
    if shared:
        shared_label = next((r['label'] for r in rows if r['id'] == shared), shared)
        lines += [f'Dent + gouge 0 deg reuses {shared_label}; counted once.']
    counts = Counter(r['state'] for r in active)
    lines += [f'Campaign: {len(active)} tasks | ' + ' | '.join(f'{state}: {n}' for state, n in sorted(counts.items())),
              'Percentages measure converged loading, NOT time remaining or validation accuracy.',
              'Applied pressure and solver stops are NOT confirmed burst pressures.',
              'PREPARED does not mean queued. SOLVED means native completion, not a passed diagnostic.',
              'Superseded attempts with files are shown; --all-attempts also lists unbuilt superseded inputs.',
              'NO SOLVER? = recorded running, but no solver observed; may be stale or starting.']
    return lines


def refresh(args, width):
    plan, rows = snapshot(include_earlier=not args.campaign_only, all_attempts=args.all_attempts,
                          include_other=not args.campaign_only)
    stamp = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
    return stamp, render(plan, rows, width, management=None if args.campaign_only else management_snapshot(FINAL))


def terminal(screen, args):
    import curses
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    screen.keypad(True)
    screen.timeout(200)
    offset = 0
    stamp, lines, error = 'waiting', [], ''
    next_refresh = 0.
    while True:
        height, width = screen.getmaxyx()
        if time.monotonic() >= next_refresh:
            try:
                stamp, lines = refresh(args, width)
                error = ''
            except (OSError, ValueError, KeyError) as exc:
                error = f'Refresh failed; retaining previous snapshot: {exc}'
            next_refresh = time.monotonic() + args.interval
        pinned, body = lines[:3], lines[3:]
        page = max(1, height-8)
        offset = min(max(0, offset), max(0, len(body)-page))
        screen.erase()
        def put(y, text):
            if 0 <= y < height:
                try:
                    screen.addnstr(y, 0, text, max(0, width-1))
                except curses.error:
                    pass
        for index, line in enumerate(pinned):
            put(index, line)
        put(3, f'THESIS SIMULATION PROGRESS | {stamp}')
        put(4, f'Refresh: {args.interval:g}s | Read-only | Ctrl+C or q closes dashboard ONLY')
        put(5, error or 'Case bars show converged loading; overall completion excludes failed attempts.')
        for index, line in enumerate(body[offset:offset+page], start=7):
            put(index, line)
        put(height-1, f'Lines {offset+1}-{min(offset+page,len(body))}/{len(body)} | Up/Down, PgUp/PgDn, Home/End | r refresh | q quit')
        screen.refresh()
        key = screen.getch()
        if key in (ord('q'), ord('Q'), 3):
            return
        if key in (curses.KEY_DOWN, ord('j')):
            offset += 1
        elif key in (curses.KEY_UP, ord('k')):
            offset -= 1
        elif key in (curses.KEY_NPAGE, ord('n'), ord(' ')):
            offset += page
        elif key in (curses.KEY_PPAGE, ord('p')):
            offset -= page
        elif key == curses.KEY_HOME:
            offset = 0
        elif key == curses.KEY_END:
            offset = max(0, len(body)-page)
        elif key in (ord('r'), curses.KEY_RESIZE):
            next_refresh = 0.


def positive_seconds(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('interval must be a finite positive number')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', type=positive_seconds, default=10., help='refresh interval in seconds (default: 10)')
    parser.add_argument('--once', action='store_true', help='print one full snapshot and exit')
    parser.add_argument('--plain', action='store_true', help='print snapshots without a fullscreen terminal interface')
    parser.add_argument('--campaign-only', action='store_true', help='show only the campaign ledger; omit diagnostics/history and management')
    parser.add_argument('--all-attempts', action='store_true', help='also display superseded campaign meshes')
    args = parser.parse_args()
    try:
        if not args.once and not args.plain and sys.stdout.isatty() and sys.stdin.isatty() and os.environ.get('TERM', 'dumb') != 'dumb':
            import curses
            curses.wrapper(terminal, args)
        else:
            while True:
                try:
                    stamp, lines = refresh(args, 120)
                    print('\n'.join(lines[:3] + [f'THESIS SIMULATION PROGRESS | {stamp}'] + lines[3:]), flush=True)
                except (OSError, ValueError, KeyError) as exc:
                    print(f'Cannot refresh dashboard: {exc}', file=sys.stderr)
                    if args.once:
                        return 1
                if args.once:
                    return 0
                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    print('Dashboard closed. Simulations and queues were left running.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

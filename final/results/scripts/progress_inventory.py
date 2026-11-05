"""Read-only discovery for runs and services outside the main campaign ledger."""
from collections import Counter
import json
from pathlib import Path
import subprocess


def extra_jobs(root, known_jobs):
    """Deduplicate by run directory, preserving full paths as stable identities."""
    found = {}
    errors = []
    for filename in ['investigation.json', 'runs.json']:
        path = root / 'results/config' / filename
        if not path.exists():
            continue
        try:
            value = json.loads(path.read_text())
            records = value.get('jobs', []) if isinstance(value, dict) else value
            for item in records:
                if item.get('run'):
                    directory = (root / item['run']).resolve()
                    found[directory] = dict(item, id=item.get('id', item.get('case_id', directory.name)))
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f'{path.name}: {exc}')
    runs = root / 'sym/runs'
    if runs.exists():
        for pattern in ['status.json', 'case.json', 'job.inp', 'job.sta']:
            for path in runs.rglob(pattern):
                directory = path.parent.resolve()
                found.setdefault(directory, dict(id=directory.name, run=str(directory.relative_to(root))))
    campaign_by_run = {(root / j['run']).resolve(): j for j in known_jobs}
    for directory, job in found.items():
        if directory in campaign_by_run:
            # Executed superseded attempts retain their recorded lineage/state.
            job.update(campaign_by_run[directory])
        job['kind'] = 'additional'
        job['campaign'] = False
        if (directory / 'case.json').exists():
            job['case_file'] = str(directory / 'case.json')
        job['display_path'] = str(directory.relative_to(root))
    return [found[p] for p in sorted(found)], errors


def tail_text(path, size=65536):
    if not path.exists():
        return ''
    with path.open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell()-size))
        return stream.read().decode(errors='replace')


def deck_duration(path):
    """Only infer time for supported static/dynamic steps with explicit periods."""
    if not path.exists():
        return None
    periods = []
    count = 0
    waiting = False
    for raw in path.read_text(errors='replace').splitlines():
        line = raw.strip().upper()
        if not line or line.startswith('**'):
            continue
        keyword = line.split(',')[0]
        if keyword == '*STEP':
            count += 1
        if keyword in ('*STATIC', '*DYNAMIC'):
            waiting = True
        elif waiting:
            if line.startswith('*'):
                return None
            try:
                period = float(line.split(',')[1].replace('D', 'E'))
                if period <= 0:
                    return None
                periods.append(period)
            except (ValueError, IndexError):
                return None
            waiting = False
    return sum(periods) if count and count == len(periods) else None


def unit_state(data):
    if data.get('LoadState') == 'not-found':
        return 'ORPHANED' if data.get('ActiveState') == 'active' else 'NOT LOADED'
    if data.get('FreezerState') in ('frozen', 'freezing'):
        return 'PAUSED'
    if data.get('ActiveState') == 'failed':
        return 'FAILED'
    return {'active': 'ACTIVE', 'activating': 'STARTING', 'inactive': 'INACTIVE',
            'deactivating': 'STOPPING'}.get(data.get('ActiveState'), 'UNKNOWN')


def management_snapshot(root):
    """Stable dashboard schema without private service or AI review discovery."""
    return {'units': [], 'errors': [], 'review': None, 'review_counts': {}}

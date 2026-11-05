"""Audit bounded solver-log contact counts without touching native results."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

FINAL = Path(__file__).resolve().parents[2]


def inspect(raw):
    step_time = None
    stage = None
    records = {}
    for line in raw.decode(errors='replace').splitlines():
        match = re.search(r'actual step time=\s*(\S+)', line)
        if match:
            step_time = float(match[1])
        match = re.search(r'actual total time=\s*(\S+)', line)
        if match and step_time is not None:
            offset = float(match[1]) - step_time
            assert abs(offset - round(offset)) < 1e-5, 'Non-unit workflow steps'
            stage = str(round(offset) + 1)
        match = re.search(r'Number of contact spring elements=\s*(\d+)', line)
        if match:
            count = int(match[1])
            record = records.setdefault(stage or 'unassigned', dict(samples=0, nonzero_samples=0, minimum=count, maximum=count))
            record['samples'] += 1
            record['nonzero_samples'] += count != 0
            record['minimum'] = min(record['minimum'], count)
            record['maximum'] = max(record['maximum'], count)
    return records


def main():
    records = []
    for angle in (75, 90):
        case_id = f'orientation_combined_{angle}_adaptive'
        path = FINAL / 'sym/runs/campaign' / case_id / 'solver.log'
        size = path.stat().st_size
        with path.open('rb') as stream:
            raw = stream.read(size)
        # A concurrently appended partial line must not enter the audit.
        raw = raw[:raw.rfind(b'\n') + 1]
        records.append(dict(case_id=case_id, log_prefix_bytes=len(raw), log_prefix_sha256=hashlib.sha256(raw).hexdigest(), counts_by_step=inspect(raw)))
    source = FINAL / 'sym/vendor/CalculiX/ccx_2.23/src/nonlingeo.c'
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), records=records, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), source_reference='nonlingeo.c:1920-1921 reports ne-ne0', limitations='Unit-duration workflow step inferred from actual total minus step time. Counts include unsuccessful iterations; zero generated springs is not an independent gap/pressure test. No contact causation, structural instability or burst established.')
    out = FINAL / 'results/output/contact_activity_20261010T063002Z.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

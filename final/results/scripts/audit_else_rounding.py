"""Bound literal ELSE print rounding; does not bound external-work errors."""
import json
from decimal import Decimal
from pathlib import Path

from pipe_sim.runner import sha256

FINAL = Path(__file__).resolve().parents[2]


def audit(tag):
    terminal_path = FINAL / 'results/output' / f'{tag}_terminal.json'
    terminal = json.loads(terminal_path.read_text())
    run = FINAL / terminal['run']
    assert terminal['completed']
    assert all(sha256(run / n) == h for n, h in
               {**terminal['input_hashes'], **terminal['output_hashes']}.items())
    total = Decimal(0)
    bound = Decimal(0)
    ids = set()
    time = None
    reading = False
    with (run / 'job.dat').open() as stream:
        for line in stream:
            if line.startswith(' internal energy (element, energy) for set PIPE and time '):
                time = float(line.split('time')[1])
                total, bound, ids = Decimal(0), Decimal(0), set()
                reading = True
            elif reading:
                fields = line.split()
                if len(fields) == 2 and fields[0].isdigit():
                    assert int(fields[0]) not in ids
                    ids.add(int(fields[0]))
                    value = Decimal(fields[1])
                    total += value
                    bound += Decimal(5).scaleb(value.as_tuple().exponent - 1)
                elif line.strip():
                    reading = False
    assert time == 1.0 and len(ids) == 2880
    endpoint = terminal['work_audit']['history'][-1]
    assert abs(float(total) - endpoint['internal_work_n_mm']) < 1e-5
    gap = endpoint['external_work_n_mm'] - float(total)
    return dict(run=terminal['run'], elements=len(ids), time=time,
                printed_internal_work_n_mm=float(total),
                summed_half_decimal_intervals_n_mm=float(bound),
                external_minus_printed_internal_n_mm=gap,
                gap_interval_internal_rounding_only_n_mm=[gap-float(bound), gap+float(bound)],
                internal_rounding_alone_can_include_zero=abs(gap) <= float(bound),
                terminal_sha256=sha256(terminal_path), dat_sha256=sha256(run/'job.dat'))


if __name__ == '__main__':
    tags = ['plastic_tangential_work_p24_20261011T070440Z',
            'plastic_tangential_work_quarterdt_p24_20261011T072144Z']
    report = dict(cases=[audit(tag) for tag in tags], script_sha256=sha256(__file__),
                  source='sym/vendor/CalculiX/ccx_2.23/src/printoutelem.f:521',
                  limitations='Conservative sum of literal e13.6 half-decimal intervals, assuming nearest rounding. External work treated as fixed for this diagnostic only. No bound on displacement/geometry/time rounding, pressure quadrature, assembly residual or internal integration. No work convergence, coupled cap acceptance, production revision or physical validation.')
    out = FINAL/'results/output/else_rounding_20261011T074839Z.json'
    assert not out.exists()
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

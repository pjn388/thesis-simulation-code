"""Compare gated immutable native states at shared accepted times; no solves."""
import json
import sys
from pathlib import Path

import numpy as np
from pipe_sim.runner import sha256

F = Path(__file__).resolve().parents[2]
STANDARD = F/'results/output/plastic_pipe_replay_20261011T101242Z.json'
QUARTER = F/'results/output/plastic_state_quarter_20261011T102248Z_replay.json'


def load(gate):
    evidence = json.loads(gate.read_text())
    assert evidence['checks_passed'], 'Constitutive identity gate must pass.'
    run = F/evidence['run']
    hashes = evidence['input_output_hashes']
    assert all(sha256(run/n) == h for n, h in hashes.items())
    records = json.loads((run/'fields.json').read_text())['records']
    assert len(records) == evidence['accepted_frames']
    times = np.array([r['time'] for r in records])
    assert np.all(np.diff(times) > 0) and abs(times[-1]-1.) < 1e-12
    raw = np.memmap(run/'fields.bin', dtype='<f8', mode='r')
    return evidence, run, records, raw


def field(raw, rec, name, width):
    m = rec['fields'][name]
    values = raw[m['offset']//8:m['offset']//8+m['count']]
    if name == 'u':
        return values.reshape((-1, width))[:, 1:]
    return values.reshape((rec['elements'], rec['ips'], width))[:, :8]


def retained_work(evidence, run):
    lane = 'quarter' if run.parent.name.startswith('plastic_state_quarter_') else 'standard'
    work_path = F/'results/output'/f'internal_pipe_work_verified_{lane}_20261011T093005Z.json'
    work = json.loads(work_path.read_text())
    capture_path = F/'results/output'/f'{run.parent.name}_audit.json'
    capture = json.loads(capture_path.read_text())
    assert capture['checks_passed'] and work['field_checks_passed']
    assert capture['identical_parent_raw_fields'] and capture['identical_parent_dat_sta']
    assert all(capture['input_output_hashes'][n] == h
               for n, h in evidence['input_output_hashes'].items())
    assert capture['retained_field_evidence_sha256'] == sha256(work_path)
    parent = F/work['run']
    assert all(sha256(parent/n) == h for n, h in work['input_output_hashes'].items())
    endpoint = work['history'][-1]
    assert abs(endpoint['internal_energy_n_mm']-evidence['history'][-1]['native_energy_n_mm']) < 1e-4
    return dict(evidence=str(work_path.relative_to(F)), sha256=sha256(work_path),
                endpoint=endpoint,
                interpretation='External Gauss2 work and linear material work follow piecewise linear output paths; these are assumptions, not recovered intermediate native trajectories.')


def compare(standard=STANDARD, quarter=QUARTER):
    a, ar, records_a, raw_a = load(standard)
    b, br, records_b, raw_b = load(quarter)
    assert all(a['input_output_hashes'][n] == b['input_output_hashes'][n]
               for n in ('mesh.inp', 'model.json', 'case.json'))
    deck_a, deck_b = (ar/'job.inp').read_text(), (br/'job.inp').read_text()
    if standard != quarter:
        old = '0.02291666666666667, 1., 1e-06, 0.02291666666666667'
        new = '0.005729166666666668, 1., 1e-06, 0.005729166666666668'
        assert deck_a.count(old) == 1 and deck_b == deck_a.replace(old, new)
    else:
        assert deck_a == deck_b
    assert a['source_sha256'] == b['source_sha256']
    assert a['return_map_script_sha256'] == b['return_map_script_sha256']
    assert a['fixed_limits'] == b['fixed_limits']
    times_b = np.array([r['time'] for r in records_b])
    history = []
    for ra in records_a:
        j = int(np.argmin(abs(times_b-ra['time'])))
        rb = records_b[j]
        # Numerical time-match slack only; never interpolate constitutive states.
        if abs(rb['time']-ra['time']) > 1e-10:
            continue
        assert (ra['elements'], ra['ips'], ra['states'], ra['nodes']) == (
            rb['elements'], rb['ips'], rb['states'], rb['nodes'])
        differences = {}
        for name, width in (('u', 4), ('stress', 6), ('strain', 6),
                            ('energy', 2), ('state', ra['states'])):
            fa, fb = field(raw_a, ra, name, width), field(raw_b, rb, name, width)
            assert np.isfinite(fa).all() and np.isfinite(fb).all()
            differences[name] = float(abs(fb-fa).max())
        sa = field(raw_a, ra, 'state', ra['states'])
        sb = field(raw_b, rb, 'state', rb['states'])
        def hotspot(s):
            e, ip = np.unravel_index(np.argmax(s[..., 0]), s[..., 0].shape)
            return dict(element=int(e)+1, integration_point=int(ip)+1,
                        peeq=float(s[e, ip, 0]))
        history.append(dict(time_standard=ra['time'], time_quarter=rb['time'],
                            pressure_mpa=24*ra['time'],
                            maximum_absolute_field_differences=differences,
                            maximum_pointwise_peeq_difference=float(abs(sb[..., 0]-sa[..., 0]).max()),
                            standard_hotspot=hotspot(sa), quarter_hotspot=hotspot(sb)))
    assert history and abs(history[-1]['time_standard']-1.) < 1e-12
    work_a, work_b = retained_work(a, ar), retained_work(b, br)
    for evidence, run in ((a, ar), (b, br)):
        assert all(sha256(run/n) == h for n, h in evidence['input_output_hashes'].items())
    return dict(provenance_checks_passed=True, physical_or_convergence_acceptance=False,
                standard_gate=str(standard.relative_to(F)), quarter_gate=str(quarter.relative_to(F)),
                gate_hashes=[sha256(standard), sha256(quarter)],
                runs=[a['run'], b['run']], accepted_frames=[len(records_a), len(records_b)],
                shared_accepted_times=len(history), shared_time_numerical_slack=1e-10,
                unmatched_standard_frames=len(records_a)-len(history), history=history,
                endpoint_native_energy_n_mm=[e['history'][-1]['native_energy_n_mm'] for e in (a, b)],
                endpoint_native_energy_difference_n_mm=b['history'][-1]['native_energy_n_mm']-a['history'][-1]['native_energy_n_mm'],
                retained_work_evidence=[work_a, work_b],
                endpoint_work_differences_n_mm={key:work_b['endpoint'][key]-work_a['endpoint'][key]
                    for key in ('internal_energy_n_mm', 'linear_output_path_material_work_n_mm',
                                'external_minus_native_n_mm', 'external_minus_linear_material_n_mm')},
                script_sha256=sha256(__file__),
                limitations='Only shared native accepted times; no interpolation or extrapolation of plastic states. Stress is captured Cauchy MPa, strain is Green strain, energy array includes native fields, state difference covers captured constitutive variables. Two time resolutions do not establish asymptotic convergence, physical dissipation, cap/contact/MPC/Abaqus transfer or burst. Work-path gaps require independent temporal-path analysis; no tolerance fitted.')


if __name__ == '__main__':
    out = Path(sys.argv[1])
    assert not out.exists()
    report = compare()
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'history'}, indent=2))

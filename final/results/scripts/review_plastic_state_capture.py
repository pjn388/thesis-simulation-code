"""Capture missing constitutive states, gated by an independent plastic coupon."""
import json
import sys
import numpy as np
import review_internal_pipe_fields as base
from pipe_sim.results import dat_blocks
from pipe_sim.runner import sha256
F = base.F
TAG = 'plastic_state_capture_20261011T095613Z'
CAP = F/'results/scripts/capture_plastic_state_gdb.py'


def state_audit(run):
    records = json.loads((run/'fields.json').read_text())['records']
    raw = np.memmap(run/'fields.bin', dtype='<f8', mode='r')
    printed = [(t, rows) for kind, group, t, rows in dat_blocks(run/'job.dat') if kind == 'PEEQ']
    maximum = 0.
    for record in records:
        meta = record['fields']['state']
        state = raw[meta['offset']//8:meta['offset']//8+meta['count']].reshape((record['elements'], record['ips'], record['states']))
        assert np.isfinite(state).all() and record['states'] >= 13
        time, rows = min(printed, key=lambda item:abs(item[0]-record['time']))
        assert abs(time-record['time']) < 1e-6
        for row in rows:
            value = row[-1]
            actual = state[int(row[0])-1, int(row[1])-1, 0]
            allowance = 0. if value == 0 else .5*10**(int(np.floor(np.log10(abs(value))))-6)
            excess = abs(actual-value)-allowance
            maximum = max(maximum, excess)
            assert excess < 1e-12
    return dict(accepted_frames=len(records),maximum_peeq_rounding_excess=maximum,minimum_states=min(record['states'] for record in records),checks_passed=True)


def coupon():
    import verify_internal_work_coupon as old
    old.TAG = 'plastic_state_coupon_20261011T095613Z'
    old.R = F/'sym/runs/investigation'/old.TAG
    old.CAP = CAP
    old.run()
    result = state_audit(old.R)
    result.update(run=str(old.R.relative_to(F)),capture_sha256=sha256(CAP),fields_sha256=sha256(old.R/'fields.bin'),index_sha256=sha256(old.R/'fields.json'),source_sha256=sha256(F/'sym/vendor/CalculiX/ccx_2.23/src/printout.f'),limitations='PEEQ correspondence verifies state pointer/stride only. Other state meanings follow incplas source and require constitutive replay before physical/path acceptance.')
    out = F/'results/output/plastic_state_coupon_20261011T095613Z_state.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


def paths(lane):
    return TAG, F/'sym/runs/investigation'/TAG/'intact_support_bounded_20261010T170002Z'


def configure():
    base.paths = paths
    base.CAPTURE = CAP


def prepare():
    gate = json.loads((F/'results/output/plastic_state_coupon_20261011T095613Z_state.json').read_text())
    assert gate['checks_passed'] and gate['capture_sha256'] == sha256(CAP)
    decomposition = json.loads((F/'results/output/stress_path_decomposition_standard_20261011T095613Z.json').read_text())
    assert decomposition['checks_passed']
    configure()
    base.prepare('standard')
    tag, run = paths('standard')
    path = run/'diagnostic_revision.json'
    manifest = json.loads(path.read_text())
    manifest.update(change='New output-only xstate capture of a completed24MPa parent, with all frozen inputs identical. Captures missing plastic inverse-metric/backstress/PEEQ states needed to investigate actual constitutive path; no rerun of failed input.',acceptance='Require coupon PEEQ pointer/stride verification, identical parent DAT/STA and raw U/S/E/energy arrays against retained standard field capture, literal PEEQ intervals, finite states, input/solver hashes. Capture alone does not accept plastic trajectory or change production loads.',successor='Audit states and formulate source-consistent constitutive replay before any new pressure/refinement/cap revision.',state_coupon_sha256=sha256(F/'results/output/plastic_state_coupon_20261011T095613Z_state.json'),state_sources=['printout.f:19-25,49-50 xstate argument16/state count19','incplas.f:65-74,1029-1032 state interpretation'])
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    (F/'results/output'/f'{tag}_launch.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest))


def run():
    configure()
    base.run('standard')


def audit(tag=None, run=None, retained_path=None):
    if tag is None:
        tag, run = paths('standard')
    terminal_path = F/'results/output'/f'{tag}_terminal.json'
    terminal = json.loads(terminal_path.read_text())
    assert terminal['completed_native']
    assert sha256(CAP) == terminal['capture_script_sha256']
    hashes = {**terminal['input_hashes'], **terminal['output_hashes']}
    assert all(sha256(run/name) == value for name, value in hashes.items())
    if retained_path is None:
        retained_path = F/'results/output/internal_pipe_work_verified_standard_20261011T093005Z.json'
    retained = json.loads(retained_path.read_text())
    parent = F/retained['run']
    assert retained['field_checks_passed']
    assert all(sha256(parent/name) == value for name, value in retained['input_output_hashes'].items())
    assert terminal['input_hashes'] == {name:retained['input_output_hashes'][name] for name in terminal['input_hashes']}
    assert all((run/name).read_bytes() == (parent/name).read_bytes() for name in ('job.dat','job.sta'))
    records = json.loads((run/'fields.json').read_text())['records']
    prior = json.loads((parent/'fields.json').read_text())['records']
    raw = np.memmap(run/'fields.bin', dtype='<f8', mode='r')
    old = np.memmap(parent/'fields.bin', dtype='<f8', mode='r')
    assert len(records) == len(prior)
    for current, previous in zip(records, prior):
        assert current['time'] == previous['time']
        for name in ('u','stress','strain','energy'):
            a, b = current['fields'][name], previous['fields'][name]
            assert a['count'] == b['count']
            assert np.array_equal(raw[a['offset']//8:a['offset']//8+a['count']], old[b['offset']//8:b['offset']//8+b['count']])
    result = state_audit(run)
    result.update(run=terminal['run'],identical_parent_dat_sta=True,identical_parent_raw_fields=True,input_output_hashes=hashes,terminal_sha256=sha256(terminal_path),retained_field_evidence_sha256=sha256(retained_path),script_sha256=sha256(__file__),limitations='State capture accepted only as evidence acquisition; actual plastic constitutive replay/path, general support/cap/contact/MPC/Abaqus and physical validation remain unresolved.')
    assert all(sha256(run/name) == value for name, value in hashes.items())
    out = F/'results/output'/f'{tag}_audit.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    {'coupon':coupon,'prepare':prepare,'run':run,'audit':audit}[sys.argv[1]]()

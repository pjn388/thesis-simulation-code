"""Output-only quarter-step state acquisition, gated by passed pipe replay."""
import json
import sys
import review_internal_pipe_fields as base
import review_plastic_state_capture as state
from pipe_sim.runner import sha256

TAG = 'plastic_state_quarter_20261011T102248Z'


def paths(lane):
    return TAG, base.F/'sym/runs/investigation'/TAG/'intact_support_bounded_20261010T170002Z'


def configure():
    base.paths = paths
    base.CAPTURE = state.CAP


def prepare():
    gate_path = base.F/'results/output/plastic_pipe_replay_20261011T101242Z.json'
    gate = json.loads(gate_path.read_text())
    assert gate['checks_passed']
    assert all(sha256(base.F/gate['run']/name) == value for name, value in gate['input_output_hashes'].items())
    assert gate['return_map_script_sha256'] == sha256(base.F/'results/scripts/audit_plastic_return_map.py')
    configure()
    base.prepare('quarter')
    tag, run = paths('quarter')
    path = run/'diagnostic_revision.json'
    manifest = json.loads(path.read_text())
    manifest.update(
        pipe_replay_gate_sha256=sha256(gate_path),
        change='Output-only plastic-state acquisition from retained successful quarter-step24MPa parent. All frozen inputs identical; captures actual native temporal states rather than assumed interpolated trajectory.',
        acceptance='Require parent-identical DAT/STA, raw U/S/E/energy identity with retained quarter internal fields, finite13-state capture and literal PEEQ intervals, immutable hashes; then independently replay previous-native and accumulated metric/PEEQ/stress and native energy under unchanged fixed tolerances. Compare actual temporal work against standard at common pressures without fitting.',
        successor='Audit immutable terminal capture and quarter constitutive replay before native temporal comparison. No unconditional further native run.',
        limitations='Evidence acquisition only; physical dissipation, temporal convergence, coupled cap tangent, contact/MPC and Abaqus equivalence remain unresolved.')
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    (base.F/'results/output'/f'{tag}_launch.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


def run():
    configure()
    base.run('quarter')


if __name__ == '__main__':
    {'prepare': prepare, 'run': run}[sys.argv[1]]()

"""One constitutive-control diagnostic; removes plasticity only, no production edit."""
import json,re,shutil,sys
from pathlib import Path
import review_internal_pipe_fields as base
from pipe_sim.runner import sha256
F=base.F
TAG='elastic_work_control_20261011T093005Z'
R=F/'sym/runs/investigation'/TAG/'intact_support_bounded_20261010T170002Z'
PARENT=F/'results/output/internal_pipe_fields_standard_20261011T090414Z_terminal.json'
def configure(lane):
    global TAG,R,PARENT
    assert lane in ('standard','quarter')
    TAG='elastic_work_control'+('_quarter' if lane=='quarter' else '')+'_20261011T093005Z'
    R=F/'sym/runs/investigation'/TAG/'intact_support_bounded_20261010T170002Z'
    PARENT=F/'results/output'/f'internal_pipe_fields_{lane}_20261011T090414Z_terminal.json'
def paths(lane):return TAG,R
def prepare():
    assert not base.paused()
    parent=json.loads(PARENT.read_text());p=F/parent['run'];assert parent['completed_native']
    assert all(sha256(p/n)==h for n,h in {**parent['input_hashes'],**parent['output_hashes']}.items())
    lane='quarter' if '_quarter_' in TAG else 'standard'
    passed=json.loads((F/'results/output'/f'internal_pipe_work_verified_{lane}_20261011T093005Z.json').read_text());assert passed['field_checks_passed']
    R.mkdir(parents=True,exist_ok=False)
    for n in parent['input_hashes']:shutil.copyfile(p/n,R/n)
    old=(p/'job.inp').read_text();new,count=re.subn(r'\*PLASTIC\n[^*]*','',old);assert count==1
    (R/'job.inp').write_text(new)
    case=json.loads((p/'case.json').read_text());case['material']['plastic']=[]
    (R/'case.json').write_text(json.dumps(case,indent=2)+'\n')
    assert (R/'mesh.inp').read_bytes()==(p/'mesh.inp').read_bytes() and (R/'model.json').read_bytes()==(p/'model.json').read_bytes()
    assert new==old[:old.index('*PLASTIC\n')]+old[old.index('*SOLID SECTION'):]
    report=dict(parent=parent['run'],run=str(R.relative_to(F)),parent_report_sha256=sha256(PARENT),input_hashes={n:sha256(R/n) for n in parent['input_hashes']},solver_sha256=parent['solver_sha256'],capture_script_sha256=sha256(base.CAPTURE),threads=1,native_timeout_seconds=3000,memory_limit_mib=11000,change='Remove only PIPE_STEEL PLASTIC block and empty case.material.plastic; preserve E/nu,24MPa loads,NLGEOM,controls,mesh,fixtures and output. model.json retained as parent geometry provenance; constitutive override is explicit frozen job/case. Elastic control is artificial diagnostic, not proposed pipe material.',acceptance='Immutable parent/deck hashes; exact plastic-block-only deck diff; serial capture STA and literal native-coordinate U/S/E/ELSE intervals, zero constrainedU,PEEQ zero, independent C3D20R Green/material energy reconstruction. Compare external/native and linear-output-path material work gaps with plastic standard parent; histories differ intentionally under one constitutive control, not expected response equality.',limitations='Elastic control cannot validate plasticity or physical burst. Piecewise linear output trajectory remains an assumption; cap/contact/MPC/Abaqus transfer unresolved.',successor='No unconditional next solve; assess elastic versus plastic gap and quarter field audit before further diagnostics.')
    (R/'diagnostic_revision.json').write_text(json.dumps(report,indent=2)+'\n');(R/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(base.CAPTURE)+'\nquit\n')
    (F/'results/output'/f'{TAG}_launch.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
def run():
    base.paths=paths
    base.run('elastic')
if __name__=='__main__':
    configure(sys.argv[2] if len(sys.argv)>2 else 'standard')
    {'prepare':prepare,'run':run}[sys.argv[1]]()

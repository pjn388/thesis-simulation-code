"""Third temporal resolution only after paired conservative elastic audits pass."""
import json,shutil,sys
import review_internal_pipe_fields as base
from pipe_sim.runner import sha256
F=base.F
TAG='elastic_work_control_half_20261011T094510Z'
R=F/'sym/runs/investigation'/TAG/'intact_support_bounded_20261010T170002Z'
PARENT=F/'results/output/elastic_work_control_20261011T093005Z_terminal.json'
def paths(lane):return TAG,R

def prepare():
    assert not base.paused()
    standard=json.loads((F/'results/output/elastic_conservative_path_elastic_20261011T094510Z.json').read_text())
    quarter=json.loads((F/'results/output/elastic_conservative_path_elastic_quarter_20261011T094510Z.json').read_text())
    assert standard['checks_passed'] and quarter['checks_passed']
    for lane in ('elastic','elastic_quarter'):
        force=json.loads((F/'results/output'/f'internal_endpoint_force_componentwise_{lane}_20261011T093005Z.json').read_text());assert force['diagnostic_limit_passed']
    # A third resolution tests observed elastic path convergence, never fits acceptance limits.
    assert 0<abs(quarter['history'][-1]['external_minus_conservative_path_n_mm'])<abs(standard['history'][-1]['external_minus_conservative_path_n_mm'])
    parent=json.loads(PARENT.read_text());p=F/parent['run'];assert parent['completed_native']
    assert all(sha256(p/n)==h for n,h in {**parent['input_hashes'],**parent['output_hashes']}.items())
    old=(p/'job.inp').read_text();oldline='0.02291666666666667, 1., 1e-06, 0.02291666666666667';newline='0.011458333333333334, 1., 1e-06, 0.011458333333333334'
    assert old.count(oldline)==1
    R.mkdir(parents=True,exist_ok=False)
    for n in parent['input_hashes']:shutil.copyfile(p/n,R/n)
    (R/'job.inp').write_text(old.replace(oldline,newline))
    assert all((p/n).read_bytes()==(R/n).read_bytes() for n in parent['input_hashes'] if n!='job.inp')
    m=dict(parent=parent['run'],run=str(R.relative_to(F)),parent_report_sha256=sha256(PARENT),input_hashes={n:sha256(R/n) for n in parent['input_hashes']},solver_sha256=parent['solver_sha256'],capture_script_sha256=sha256(base.CAPTURE),threads=1,native_timeout_seconds=3000,memory_limit_mib=11000,change='Only initial and maximum normalized STATIC increment halved versus elastic standard; geometry,elastic material,24MPa loads,controls,mesh,fixtures and outputs preserved.',acceptance='Immutable hashes and exact STATIC-line-only isolation; accepted-frame/raw interval/Green/elastic S(E)/conservative energy/endpoint componentwise equilibrium checks under unchanged limits. Compare standard,half,quarter work gaps,actual step sequences and common-pressure elastic responses without fitting; quantify temporal scaling, not physical validation.',limitations='Artificial elastic diagnostic; does not establish plastic path convergence,cap/contact/MPC/Abaqus equivalence or physical burst.',successor='No unconditional further refinement; third resolution completes bounded elastic path diagnosis, choose supported plastic constitutive-path investigation next.',paired_standard_evidence_sha256=sha256(F/'results/output/elastic_conservative_path_elastic_20261011T094510Z.json'),paired_quarter_evidence_sha256=sha256(F/'results/output/elastic_conservative_path_elastic_quarter_20261011T094510Z.json'))
    (R/'diagnostic_revision.json').write_text(json.dumps(m,indent=2)+'\n');(R/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(base.CAPTURE)+'\nquit\n')
    (F/'results/output'/f'{TAG}_launch.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
def run():
    base.paths=paths;base.run('elastic_half')
if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()

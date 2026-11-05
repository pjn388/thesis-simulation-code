"""One bounded diagnostic capture of opt-in native assembly, not pressure acceptance."""
import json,os,shutil,subprocess
from pathlib import Path
from datetime import datetime,timezone
from audit_plastic_return_map import F,sha256
from terminal_progress import solver_directories
ROOT=F/'sym/runs/investigation/volume_material_capture_v5_20261019'
OUT=F/'results/output/volume_material_capture_v5_20261019.json'
BUILD=F/'sym/vendor/ccx_volume_material_v5_20261019'
def main():
    assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
    live=solver_directories(set(),run_root=F/'sym');assert live is not None and sum(len(v['pids']) for v in live.values())<2
    assert int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))>5*1024**2
    assert shutil.disk_usage(F).free>25*1024**3
    prerequisite=F/'results/output/volume_material_integration_v5_20261019.json';assert json.loads(prerequisite.read_text())['checks_passed']
    ROOT.mkdir(exist_ok=False);parent=F/'sym/runs/campaign/orientation_intact_00_adaptive'
    exe=BUILD/'ccx_2.23_material';metadata=json.loads((BUILD/'build.json').read_text());assert sha256(exe)==metadata['solver_sha256']
    manifest=json.loads((parent/'manifest.json').read_text())
    for name,h in manifest['files'].items():assert sha256(parent/name)==h;shutil.copy2(parent/name,ROOT/name)
    source=Path(__file__).with_name('capture_pipe_solver_gdb.py');code=source.read_text()
    anchor="snapshot['global_matrix']=dict(neq=n,nzs=nz,ad=read(p(1),n,'d'),au=read(p(2),nz,'d'),"
    assert code.count(anchor)==1
    code=code.replace(anchor,"snapshot['global_matrix']=dict(neq=n,nzs=nz,ad=read(p(1),n,'d'),au_upper=read(p(2)+8*nz,nz,'d'),au=read(p(2),nz,'d'),")
    anchor="Results('*results',internal=True);Residual('*calcresidual',internal=True);Solver('*pardiso_main',internal=True)"
    addition='''
class ScatterDone(gdb.FinishBreakpoint):
    def __init__(self,addresses):
        super().__init__(gdb.newest_frame(),internal=True)
        self.saved=[(a,n,read(a,n,'d')) for a,n in addresses]
    def stop(self):
        assert all(read(a,n,'d')==v for a,n,v in self.saved), 'Assembly modified native geometry/state'
        snapshot['assembly_state_restoration']=True
        return False
class Scatter(gdb.Breakpoint):
    def stop(self):
        if not selected:return False
        p=args()
        ScatterDone([(p(3),snapshot['nodal_count']),(p(8),snapshot['state_count']),(p(9),snapshot['state_count'])])
        return False
Scatter('*thesis_material_scatter',internal=True)
'''
    assert code.count(anchor)==1;code=code.replace(anchor,addition+anchor)
    capture=ROOT/'capture_native.py';capture.write_text(code)
    (ROOT/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
    report=dict(state='running',checks_passed=False,physical_validation=False,pressure_authorized=False,run=str(ROOT.relative_to(F)),timeout_seconds=900,sources={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),source,capture,exe,BUILD/'build.json',prerequisite,parent/'manifest.json')})
    def save():OUT.write_text(json.dumps(report,indent=2)+'\n');(ROOT/'status.json').write_text(json.dumps(dict(state=report['state'],kind='bounded_native_diagnostic',supervisor_pid=os.getpid()),indent=2)+'\n')
    save()
    try:
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CCX_NPROC_RESULTS='1',CCX_NPROC_STIFFNESS='1',CCX_NPROC_EQUATION_SOLVER='1',THESIS_VOLUME_MATERIAL='1',THESIS_MIN_TIME='.75049',THESIS_OUTPUT=str(ROOT/'assembly.json'))
        with (ROOT/'solver.log').open('w') as log:subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(exe),'-i','job'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
        b=json.loads((ROOT/'assembly.json').read_text());g=b['global_matrix']
        assert g['symmetryflag']==2 and g['inputformat']==1 and b['assembly_state_restoration']
        assert g['rhs']==b['residual']['b']
        assert b['displacement_after']==b['residual']['displacement']
        report.update(state='completed',checks_passed=True,time=b['time'],cycle=b['cycle'],assembly_sha256=sha256(ROOT/'assembly.json'),assembly_state_restoration=True)
    except Exception as exc:report.update(state='failed',error=str(exc));raise
    finally:
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),limitations='Captures selected revised-trajectory matrix and checks assembly geometry/old/current state immutability. Does not verify native residual-evaluation rollback or full-direction finite differences. No pressure-path improvement, stability or physical burst acceptance. Guard failures are retained, never bypassed.')
        save()
if __name__=='__main__':main()

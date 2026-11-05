"""Bounded unchanged24MPa raw-output diagnostics; no production revision."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from pipe_sim.runner import sha256
from campaign import request_capacity_review
F=Path(__file__).resolve().parents[2]
EXE=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
CAPTURE=F/'results/scripts/capture_raw_printout_gdb.py'
PARENTS={'standard':'plastic_tangential_work_p24_20261011T070440Z','quarter':'plastic_tangential_work_quarterdt_p24_20261011T072144Z'}
def paused():return json.loads((F/'results/automation/paused.json').read_text())['state']!='resumed'
def paths(lane):
    tag=f'raw_pipe_precision_{lane}_20261011T083005Z'
    return tag,F/'sym/runs/investigation'/tag/'intact_support_bounded_20261010T170002Z'
def prepare(lane):
    assert not paused()
    parent_report_path=F/'results/output'/f'{PARENTS[lane]}_terminal.json'
    parent=json.loads(parent_report_path.read_text());p=F/parent['run']
    assert parent['completed'] and all(sha256(p/n)==h for n,h in {**parent['input_hashes'],**parent['output_hashes']}.items())
    assert sha256(EXE)==parent['solver_sha256']
    coupon=json.loads((F/'results/output/raw_printout_coupon_20261011T083005Z.json').read_text())
    assert coupon['identical_parent_dat'] and coupon['accepted_frames']==80
    tag,r=paths(lane);r.mkdir(parents=True,exist_ok=False)
    for n in parent['input_hashes']:shutil.copyfile(p/n,r/n)
    (r/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(CAPTURE)+'\nquit\n')
    report=dict(parent=str(p.relative_to(F)),run=str(r.relative_to(F)),parent_report_sha256=sha256(parent_report_path),input_hashes={n:sha256(r/n) for n in parent['input_hashes']},solver_sha256=sha256(EXE),capture_script_sha256=sha256(CAPTURE),threads=1,native_timeout_seconds=3000,memory_limit_mib=11000,change='Identical frozen parent input; serial GDB output-only raw U/time all print frames and endpoint-only integrated element energy. No pressure, control, material, geometry or restraint change.',acceptance='STA accepted-frame identity, literal U/time/ELSE rounding intervals, parent crown/PEEQ agreement at common states, constrained U zero; quantify raw-versus-rounded Gauss2 work and energy gap with bore/both-end terms. Quantify cumulative-difference floating precision. No fitted tolerance or whole-workflow acceptance.',limitations='Nonlinear path and internal-energy integration errors, cap tangent/work, contact/MPC and Abaqus equivalence remain unresolved.',successor='No further native successor before raw evidence audit; production extensions remain blocked.')
    assert report['input_hashes']==parent['input_hashes']
    (r/'diagnostic_revision.json').write_text(json.dumps(report,indent=2)+'\n')
    (F/'results/output'/f'{tag}_launch.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
def run(lane):
    tag,r=paths(lane);m=json.loads((r/'diagnostic_revision.json').read_text())
    assert not paused() and not (r/'solver.log').exists()
    assert all(sha256(r/n)==h for n,h in m['input_hashes'].items()) and sha256(EXE)==m['solver_sha256'] and sha256(CAPTURE)==m['capture_script_sha256']
    assert len(subprocess.run(['pgrep','-x','ccx_2.23_pardis'],capture_output=True,text=True).stdout.splitlines())<2
    assert int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))>13048*1024
    assert shutil.disk_usage(F).free>25*1024**3
    env=dict(os.environ,THESIS_RAW_OUTPUT=str(r/'raw.json'),THESIS_RAW_ENDPOINT_ONLY='1',MKL_THREADING_LAYER='GNU',**{n:'1' for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','CCX_NPROC_RESULTS','CCX_NPROC_STIFFNESS','CCX_NPROC_EQUATION_SOLVER')})
    start=time.monotonic();stop_reason=None
    with (r/'solver.log').open('w') as log:
        proc=subprocess.Popen(['gdb','--batch','-x','capture.gdb','--args',str(EXE),'-i','job'],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            native=[]
            for path in Path('/proc').glob('[0-9]*/status'):
                try:
                    content=path.read_text();pid=int(path.parent.name)
                    if os.getpgid(pid)==proc.pid:
                        rss=int(next(l.split()[1] for l in content.splitlines() if l.startswith('VmRSS:')));native.append(rss)
                except (OSError,StopIteration):pass
            if time.monotonic()-start>3000:stop_reason='bounded diagnostic timeout'
            if max(native,default=0)>11000*1024:stop_reason='11000MiB observed RSS guard'
            if stop_reason:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                break
            time.sleep(5)
    log=(r/'solver.log').read_text()
    terminal=dict(**m,elapsed_seconds=time.monotonic()-start,returncode=proc.returncode,stop_reason=stop_reason,completed_native=proc.returncode==0 and 'exited normally' in log and '*ERROR' not in log and 'Python Exception' not in log and (r/'raw.json').exists(),output_hashes={n:sha256(r/n) for n in ('job.dat','job.sta','solver.log','raw.json') if (r/n).exists()},audit_status='Await accepted-frame, parent-response, rounding and work comparison; native completion is not acceptance.')
    (F/'results/output'/f'{tag}_terminal.json').write_text(json.dumps(terminal,indent=2)+'\n')
    print(json.dumps(terminal),flush=True)
    if not paused():request_capacity_review()
if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]](sys.argv[2])

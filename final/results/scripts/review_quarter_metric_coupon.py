"""One bounded quarter-step native coupon, unchanged physical inputs."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from review_plastic_state_capture import CAP, state_audit
from audit_plastic_return_map import F, sha256, tensors, update
from audit_metric_energy_gauge import candidate, MU, BULK
from pipe_sim.results import dat_blocks

TAG='plastic_state_coupon_quarter_20261011T140008Z'
R=F/'sym/runs/investigation'/TAG
P=F/'sym/runs/investigation/plastic_state_coupon_half_20261011T100722Z'
EXE=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'


def run():
    assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
    gate=F/'results/output/metric_energy_gauge_history_20261011T140008Z.json'
    evidence=json.loads(gate.read_text()); assert evidence['manufactured']['checks_passed']
    for t in evidence['trajectories']:
        assert t['prior_native_replay_checks_passed']
        assert all(sha256(F/t['run']/n)==h for n,h in t['input_output_hashes'].items())
    assert sha256(EXE)==json.loads((EXE.parent/'build.json').read_text())['solver_sha256']
    assert len(subprocess.run(['pgrep','-x','ccx_2.23_pardis'],capture_output=True,text=True).stdout.splitlines())<2
    assert shutil.disk_usage(F).free>25*1024**3
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
    assert available>13048*1024
    R.mkdir(exist_ok=False)
    original=(P/'job.inp').read_text()
    old='0.0125,1.,1.e-8,0.0125'; new='0.00625,1.,1.e-8,0.00625'
    assert original.count(old)==2
    deck=original.replace(old,new); assert deck.replace(new,old)==original
    (R/'job.inp').write_text(deck)
    (R/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(CAP)+'\nquit\n')
    manifest=dict(run=str(R.relative_to(F)),parent=str(P.relative_to(F)),parent_input_sha256=sha256(P/'job.inp'),input_sha256=sha256(R/'job.inp'),solver_sha256=sha256(EXE),capture_sha256=sha256(CAP),gate_sha256=sha256(gate),script_sha256=sha256(__file__),threads=1,native_timeout_seconds=120,change='Only both STATIC initial/maximum increments halve from0.0125 to0.00625; same geometry/material/loads/unload/output/controls.320frame state capture.',acceptance='Native normal completion, literal PEEQ state gate, accumulated PK2/metric/PEEQ replay1e-4MPa/1e-8/1e-8, native work replay1e-8Nmm; quarter-half work difference less than half-coarse difference. Report candidate energy and external work without dissipation claim.',limitations='Homogeneous coupon temporal check only; no production, hardening storage, cap/contact/MPC/pipe/Abaqus or experimental acceptance.')
    (R/'diagnostic_revision.json').write_text(json.dumps(manifest,indent=2)+'\n')
    env=dict(os.environ,THESIS_INTERNAL_BINARY=str(R/'fields.bin'),THESIS_INTERNAL_INDEX=str(R/'fields.json'),MKL_THREADING_LAYER='GNU',**{n:'1' for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','CCX_NPROC_RESULTS','CCX_NPROC_STIFFNESS','CCX_NPROC_EQUATION_SOLVER')})
    with (R/'solver.log').open('w') as log:
        result=subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(EXE),'-i','job'],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    log=(R/'solver.log').read_text()
    assert result.returncode==0 and 'exited normally' in log and '*ERROR' not in log and 'Python Exception' not in log
    audit()


def audit():
    manifest=json.loads((R/'diagnostic_revision.json').read_text())
    assert sha256(R/'job.inp')==manifest['input_sha256']
    assert 'exited normally' in (R/'solver.log').read_text() and '*ERROR' not in (R/'solver.log').read_text()
    deck=(R/'job.inp').read_text()
    evidence=json.loads((F/'results/output/metric_energy_gauge_history_20261011T140008Z.json').read_text())
    state_gate=state_audit(R); assert state_gate['accepted_frames']==320
    records=json.loads((R/'fields.json').read_text())['records'];raw=np.memmap(R/'fields.bin',dtype='<f8',mode='r')
    lines=deck.splitlines(); curve=[]
    for line in lines[lines.index('*PLASTIC')+1:]:
        if line.startswith('*'):break
        curve.append(list(map(float,line.split(','))))
    curve=np.array(curve)
    xyz=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    metric=np.broadcast_to(np.eye(3),(1,8,3,3)).copy();peeq=np.zeros((1,8));c0=metric.copy();s0=np.zeros_like(metric)
    work=external=0.; old_u=np.zeros((8,3));old_rf=old_u.copy(); maxima=dict(stress_mpa=0.,metric=0.,peeq=0.,native_work_n_mm=0.)
    printed={}
    for kind,group,t,rows in dat_blocks(R/'job.dat'):
        if kind in ('U','RF'):printed.setdefault(t,{})[kind]=np.array(rows)[:,1:]
    assert len(printed)==320
    history=[]
    for rec in records:
        def field(name,width):
            meta=rec['fields'][name];a=raw[meta['offset']//8:meta['offset']//8+meta['count']]
            return a.reshape((-1,width)) if name in ('u','rf') else a.reshape((1,8,width))
        u=field('u',4)[:,1:]
        t=min(printed,key=lambda t:abs(t-rec['time']))
        assert abs(t-rec['time'])<1e-6
        printed_u=printed[t]['U'];rf=printed[t]['RF']
        coef=np.linalg.lstsq(np.c_[np.ones(8),xyz],u,rcond=None)[0];deformation=np.eye(3)+coef[1:].T
        c=np.broadcast_to(deformation.T@deformation,c0.shape);inverse=np.linalg.inv(deformation)
        actual=np.linalg.det(deformation)*(inverse@tensors(field('stress',6))@inverse.T)
        stress,metric,peeq,_=update(c,metric,peeq,curve,MU,BULK)
        state=field('state',rec['states']);native_metric=np.eye(3)-2*tensors(state[...,1:7])
        work+=float((.25*np.einsum('...ij,...ij->...',stress+s0,c-c0)).mean())
        external+=float((.5*(rf+old_rf)*(printed_u-old_u)).sum())
        native_work=float(field('energy',2)[...,0].mean())
        errors=dict(stress_mpa=float(abs(stress-actual).max()),metric=float(abs(metric-native_metric).max()),peeq=float(abs(peeq-state[...,0]).max()),native_work_n_mm=abs(work-native_work))
        for k,v in errors.items():maxima[k]=max(maxima[k],v)
        energy,_=candidate(c,native_metric)
        history.append(dict(time=rec['time'],native_work_n_mm=native_work,external_work_n_mm=external,candidate_energy_n_mm=float(energy.mean()),peeq=float(peeq.max())))
        c0,s0,old_u,old_rf=c,stress,printed_u,rf
    limits=dict(stress_mpa=1e-4,metric=1e-8,peeq=1e-8,native_work_n_mm=1e-8)
    coarse,half=[t['native_work_n_mm'] for t in evidence['trajectories']]
    decreasing=abs(native_work-half)<abs(half-coarse)
    hashes={n:sha256(R/n) for n in ('job.inp','job.dat','job.sta','solver.log','fields.bin','fields.json')}
    report=dict(**manifest,audit_script_sha256=sha256(__file__),external_work_precision='DAT U/RF rounded; descriptive only, not raw identity',state_gate=state_gate,maximum_errors=maxima,fixed_limits=limits,replay_checks_passed=all(maxima[k]<v for k,v in limits.items()),native_work_series_n_mm=[coarse,half,native_work],temporal_difference_reduced=decreasing,final=history[-1],history=history,hashes=hashes)
    out=F/'results/output'/f'{TAG}_audit.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('history','hashes')},indent=2))
    assert report['replay_checks_passed'] and decreasing


if __name__=='__main__':{'run':run,'audit':audit}[sys.argv[1] if len(sys.argv)>1 else 'run']()
